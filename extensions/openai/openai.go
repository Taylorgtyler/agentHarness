package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/taylorgtyler/agentHarness/pkg/provider"
	"github.com/taylorgtyler/agentHarness/pkg/types"
)

var _ provider.StreamProvider = (*Provider)(nil)

type Provider struct {
	model       string
	baseURL     string
	apiKey      string
	streamUsage bool
}

func New(model, baseURL string) *Provider {
	return &Provider{model: model, baseURL: baseURL}
}

func (p *Provider) WithAPIKey(key string) *Provider {
	p.apiKey = key
	return p
}

// WithStreamUsage opts into sending `stream_options.include_usage=true` on
// streaming requests so usage data is returned in the final chunk. This is an
// OpenAI-specific extension — some OpenAI-compatible backends reject unknown
// request fields with HTTP 400. Off by default for maximum compatibility.
func (p *Provider) WithStreamUsage() *Provider {
	p.streamUsage = true
	return p
}

type APIError struct {
	StatusCode int
	Body       string
}

func (e *APIError) Error() string   { return fmt.Sprintf("api returned %d: %s", e.StatusCode, e.Body) }
func (e *APIError) Retryable() bool { return e.StatusCode == 429 || e.StatusCode >= 500 }

type chatRequest struct {
	Model         string          `json:"model"`
	Messages      []types.Message `json:"messages"`
	Tools         []toolSchema    `json:"tools,omitempty"`
	Stream        bool            `json:"stream,omitempty"`
	StreamOptions *streamOptions  `json:"stream_options,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type toolSchema struct {
	Type     string          `json:"type"`
	Function json.RawMessage `json:"function"`
}

type chatResponse struct {
	Choices []struct {
		Message types.Message `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

func (p *Provider) Invoke(ctx context.Context, messages []types.Message, tools []types.Tool) (types.Message, error) {
	schemas := make([]toolSchema, 0, len(tools))
	for _, t := range tools {
		schemas = append(schemas, toolSchema{Type: "function", Function: t.Schema()})
	}

	reqBody := chatRequest{
		Model:    p.model,
		Messages: messages,
		Tools:    schemas,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return types.Message{}, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return types.Message{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return types.Message{}, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return types.Message{}, err
	}

	if resp.StatusCode != 200 {
		return types.Message{}, &APIError{StatusCode: resp.StatusCode, Body: string(respBody)}
	}

	var parsed chatResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return types.Message{}, err
	}

	if len(parsed.Choices) == 0 {
		return types.Message{}, errors.New("no choices in response")
	}

	msg := parsed.Choices[0].Message
	if parsed.Usage.PromptTokens > 0 || parsed.Usage.CompletionTokens > 0 {
		msg.Usage = &types.Usage{
			PromptTokens:     parsed.Usage.PromptTokens,
			CompletionTokens: parsed.Usage.CompletionTokens,
		}
	}
	return msg, nil
}

type streamChunkWire struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

func (p *Provider) InvokeStream(ctx context.Context, messages []types.Message, tools []types.Tool) (<-chan types.StreamChunk, error) {
	schemas := make([]toolSchema, 0, len(tools))
	for _, t := range tools {
		schemas = append(schemas, toolSchema{Type: "function", Function: t.Schema()})
	}

	reqBody := chatRequest{
		Model:    p.model,
		Messages: messages,
		Tools:    schemas,
		Stream:   true,
	}
	if p.streamUsage {
		reqBody.StreamOptions = &streamOptions{IncludeUsage: true}
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(respBody)}
	}

	out := make(chan types.StreamChunk)
	go func() {
		defer resp.Body.Close()
		defer close(out)

		send := func(c types.StreamChunk) bool {
			select {
			case out <- c:
				return true
			case <-ctx.Done():
				return false
			}
		}

		reader := bufio.NewReader(resp.Body)
		for {
			lineBytes, readErr := reader.ReadBytes('\n')
			// ReadBytes returns any data accumulated before the error (including
			// the delimiter, if found). Process that data first, then decide
			// whether the error means we stop.
			if len(lineBytes) > 0 {
				line := strings.TrimRight(string(lineBytes), "\r\n")
				if !handleSSELine(line, send) {
					return
				}
			}
			if readErr != nil {
				if !errors.Is(readErr, io.EOF) {
					send(types.StreamChunk{Err: readErr})
				}
				return
			}
		}
	}()

	return out, nil
}

// handleSSELine parses one SSE line and sends a StreamChunk if it yields data.
// Returns false if the consumer has gone away (send failed) or the line was
// malformed and an error was sent — caller should stop reading in either case.
func handleSSELine(line string, send func(types.StreamChunk) bool) bool {
	if !strings.HasPrefix(line, "data:") {
		return true
	}
	payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
	if payload == "" || payload == "[DONE]" {
		return true
	}

	var wire streamChunkWire
	if err := json.Unmarshal([]byte(payload), &wire); err != nil {
		send(types.StreamChunk{Err: fmt.Errorf("decode sse chunk: %w", err)})
		return false
	}

	chunk := types.StreamChunk{}
	if wire.Usage != nil {
		chunk.Usage = &types.Usage{
			PromptTokens:     wire.Usage.PromptTokens,
			CompletionTokens: wire.Usage.CompletionTokens,
		}
	}
	if len(wire.Choices) > 0 {
		choice := wire.Choices[0]
		chunk.ContentDelta = choice.Delta.Content
		chunk.FinishReason = choice.FinishReason
		if len(choice.Delta.ToolCalls) > 0 {
			frags := make([]types.StreamToolCallFragment, len(choice.Delta.ToolCalls))
			for i, tc := range choice.Delta.ToolCalls {
				frags[i] = types.StreamToolCallFragment{
					Index:     tc.Index,
					ID:        tc.ID,
					Type:      tc.Type,
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				}
			}
			chunk.ToolCalls = frags
		}
	}

	if chunk.ContentDelta == "" && len(chunk.ToolCalls) == 0 && chunk.FinishReason == "" && chunk.Usage == nil {
		return true
	}
	return send(chunk)
}
