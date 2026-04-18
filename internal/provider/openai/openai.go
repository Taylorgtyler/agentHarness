package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"agentHarness/internal/types"
)

type Provider struct {
	model   string
	baseURL string
	apiKey  string
}

func New(model, baseURL string) *Provider {
	return &Provider{model: model, baseURL: baseURL}
}

func (p *Provider) WithAPIKey(key string) *Provider {
	p.apiKey = key
	return p
}

type chatRequest struct {
	Model    string          `json:"model"`
	Messages []types.Message `json:"messages"`
	Tools    []toolSchema    `json:"tools,omitempty"`
}

type toolSchema struct {
	Type     string          `json:"type"`
	Function json.RawMessage `json:"function"`
}

type chatResponse struct {
	Choices []struct {
		Message types.Message `json:"message"`
	} `json:"choices"`
}

func (p *Provider) Invoke(_ context.Context, messages []types.Message, tools []types.Tool) (types.Message, error) {
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

	req, err := http.NewRequest("POST", p.baseURL+"/chat/completions", bytes.NewReader(body))
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
		return types.Message{}, fmt.Errorf("api returned %d: %s", resp.StatusCode, string(respBody))
	}

	var parsed chatResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return types.Message{}, err
	}

	if len(parsed.Choices) == 0 {
		return types.Message{}, errors.New("no choices in response")
	}

	return parsed.Choices[0].Message, nil
}
