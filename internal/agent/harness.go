package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"golang.org/x/sync/errgroup"
)

type Harness struct {
	messages []Message
	tools    map[string]Tool
	model    string
	apiKey   string
	baseURL  string
	maxSteps int
}

type Tool interface {
	Name() string
	Schema() json.RawMessage
	Execute(ctx context.Context, args string) (string, error)
}

func New(model, baseURL string) *Harness {
	return &Harness{
		tools:   make(map[string]Tool),
		model:   model,
		baseURL: baseURL,
	}
}

func (h *Harness) WithAPIKey(key string) *Harness {
	h.apiKey = key
	return h
}

func (h *Harness) WithSystemPrompt(prompt string) *Harness {
	h.messages = append([]Message{SystemMessage(prompt)}, h.messages...)
	return h
}

func (h *Harness) WithMaxSteps(maxSteps int) *Harness {
	h.maxSteps = maxSteps
	return h
}

func (h *Harness) RegisterTool(t Tool) {
	h.tools[t.Name()] = t
}

func (h *Harness) RegisterFunc(name, description string, fn func(context.Context) string) {
	h.RegisterTool(Func(name, description, func(ctx context.Context, _ struct{}) (string, error) {
		return fn(ctx), nil
	}))
}

func (h *Harness) RunBackground(task string) (string, error) {
	return h.Run(context.Background(), task)
}

func (h *Harness) Run(ctx context.Context, task string) (string, error) {
	h.messages = append(h.messages, UserMessage(task))

	for step := 0; step < h.maxSteps; step++ {
		response, err := h.callAPI()
		if err != nil {
			return "", fmt.Errorf("api call failed at step %d: %w", step, err)
		}

		h.messages = append(h.messages, response)

		if len(response.ToolCalls) == 0 {
			if response.Content == nil {
				return "", errors.New("assistant returned no content and no tool calls")
			}
			return *response.Content, nil
		}

		results := make([]Message, len(response.ToolCalls))
		g, gctx := errgroup.WithContext(ctx)
		for i, call := range response.ToolCalls {
			g.Go(func() error {
				result, err := h.executeTool(gctx, call)
				if err != nil {
					return err
				}
				results[i] = ToolResultMessage(call.ID, result)
				return nil
			})
		}
		if err := g.Wait(); err != nil {
			return "", err
		}
		h.messages = append(h.messages, results...)
	}

	return "", fmt.Errorf("exceeded max steps (%d)", h.maxSteps)
}

func (h *Harness) executeTool(ctx context.Context, call ToolCall) (string, error) {
	tool, ok := h.tools[call.Function.Name]
	if !ok {
		return "", fmt.Errorf("tool %q not found", call.Function.Name)
	}

	result, err := tool.Execute(ctx, call.Function.Arguments)
	if err != nil {
		return fmt.Sprintf("error: %s", err.Error()), nil
	}
	return result, nil
}
