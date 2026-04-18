package agent

import (
	"encoding/json"
	"errors"
	"fmt"
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
	Execute(args string) (string, error)
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

func (h *Harness) RegisterFunc(name, description string, fn func() string) {
	h.RegisterTool(Func(name, description, func(struct{}) (string, error) {
		return fn(), nil
	}))
}

func (h *Harness) Run(task string) (string, error) {
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

		for _, call := range response.ToolCalls {
			result, err := h.executeTool(call)
			if err != nil {
				return "", err
			}
			h.messages = append(h.messages, ToolResultMessage(call.ID, result))
		}
	}

	return "", fmt.Errorf("exceeded max steps (%d)", h.maxSteps)
}

func (h *Harness) executeTool(call ToolCall) (string, error) {
	tool, ok := h.tools[call.Function.Name]
	if !ok {
		return "", fmt.Errorf("tool %q not found", call.Function.Name)
	}

	result, err := tool.Execute(call.Function.Arguments)
	if err != nil {
		return fmt.Sprintf("error: %s", err.Error()), nil
	}
	return result, nil
}
