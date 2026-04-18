package agent

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"golang.org/x/sync/errgroup"

	"agentHarness/internal/provider"
	"agentHarness/internal/types"
)

type Harness struct {
	messages []types.Message
	tools    map[string]types.Tool
	provider provider.Provider
	maxSteps int
	log      *slog.Logger
}

func New(p provider.Provider) *Harness {
	return &Harness{
		tools:    make(map[string]types.Tool),
		provider: p,
		log:      slog.Default(),
	}
}

func (h *Harness) WithLogger(l *slog.Logger) *Harness {
	h.log = l
	return h
}

func (h *Harness) WithSystemPrompt(prompt string) *Harness {
	h.messages = append([]types.Message{types.SystemMessage(prompt)}, h.messages...)
	return h
}

func (h *Harness) WithMaxSteps(maxSteps int) *Harness {
	h.maxSteps = maxSteps
	return h
}

func (h *Harness) RegisterTool(t types.Tool) {
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
	h.log.InfoContext(ctx, "starting run", "task", task, "max_steps", h.maxSteps)
	h.messages = append(h.messages, types.UserMessage(task))

	tools := make([]types.Tool, 0, len(h.tools))
	for _, t := range h.tools {
		tools = append(tools, t)
	}

	for step := 0; step < h.maxSteps; step++ {
		h.log.DebugContext(ctx, "invoking provider", "step", step)
		response, err := h.provider.Invoke(ctx, h.messages, tools)
		if err != nil {
			h.log.ErrorContext(ctx, "provider invocation failed", "step", step, "err", err)
			return "", fmt.Errorf("api call failed at step %d: %w", step, err)
		}

		h.messages = append(h.messages, response)

		if len(response.ToolCalls) == 0 {
			if response.Content == nil {
				return "", errors.New("assistant returned no content and no tool calls")
			}
			h.log.InfoContext(ctx, "run complete", "steps", step+1)
			return *response.Content, nil
		}

		h.log.DebugContext(ctx, "executing tools", "step", step, "count", len(response.ToolCalls))
		results := make([]types.Message, len(response.ToolCalls))
		g, gctx := errgroup.WithContext(ctx)
		for i, call := range response.ToolCalls {
			g.Go(func() error {
				result, err := h.executeTool(gctx, call)
				if err != nil {
					return err
				}
				results[i] = types.ToolResultMessage(call.ID, result)
				return nil
			})
		}
		if err := g.Wait(); err != nil {
			return "", err
		}
		h.messages = append(h.messages, results...)
	}

	h.log.WarnContext(ctx, "exceeded max steps", "max_steps", h.maxSteps)
	return "", fmt.Errorf("exceeded max steps (%d)", h.maxSteps)
}

func (h *Harness) executeTool(ctx context.Context, call types.ToolCall) (string, error) {
	tool, ok := h.tools[call.Function.Name]
	if !ok {
		return "", fmt.Errorf("tool %q not found", call.Function.Name)
	}

	h.log.DebugContext(ctx, "executing tool", "tool", call.Function.Name, "args", call.Function.Arguments)
	result, err := tool.Execute(ctx, call.Function.Arguments)
	if err != nil {
		h.log.WarnContext(ctx, "tool execution error", "tool", call.Function.Name, "err", err)
		return fmt.Sprintf("error: %s", err.Error()), nil
	}
	h.log.DebugContext(ctx, "tool result", "tool", call.Function.Name, "result", result)
	return result, nil
}
