package agent

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"golang.org/x/sync/errgroup"

	"github.com/taylortyler/agentHarness/pkg/provider"
	"github.com/taylortyler/agentHarness/pkg/retry"
	"github.com/taylortyler/agentHarness/pkg/tracing"
	"github.com/taylortyler/agentHarness/pkg/types"
)

type Harness struct {
	messages []types.Message
	tools    map[string]types.Tool
	provider provider.Provider
	maxSteps int
	retryCfg retry.Config
	tracer   tracing.Tracer
	log      *slog.Logger
}

func New(p provider.Provider) *Harness {
	return &Harness{
		tools:    make(map[string]types.Tool),
		provider: p,
		tracer:   tracing.Noop,
		log:      slog.Default(),
	}
}

func (h *Harness) WithLogger(l *slog.Logger) *Harness {
	h.log = l
	return h
}

func (h *Harness) WithTracer(t tracing.Tracer) *Harness {
	h.tracer = t
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

func (h *Harness) WithRetry(cfg retry.Config) *Harness {
	h.retryCfg = cfg
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

func (h *Harness) AsTool(name, description string) types.Tool {
	type params struct {
		Task string `json:"task" desc:"The task for the sub-agent to perform"`
	}
	initMessages := make([]types.Message, len(h.messages))
	copy(initMessages, h.messages)

	return Func(name, description, func(ctx context.Context, p params) (string, error) {
		sub := &Harness{
			messages: append([]types.Message(nil), initMessages...),
			tools:    h.tools,
			provider: h.provider,
			maxSteps: h.maxSteps,
			retryCfg: h.retryCfg,
			tracer:   h.tracer,
			log:      h.log,
		}
		return sub.Run(ctx, p.Task)
	})
}

func (h *Harness) RunBackground(task string) (string, error) {
	return h.Run(context.Background(), task)
}

func (h *Harness) Run(ctx context.Context, task string) (string, error) {
	if h.maxSteps <= 0 {
		return "", errors.New("maxSteps must be greater than 0; use WithMaxSteps to configure")
	}

	ctx, span := h.tracer.Start(ctx, "agent.run",
		tracing.String("task", task),
		tracing.Int("max_steps", h.maxSteps),
	)
	defer span.End()

	log := spanLogger(span, h.log)
	log.InfoContext(ctx, "starting run", "task", task, "max_steps", h.maxSteps)

	messages := make([]types.Message, len(h.messages), len(h.messages)+1)
	copy(messages, h.messages)
	messages = append(messages, types.UserMessage(task))

	tools := make([]types.Tool, 0, len(h.tools))
	for _, t := range h.tools {
		tools = append(tools, t)
	}

	for step := 0; step < h.maxSteps; step++ {
		log.DebugContext(ctx, "invoking provider", "step", step)

		invokeCtx, invokeSpan := h.tracer.Start(ctx, "provider.invoke",
			tracing.Int("step", step),
		)
		response, err := retry.Do(invokeCtx, h.retryCfg, func() (types.Message, error) {
			return h.provider.Invoke(invokeCtx, messages, tools)
		})
		if err != nil {
			invokeSpan.RecordError(err)
			invokeSpan.SetStatus(err)
			invokeSpan.End()
			log.ErrorContext(ctx, "provider invocation failed", "step", step, "err", err)
			return "", fmt.Errorf("api call failed at step %d: %w", step, err)
		}
		if response.Usage != nil {
			invokeSpan.SetAttributes(
				tracing.Int("prompt_tokens", response.Usage.PromptTokens),
				tracing.Int("completion_tokens", response.Usage.CompletionTokens),
			)
			log.DebugContext(ctx, "provider response", "step", step,
				"prompt_tokens", response.Usage.PromptTokens,
				"completion_tokens", response.Usage.CompletionTokens,
			)
		}
		invokeSpan.End()

		messages = append(messages, response)

		if len(response.ToolCalls) == 0 {
			if response.Content == nil {
				return "", errors.New("assistant returned no content and no tool calls")
			}
			span.SetAttributes(tracing.Int("steps", step+1))
			log.InfoContext(ctx, "run complete", "steps", step+1)
			return *response.Content, nil
		}

		log.DebugContext(ctx, "executing tools", "step", step, "count", len(response.ToolCalls))
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
			span.RecordError(err)
			span.SetStatus(err)
			return "", err
		}
		messages = append(messages, results...)
	}

	err := fmt.Errorf("exceeded max steps (%d)", h.maxSteps)
	span.SetStatus(err)
	log.WarnContext(ctx, "exceeded max steps", "max_steps", h.maxSteps)
	return "", err
}

func (h *Harness) executeTool(ctx context.Context, call types.ToolCall) (string, error) {
	tool, ok := h.tools[call.Function.Name]
	if !ok {
		return "", fmt.Errorf("tool %q not found", call.Function.Name)
	}

	ctx, span := h.tracer.Start(ctx, "tool.execute",
		tracing.String("tool", call.Function.Name),
	)
	defer span.End()

	log := spanLogger(span, h.log)
	log.DebugContext(ctx, "executing tool", "tool", call.Function.Name, "args", call.Function.Arguments)
	result, err := retry.Do(ctx, h.retryCfg, func() (string, error) {
		return tool.Execute(ctx, call.Function.Arguments)
	})
	if err != nil {
		span.RecordError(err)
		span.SetStatus(err)
		log.WarnContext(ctx, "tool execution error", "tool", call.Function.Name, "err", err)
		return fmt.Sprintf("error: %s", err.Error()), nil
	}
	log.DebugContext(ctx, "tool result", "tool", call.Function.Name, "result", result)
	return result, nil
}

func spanLogger(span tracing.Span, log *slog.Logger) *slog.Logger {
	traceID, spanID := span.TraceIDs()
	if traceID == "" {
		return log
	}
	return log.With("trace_id", traceID, "span_id", spanID)
}
