package agent

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"golang.org/x/sync/errgroup"

	"agentHarness/internal/provider"
	"agentHarness/internal/retry"
	"agentHarness/internal/types"
)

type Harness struct {
	messages []types.Message
	tools    map[string]types.Tool
	provider provider.Provider
	maxSteps int
	retryCfg retry.Config
	tracer   trace.Tracer
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

func (h *Harness) WithTracer(t trace.Tracer) *Harness {
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
	ctx, span := h.startSpan(ctx, "agent.run",
		attribute.String("task", task),
		attribute.Int("max_steps", h.maxSteps),
	)
	defer span.End()

	log := spanLogger(ctx, h.log)
	log.InfoContext(ctx, "starting run", "task", task, "max_steps", h.maxSteps)
	h.messages = append(h.messages, types.UserMessage(task))

	tools := make([]types.Tool, 0, len(h.tools))
	for _, t := range h.tools {
		tools = append(tools, t)
	}

	for step := 0; step < h.maxSteps; step++ {
		log.DebugContext(ctx, "invoking provider", "step", step)

		invokeCtx, invokeSpan := h.startSpan(ctx, "provider.invoke",
			attribute.Int("step", step),
		)
		response, err := retry.Do(invokeCtx, h.retryCfg, func() (types.Message, error) {
			return h.provider.Invoke(invokeCtx, h.messages, tools)
		})
		if err != nil {
			invokeSpan.RecordError(err)
			invokeSpan.SetStatus(codes.Error, err.Error())
			invokeSpan.End()
			log.ErrorContext(ctx, "provider invocation failed", "step", step, "err", err)
			return "", fmt.Errorf("api call failed at step %d: %w", step, err)
		}
		if response.Usage != nil {
			invokeSpan.SetAttributes(
				attribute.Int("prompt_tokens", response.Usage.PromptTokens),
				attribute.Int("completion_tokens", response.Usage.CompletionTokens),
			)
		}
		invokeSpan.End()

		h.messages = append(h.messages, response)

		if len(response.ToolCalls) == 0 {
			if response.Content == nil {
				return "", errors.New("assistant returned no content and no tool calls")
			}
			span.SetAttributes(attribute.Int("steps", step+1))
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
			span.SetStatus(codes.Error, err.Error())
			return "", err
		}
		h.messages = append(h.messages, results...)
	}

	err := fmt.Errorf("exceeded max steps (%d)", h.maxSteps)
	span.SetStatus(codes.Error, err.Error())
	log.WarnContext(ctx, "exceeded max steps", "max_steps", h.maxSteps)
	return "", err
}

func (h *Harness) executeTool(ctx context.Context, call types.ToolCall) (string, error) {
	tool, ok := h.tools[call.Function.Name]
	if !ok {
		return "", fmt.Errorf("tool %q not found", call.Function.Name)
	}

	ctx, span := h.startSpan(ctx, "tool.execute",
		attribute.String("tool", call.Function.Name),
	)
	defer span.End()

	log := spanLogger(ctx, h.log)
	log.DebugContext(ctx, "executing tool", "tool", call.Function.Name, "args", call.Function.Arguments)
	result, err := retry.Do(ctx, h.retryCfg, func() (string, error) {
		return tool.Execute(ctx, call.Function.Arguments)
	})
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		log.WarnContext(ctx, "tool execution error", "tool", call.Function.Name, "err", err)
		return fmt.Sprintf("error: %s", err.Error()), nil
	}
	log.DebugContext(ctx, "tool result", "tool", call.Function.Name, "result", result)
	return result, nil
}

// startSpan starts a span if a tracer is configured, otherwise returns a noop span.
func (h *Harness) startSpan(ctx context.Context, name string, attrs ...attribute.KeyValue) (context.Context, trace.Span) {
	if h.tracer == nil {
		return ctx, trace.SpanFromContext(ctx)
	}
	return h.tracer.Start(ctx, name, trace.WithAttributes(attrs...))
}

// spanLogger returns a logger enriched with trace_id and span_id when a valid span is active,
// correlating log lines with traces in the observability backend.
func spanLogger(ctx context.Context, log *slog.Logger) *slog.Logger {
	sc := trace.SpanFromContext(ctx).SpanContext()
	if !sc.IsValid() {
		return log
	}
	return log.With(
		"trace_id", sc.TraceID().String(),
		"span_id", sc.SpanID().String(),
	)
}
