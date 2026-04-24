/*
Copyright 2026 Taylor Tyler

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package agent

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"golang.org/x/sync/errgroup"

	"github.com/taylorgtyler/agentHarness/pkg/provider"
	"github.com/taylorgtyler/agentHarness/pkg/retry"
	"github.com/taylorgtyler/agentHarness/pkg/tracing"
	"github.com/taylorgtyler/agentHarness/pkg/types"
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

// DefaultMaxSteps is used when the Harness is constructed without calling
// WithMaxSteps. Chosen to cover typical multi-step tool-using agents without
// letting a runaway loop burn tokens indefinitely.
const DefaultMaxSteps = 10

// Sentinel errors returned by Run / RunStream. Callers can use errors.Is to
// distinguish them from provider or tool errors.
var (
	// ErrInvalidMaxSteps is returned when Run is called with maxSteps <= 0.
	ErrInvalidMaxSteps = errors.New("maxSteps must be greater than 0")

	// ErrMaxStepsExceeded is returned when the model keeps requesting tools
	// and the step budget is exhausted before a final response.
	ErrMaxStepsExceeded = errors.New("exceeded max steps")

	// ErrEmptyResponse is returned when the assistant returns a message with
	// neither content nor tool calls — usually a provider bug or an
	// over-aggressive safety filter.
	ErrEmptyResponse = errors.New("assistant returned no content and no tool calls")

	// ErrToolNotFound is returned when the model requests a tool the harness
	// does not have registered.
	ErrToolNotFound = errors.New("tool not found")
)

func New(p provider.Provider) *Harness {
	return &Harness{
		tools:    make(map[string]types.Tool),
		provider: p,
		tracer:   tracing.Noop,
		log:      slog.Default(),
		maxSteps: DefaultMaxSteps,
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

// RunResult is delivered on the channel returned by RunBackground.
type RunResult struct {
	Output string
	Err    error
}

// RunBackground runs the agent in a goroutine and returns a buffered channel
// that receives the result when the run completes. The channel is closed after
// the single result is sent. Cancel the run via ctx.
func (h *Harness) RunBackground(ctx context.Context, task string) <-chan RunResult {
	ch := make(chan RunResult, 1)
	go func() {
		defer close(ch)
		out, err := h.Run(ctx, task)
		ch <- RunResult{Output: out, Err: err}
	}()
	return ch
}

// invokeFn performs one provider turn: send messages/tools, return the assistant
// reply. Implementations own their own retry policy because streaming and
// non-streaming retry differently (see terminalStreamError).
type invokeFn func(ctx context.Context, log *slog.Logger, messages []types.Message, tools []types.Tool) (types.Message, error)

func (h *Harness) Run(ctx context.Context, task string) (string, error) {
	return h.run(ctx, task, "sync", func(ctx context.Context, _ *slog.Logger, messages []types.Message, tools []types.Tool) (types.Message, error) {
		return retry.Do(ctx, h.retryCfg, func() (types.Message, error) {
			return h.provider.Invoke(ctx, messages, tools)
		})
	})
}

func (h *Harness) run(ctx context.Context, task, mode string, invoke invokeFn) (string, error) {
	if h.maxSteps <= 0 {
		return "", fmt.Errorf("%w; use WithMaxSteps to configure", ErrInvalidMaxSteps)
	}

	ctx, span := h.tracer.Start(ctx, "agent.run",
		tracing.String("task", task),
		tracing.Int("max_steps", h.maxSteps),
		tracing.String("mode", mode),
	)
	defer span.End()

	log := spanLogger(span, h.log)
	log.InfoContext(ctx, "starting run", "task", task, "max_steps", h.maxSteps, "mode", mode)

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
		response, err := invoke(invokeCtx, log, messages, tools)
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
				return "", ErrEmptyResponse
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

	err := fmt.Errorf("%w (%d)", ErrMaxStepsExceeded, h.maxSteps)
	span.SetStatus(err)
	log.WarnContext(ctx, "exceeded max steps", "max_steps", h.maxSteps)
	return "", err
}

func (h *Harness) executeTool(ctx context.Context, call types.ToolCall) (string, error) {
	tool, ok := h.tools[call.Function.Name]
	if !ok {
		return "", fmt.Errorf("%w: %q", ErrToolNotFound, call.Function.Name)
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
