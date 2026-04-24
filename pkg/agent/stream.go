package agent

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"golang.org/x/sync/errgroup"

	"github.com/taylorgtyler/agentHarness/pkg/provider"
	"github.com/taylorgtyler/agentHarness/pkg/retry"
	"github.com/taylorgtyler/agentHarness/pkg/tracing"
	"github.com/taylorgtyler/agentHarness/pkg/types"
)

type partialToolCall struct {
	id        string
	callType  string
	name      string
	arguments strings.Builder
}

func (h *Harness) RunStream(ctx context.Context, task string, onChunk func(string)) (string, error) {
	if h.maxSteps <= 0 {
		return "", errors.New("maxSteps must be greater than 0; use WithMaxSteps to configure")
	}
	if onChunk == nil {
		onChunk = func(string) {}
	}

	sp, streamOK := h.provider.(provider.StreamProvider)

	ctx, span := h.tracer.Start(ctx, "agent.run_stream",
		tracing.String("task", task),
		tracing.Int("max_steps", h.maxSteps),
		tracing.String("mode", streamingModeLabel(streamOK)),
	)
	defer span.End()

	log := spanLogger(span, h.log)
	log.InfoContext(ctx, "starting stream run", "task", task, "max_steps", h.maxSteps, "streaming", streamOK)

	messages := make([]types.Message, len(h.messages), len(h.messages)+1)
	copy(messages, h.messages)
	messages = append(messages, types.UserMessage(task))

	tools := make([]types.Tool, 0, len(h.tools))
	for _, t := range h.tools {
		tools = append(tools, t)
	}

	for step := 0; step < h.maxSteps; step++ {
		log.DebugContext(ctx, "invoking provider", "step", step)

		invokeCtx, invokeSpan := h.tracer.Start(ctx, "provider.invoke_stream",
			tracing.Int("step", step),
		)

		var response types.Message
		var err error
		if streamOK {
			var emitted bool
			streamOnChunk := func(delta string) {
				emitted = true
				onChunk(delta)
			}
			response, err = retry.Do(invokeCtx, h.retryCfg, func() (types.Message, error) {
				msg, cerr := consumeStream(invokeCtx, sp, messages, tools, streamOnChunk, log)
				if cerr != nil && emitted {
					return types.Message{}, &terminalStreamError{err: cerr}
				}
				return msg, cerr
			})
			if err != nil {
				var t *terminalStreamError
				if errors.As(err, &t) {
					err = t.err
				}
			}
		} else {
			response, err = retry.Do(invokeCtx, h.retryCfg, func() (types.Message, error) {
				return h.provider.Invoke(invokeCtx, messages, tools)
			})
			if err == nil && response.Content != nil && *response.Content != "" {
				onChunk(*response.Content)
			}
		}
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

func consumeStream(ctx context.Context, sp provider.StreamProvider, messages []types.Message, tools []types.Tool, onChunk func(string), log *slog.Logger) (types.Message, error) {
	ch, err := sp.InvokeStream(ctx, messages, tools)
	if err != nil {
		return types.Message{}, err
	}

	var content strings.Builder
	var partials []partialToolCall
	var usage *types.Usage
	var finish string

	for {
		select {
		case <-ctx.Done():
			go drain(ch)
			return types.Message{}, ctx.Err()
		case chunk, ok := <-ch:
			if !ok {
				for i, p := range partials {
					log.DebugContext(ctx, "tool call assembled",
						"index", i,
						"id", p.id,
						"name", p.name,
						"args", p.arguments.String(),
					)
				}
				return buildAssistantMessage(content.String(), partials, usage, finish), nil
			}
			if chunk.Err != nil {
				go drain(ch)
				return types.Message{}, chunk.Err
			}
			if chunk.ContentDelta != "" {
				content.WriteString(chunk.ContentDelta)
				onChunk(chunk.ContentDelta)
			}
			for _, frag := range chunk.ToolCalls {
				for len(partials) <= frag.Index {
					partials = append(partials, partialToolCall{})
				}
				slot := &partials[frag.Index]
				isFirst := frag.ID != "" || frag.Name != ""
				if frag.ID != "" {
					slot.id = frag.ID
				}
				if frag.Type != "" {
					slot.callType = frag.Type
				}
				if frag.Name != "" {
					slot.name = frag.Name
				}
				if frag.Arguments != "" {
					slot.arguments.WriteString(frag.Arguments)
				}
				if isFirst {
					log.DebugContext(ctx, "tool call started",
						"index", frag.Index,
						"id", slot.id,
						"name", slot.name,
					)
				}
				if frag.Arguments != "" {
					log.DebugContext(ctx, "tool call args fragment",
						"index", frag.Index,
						"name", slot.name,
						"fragment", frag.Arguments,
						"total_len", slot.arguments.Len(),
					)
				}
			}
			if chunk.Usage != nil {
				usage = chunk.Usage
			}
			if chunk.FinishReason != "" {
				finish = chunk.FinishReason
			}
		}
	}
}

func drain(ch <-chan types.StreamChunk) {
	for range ch {
	}
}

func buildAssistantMessage(content string, partials []partialToolCall, usage *types.Usage, finish string) types.Message {
	msg := types.Message{Role: "assistant"}
	if content != "" {
		c := content
		msg.Content = &c
	}
	if len(partials) > 0 {
		calls := make([]types.ToolCall, len(partials))
		for i, p := range partials {
			calls[i] = types.ToolCall{
				ID:   p.id,
				Type: p.callType,
				Function: types.FunctionCall{
					Name:      p.name,
					Arguments: p.arguments.String(),
				},
			}
		}
		msg.ToolCalls = calls
	}
	msg.Usage = usage
	_ = finish
	return msg
}

// terminalStreamError marks an error from consumeStream as non-retryable because
// onChunk has already emitted content. Retrying would cause the caller to see
// duplicate or divergent content across attempts — see docs/streaming-followups.md.
type terminalStreamError struct{ err error }

func (e *terminalStreamError) Error() string   { return e.err.Error() }
func (e *terminalStreamError) Unwrap() error   { return e.err }
func (e *terminalStreamError) Retryable() bool { return false }

func streamingModeLabel(streaming bool) string {
	if streaming {
		return "stream"
	}
	return "fallback"
}

