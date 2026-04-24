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
	"log/slog"
	"strings"

	"github.com/taylorgtyler/agentHarness/pkg/provider"
	"github.com/taylorgtyler/agentHarness/pkg/retry"
	"github.com/taylorgtyler/agentHarness/pkg/types"
)

type partialToolCall struct {
	id        string
	callType  string
	name      string
	arguments strings.Builder
}

func (h *Harness) RunStream(ctx context.Context, task string, onChunk func(string)) (string, error) {
	if onChunk == nil {
		onChunk = func(string) {}
	}

	sp, streamOK := h.provider.(provider.StreamProvider)

	if !streamOK {
		return h.run(ctx, task, "fallback", func(ctx context.Context, _ *slog.Logger, messages []types.Message, tools []types.Tool) (types.Message, error) {
			msg, err := retry.Do(ctx, h.retryCfg, func() (types.Message, error) {
				return h.provider.Invoke(ctx, messages, tools)
			})
			if err == nil && msg.Content != nil && *msg.Content != "" {
				onChunk(*msg.Content)
			}
			return msg, err
		})
	}

	return h.run(ctx, task, "stream", func(ctx context.Context, log *slog.Logger, messages []types.Message, tools []types.Tool) (types.Message, error) {
		var emitted bool
		streamOnChunk := func(delta string) {
			emitted = true
			onChunk(delta)
		}
		msg, err := retry.Do(ctx, h.retryCfg, func() (types.Message, error) {
			m, cerr := consumeStream(ctx, sp, messages, tools, streamOnChunk, log)
			if cerr != nil && emitted {
				return types.Message{}, &terminalStreamError{err: cerr}
			}
			return m, cerr
		})
		if err != nil {
			var t *terminalStreamError
			if errors.As(err, &t) {
				err = t.err
			}
		}
		return msg, err
	})
}

func consumeStream(ctx context.Context, sp provider.StreamProvider, messages []types.Message, tools []types.Tool, onChunk func(string), log *slog.Logger) (types.Message, error) {
	ch, err := sp.InvokeStream(ctx, messages, tools)
	if err != nil {
		return types.Message{}, err
	}

	var content strings.Builder
	var partials []partialToolCall
	var usage *types.Usage

	for {
		select {
		case <-ctx.Done():
			// Producer's send selects on ctx.Done() too, so it will exit on its own
			// once it notices cancellation. Drain defensively in case a chunk is
			// already in flight.
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
				return buildAssistantMessage(content.String(), partials, usage), nil
			}
			if chunk.Err != nil {
				// Abandon the stream — producer may still have unsent chunks.
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
		}
	}
}

func drain(ch <-chan types.StreamChunk) {
	for range ch {
	}
}

func buildAssistantMessage(content string, partials []partialToolCall, usage *types.Usage) types.Message {
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
	return msg
}

// terminalStreamError marks an error from consumeStream as non-retryable because
// onChunk has already emitted content. Retrying would cause the caller to see
// duplicate or divergent content across attempts
type terminalStreamError struct{ err error }

func (e *terminalStreamError) Error() string   { return e.err.Error() }
func (e *terminalStreamError) Unwrap() error   { return e.err }
func (e *terminalStreamError) Retryable() bool { return false }
