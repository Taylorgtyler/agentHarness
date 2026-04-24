package agent_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/taylorgtyler/agentHarness/pkg/agent"
	"github.com/taylorgtyler/agentHarness/pkg/retry"
	"github.com/taylorgtyler/agentHarness/pkg/types"
)

// retryableError implements retry.Retryable so the retry package will retry it.
type retryableError struct{ msg string }

func (e *retryableError) Error() string   { return e.msg }
func (e *retryableError) Retryable() bool { return true }

// streamProvider is a test double for provider.StreamProvider. Each Invoke /
// InvokeStream call consumes the next scripted response.
type streamProvider struct {
	streams   [][]types.StreamChunk // one slice per call
	fallbacks []types.Message          // used when Invoke is called (non-stream fallback path)
	mu        sync.Mutex
	streamErr error
	delay     time.Duration // per-chunk delay, useful for cancellation tests
}

func (s *streamProvider) Invoke(_ context.Context, _ []types.Message, _ []types.Tool) (types.Message, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.fallbacks) == 0 {
		return types.Message{}, errors.New("no fallback responses")
	}
	r := s.fallbacks[0]
	s.fallbacks = s.fallbacks[1:]
	return r, nil
}

func (s *streamProvider) InvokeStream(ctx context.Context, _ []types.Message, _ []types.Tool) (<-chan types.StreamChunk, error) {
	s.mu.Lock()
	if s.streamErr != nil {
		err := s.streamErr
		s.mu.Unlock()
		return nil, err
	}
	if len(s.streams) == 0 {
		s.mu.Unlock()
		return nil, errors.New("no more stream responses")
	}
	chunks := s.streams[0]
	s.streams = s.streams[1:]
	delay := s.delay
	s.mu.Unlock()

	ch := make(chan types.StreamChunk)
	go func() {
		defer close(ch)
		for _, c := range chunks {
			if delay > 0 {
				select {
				case <-time.After(delay):
				case <-ctx.Done():
					return
				}
			}
			select {
			case ch <- c:
			case <-ctx.Done():
				return
			}
		}
	}()
	return ch, nil
}

// nonStreamProvider satisfies provider.Provider but NOT provider.StreamProvider.
// Used to verify RunStream falls back to Invoke.
type nonStreamProvider struct {
	responses []types.Message
}

func (n *nonStreamProvider) Invoke(_ context.Context, _ []types.Message, _ []types.Tool) (types.Message, error) {
	if len(n.responses) == 0 {
		return types.Message{}, errors.New("no responses")
	}
	r := n.responses[0]
	n.responses = n.responses[1:]
	return r, nil
}

func contentChunk(s string) types.StreamChunk {
	return types.StreamChunk{ContentDelta: s}
}

func toolChunk(frags ...types.StreamToolCallFragment) types.StreamChunk {
	return types.StreamChunk{ToolCalls: frags}
}

func finishChunk(reason string) types.StreamChunk {
	return types.StreamChunk{FinishReason: reason}
}

// TestRunStream_ContentOnly verifies content deltas are forwarded to onChunk in order
// and the full content is returned as the final result.
func TestRunStream_ContentOnly(t *testing.T) {
	p := &streamProvider{
		streams: [][]types.StreamChunk{{
			contentChunk("hello "),
			contentChunk("world"),
			finishChunk("stop"),
		}},
	}

	var got strings.Builder
	h := agent.New(p).WithMaxSteps(5)
	result, err := h.RunStream(context.Background(), "task", func(s string) { got.WriteString(s) })
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "hello world" {
		t.Fatalf("final = %q, want %q", result, "hello world")
	}
	if got.String() != "hello world" {
		t.Fatalf("onChunk accumulated = %q, want %q", got.String(), "hello world")
	}
}

// TestRunStream_SingleToolCallMultiFragmentArgs verifies that a tool call whose
// argument JSON is split across many fragments reassembles into valid JSON that
// the tool can execute.
func TestRunStream_SingleToolCallMultiFragmentArgs(t *testing.T) {
	p := &streamProvider{
		streams: [][]types.StreamChunk{
			{
				// First fragment carries id/type/name, empty args.
				toolChunk(types.StreamToolCallFragment{Index: 0, ID: "call_1", Type: "function", Name: "echo"}),
				// Subsequent fragments carry only argument pieces.
				toolChunk(types.StreamToolCallFragment{Index: 0, Arguments: `{"te`}),
				toolChunk(types.StreamToolCallFragment{Index: 0, Arguments: `xt":"`}),
				toolChunk(types.StreamToolCallFragment{Index: 0, Arguments: `hi"}`}),
				finishChunk("tool_calls"),
			},
			{
				contentChunk("done"),
				finishChunk("stop"),
			},
		},
	}

	var gotArgs string
	h := agent.New(p).WithMaxSteps(5)
	h.RegisterTool(agent.Func("echo", "echo", func(_ context.Context, p struct {
		Text string `json:"text"`
	}) (string, error) {
		gotArgs = p.Text
		return p.Text, nil
	}))

	result, err := h.RunStream(context.Background(), "task", func(string) {})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "done" {
		t.Fatalf("final = %q, want %q", result, "done")
	}
	if gotArgs != "hi" {
		t.Fatalf("tool received args %q, want %q — argument reassembly failed", gotArgs, "hi")
	}
}

// TestRunStream_OnChunkNotCalledForToolCalls verifies that onChunk only fires for
// visible content — tool-call fragments must not leak through as content.
func TestRunStream_OnChunkNotCalledForToolCalls(t *testing.T) {
	p := &streamProvider{
		streams: [][]types.StreamChunk{
			{
				toolChunk(types.StreamToolCallFragment{Index: 0, ID: "c1", Type: "function", Name: "noop", Arguments: "{}"}),
				finishChunk("tool_calls"),
			},
			{
				contentChunk("final"),
				finishChunk("stop"),
			},
		},
	}

	var chunks []string
	h := agent.New(p).WithMaxSteps(5)
	h.RegisterTool(agent.Func("noop", "noop", func(_ context.Context, _ struct{}) (string, error) {
		return "ok", nil
	}))

	if _, err := h.RunStream(context.Background(), "task", func(s string) { chunks = append(chunks, s) }); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Only "final" from the second step should have fired onChunk.
	if len(chunks) != 1 || chunks[0] != "final" {
		t.Fatalf("onChunk fired with %v, want exactly [\"final\"]", chunks)
	}
}

// TestRunStream_MultipleToolCallsOutOfOrderIndexes verifies that when fragments
// arrive for index 1 before index 0 ever sees a subsequent fragment, both slots
// end up with the correct data.
func TestRunStream_MultipleToolCallsOutOfOrderIndexes(t *testing.T) {
	p := &streamProvider{
		streams: [][]types.StreamChunk{
			{
				// Interleaved fragments across two tool calls.
				toolChunk(types.StreamToolCallFragment{Index: 0, ID: "c_a", Type: "function", Name: "toolA"}),
				toolChunk(types.StreamToolCallFragment{Index: 1, ID: "c_b", Type: "function", Name: "toolB"}),
				toolChunk(types.StreamToolCallFragment{Index: 1, Arguments: `{"v":"b"}`}),
				toolChunk(types.StreamToolCallFragment{Index: 0, Arguments: `{"v":"a"}`}),
				finishChunk("tool_calls"),
			},
			{
				contentChunk("done"),
				finishChunk("stop"),
			},
		},
	}

	var mu sync.Mutex
	seen := map[string]string{}
	h := agent.New(p).WithMaxSteps(5)
	for _, name := range []string{"toolA", "toolB"} {
		n := name
		h.RegisterTool(agent.Func(n, n, func(_ context.Context, args struct {
			V string `json:"v"`
		}) (string, error) {
			mu.Lock()
			seen[n] = args.V
			mu.Unlock()
			return "ok", nil
		}))
	}

	if _, err := h.RunStream(context.Background(), "task", func(string) {}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if seen["toolA"] != "a" {
		t.Fatalf("toolA got v=%q, want %q", seen["toolA"], "a")
	}
	if seen["toolB"] != "b" {
		t.Fatalf("toolB got v=%q, want %q", seen["toolB"], "b")
	}
}

// TestRunStream_IndexGrowthFromOne verifies the slice grows correctly when the
// first fragment seen is at a non-zero index (partials[0] gets a zero-value slot,
// partials[1] gets the real data).
func TestRunStream_IndexGrowthFromOne(t *testing.T) {
	p := &streamProvider{
		streams: [][]types.StreamChunk{
			{
				// This is a pathological sequence: no index 0 ever appears. The assembled
				// ToolCall at index 0 will have empty id/name and the harness will reject
				// it via "tool not found". We use this purely to test slice growth doesn't
				// panic.
				toolChunk(types.StreamToolCallFragment{Index: 2, ID: "c_c", Type: "function", Name: "toolC", Arguments: `{}`}),
				finishChunk("tool_calls"),
			},
		},
	}

	h := agent.New(p).WithMaxSteps(5)
	h.RegisterTool(agent.Func("toolC", "toolC", func(_ context.Context, _ struct{}) (string, error) {
		return "ok", nil
	}))

	_, err := h.RunStream(context.Background(), "task", func(string) {})
	// Expected to fail because index 0 slot is empty — tool name "" is unknown.
	// The important assertion is that we did not panic during slice growth.
	if err == nil {
		t.Fatal("expected error from empty tool slot, got nil")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Fatalf("expected tool-not-found error, got: %v", err)
	}
}

// TestRunStream_ErrChunkMidStream verifies that an error chunk returned mid-stream
// causes RunStream to fail and no partial dispatch happens.
func TestRunStream_ErrChunkMidStream(t *testing.T) {
	p := &streamProvider{
		streams: [][]types.StreamChunk{{
			contentChunk("partial "),
			{Err: errors.New("network hiccup")},
		}},
	}

	var toolCalled bool
	h := agent.New(p).WithMaxSteps(5)
	h.RegisterTool(agent.Func("noop", "noop", func(_ context.Context, _ struct{}) (string, error) {
		toolCalled = true
		return "ok", nil
	}))

	_, err := h.RunStream(context.Background(), "task", func(string) {})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "network hiccup") {
		t.Fatalf("expected wrapped error, got: %v", err)
	}
	if toolCalled {
		t.Fatal("tool was executed despite mid-stream error")
	}
}

// TestRunStream_ContextCancellation verifies that cancelling the context mid-stream
// unblocks the consumer and returns ctx.Err().
func TestRunStream_ContextCancellation(t *testing.T) {
	p := &streamProvider{
		streams: [][]types.StreamChunk{{
			contentChunk("a"),
			contentChunk("b"),
			contentChunk("c"),
			finishChunk("stop"),
		}},
		delay: 50 * time.Millisecond,
	}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()

	h := agent.New(p).WithMaxSteps(5)
	_, err := h.RunStream(ctx, "task", func(string) {})
	if err == nil {
		t.Fatal("expected error from cancellation, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled in error chain, got: %v", err)
	}
}

// TestRunStream_NonStreamProviderFallback verifies that a provider without
// InvokeStream falls back to Invoke and still fires onChunk once with the full
// response.
func TestRunStream_NonStreamProviderFallback(t *testing.T) {
	p := &nonStreamProvider{
		responses: []types.Message{contentMsg("all at once")},
	}

	var chunks []string
	h := agent.New(p).WithMaxSteps(5)
	result, err := h.RunStream(context.Background(), "task", func(s string) { chunks = append(chunks, s) })
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "all at once" {
		t.Fatalf("final = %q, want %q", result, "all at once")
	}
	if len(chunks) != 1 || chunks[0] != "all at once" {
		t.Fatalf("onChunk fired with %v, want exactly [\"all at once\"]", chunks)
	}
}

// TestRunStream_MixedContentAndToolCallsSameStep exercises a model that emits
// both visible prose and a tool call in the same assistant turn — content deltas
// go to onChunk, tool fragments accumulate silently, both make it into the
// assembled message.
func TestRunStream_MixedContentAndToolCallsSameStep(t *testing.T) {
	p := &streamProvider{
		streams: [][]types.StreamChunk{
			{
				contentChunk("thinking... "),
				toolChunk(types.StreamToolCallFragment{Index: 0, ID: "c1", Type: "function", Name: "echo", Arguments: `{"text":"x"}`}),
				finishChunk("tool_calls"),
			},
			{
				contentChunk("final answer"),
				finishChunk("stop"),
			},
		},
	}

	var streamed strings.Builder
	var toolRan bool
	h := agent.New(p).WithMaxSteps(5)
	h.RegisterTool(agent.Func("echo", "echo", func(_ context.Context, args struct {
		Text string `json:"text"`
	}) (string, error) {
		toolRan = true
		return args.Text, nil
	}))

	result, err := h.RunStream(context.Background(), "task", func(s string) { streamed.WriteString(s) })
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !toolRan {
		t.Fatal("tool did not execute")
	}
	if result != "final answer" {
		t.Fatalf("final = %q, want %q", result, "final answer")
	}
	// Both "thinking... " (step 1) and "final answer" (step 2) should have been streamed.
	if got := streamed.String(); got != "thinking... final answer" {
		t.Fatalf("streamed content = %q, want %q", got, "thinking... final answer")
	}
}

// TestRunStream_UsagePropagated verifies that usage info from a final chunk
// survives onto the assembled Message.
func TestRunStream_UsagePropagated(t *testing.T) {
	p := &streamProvider{
		streams: [][]types.StreamChunk{{
			contentChunk("ok"),
			{Usage: &types.Usage{PromptTokens: 10, CompletionTokens: 5}, FinishReason: "stop"},
		}},
	}

	h := agent.New(p).WithMaxSteps(5)
	if _, err := h.RunStream(context.Background(), "task", func(string) {}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Usage lands on the span/logger path; we don't have a direct accessor here.
	// The assertion is simply that the stream completes — usage-carrying chunks
	// with no content/tool fields must not break consumption.
}

// TestRunStream_RetriesWhenNoContentEmitted verifies that a stream error that
// occurs before any content is emitted is retryable — we still get resilience
// for pre-first-token failures (connection refused, 429, auth hiccups).
func TestRunStream_RetriesWhenNoContentEmitted(t *testing.T) {
	p := &streamProvider{
		streams: [][]types.StreamChunk{
			// First attempt: error with no prior emission. Must be retried.
			{{Err: &retryableError{msg: "transient"}}},
			// Second attempt: succeeds.
			{
				contentChunk("recovered"),
				finishChunk("stop"),
			},
		},
	}

	var got strings.Builder
	h := agent.New(p).
		WithMaxSteps(5).
		WithRetry(retry.Config{MaxAttempts: 3, InitialDelay: time.Millisecond})

	result, err := h.RunStream(context.Background(), "task", func(s string) { got.WriteString(s) })
	if err != nil {
		t.Fatalf("expected retry to succeed, got error: %v", err)
	}
	if result != "recovered" {
		t.Fatalf("final = %q, want %q", result, "recovered")
	}
	if got.String() != "recovered" {
		t.Fatalf("onChunk received %q, want %q", got.String(), "recovered")
	}
	// Both streams should have been consumed.
	if len(p.streams) != 0 {
		t.Fatalf("%d streams left unconsumed, expected retry to use both", len(p.streams))
	}
}

// TestRunStream_DoesNotRetryAfterEmission verifies option B: once onChunk has
// fired, a subsequent error terminates the step rather than retrying. Retrying
// would force the caller to receive duplicate or divergent content.
func TestRunStream_DoesNotRetryAfterEmission(t *testing.T) {
	p := &streamProvider{
		streams: [][]types.StreamChunk{
			// First attempt: emits content, THEN errors. Must NOT be retried.
			{
				contentChunk("partial "),
				{Err: &retryableError{msg: "mid-stream"}},
			},
			// Second attempt: exists but must not be consumed.
			{
				contentChunk("should not be seen"),
				finishChunk("stop"),
			},
		},
	}

	var got strings.Builder
	h := agent.New(p).
		WithMaxSteps(5).
		WithRetry(retry.Config{MaxAttempts: 3, InitialDelay: time.Millisecond})

	_, err := h.RunStream(context.Background(), "task", func(s string) { got.WriteString(s) })
	if err == nil {
		t.Fatal("expected terminal error after emission, got nil")
	}
	if !strings.Contains(err.Error(), "mid-stream") {
		t.Fatalf("expected underlying error to surface, got: %v", err)
	}
	if got.String() != "partial " {
		t.Fatalf("onChunk received %q, want %q — retry may have re-emitted", got.String(), "partial ")
	}
	// The second stream must still be present — retry should have bailed.
	if len(p.streams) != 1 {
		t.Fatalf("%d streams left, expected 1 (second stream must not be consumed)", len(p.streams))
	}
}

// TestRunStream_NameOnlyOnFirstFragment locks in that a later empty-string name
// fragment does not clobber the real name set by the first fragment. (If the
// guard `if frag.Name != ""` ever regresses to unconditional assignment, this
// test catches it.)
func TestRunStream_NameOnlyOnFirstFragment(t *testing.T) {
	p := &streamProvider{
		streams: [][]types.StreamChunk{
			{
				toolChunk(types.StreamToolCallFragment{Index: 0, ID: "c1", Type: "function", Name: "echo"}),
				// Later fragment with empty Name must NOT overwrite the real name.
				toolChunk(types.StreamToolCallFragment{Index: 0, Arguments: `{"text":"ok"}`}),
				finishChunk("tool_calls"),
			},
			{
				contentChunk("done"),
				finishChunk("stop"),
			},
		},
	}

	var ran bool
	h := agent.New(p).WithMaxSteps(5)
	h.RegisterTool(agent.Func("echo", "echo", func(_ context.Context, _ struct {
		Text string `json:"text"`
	}) (string, error) {
		ran = true
		return "ok", nil
	}))

	if _, err := h.RunStream(context.Background(), "task", func(string) {}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ran {
		t.Fatal("tool never ran — name was likely overwritten by empty-string fragment")
	}
}
