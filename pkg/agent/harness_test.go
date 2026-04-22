package agent_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/taylortyler/agentHarness/pkg/agent"
	"github.com/taylortyler/agentHarness/pkg/types"
)

type testProvider struct {
	responses []types.Message
	calls     [][]types.Message
	mu        sync.Mutex
}

func (f *testProvider) Invoke(_ context.Context, msgs []types.Message, _ []types.Tool) (types.Message, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, msgs)
	if len(f.responses) == 0 {
		return types.Message{}, errors.New("no more responses")
	}
	r := f.responses[0]
	f.responses = f.responses[1:]
	return r, nil
}

func strptr(s string) *string { return &s }

func toolCall(id, name, args string) types.ToolCall {
	return types.ToolCall{ID: id, Type: "function", Function: types.FunctionCall{Name: name, Arguments: args}}
}

func contentMsg(s string) types.Message {
	return types.Message{Role: "assistant", Content: strptr(s)}
}

func toolCallMsg(calls ...types.ToolCall) types.Message {
	return types.Message{Role: "assistant", ToolCalls: calls}
}

func newHarness(p *testProvider) *agent.Harness {
	return agent.New(p).WithMaxSteps(10)
}

// TestRun_NoToolCalls verifies that when the provider returns content immediately,
// Run returns it without calling any tools.
func TestRun_NoToolCalls(t *testing.T) {
	p := &testProvider{responses: []types.Message{contentMsg("hello")}}
	result, err := newHarness(p).Run(context.Background(), "say hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "hello" {
		t.Fatalf("got %q, want %q", result, "hello")
	}
}

// TestRun_SingleToolCall verifies the full tool-call cycle: model calls a tool,
// the tool executes, the result is fed back, and the model returns final content.
func TestRun_SingleToolCall(t *testing.T) {
	var gotArgs string
	p := &testProvider{
		responses: []types.Message{
			toolCallMsg(toolCall("c1", "echo", `{"text":"hi"}`)),
			contentMsg("done"),
		},
	}
	h := newHarness(p)
	h.RegisterTool(agent.Func("echo", "echo text", func(_ context.Context, p struct {
		Text string `json:"text"`
	}) (string, error) {
		gotArgs = p.Text
		return p.Text, nil
	}))

	result, err := h.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "done" {
		t.Fatalf("got %q, want %q", result, "done")
	}
	if gotArgs != "hi" {
		t.Fatalf("tool got args %q, want %q", gotArgs, "hi")
	}
}

// TestRun_MultipleToolCalls verifies that multiple tool calls in one response all execute.
func TestRun_MultipleToolCalls(t *testing.T) {
	var mu sync.Mutex
	called := map[string]bool{}

	p := &testProvider{
		responses: []types.Message{
			toolCallMsg(
				toolCall("c1", "toolA", "{}"),
				toolCall("c2", "toolB", "{}"),
			),
			contentMsg("done"),
		},
	}
	h := newHarness(p)
	for _, name := range []string{"toolA", "toolB"} {
		n := name
		h.RegisterTool(agent.Func(n, n, func(_ context.Context, _ struct{}) (string, error) {
			mu.Lock()
			called[n] = true
			mu.Unlock()
			return "ok", nil
		}))
	}

	if _, err := h.Run(context.Background(), "task"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called["toolA"] || !called["toolB"] {
		t.Fatalf("not all tools called: %v", called)
	}
}

// TestRun_MaxStepsZero verifies that maxSteps <= 0 returns an error before invoking the provider.
func TestRun_MaxStepsZero(t *testing.T) {
	p := &testProvider{}
	_, err := agent.New(p).Run(context.Background(), "task")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if len(p.calls) != 0 {
		t.Fatal("provider should not be called when maxSteps <= 0")
	}
}

// TestRun_ExceedsMaxSteps verifies that Run returns an error when the model never stops calling tools.
func TestRun_ExceedsMaxSteps(t *testing.T) {
	responses := make([]types.Message, 3)
	for i := range responses {
		responses[i] = toolCallMsg(toolCall("c1", "noop", "{}"))
	}
	p := &testProvider{responses: responses}
	h := agent.New(p).WithMaxSteps(3)
	h.RegisterTool(agent.Func("noop", "noop", func(_ context.Context, _ struct{}) (string, error) {
		return "ok", nil
	}))

	_, err := h.Run(context.Background(), "task")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "exceeded max steps") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// TestRun_ProviderError verifies that a provider error propagates with the step number.
func TestRun_ProviderError(t *testing.T) {
	p := &testProvider{} // no responses → "no more responses" error on first call
	_, err := newHarness(p).Run(context.Background(), "task")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "step 0") {
		t.Fatalf("expected step number in error, got: %v", err)
	}
}

// TestRun_UnknownTool verifies that calling an unregistered tool hard-errors Run.
func TestRun_UnknownTool(t *testing.T) {
	p := &testProvider{
		responses: []types.Message{
			toolCallMsg(toolCall("c1", "ghost", "{}")),
		},
	}
	_, err := newHarness(p).Run(context.Background(), "task")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

// TestRun_ToolExecuteError verifies that a tool execution error is passed to the model as a string
// rather than hard-erroring Run.
func TestRun_ToolExecuteError(t *testing.T) {
	var gotResult string
	p := &testProvider{
		responses: []types.Message{
			toolCallMsg(toolCall("c1", "boom", "{}")),
			contentMsg("handled"),
		},
	}
	h := newHarness(p)
	h.RegisterTool(agent.Func("boom", "boom", func(_ context.Context, _ struct{}) (string, error) {
		return "", errors.New("something broke")
	}))

	result, err := h.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("unexpected hard error: %v", err)
	}
	if result != "handled" {
		t.Fatalf("got %q, want %q", result, "handled")
	}

	// The second invocation should have received the error string as the tool result.
	secondCall := p.calls[1]
	for _, msg := range secondCall {
		if msg.Role == "tool" && msg.Content != nil {
			gotResult = *msg.Content
		}
	}
	if !strings.HasPrefix(gotResult, "error:") {
		t.Fatalf("expected tool result to start with 'error:', got %q", gotResult)
	}
}

// TestRun_SystemPromptFirst verifies that WithSystemPrompt places the system message
// first in every provider invocation.
func TestRun_SystemPromptFirst(t *testing.T) {
	p := &testProvider{responses: []types.Message{contentMsg("ok")}}
	h := agent.New(p).WithMaxSteps(10).WithSystemPrompt("be helpful")

	if _, err := h.Run(context.Background(), "task"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	first := p.calls[0][0]
	if first.Role != "system" {
		t.Fatalf("first message role = %q, want %q", first.Role, "system")
	}
	if first.Content == nil || *first.Content != "be helpful" {
		t.Fatalf("system message content = %v, want %q", first.Content, "be helpful")
	}
}

// TestRun_MessageSequence verifies that after a tool call the next invocation contains
// the assistant tool-call message followed by the tool result message.
func TestRun_MessageSequence(t *testing.T) {
	p := &testProvider{
		responses: []types.Message{
			toolCallMsg(toolCall("c1", "echo", `{"text":"hi"}`)),
			contentMsg("done"),
		},
	}
	h := newHarness(p)
	h.RegisterTool(agent.Func("echo", "echo", func(_ context.Context, p struct {
		Text string `json:"text"`
	}) (string, error) {
		return p.Text, nil
	}))

	if _, err := h.Run(context.Background(), "task"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	second := p.calls[1]
	// Expected: user message, assistant tool-call message, tool result message.
	if len(second) < 3 {
		t.Fatalf("second call has %d messages, want at least 3", len(second))
	}
	assistantMsg := second[len(second)-2]
	toolResultMsg := second[len(second)-1]

	if assistantMsg.Role != "assistant" || len(assistantMsg.ToolCalls) == 0 {
		t.Fatalf("expected assistant tool-call message, got role=%q", assistantMsg.Role)
	}
	if toolResultMsg.Role != "tool" {
		t.Fatalf("expected tool result message, got role=%q", toolResultMsg.Role)
	}
	if toolResultMsg.Content == nil || *toolResultMsg.Content != "hi" {
		t.Fatalf("tool result content = %v, want %q", toolResultMsg.Content, "hi")
	}
}

// TestRun_EmptyResponse verifies that Run errors when the model returns neither content nor tool calls.
func TestRun_EmptyResponse(t *testing.T) {
	p := &testProvider{responses: []types.Message{{Role: "assistant"}}}
	_, err := newHarness(p).Run(context.Background(), "task")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "no content and no tool calls") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// TestRun_DoesNotMutateMessages verifies that calling Run twice on the same harness
// starts each run with a clean message slate, not the prior run's conversation appended.
func TestRun_DoesNotMutateMessages(t *testing.T) {
	p := &testProvider{responses: []types.Message{contentMsg("first"), contentMsg("second")}}
	h := newHarness(p)

	if _, err := h.Run(context.Background(), "task one"); err != nil {
		t.Fatalf("first run error: %v", err)
	}
	if _, err := h.Run(context.Background(), "task two"); err != nil {
		t.Fatalf("second run error: %v", err)
	}

	if len(p.calls[0]) != len(p.calls[1]) {
		t.Fatalf("second run received %d messages, first run received %d — h.messages was mutated",
			len(p.calls[1]), len(p.calls[0]))
	}
}

// TestAsTool verifies that AsTool wraps the harness as a tool that runs a sub-agent
// and returns its final content.
func TestAsTool(t *testing.T) {
	inner := &testProvider{responses: []types.Message{contentMsg("sub result")}}
	h := agent.New(inner).WithMaxSteps(10)

	outer := &testProvider{
		responses: []types.Message{
			toolCallMsg(toolCall("c1", "subagent", `{"task":"do something"}`)),
			contentMsg("outer done"),
		},
	}
	outerH := agent.New(outer).WithMaxSteps(10)
	outerH.RegisterTool(h.AsTool("subagent", "run sub-agent"))

	result, err := outerH.Run(context.Background(), "main task")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "outer done" {
		t.Fatalf("got %q, want %q", result, "outer done")
	}
	if len(inner.calls) == 0 {
		t.Fatal("inner provider was never called")
	}
}
