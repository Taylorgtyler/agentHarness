package provider

import (
	"context"

	"github.com/taylorgtyler/agentHarness/pkg/types"
)

type Provider interface {
	Invoke(ctx context.Context, messages []types.Message, tools []types.Tool) (types.Message, error)
}

type StreamProvider interface {
	Provider
	InvokeStream(ctx context.Context, messages []types.Message, tools []types.Tool) (<-chan StreamChunk, error)
}

type StreamChunk struct {
	ContentDelta string
	ToolCalls    []StreamToolCallFragment
	FinishReason string
	Usage        *types.Usage
	Err          error
}

type StreamToolCallFragment struct {
	Index     int
	ID        string
	Type      string
	Name      string
	Arguments string
}
