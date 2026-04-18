package provider

import (
	"context"

	"agentHarness/internal/types"
)

type Provider interface {
	Invoke(ctx context.Context, messages []types.Message, tools []types.Tool) (types.Message, error)
}
