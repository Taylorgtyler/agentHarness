package provider

import (
	"context"

	"github.com/taylortyler/agentHarness/pkg/types"
)

type Provider interface {
	Invoke(ctx context.Context, messages []types.Message, tools []types.Tool) (types.Message, error)
}
