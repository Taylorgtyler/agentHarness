package tracing

import "context"

type Attr struct {
	Key   string
	Value any
}

func String(key, value string) Attr { return Attr{Key: key, Value: value} }
func Int(key string, value int) Attr { return Attr{Key: key, Value: value} }

type Span interface {
	End()
	RecordError(err error)
	// SetStatus marks the span as failed if err is non-nil, successful otherwise.
	SetStatus(err error)
	SetAttributes(attrs ...Attr)
	// TraceIDs returns trace and span IDs for log correlation, empty strings if unavailable.
	TraceIDs() (traceID, spanID string)
}

type Tracer interface {
	Start(ctx context.Context, name string, attrs ...Attr) (context.Context, Span)
}

// Noop is a no-op Tracer that produces no spans. Used when no tracer is configured.
var Noop Tracer = noopTracer{}

type noopTracer struct{}

func (noopTracer) Start(ctx context.Context, _ string, _ ...Attr) (context.Context, Span) {
	return ctx, noopSpan{}
}

type noopSpan struct{}

func (noopSpan) End()                    {}
func (noopSpan) RecordError(error)       {}
func (noopSpan) SetStatus(error)         {}
func (noopSpan) SetAttributes(...Attr)   {}
func (noopSpan) TraceIDs() (string, string) { return "", "" }
