package observability

import (
	"context"
	"fmt"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"

	"agentHarness/internal/tracing"
)

type Config struct {
	ServiceName    string
	ServiceVersion string
}

// Setup initializes the global OTel tracer provider and propagator.
// The OTLP endpoint is read from OTEL_EXPORTER_OTLP_ENDPOINT (default: http://localhost:4318).
// Returns a shutdown function that must be called on exit to flush pending spans.
func Setup(ctx context.Context, cfg Config) (func(context.Context) error, error) {
	exp, err := otlptracehttp.New(ctx)
	if err != nil {
		return nil, err
	}

	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceName(cfg.ServiceName),
			semconv.ServiceVersion(cfg.ServiceVersion),
		),
	)
	if err != nil {
		return nil, err
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp),
		sdktrace.WithResource(res),
	)

	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	return tp.Shutdown, nil
}

// NewTracer wraps an OTel trace.Tracer as a tracing.Tracer for use with the harness.
func NewTracer(t trace.Tracer) tracing.Tracer {
	return &otelTracer{t: t}
}

type otelTracer struct {
	t trace.Tracer
}

func (ot *otelTracer) Start(ctx context.Context, name string, attrs ...tracing.Attr) (context.Context, tracing.Span) {
	otelAttrs := make([]attribute.KeyValue, 0, len(attrs))
	for _, a := range attrs {
		otelAttrs = append(otelAttrs, toOtelAttr(a))
	}
	ctx, span := ot.t.Start(ctx, name, trace.WithAttributes(otelAttrs...))
	return ctx, &otelSpan{s: span}
}

type otelSpan struct {
	s trace.Span
}

func (os *otelSpan) End() { os.s.End() }

func (os *otelSpan) RecordError(err error) { os.s.RecordError(err) }

func (os *otelSpan) SetStatus(err error) {
	if err != nil {
		os.s.SetStatus(codes.Error, err.Error())
	} else {
		os.s.SetStatus(codes.Ok, "")
	}
}

func (os *otelSpan) SetAttributes(attrs ...tracing.Attr) {
	otelAttrs := make([]attribute.KeyValue, 0, len(attrs))
	for _, a := range attrs {
		otelAttrs = append(otelAttrs, toOtelAttr(a))
	}
	os.s.SetAttributes(otelAttrs...)
}

func (os *otelSpan) TraceIDs() (string, string) {
	sc := os.s.SpanContext()
	if !sc.IsValid() {
		return "", ""
	}
	return sc.TraceID().String(), sc.SpanID().String()
}

func toOtelAttr(a tracing.Attr) attribute.KeyValue {
	switch v := a.Value().(type) {
	case string:
		return attribute.String(a.Key(), v)
	case int:
		return attribute.Int(a.Key(), v)
	case bool:
		return attribute.Bool(a.Key(), v)
	case float64:
		return attribute.Float64(a.Key(), v)
	default:
		return attribute.String(a.Key(), fmt.Sprintf("%v", v))
	}
}
