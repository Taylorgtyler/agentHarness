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

package tracing

import "context"

type Attr struct {
	key   string
	value any
}

func (a Attr) Key() string { return a.key }
func (a Attr) Value() any  { return a.value }

func String(key, value string) Attr          { return Attr{key, value} }
func Int(key string, value int) Attr         { return Attr{key, value} }
func Bool(key string, value bool) Attr       { return Attr{key, value} }
func Float64(key string, value float64) Attr { return Attr{key, value} }

type Span interface {
	End()
	RecordError(err error)
	// SetStatus marks the span failed if err is non-nil, successful otherwise.
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

func (noopSpan) End()                       {}
func (noopSpan) RecordError(error)          {}
func (noopSpan) SetStatus(error)            {}
func (noopSpan) SetAttributes(...Attr)      {}
func (noopSpan) TraceIDs() (string, string) { return "", "" }
