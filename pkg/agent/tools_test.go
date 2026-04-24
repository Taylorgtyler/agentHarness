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

package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/taylorgtyler/agentHarness/pkg/agent"
)

type schemaDoc struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  struct {
		Type       string                       `json:"type"`
		Properties map[string]map[string]string `json:"properties"`
		Required   []string                     `json:"required"`
	} `json:"parameters"`
}

func parseSchema(t *testing.T, tool interface{ Schema() json.RawMessage }) schemaDoc {
	t.Helper()
	var doc schemaDoc
	if err := json.Unmarshal(tool.Schema(), &doc); err != nil {
		t.Fatalf("schema is not valid JSON: %v", err)
	}
	return doc
}

func hasRequired(doc schemaDoc, field string) bool {
	return slices.Contains(doc.Parameters.Required, field)
}

func TestFunc_Name(t *testing.T) {
	tool := agent.Func("mytool", "desc", func(_ context.Context, _ struct{}) (string, error) {
		return "", nil
	})
	if tool.Name() != "mytool" {
		t.Fatalf("got %q, want %q", tool.Name(), "mytool")
	}
}

func TestFunc_SchemaNameAndDescription(t *testing.T) {
	tool := agent.Func("mytool", "does stuff", func(_ context.Context, _ struct{}) (string, error) {
		return "", nil
	})
	doc := parseSchema(t, tool)
	if doc.Name != "mytool" {
		t.Fatalf("schema name = %q, want %q", doc.Name, "mytool")
	}
	if doc.Description != "does stuff" {
		t.Fatalf("schema description = %q, want %q", doc.Description, "does stuff")
	}
}

func TestFunc_Execute_ValidArgs(t *testing.T) {
	type params struct {
		Text string `json:"text"`
	}
	tool := agent.Func("echo", "echo", func(_ context.Context, p params) (string, error) {
		return p.Text, nil
	})
	result, err := tool.Execute(context.Background(), `{"text":"hello"}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "hello" {
		t.Fatalf("got %q, want %q", result, "hello")
	}
}

func TestFunc_Execute_EmptyArgs(t *testing.T) {
	called := false
	tool := agent.Func("noop", "noop", func(_ context.Context, _ struct{}) (string, error) {
		called = true
		return "ok", nil
	})
	for _, args := range []string{"", "null", "{}"} {
		called = false
		if _, err := tool.Execute(context.Background(), args); err != nil {
			t.Fatalf("args=%q: unexpected error: %v", args, err)
		}
		if !called {
			t.Fatalf("args=%q: handler was not called", args)
		}
	}
}

func TestFunc_Execute_InvalidArgs(t *testing.T) {
	tool := agent.Func("t", "t", func(_ context.Context, _ struct{ X string }) (string, error) {
		return "", nil
	})
	_, err := tool.Execute(context.Background(), `not json`)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "invalid args") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestFunc_Execute_HandlerError(t *testing.T) {
	tool := agent.Func("t", "t", func(_ context.Context, _ struct{}) (string, error) {
		return "", errors.New("boom")
	})
	_, err := tool.Execute(context.Background(), "")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "boom") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestSchema_FieldTypes(t *testing.T) {
	type params struct {
		S string  `json:"s"`
		B bool    `json:"b"`
		I int     `json:"i"`
		F float64 `json:"f"`
		A []int   `json:"a"`
	}
	tool := agent.Func("t", "t", func(_ context.Context, _ params) (string, error) { return "", nil })
	doc := parseSchema(t, tool)

	want := map[string]string{"s": "string", "b": "boolean", "i": "integer", "f": "number", "a": "array"}
	for field, wantType := range want {
		prop, ok := doc.Parameters.Properties[field]
		if !ok {
			t.Fatalf("field %q missing from schema", field)
		}
		if prop["type"] != wantType {
			t.Fatalf("field %q: got type %q, want %q", field, prop["type"], wantType)
		}
	}
}

func TestSchema_RequiredVsOptional(t *testing.T) {
	type params struct {
		Required string `json:"required_field"`
		Optional string `json:"optional_field,omitempty"`
	}
	tool := agent.Func("t", "t", func(_ context.Context, _ params) (string, error) { return "", nil })
	doc := parseSchema(t, tool)

	if !hasRequired(doc, "required_field") {
		t.Fatal("expected required_field in required")
	}
	if hasRequired(doc, "optional_field") {
		t.Fatal("optional_field should not be in required")
	}
}

func TestSchema_DescTag(t *testing.T) {
	type params struct {
		X string `json:"x" desc:"the x value"`
	}
	tool := agent.Func("t", "t", func(_ context.Context, _ params) (string, error) { return "", nil })
	doc := parseSchema(t, tool)

	prop, ok := doc.Parameters.Properties["x"]
	if !ok {
		t.Fatal("field x missing from schema")
	}
	if prop["description"] != "the x value" {
		t.Fatalf("got description %q, want %q", prop["description"], "the x value")
	}
}

func TestSchema_UnexportedFieldsExcluded(t *testing.T) {
	type params struct {
		Exported   string `json:"exported"`
		unexported string //nolint
	}
	tool := agent.Func("t", "t", func(_ context.Context, _ params) (string, error) { return "", nil })
	doc := parseSchema(t, tool)

	if _, ok := doc.Parameters.Properties["unexported"]; ok {
		t.Fatal("unexported field should not appear in schema")
	}
	if _, ok := doc.Parameters.Properties["Exported"]; !ok {
		if _, ok := doc.Parameters.Properties["exported"]; !ok {
			t.Fatal("exported field missing from schema")
		}
	}
}

func TestSchema_ParametersTypeIsObject(t *testing.T) {
	tool := agent.Func("t", "t", func(_ context.Context, _ struct{}) (string, error) { return "", nil })
	doc := parseSchema(t, tool)
	if doc.Parameters.Type != "object" {
		t.Fatalf("parameters.type = %q, want %q", doc.Parameters.Type, "object")
	}
}

func TestSchema_JSONTagRenamesField(t *testing.T) {
	type params struct {
		MyField string `json:"my_field"`
	}
	tool := agent.Func("t", "t", func(_ context.Context, _ params) (string, error) { return "", nil })
	doc := parseSchema(t, tool)

	if _, ok := doc.Parameters.Properties["my_field"]; !ok {
		t.Fatal("expected field named my_field in schema")
	}
	if _, ok := doc.Parameters.Properties["MyField"]; ok {
		t.Fatal("field should be renamed to my_field, not MyField")
	}
}

func TestSchema_JSONDashExcludesField(t *testing.T) {
	type params struct {
		APIKey string `json:"-"`
		Query  string `json:"query"`
	}
	tool := agent.Func("t", "t", func(_ context.Context, _ params) (string, error) { return "", nil })
	doc := parseSchema(t, tool)

	if _, ok := doc.Parameters.Properties["APIKey"]; ok {
		t.Fatal("field with json:\"-\" should be excluded from schema")
	}
	if _, ok := doc.Parameters.Properties["-"]; ok {
		t.Fatal("field with json:\"-\" should not appear as \"-\" in schema")
	}
	if hasRequired(doc, "APIKey") || hasRequired(doc, "-") {
		t.Fatal("field with json:\"-\" should not appear in required")
	}
}

func TestSchema_EmptyParams(t *testing.T) {
	tool := agent.Func("t", "t", func(_ context.Context, _ struct{}) (string, error) { return "", nil })
	doc := parseSchema(t, tool)

	if len(doc.Parameters.Properties) != 0 {
		t.Fatalf("expected empty properties, got %v", doc.Parameters.Properties)
	}
	if len(doc.Parameters.Required) != 0 {
		t.Fatalf("expected empty required, got %v", doc.Parameters.Required)
	}
}
