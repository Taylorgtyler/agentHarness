package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"agentHarness/internal/types"
)

type funcTool[P any] struct {
	name    string
	schema  json.RawMessage
	handler func(context.Context, P) (string, error)
}

func (t *funcTool[P]) Name() string            { return t.name }
func (t *funcTool[P]) Schema() json.RawMessage { return t.schema }

func (t *funcTool[P]) Execute(ctx context.Context, args string) (string, error) {
	var params P
	if args != "" && args != "null" && args != "{}" {
		if err := json.Unmarshal([]byte(args), &params); err != nil {
			return "", fmt.Errorf("invalid args: %w", err)
		}
	}
	return t.handler(ctx, params)
}

func Func[P any](name, description string, handler func(context.Context, P) (string, error)) types.Tool {
	var zero P
	schema := buildSchema(name, description, reflect.TypeOf(zero))
	return &funcTool[P]{name: name, schema: schema, handler: handler}
}

func buildSchema(name, description string, t reflect.Type) json.RawMessage {
	props := map[string]any{}
	required := []string{}

	if t != nil {
		for i := range t.NumField() {
			field := t.Field(i)
			if !field.IsExported() {
				continue
			}

			jsonName := field.Name
			optional := false
			if tag := field.Tag.Get("json"); tag != "" {
				parts := strings.Split(tag, ",")
				if parts[0] != "" && parts[0] != "-" {
					jsonName = parts[0]
				}
				optional = strings.Contains(tag, "omitempty")
			}

			prop := map[string]any{"type": goTypeToJSONType(field.Type)}
			if desc := field.Tag.Get("desc"); desc != "" {
				prop["description"] = desc
			}
			props[jsonName] = prop

			if !optional {
				required = append(required, jsonName)
			}
		}
	}

	schema := map[string]any{
		"name":        name,
		"description": description,
		"parameters": map[string]any{
			"type":       "object",
			"properties": props,
			"required":   required,
		},
	}
	raw, _ := json.Marshal(schema)
	return raw
}

func goTypeToJSONType(t reflect.Type) string {
	switch t.Kind() {
	case reflect.String:
		return "string"
	case reflect.Bool:
		return "boolean"
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return "integer"
	case reflect.Float32, reflect.Float64:
		return "number"
	case reflect.Slice, reflect.Array:
		return "array"
	default:
		return "object"
	}
}
