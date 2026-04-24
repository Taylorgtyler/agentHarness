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

package types

import (
	"context"
	"encoding/json"
)

type Usage struct {
	PromptTokens     int
	CompletionTokens int
}

type Message struct {
	Role       string     `json:"role"`
	Content    *string    `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Usage      *Usage     `json:"usage,omitempty"`
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func UserMessage(content string) Message {
	return Message{Role: "user", Content: &content}
}

func SystemMessage(content string) Message {
	return Message{Role: "system", Content: &content}
}

func ToolResultMessage(toolCallID, content string) Message {
	return Message{Role: "tool", Content: &content, ToolCallID: toolCallID}
}

type Tool interface {
	Name() string
	Schema() json.RawMessage
	Execute(ctx context.Context, args string) (string, error)
}

type StreamChunk struct {
	ContentDelta string
	ToolCalls    []StreamToolCallFragment
	FinishReason string
	Usage        *Usage
	Err          error
}

type StreamToolCallFragment struct {
	Index     int
	ID        string
	Type      string
	Name      string
	Arguments string
}
