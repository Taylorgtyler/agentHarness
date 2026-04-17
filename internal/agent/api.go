package agent

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
)

type chatRequest struct {
	Model    string       `json:"model"`
	Messages []Message    `json:"messages"`
	Tools    []toolSchema `json:"tools,omitempty"`
}

type toolSchema struct {
	Type     string          `json:"type"`
	Function json.RawMessage `json:"function"`
}

type chatResponse struct {
	Choices []struct {
		Message Message `json:"message"`
	} `json:"choices"`
}

func (h *Harness) callAPI() (Message, error) {
	tools := make([]toolSchema, 0, len(h.tools))
	for _, t := range h.tools {
		tools = append(tools, toolSchema{
			Type:     "function",
			Function: t.Schema(),
		})
	}

	reqBody := chatRequest{
		Model:    h.model,
		Messages: h.messages,
		Tools:    tools,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return Message{}, err
	}

	req, err := http.NewRequest("POST", h.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Message{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if h.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+h.apiKey)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return Message{}, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return Message{}, err
	}

	if resp.StatusCode != 200 {
		return Message{}, fmt.Errorf("api returned %d: %s", resp.StatusCode, string(respBody))
	}

	var parsed chatResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return Message{}, err
	}

	if len(parsed.Choices) == 0 {
		return Message{}, errors.New("no choices in response")
	}

	return parsed.Choices[0].Message, nil
}
