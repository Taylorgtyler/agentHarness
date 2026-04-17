package main

import (
	"agentHarness/internal/agent"
	"encoding/json"
	"fmt"
	"log"
	"time"
)

type GetTimeTool struct{}

func (t *GetTimeTool) Name() string { return "get_time" }

func (t *GetTimeTool) Schema() json.RawMessage {
	return json.RawMessage(`{
		"name": "get_time",
		"description": "Returns the current time in UTC",
		"parameters": {
			"type": "object",
			"properties": {},
			"required": []
		}
	}`)
}

func (t *GetTimeTool) Execute(args string) (string, error) {
	return time.Now().UTC().Format(time.RFC3339), nil
}

func main() {
	h := agent.New(
		"ministral-3:8b",
		"http://localhost:11434/v1",
	)

	h.RegisterTool(&GetTimeTool{})

	answer, err := h.Run("What time is it?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(answer)
}
