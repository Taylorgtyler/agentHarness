package main

import (
	"agentHarness/internal/agent"
	"fmt"
	"log"
	"time"
)

func main() {
	h := agent.New(
		"ministral-3b:latest",
		"http://localhost:11434/v1",
	)

	// Zero-param convenience wrapper
	h.RegisterFunc("get_time", "Returns the current time in UTC", func() string {
		return time.Now().UTC().Format(time.RFC3339)
	})

	// Typed params — schema auto-generated from struct tags
	h.RegisterTool(agent.Func("add", "Adds two integers", func(p struct {
		A int `json:"a" desc:"first number"`
		B int `json:"b" desc:"second number"`
	}) (string, error) {
		return fmt.Sprintf("%d", p.A+p.B), nil
	}))

	answer, err := h.Run("What time is it? Also what is 4 + 7?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(answer)
}
