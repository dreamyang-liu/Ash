package tools

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"
)

func TestBackgroundProcessExitWaitsForCapturedOutput(t *testing.T) {
	shell := &ShellTool{}
	process := &ProcessTool{}

	for i := 0; i < 25; i++ {
		stdout := fmt.Sprintf("stdout-%d", i)
		stderr := fmt.Sprintf("stderr-%d", i)
		started := shell.Execute(map[string]any{
			"command":    fmt.Sprintf("printf %s; printf %s >&2", stdout, stderr),
			"background": true,
		})
		if !started.Success {
			t.Fatalf("start background process: %s", started.Error)
		}

		var startData struct {
			PID string `json:"pid"`
		}
		if err := json.Unmarshal([]byte(started.Output), &startData); err != nil {
			t.Fatalf("decode start result: %v", err)
		}

		deadline := time.Now().Add(2 * time.Second)
		for {
			result := process.Execute(map[string]any{
				"pid":    startData.PID,
				"action": "read",
			})
			if !result.Success {
				t.Fatalf("read process %s: %s", startData.PID, result.Error)
			}

			var snapshot struct {
				Stdout  string `json:"stdout"`
				Stderr  string `json:"stderr"`
				Running bool   `json:"running"`
			}
			if err := json.Unmarshal([]byte(result.Output), &snapshot); err != nil {
				t.Fatalf("decode process snapshot: %v", err)
			}
			if !snapshot.Running {
				if snapshot.Stdout != stdout || snapshot.Stderr != stderr {
					t.Fatalf(
						"process exited before output capture completed: stdout=%q stderr=%q",
						snapshot.Stdout, snapshot.Stderr,
					)
				}
				break
			}
			if time.Now().After(deadline) {
				t.Fatalf("background process %s did not exit", startData.PID)
			}
			time.Sleep(time.Millisecond)
		}
	}
}
