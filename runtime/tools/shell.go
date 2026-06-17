package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/dreamyang-liu/ash/runtime/events"
	"github.com/google/uuid"
)

// Process holds state for a background process.
type Process struct {
	pid        string
	cmd        *exec.Cmd
	stdout     []string
	stderr     []string
	exitCode   *int
	readCursor int
	mu         sync.Mutex
}

var (
	processes   = make(map[string]*Process)
	processesMu sync.Mutex
)

// ShellTool executes shell commands.
type ShellTool struct{}

func (s *ShellTool) Name() string { return "shell" }

func (s *ShellTool) Description() string {
	return "Execute a shell command synchronously or in the background"
}

func (s *ShellTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"command":     map[string]any{"type": "string", "description": "Shell command to execute"},
			"background":  map[string]any{"type": "boolean", "default": false, "description": "Run in background, returns pid"},
			"timeout":     map[string]any{"type": "integer", "default": 300, "description": "Timeout in seconds"},
			"tail":        map[string]any{"type": "integer", "description": "Only return last N lines of output"},
			"working_dir": map[string]any{"type": "string", "description": "Working directory"},
		},
		"required": []string{"command"},
	}
}

func (s *ShellTool) Execute(args map[string]any) Result {
	command, _ := args["command"].(string)
	if command == "" {
		return Err("command is required")
	}

	background, _ := args["background"].(bool)
	timeout := 300
	if t, ok := args["timeout"].(float64); ok {
		timeout = int(t)
	}
	tail := 0
	if n, ok := args["tail"].(float64); ok {
		tail = int(n)
	}
	workingDir, _ := args["working_dir"].(string)

	agentID, _ := args["agent_id"].(string)

	if background {
		return s.runBackground(command, workingDir, agentID)
	}
	return s.runSync(command, workingDir, timeout, tail)
}

func (s *ShellTool) runSync(command, workingDir string, timeout, tail int) Result {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sh", "-c", command)
	if workingDir != "" {
		cmd.Dir = workingDir
	}

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	output := stdout.String()
	if stderr.Len() > 0 {
		output += "\n[stderr]\n" + stderr.String()
	}

	if tail > 0 {
		output = lastNLines(output, tail)
	}

	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return Err("command timed out after " + fmt.Sprintf("%d", timeout) + "s")
		}
		return Err(output)
	}
	return Ok(output)
}

func (s *ShellTool) runBackground(command, workingDir, agentID string) Result {
	pid := uuid.New().String()[:8]

	cmd := exec.Command("sh", "-c", command)
	if workingDir != "" {
		cmd.Dir = workingDir
	}

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	proc := &Process{pid: pid, cmd: cmd}

	if err := cmd.Start(); err != nil {
		return Err("failed to start: " + err.Error())
	}

	processesMu.Lock()
	processes[pid] = proc
	processesMu.Unlock()

	go func() {
		err := cmd.Wait()

		proc.mu.Lock()
		proc.stdout = strings.Split(stdout.String(), "\n")
		proc.stderr = strings.Split(stderr.String(), "\n")
		code := 0
		if err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok {
				code = exitErr.ExitCode()
			} else {
				code = 1
			}
		}
		proc.exitCode = &code
		proc.mu.Unlock()

		data := map[string]any{"pid": pid, "exitCode": code}
		events.PushTo(agentID, "process_exited", pid, data)
	}()

	resp, _ := json.Marshal(map[string]any{"pid": pid})
	return Ok(string(resp))
}

func lastNLines(s string, n int) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) <= n {
		return s
	}
	return strings.Join(lines[len(lines)-n:], "\n")
}

// GetProcess returns a background process by pid.
func GetProcess(pid string) *Process {
	processesMu.Lock()
	defer processesMu.Unlock()
	return processes[pid]
}
