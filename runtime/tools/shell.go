package tools

import (
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
	pid      string
	cmd      *exec.Cmd
	stdout   *BoundedLog
	stderr   *BoundedLog
	exitCode *int
	mu       sync.Mutex
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
			"command":    map[string]any{"type": "string", "description": "Shell command to execute"},
			"background": map[string]any{"type": "boolean", "default": false, "description": "Run in background, returns pid"},
			"timeout":    map[string]any{"type": "integer", "default": 300, "description": "Timeout in seconds"},
			"tail":       map[string]any{"type": "integer", "description": "Only return last N lines of output"},
			"max_output_bytes": map[string]any{
				"type":        "integer",
				"default":     defaultMaxOutputBytes,
				"description": "Total captured bytes per output stream. Larger output is truncated per truncate_mode.",
			},
			"truncate_mode": map[string]any{
				"type":        "string",
				"default":     defaultTruncateMode,
				"description": "How to divide the byte budget when output is too long: \"H<n>T<n>\" with weights for the head and tail sections. H2T3 keeps the first 40% and last 60%; T1 keeps only the tail (useful for build/test errors); H1 keeps only the beginning.",
			},
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
	maxOutputBytes := outputBytesArg(args)
	mode := truncateModeArg(args)

	agentID, _ := args["agent_id"].(string)

	if background {
		return s.runBackground(command, workingDir, agentID, maxOutputBytes, mode)
	}
	return s.runSync(command, workingDir, timeout, tail, maxOutputBytes, mode)
}

func (s *ShellTool) runSync(command, workingDir string, timeout, tail int, maxOutputBytes int, mode truncateMode) Result {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sh", "-c", command)
	if workingDir != "" {
		cmd.Dir = workingDir
	}

	stdout := NewBoundedLogMode(maxOutputBytes, mode)
	stderr := NewBoundedLogMode(maxOutputBytes, mode)
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	err := cmd.Run()
	output := renderCommandOutput(stdout, stderr, tail)

	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			if output != "" {
				return Err(output + "\ncommand timed out after " + fmt.Sprintf("%d", timeout) + "s")
			}
			return Err("command timed out after " + fmt.Sprintf("%d", timeout) + "s")
		}
		return Err(output)
	}
	return Ok(output)
}

func (s *ShellTool) runBackground(command, workingDir, agentID string, maxOutputBytes int, mode truncateMode) Result {
	pid := uuid.New().String()[:8]

	cmd := exec.Command("sh", "-c", command)
	if workingDir != "" {
		cmd.Dir = workingDir
	}

	proc := &Process{
		pid:    pid,
		cmd:    cmd,
		stdout: NewBoundedLogMode(maxOutputBytes, mode),
		stderr: NewBoundedLogMode(maxOutputBytes, mode),
	}
	cmd.Stdout = proc.stdout
	cmd.Stderr = proc.stderr

	if err := cmd.Start(); err != nil {
		return Err("failed to start: " + err.Error())
	}

	processesMu.Lock()
	processes[pid] = proc
	processesMu.Unlock()

	// Wait also joins os/exec's stdout/stderr copy goroutines. Publish exit only
	// after both logs contain the process's complete captured output.
	go func() {
		err := cmd.Wait()
		code := 0
		if err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok {
				code = exitErr.ExitCode()
			} else {
				code = 1
			}
		}
		proc.mu.Lock()
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
