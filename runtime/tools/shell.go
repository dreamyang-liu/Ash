package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"sort"
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
			"tail": map[string]any{
				"type":        "integer",
				"description": "Return only the last N lines of each stream. Applied after the byte budget, so if the command produced more than max_output_bytes these are the last N lines of what was captured. Setting this makes the capture keep the tail of the output (truncate_mode T1) unless you name a mode yourself.",
			},
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
			"stdin": map[string]any{
				"type":        "string",
				"description": "Data to feed the command on standard input, then close it. Lets you run 'python -', 'patch -p1', or 'sh -s' without first writing a temporary file.",
			},
			"env": map[string]any{
				"type":                 "object",
				"additionalProperties": map[string]any{"type": "string"},
				"description":          "Extra environment variables, e.g. {\"PYTHONPATH\": \"/testbed\"}. Added to the existing environment rather than replacing it. Prefer this over a 'KEY=value cmd' prefix: values needing quotes or spaces are handled correctly.",
			},
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
	// Asking for the last N lines means the head of the output is going to be
	// discarded at render time, so spending part of the byte budget keeping it
	// would be waste. An explicit truncate_mode still wins.
	if tail > 0 && !hasTruncateModeArg(args) {
		mode = tailOnlyMode
	}
	stdin, _ := args["stdin"].(string)
	env, err := envArg(args)
	if err != nil {
		return Err(err.Error())
	}

	agentID, _ := args["agent_id"].(string)

	opts := runOpts{
		workingDir:     workingDir,
		stdin:          stdin,
		env:            env,
		maxOutputBytes: maxOutputBytes,
		mode:           mode,
	}
	if background {
		return s.runBackground(command, agentID, opts)
	}
	return s.runSync(command, timeout, tail, opts)
}

// runOpts carries the shared per-call execution settings, so adding another
// one does not mean threading a new parameter through both run paths.
type runOpts struct {
	workingDir     string
	stdin          string
	env            []string // extra KEY=VALUE entries, appended to the host env
	maxOutputBytes int
	mode           truncateMode
}

// apply wires the options onto a command, leaving stdout/stderr to the caller.
func (o runOpts) apply(cmd *exec.Cmd) {
	if o.workingDir != "" {
		cmd.Dir = o.workingDir
	}
	if o.stdin != "" {
		cmd.Stdin = strings.NewReader(o.stdin)
	}
	if len(o.env) > 0 {
		// Append rather than replace: an agent that wanted PYTHONPATH set
		// should not lose PATH and break every subsequent command.
		cmd.Env = append(os.Environ(), o.env...)
	}
}

// envArg converts the env object into KEY=VALUE entries. Passing them
// structurally avoids the quoting hazards of prefixing the command string.
func envArg(args map[string]any) ([]string, error) {
	raw, ok := args["env"].(map[string]any)
	if !ok {
		return nil, nil
	}
	out := make([]string, 0, len(raw))
	for k, v := range raw {
		if k == "" || strings.ContainsAny(k, "=\x00") {
			return nil, fmt.Errorf("invalid env name: %q", k)
		}
		s, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("env value for %q must be a string", k)
		}
		if strings.ContainsRune(s, 0) {
			return nil, fmt.Errorf("env value for %q must not contain NUL", k)
		}
		out = append(out, k+"="+s)
	}
	sort.Strings(out) // deterministic for traces and tests
	return out, nil
}

func (s *ShellTool) runSync(command string, timeout, tail int, opts runOpts) Result {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sh", "-c", command)
	opts.apply(cmd)

	stdout := NewBoundedLogMode(opts.maxOutputBytes, opts.mode)
	stderr := NewBoundedLogMode(opts.maxOutputBytes, opts.mode)
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

func (s *ShellTool) runBackground(command, agentID string, opts runOpts) Result {
	pid := uuid.New().String()[:8]

	cmd := exec.Command("sh", "-c", command)
	opts.apply(cmd)

	proc := &Process{
		pid:    pid,
		cmd:    cmd,
		stdout: NewBoundedLogMode(opts.maxOutputBytes, opts.mode),
		stderr: NewBoundedLogMode(opts.maxOutputBytes, opts.mode),
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
