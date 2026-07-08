package tools

import (
	"encoding/json"
)

// ProcessTool reads output from or kills background processes.
type ProcessTool struct{}

func (p *ProcessTool) Name() string { return "process" }

func (p *ProcessTool) Description() string {
	return "Manage background processes: read bounded output snapshots or kill"
}

func (p *ProcessTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"pid":    map[string]any{"type": "string", "description": "Process ID from shell(background=true)"},
			"action": map[string]any{"type": "string", "enum": []string{"read", "kill"}, "description": "read: get current bounded output snapshot. kill: terminate."},
			"tail":   map[string]any{"type": "integer", "description": "Only return last N lines from each stream (read only)"},
		},
		"required": []string{"pid", "action"},
	}
}

func (p *ProcessTool) Execute(args map[string]any) Result {
	pid, _ := args["pid"].(string)
	action, _ := args["action"].(string)

	proc := GetProcess(pid)
	if proc == nil {
		return Err("process not found: " + pid)
	}

	switch action {
	case "read":
		tail := 0
		if n, ok := args["tail"].(float64); ok {
			tail = int(n)
		}
		return p.read(proc, tail)
	case "kill":
		return p.kill(proc)
	default:
		return Err("unknown action: " + action)
	}
}

func (p *ProcessTool) read(proc *Process, tail int) Result {
	proc.mu.Lock()
	running := proc.exitCode == nil
	exitCode := proc.exitCode
	proc.mu.Unlock()

	stdout, stdoutTruncated := proc.stdout.Render()
	stderr, stderrTruncated := proc.stderr.Render()
	if tail > 0 {
		stdout = lastNLines(stdout, tail)
		stderr = lastNLines(stderr, tail)
	}

	resp := map[string]any{
		"stdout":           stdout,
		"stderr":           stderr,
		"running":          running,
		"exit_code":        exitCode,
		"stdout_bytes":     proc.stdout.TotalBytes(),
		"stderr_bytes":     proc.stderr.TotalBytes(),
		"stdout_truncated": stdoutTruncated,
		"stderr_truncated": stderrTruncated,
	}
	out, _ := json.Marshal(resp)
	return Ok(string(out))
}

func (p *ProcessTool) kill(proc *Process) Result {
	proc.mu.Lock()
	defer proc.mu.Unlock()

	if proc.exitCode != nil {
		return Ok("process already exited")
	}

	_ = proc.cmd.Process.Kill()
	code := -9
	proc.exitCode = &code

	resp := map[string]any{"killed": true, "exit_code": code}
	out, _ := json.Marshal(resp)
	return Ok(string(out))
}
