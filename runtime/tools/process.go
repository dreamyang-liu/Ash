package tools

import (
	"encoding/json"
	"strings"
)

// ProcessTool reads output from or kills background processes.
type ProcessTool struct{}

func (p *ProcessTool) Name() string { return "process" }

func (p *ProcessTool) Description() string {
	return "Manage background processes: read new output or kill"
}

func (p *ProcessTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"pid":    map[string]any{"type": "string", "description": "Process ID from shell(background=true)"},
			"action": map[string]any{"type": "string", "enum": []string{"read", "kill"}, "description": "read: get output. kill: terminate."},
			"tail":   map[string]any{"type": "integer", "description": "Only return last N lines (read only)"},
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
	defer proc.mu.Unlock()

	// Get new stdout lines since last read.
	var newLines []string
	if proc.readCursor < len(proc.stdout) {
		newLines = proc.stdout[proc.readCursor:]
		proc.readCursor = len(proc.stdout)
	}

	stdout := strings.Join(newLines, "\n")
	if tail > 0 && len(newLines) > tail {
		stdout = strings.Join(newLines[len(newLines)-tail:], "\n")
	}

	resp := map[string]any{
		"stdout":    stdout,
		"stderr":    strings.Join(proc.stderr, "\n"),
		"running":   proc.exitCode == nil,
		"exit_code": proc.exitCode,
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
