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
			"tail": map[string]any{
				"type":        "integer",
				"description": "Return only the last N lines from each stream (read only). Applied to what the process captured, which was bounded by the max_output_bytes and truncate_mode given when it was started.",
			},
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
	exitCode := proc.exitCode
	proc.mu.Unlock()

	// The same CommandOutcome `shell` reports, so reading a background pid and
	// running a command in the foreground answer in one shape.
	outcome := commandOutcome(proc.stdout, proc.stderr, tail)
	outcome.ExitCode = exitCode
	outcome.Running = exitCode == nil
	return outcome.Result()
}

func (p *ProcessTool) kill(proc *Process) Result {
	proc.mu.Lock()
	if proc.exitCode != nil {
		proc.mu.Unlock()
		return Ok("process already exited")
	}
	cmd := proc.cmd
	proc.mu.Unlock()

	// Signal the whole group: a backgrounded or piped command leaves
	// grandchildren that outlive their sh, and those held the output pipes open
	// so cmd.Wait() never returned.
	killProcessGroup(cmd)

	// The exit code is deliberately NOT recorded here. cmd.Wait() is the single
	// authority and is already waiting; writing a guess meant the same pid
	// reported one code right after kill and a different one once Wait
	// overwrote it. Callers read the settled value from `process read` or from
	// the process_exited event.
	resp := map[string]any{"killed": true}
	out, _ := json.Marshal(resp)
	return Ok(string(out))
}
