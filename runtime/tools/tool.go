package tools

// Tool is the interface all sandbox tools implement.
type Tool interface {
	Name() string
	Description() string
	Schema() map[string]any
	Execute(args map[string]any) Result
}

// Result of a tool execution.
type Result struct {
	Success bool
	Output  string
	Error   string
}

func Ok(output string) Result {
	return Result{Success: true, Output: output}
}

func Err(msg string) Result {
	return Result{Success: false, Error: msg}
}

// All returns the complete RL tool set.
func All() []Tool {
	return []Tool{
		&ShellTool{},
		&ProcessTool{},
		&EditTool{},
		&GrepTool{},
		&WebFetchTool{},
		&WebSearchTool{},
		&ArtifactTool{}, // primitive for SDK-side custom tools (binary download+verify)
	}
}

// Find returns a tool by name, or nil.
func Find(name string) Tool {
	for _, t := range All() {
		if t.Name() == name {
			return t
		}
	}
	return nil
}
