package tools

import (
	"fmt"
	"os"
	"strings"
)

// ReadFileTool reads a file and returns contents with line numbers.
type ReadFileTool struct{}

func (r *ReadFileTool) Name() string { return "read_file" }

func (r *ReadFileTool) Description() string {
	return "Read a file and return contents with line numbers"
}

func (r *ReadFileTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path":   map[string]any{"type": "string", "description": "File path"},
			"offset": map[string]any{"type": "integer", "description": "Start line (1-based, default: 1)"},
			"limit":  map[string]any{"type": "integer", "description": "Number of lines to read (default: all)"},
		},
		"required": []string{"path"},
	}
}

func (r *ReadFileTool) Execute(args map[string]any) Result {
	path, _ := args["path"].(string)
	if path == "" {
		return Err("path is required")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return Err(err.Error())
	}

	lines := strings.Split(string(data), "\n")
	// Remove trailing empty element from final newline
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}

	offset := 1
	if o, ok := args["offset"].(float64); ok && int(o) > 1 {
		offset = int(o)
	}

	limit := 0
	if l, ok := args["limit"].(float64); ok && int(l) > 0 {
		limit = int(l)
	}

	start := offset - 1
	if start >= len(lines) {
		return Ok("")
	}

	end := len(lines)
	if limit > 0 && start+limit < end {
		end = start + limit
	}

	var b strings.Builder
	for i := start; i < end; i++ {
		fmt.Fprintf(&b, "  %6d | %s\n", i+1, lines[i])
	}

	if end < len(lines) {
		fmt.Fprintf(&b, "\n... (%d more lines)\n", len(lines)-end)
	}

	return Ok(b.String())
}
