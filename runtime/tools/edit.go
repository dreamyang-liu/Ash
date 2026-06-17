package tools

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/dreamyang-liu/ash/runtime/events"
)

// EditTool provides file editing with view, str_replace, insert, and create commands.
type EditTool struct{}

func (e *EditTool) Name() string { return "text_editor" }

func (e *EditTool) Description() string {
	return "View, create, or edit files using commands: view, str_replace, insert, create"
}

func (e *EditTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"command":     map[string]any{"type": "string", "enum": []string{"view", "str_replace", "insert", "write"}, "description": "Command to execute"},
			"path":        map[string]any{"type": "string", "description": "File path"},
			"view_range":  map[string]any{"type": "array", "items": map[string]any{"type": "integer"}, "description": "[start, end] lines for view"},
			"old_str":     map[string]any{"type": "string", "description": "Text to find (str_replace)"},
			"new_str":     map[string]any{"type": "string", "description": "Replacement text (str_replace)"},
			"insert_line": map[string]any{"type": "integer", "description": "Line number to insert after"},
			"insert_text": map[string]any{"type": "string", "description": "Text to insert"},
			"file_text":   map[string]any{"type": "string", "description": "Full file content (write)"},
		},
		"required": []string{"command", "path"},
	}
}

func (e *EditTool) Execute(args map[string]any) Result {
	command, _ := args["command"].(string)
	path, _ := args["path"].(string)
	if path == "" {
		return Err("path is required")
	}

	switch command {
	case "view":
		return e.view(path, args)
	case "str_replace":
		return e.strReplace(path, args)
	case "insert":
		return e.insert(path, args)
	case "write":
		return e.write(path, args)
	default:
		return Err("unknown command: " + command)
	}
}

func (e *EditTool) view(path string, args map[string]any) Result {
	data, err := os.ReadFile(path)
	if err != nil {
		return Err(err.Error())
	}

	lines := strings.Split(string(data), "\n")
	start, end := 1, len(lines)

	if vr, ok := args["view_range"].([]any); ok && len(vr) == 2 {
		if s, ok := vr[0].(float64); ok {
			start = int(s)
		}
		if e, ok := vr[1].(float64); ok {
			end = int(e)
		}
	}

	if start < 1 {
		start = 1
	}
	if end > len(lines) {
		end = len(lines)
	}

	var b strings.Builder
	for i := start - 1; i < end; i++ {
		fmt.Fprintf(&b, "  %6d | %s\n", i+1, lines[i])
	}
	return Ok(b.String())
}

func (e *EditTool) strReplace(path string, args map[string]any) Result {
	oldStr, _ := args["old_str"].(string)
	newStr, _ := args["new_str"].(string)

	data, err := os.ReadFile(path)
	if err != nil {
		return Err(err.Error())
	}

	content := string(data)
	count := strings.Count(content, oldStr)

	if count == 0 {
		return Err("No match found")
	}
	if count > 1 {
		return Err(fmt.Sprintf("Multiple matches (%d). old_str must be unique.", count))
	}

	content = strings.Replace(content, oldStr, newStr, 1)
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return Err(err.Error())
	}

	events.Push("file_change", path, map[string]any{"path": path, "operation": "str_replace"})
	return Ok("OK")
}

func (e *EditTool) insert(path string, args map[string]any) Result {
	insertLine := 0
	if n, ok := args["insert_line"].(float64); ok {
		insertLine = int(n)
	}
	insertText, _ := args["insert_text"].(string)

	data, err := os.ReadFile(path)
	if err != nil {
		return Err(err.Error())
	}

	lines := strings.Split(string(data), "\n")
	newLines := strings.Split(insertText, "\n")

	result := make([]string, 0, len(lines)+len(newLines))
	result = append(result, lines[:insertLine]...)
	result = append(result, newLines...)
	result = append(result, lines[insertLine:]...)

	if err := os.WriteFile(path, []byte(strings.Join(result, "\n")), 0644); err != nil {
		return Err(err.Error())
	}

	events.Push("file_change", path, map[string]any{"path": path, "operation": "insert"})
	return Ok("OK")
}

func (e *EditTool) write(path string, args map[string]any) Result {
	fileText, _ := args["file_text"].(string)

	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return Err(err.Error())
	}
	if err := os.WriteFile(path, []byte(fileText), 0644); err != nil {
		return Err(err.Error())
	}

	events.Push("file_change", path, map[string]any{"path": path, "operation": "write"})
	return Ok("OK")
}
