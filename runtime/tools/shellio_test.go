package tools

// Tests for the shell tool's stdin and env arguments (run by
// `go test ./tools`; shell_test.go covers execution, timeouts and output
// bounding). Both exist so an agent need not write a temporary file to feed
// a command, nor splice variables into the command string by hand.
// User instruction: "shell 不能传 stdin 和 env ... 这个好加吗，stdin 这个
// 怎么做呢".

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func runShell(t *testing.T, args map[string]any) Result {
	t.Helper()
	return (&ShellTool{}).Execute(args)
}

func TestShellStdinIsFedToCommand(t *testing.T) {
	r := runShell(t, map[string]any{"command": "cat", "stdin": "hello from stdin\n"})
	if !r.Success {
		t.Fatalf("command failed: %s", r.Error)
	}
	if !strings.Contains(r.Output, "hello from stdin") {
		t.Errorf("stdin was not delivered, got %q", r.Output)
	}
}

func TestShellStdinIsClosedAfterWriting(t *testing.T) {
	// `cat` exits at EOF: if stdin were left open the call would hang until
	// the timeout instead of returning promptly.
	start := time.Now()
	r := runShell(t, map[string]any{"command": "cat", "stdin": "x", "timeout": float64(10)})
	if !r.Success {
		t.Fatalf("command failed: %s", r.Error)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("stdin was not closed: command took %v", elapsed)
	}
}

func TestShellStdinAvoidsTempFile(t *testing.T) {
	// The motivating case: run a script without first writing it to disk.
	r := runShell(t, map[string]any{
		"command": "sh -s",
		"stdin":   "echo scripted-$((2+3))\n",
	})
	if !r.Success {
		t.Fatalf("command failed: %s", r.Error)
	}
	if !strings.Contains(r.Output, "scripted-5") {
		t.Errorf("script from stdin did not run, got %q", r.Output)
	}
}

func TestShellNoStdinMeansImmediateEOF(t *testing.T) {
	// Without stdin the command must not block waiting for input.
	r := runShell(t, map[string]any{"command": "cat", "timeout": float64(5)})
	if !r.Success {
		t.Fatalf("expected clean exit on empty stdin, got %#v", r)
	}
}

func TestShellEnvIsAddedNotReplaced(t *testing.T) {
	r := runShell(t, map[string]any{
		"command": "echo \"$MY_VAR|$(test -n \"$PATH\" && echo path-kept)\"",
		"env":     map[string]any{"MY_VAR": "custom"},
	})
	if !r.Success {
		t.Fatalf("command failed: %s", r.Error)
	}
	if !strings.Contains(r.Output, "custom") {
		t.Errorf("env var was not set, got %q", r.Output)
	}
	if !strings.Contains(r.Output, "path-kept") {
		t.Errorf("existing environment should survive, got %q", r.Output)
	}
}

func TestShellEnvHandlesAwkwardValues(t *testing.T) {
	// The point of a structured argument: values needing quotes or spaces
	// would have to be escaped by hand in a 'KEY=value cmd' prefix.
	awkward := `a b "c" 'd' $e`
	r := runShell(t, map[string]any{
		"command": "printf '%s' \"$AWKWARD\"",
		"env":     map[string]any{"AWKWARD": awkward},
	})
	if !r.Success {
		t.Fatalf("command failed: %s", r.Error)
	}
	if r.Output != awkward {
		t.Errorf("value was mangled:\n got  %q\n want %q", r.Output, awkward)
	}
}

func TestShellEnvValidation(t *testing.T) {
	if r := runShell(t, map[string]any{
		"command": "true", "env": map[string]any{"BAD=NAME": "x"},
	}); r.Success || !strings.Contains(r.Error, "invalid env name") {
		t.Errorf("expected invalid name error, got %#v", r)
	}
	if r := runShell(t, map[string]any{
		"command": "true", "env": map[string]any{"NUM": float64(5)},
	}); r.Success || !strings.Contains(r.Error, "must be a string") {
		t.Errorf("expected non-string value error, got %#v", r)
	}
}

func TestShellStdinAndEnvWorkInBackground(t *testing.T) {
	r := runShell(t, map[string]any{
		"command":    "cat; echo \"env=$BG_VAR\"",
		"stdin":      "piped\n",
		"env":        map[string]any{"BG_VAR": "yes"},
		"background": true,
	})
	if !r.Success {
		t.Fatalf("background start failed: %s", r.Error)
	}
	var started struct{ Pid string }
	if err := json.Unmarshal([]byte(r.Output), &started); err != nil {
		t.Fatalf("bad start payload %q: %v", r.Output, err)
	}

	// Poll until the process finishes, then check both features applied.
	deadline := time.Now().Add(10 * time.Second)
	for {
		out := (&ProcessTool{}).Execute(map[string]any{"pid": started.Pid, "action": "read"})
		if !out.Success {
			t.Fatalf("process read failed: %s", out.Error)
		}
		if strings.Contains(out.Output, "piped") && strings.Contains(out.Output, "env=yes") {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("background process did not show stdin/env effects: %s", out.Output)
		}
		time.Sleep(100 * time.Millisecond)
	}
}
