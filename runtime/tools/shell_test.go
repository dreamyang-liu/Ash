package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestKillReportsOneStableExitCodeAndLeavesNoDescendants(t *testing.T) {
	// A backgrounded job leaves a grandchild that outlives its sh. Killing only
	// the direct child left it running, and because it held the output pipes
	// open cmd.Wait() never returned -- so `kill` writing its own -9 and Wait
	// writing -1 raced, and the same pid answered differently depending on
	// timing. One authority (Wait), one convention (128+signal), whole group.
	shell := &ShellTool{}
	process := &ProcessTool{}

	// A marker unique to this run: counting every "sleep N" on the host would
	// also count a stray left by some earlier run, failing this test for
	// someone else's mess. The env var is never read -- it exists only to make
	// the descendant's cmdline identifiable.
	marker := fmt.Sprintf("ASH_KILL_PROBE_%d_%d", os.Getpid(), time.Now().UnixNano())
	started := shell.Execute(map[string]any{
		"command":    "sh -c 'sleep 7117 # " + marker + "' & wait",
		"background": true,
	})
	if !started.Success {
		t.Fatalf("start background job: %s", started.Error)
	}
	var start struct{ PID string }
	if err := json.Unmarshal([]byte(started.Output), &start); err != nil {
		t.Fatalf("decode start: %v", err)
	}

	// Let the grandchild come up before killing, or there is nothing to orphan.
	time.Sleep(200 * time.Millisecond)
	if killed := process.Execute(map[string]any{
		"pid": start.PID, "action": "kill",
	}); !killed.Success {
		t.Fatalf("kill: %s", killed.Error)
	}

	readCode := func() (int, bool) {
		r := process.Execute(map[string]any{"pid": start.PID, "action": "read"})
		if !r.Success {
			t.Fatalf("read: %s", r.Error)
		}
		var snap struct {
			ExitCode *int `json:"exit_code"`
			Running  bool `json:"running"`
		}
		if err := json.Unmarshal([]byte(r.Output), &snap); err != nil {
			t.Fatalf("decode snapshot: %v", err)
		}
		if snap.ExitCode == nil {
			return 0, false
		}
		return *snap.ExitCode, true
	}

	// Wait for the code to settle, then confirm it does not change afterwards.
	deadline := time.Now().Add(3 * time.Second)
	code, ok := readCode()
	for !ok && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
		code, ok = readCode()
	}
	if !ok {
		t.Fatal("killed process never reported an exit code")
	}
	const wantCode = 128 + 9 // SIGKILL, the convention every shell reports
	if code != wantCode {
		t.Errorf("exit code %d, want %d", code, wantCode)
	}
	for i := 0; i < 5; i++ {
		time.Sleep(100 * time.Millisecond)
		if again, _ := readCode(); again != code {
			t.Fatalf("exit code changed after settling: %d then %d", code, again)
		}
	}

	// And nothing is left running.
	if n := countProcessesMatching(t, marker); n != 0 {
		t.Errorf("%d descendant(s) survived the kill", n)
	}
}

// countProcessesMatching counts live processes whose cmdline contains needle,
// skipping the scanning process itself.
func countProcessesMatching(t *testing.T, needle string) int {
	t.Helper()
	entries, err := os.ReadDir("/proc")
	if err != nil {
		t.Skipf("no /proc on this platform: %v", err)
	}
	self := strconv.Itoa(os.Getpid())
	count := 0
	for _, entry := range entries {
		if !entry.IsDir() || entry.Name() == self {
			continue
		}
		if _, err := strconv.Atoi(entry.Name()); err != nil {
			continue
		}
		raw, err := os.ReadFile("/proc/" + entry.Name() + "/cmdline")
		if err != nil {
			continue // exited between listing and reading
		}
		if strings.Contains(strings.ReplaceAll(string(raw), "\x00", " "), needle) {
			count++
		}
	}
	return count
}

func TestBackgroundProcessExitWaitsForCapturedOutput(t *testing.T) {
	shell := &ShellTool{}
	process := &ProcessTool{}

	for i := 0; i < 25; i++ {
		stdout := fmt.Sprintf("stdout-%d", i)
		stderr := fmt.Sprintf("stderr-%d", i)
		started := shell.Execute(map[string]any{
			"command":    fmt.Sprintf("printf %s; printf %s >&2", stdout, stderr),
			"background": true,
		})
		if !started.Success {
			t.Fatalf("start background process: %s", started.Error)
		}

		var startData struct {
			PID string `json:"pid"`
		}
		if err := json.Unmarshal([]byte(started.Output), &startData); err != nil {
			t.Fatalf("decode start result: %v", err)
		}

		deadline := time.Now().Add(2 * time.Second)
		for {
			result := process.Execute(map[string]any{
				"pid":    startData.PID,
				"action": "read",
			})
			if !result.Success {
				t.Fatalf("read process %s: %s", startData.PID, result.Error)
			}

			var snapshot struct {
				Stdout  string `json:"stdout"`
				Stderr  string `json:"stderr"`
				Running bool   `json:"running"`
			}
			if err := json.Unmarshal([]byte(result.Output), &snapshot); err != nil {
				t.Fatalf("decode process snapshot: %v", err)
			}
			if !snapshot.Running {
				if snapshot.Stdout != stdout || snapshot.Stderr != stderr {
					t.Fatalf(
						"process exited before output capture completed: stdout=%q stderr=%q",
						snapshot.Stdout, snapshot.Stderr,
					)
				}
				break
			}
			if time.Now().After(deadline) {
				t.Fatalf("background process %s did not exit", startData.PID)
			}
			time.Sleep(time.Millisecond)
		}
	}
}
