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

func TestForegroundTimeoutKillsDescendantsAndReturns(t *testing.T) {
	// This is the synchronous form of the background kill regression above.
	// CommandContext's default cancellation kills only the direct `sh`. A
	// backgrounded descendant then keeps stdout/stderr open, so cmd.Run waits
	// forever even though the requested timeout elapsed.
	marker := fmt.Sprintf("ASH_TIMEOUT_PROBE_%d_%d", os.Getpid(), time.Now().UnixNano())
	start := time.Now()
	result := (&ShellTool{}).Execute(map[string]any{
		"command": "sh -c 'sleep 7117 # " + marker + "' & wait",
		"timeout": float64(1),
	})
	elapsed := time.Since(start)

	if result.Success {
		t.Fatal("a timed-out command must fail")
	}
	if elapsed > 4*time.Second {
		t.Fatalf("timeout returned after %v, want no more than 4s", elapsed)
	}
	var outcome struct {
		ExitCode *int `json:"exit_code"`
		TimedOut bool `json:"timed_out"`
		Running  bool `json:"running"`
	}
	if err := json.Unmarshal([]byte(result.Output), &outcome); err != nil {
		t.Fatalf("decode timeout outcome %q: %v", result.Output, err)
	}
	if !outcome.TimedOut || outcome.Running {
		t.Fatalf("timeout outcome = %+v, want timed_out and not running", outcome)
	}
	if outcome.ExitCode == nil || *outcome.ExitCode != 128+9 {
		t.Fatalf("exit_code = %v, want %d", outcome.ExitCode, 128+9)
	}
	if n := countProcessesMatching(t, marker); n != 0 {
		t.Fatalf("%d descendant(s) survived the foreground timeout", n)
	}

	// A leaked pipe or process must not poison the runtime after the timeout.
	if followUp := (&ShellTool{}).Execute(map[string]any{
		"command": "printf follow-up-ok",
		"timeout": float64(2),
	}); !followUp.Success || followUp.Output != "follow-up-ok" {
		t.Fatalf("follow-up shell failed after timeout: %#v", followUp)
	}
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

// TestSnapshotOfANonZeroProcessIsASuccessfulRead pins the distinction between
// "the command failed" and "reading about the command failed". Folding them
// together made `process read` report a tool error for every non-zero pid, so a
// caller polling for an exit code saw a failure instead of the code it wanted --
// and the kill test above caught it only indirectly.
func TestSnapshotOfANonZeroProcessIsASuccessfulRead(t *testing.T) {
	shell := &ShellTool{}
	process := &ProcessTool{}

	started := shell.Execute(map[string]any{
		"command": "exit 9", "background": true,
	})
	if !started.Success {
		t.Fatalf("spawn: %s", started.Error)
	}
	var start struct{ PID string }
	if err := json.Unmarshal([]byte(started.Output), &start); err != nil {
		t.Fatalf("decode start: %v", err)
	}

	deadline := time.Now().Add(3 * time.Second)
	for {
		r := process.Execute(map[string]any{"pid": start.PID, "action": "read"})
		if !r.Success {
			t.Fatalf("reading a non-zero process must succeed, got error: %s", r.Error)
		}
		var snap struct {
			ExitCode *int `json:"exit_code"`
			Running  bool `json:"running"`
		}
		// Always structured: "is it running, and with what code" cannot be read
		// off the output, so a poller parses every reply.
		if err := json.Unmarshal([]byte(r.Output), &snap); err != nil {
			t.Fatalf("a snapshot must be JSON, got %q: %v", r.Output, err)
		}
		if snap.ExitCode != nil {
			if *snap.ExitCode != 9 {
				t.Fatalf("exit_code = %d, want 9", *snap.ExitCode)
			}
			if snap.Running {
				t.Fatal("running should be false once an exit code is known")
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("exit code never settled")
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// TestForegroundFailureStillFails is the other half: a command that ran and came
// out non-zero is a failed tool call, because for `shell` the command's success
// IS the tool's success.
func TestForegroundFailureStillFails(t *testing.T) {
	r := (&ShellTool{}).Execute(map[string]any{"command": "exit 7"})
	if r.Success {
		t.Fatal("a command exiting 7 must report failure")
	}
	var snap struct {
		ExitCode *int `json:"exit_code"`
	}
	if err := json.Unmarshal([]byte(r.Output), &snap); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if snap.ExitCode == nil || *snap.ExitCode != 7 {
		t.Fatalf("exit_code = %v, want 7", snap.ExitCode)
	}
}
