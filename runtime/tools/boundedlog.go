package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
)

const (
	// fallbackMaxOutputBytes is the shared default budget for tool output:
	// generous, because deciding how much belongs in an LLM context is a
	// consumer-side concern. The runtime's job is only to stay alive.
	fallbackMaxOutputBytes = 15 << 20 // 15 MiB
	minMaxOutputBytes      = 1024
	// fallbackMaxOutputCeiling caps what a caller may ask for, so a single
	// call cannot OOM the runtime (and with it the sandbox).
	fallbackMaxOutputCeiling = 64 << 20 // 64 MiB
	// defaultTruncateMode keeps the historical 40/60 head:tail split.
	defaultTruncateMode = "H2T3"
)

// Shared output bounds, overridable per deployment. Tools that produce byte
// output (shell, text_editor view, grep_files, web_fetch) all resolve their
// budget through these, so one knob applies everywhere.
var (
	defaultMaxOutputBytes = envInt("ASH_MAX_OUTPUT_BYTES", fallbackMaxOutputBytes)
	maxMaxOutputBytes     = envInt("ASH_MAX_OUTPUT_CEILING_BYTES", fallbackMaxOutputCeiling)
)

// envInt reads a positive integer from the environment, falling back to def.
func envInt(name string, def int) int {
	if raw := os.Getenv(name); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			return n
		}
	}
	return def
}

// boundText applies the byte budget and truncate mode to output a tool has
// already assembled in memory, reusing the same head/tail logic as streamed
// output so every tool truncates identically.
func boundText(text string, max int, mode truncateMode) (string, bool) {
	log := NewBoundedLogMode(max, mode)
	log.Write([]byte(text))
	return log.Render()
}

// boundToolOutput is the one-call form used by non-streaming tools: it reads
// the budget and mode from the call arguments and bounds the text.
func boundToolOutput(text string, args map[string]any) string {
	out, _ := boundText(text, outputBytesArg(args), truncateModeArg(args))
	return out
}

// outputBoundSchema returns the two schema properties every byte-producing
// tool accepts, so their surface stays identical.
func outputBoundSchema() map[string]any {
	return map[string]any{
		"max_output_bytes": map[string]any{
			"type":        "integer",
			"default":     defaultMaxOutputBytes,
			"description": "Total bytes of output to return. Larger output is truncated per truncate_mode.",
		},
		"truncate_mode": map[string]any{
			"type":        "string",
			"default":     defaultTruncateMode,
			"description": "How to divide the byte budget when output is too long: \"H<n>T<n>\" with weights for the head and tail sections. H2T3 keeps the first 40% and last 60%; T1 keeps only the tail (useful for errors at the end); H1 keeps only the beginning.",
		},
	}
}

// truncateMode says which parts of an over-long output to keep and in what
// proportion: "H<n>T<n>", where H is the head, T is the tail, and the numbers
// are weights (H1T1 and H50T50 both mean half each). Either section may be
// omitted: "T1" keeps only the tail, "H1" only the beginning.
//
// Total size is a separate, orthogonal parameter (max_output_bytes): the mode
// decides how the budget is divided, not how large it is.
type truncateMode struct {
	headWeight int
	tailWeight int
}

var truncateModeRe = regexp.MustCompile(`^(?:H(\d+))?(?:T(\d+))?$`)

// parseTruncateMode parses "H2T3" style modes. Unparseable input or zero total
// weight falls back to def, so a bad value degrades instead of failing a call.
func parseTruncateMode(s string, def truncateMode) truncateMode {
	m := truncateModeRe.FindStringSubmatch(strings.ToUpper(strings.TrimSpace(s)))
	if m == nil {
		return def
	}
	head, tail := 0, 0
	if m[1] != "" {
		head, _ = strconv.Atoi(m[1])
	}
	if m[2] != "" {
		tail, _ = strconv.Atoi(m[2])
	}
	if head+tail == 0 {
		return def
	}
	return truncateMode{headWeight: head, tailWeight: tail}
}

// envTruncateMode is the process-wide default, overridable per call via the
// truncate_mode argument.
var envTruncateMode = parseTruncateMode(
	os.Getenv("ASH_TRUNCATE_MODE"),
	truncateMode{headWeight: 2, tailWeight: 3},
)

// truncateModeArg reads a per-call mode override.
func truncateModeArg(args map[string]any) truncateMode {
	if s, ok := args["truncate_mode"].(string); ok && s != "" {
		return parseTruncateMode(s, envTruncateMode)
	}
	return envTruncateMode
}

// hasTruncateModeArg reports whether the caller named a mode explicitly, so a
// tool can pick a better default without overriding a stated preference.
func hasTruncateModeArg(args map[string]any) bool {
	s, ok := args["truncate_mode"].(string)
	return ok && s != ""
}

// tailOnlyMode keeps the whole budget at the end of the output.
var tailOnlyMode = truncateMode{tailWeight: 1}

// BoundedLog stores command output with a fixed memory ceiling. Once the
// output exceeds the limit, Render keeps the head and tail sections in the
// proportion given by its truncate mode.
type BoundedLog struct {
	mu    sync.Mutex
	max   int
	mode  truncateMode
	head  []byte
	tail  []byte
	total int64
}

// NewBoundedLog creates a log bounded to max bytes using the process-default
// truncate mode.
func NewBoundedLog(max int) *BoundedLog {
	return NewBoundedLogMode(max, envTruncateMode)
}

// NewBoundedLogMode creates a log bounded to max bytes with an explicit
// truncate mode.
func NewBoundedLogMode(max int, mode truncateMode) *BoundedLog {
	return &BoundedLog{max: normalizeMaxOutputBytes(max), mode: mode}
}

func normalizeMaxOutputBytes(max int) int {
	if max <= 0 {
		return defaultMaxOutputBytes
	}
	if max < minMaxOutputBytes {
		return minMaxOutputBytes
	}
	if max > maxMaxOutputBytes {
		return maxMaxOutputBytes
	}
	return max
}

func outputBytesArg(args map[string]any) int {
	max := defaultMaxOutputBytes
	if v, ok := args["max_output_bytes"].(float64); ok {
		max = int(v)
	}
	return normalizeMaxOutputBytes(max)
}

func (b *BoundedLog) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	n := len(p)
	b.total += int64(n)

	headLimit := b.headLimit()
	if len(b.head) < headLimit {
		remaining := headLimit - len(b.head)
		if remaining > n {
			remaining = n
		}
		b.head = append(b.head, p[:remaining]...)
	}

	tailLimit := b.tailLimit()
	if tailLimit > 0 {
		b.tail = append(b.tail, p...)
		if len(b.tail) > tailLimit {
			copy(b.tail, b.tail[len(b.tail)-tailLimit:])
			b.tail = b.tail[:tailLimit]
		}
	}

	return n, nil
}

func (b *BoundedLog) Render() (string, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.total <= int64(b.max) {
		total := int(b.total)
		headLimit := b.headLimit()
		tailLimit := b.tailLimit()
		if total <= headLimit {
			return string(b.head), false
		}
		if total <= tailLimit {
			return string(b.tail), false
		}
		overlap := b.max - total
		if overlap < 0 {
			overlap = 0
		}
		if overlap > len(b.tail) {
			overlap = len(b.tail)
		}
		return string(b.head) + string(b.tail[overlap:]), false
	}

	headLimit := b.headLimit()
	tailLimit := b.tailLimit()
	head := b.head
	if len(head) > headLimit {
		head = head[:headLimit]
	}
	tail := b.tail
	if len(tail) > tailLimit {
		tail = tail[len(tail)-tailLimit:]
	}

	marker := fmt.Sprintf(
		"\n... [output truncated: %d bytes total, showing first %d bytes and last %d bytes] ...\n",
		b.total, len(head), len(tail),
	)
	return string(head) + marker + string(tail), true
}

func (b *BoundedLog) TotalBytes() int64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.total
}

// headLimit is the byte budget for the head section, derived from the mode's
// weights. A mode with no head weight (e.g. "T1") yields 0.
func (b *BoundedLog) headLimit() int {
	total := b.mode.headWeight + b.mode.tailWeight
	if total <= 0 {
		return b.max * 2 / 5 // defensive: fall back to the 40/60 default
	}
	if b.mode.headWeight == 0 {
		return 0
	}
	if b.mode.tailWeight == 0 {
		return b.max
	}
	return b.max * b.mode.headWeight / total
}

// tailLimit is whatever the head does not claim.
func (b *BoundedLog) tailLimit() int {
	return b.max - b.headLimit()
}

// CommandOutcome is what running a command produced. One schema for both ways
// of running one: `shell` in the foreground and `process read` on a background
// pid report the same fields, so a consumer written for either works with both.
//
// Streams stay separate. Merging them into one blob was lossy in three ways: the
// "[stderr]" divider it inserted is forgeable by any command that prints one,
// the exit code had nowhere to go (`exit 5` prints nothing and arrived as an
// empty string), and truncation was described in prose a consumer had to grep.
//
// ExitCode is nil while a process is still running -- 0 and "not known yet" are
// different answers.
//
// Every field is always present. `running: false` and `timed_out: false` are
// answers, not absences, and omitempty would make a consumer unable to tell "it
// finished" from "this runtime never said" -- so a poller would have to treat a
// missing key as either, and pick wrong on one of them.
type CommandOutcome struct {
	Stdout          string `json:"stdout"`
	Stderr          string `json:"stderr"`
	ExitCode        *int   `json:"exit_code"`
	Running         bool   `json:"running"`
	TimedOut        bool   `json:"timed_out"`
	StdoutBytes     int64  `json:"stdout_bytes"`
	StderrBytes     int64  `json:"stderr_bytes"`
	StdoutTruncated bool   `json:"stdout_truncated"`
	StderrTruncated bool   `json:"stderr_truncated"`
	MaxPerStream    int    `json:"max_per_stream"`
}

// commandOutcome reads both logs into the shared schema.
func commandOutcome(stdout, stderr *BoundedLog, tailLines int) CommandOutcome {
	stdoutText, stdoutTruncated := stdout.Render()
	stderrText, stderrTruncated := stderr.Render()
	if tailLines > 0 {
		stdoutText = lastNLines(stdoutText, tailLines)
		stderrText = lastNLines(stderrText, tailLines)
	}
	return CommandOutcome{
		Stdout:          stdoutText,
		Stderr:          stderrText,
		StdoutBytes:     stdout.TotalBytes(),
		StderrBytes:     stderr.TotalBytes(),
		StdoutTruncated: stdoutTruncated,
		StderrTruncated: stderrTruncated,
		MaxPerStream:    stdout.max,
	}
}

// Result renders an outcome for the wire, for a tool whose own success is the
// command's success -- `shell` running one in the foreground.
//
// A plain success -- exit 0, nothing on stderr, nothing clipped -- is returned
// as bare stdout: there is nothing to say beyond it, and wrapping `echo hello`
// in JSON would make every consumer parse a payload to find one word. Anything
// else carries the schema, because something about the run cannot be read off
// the text.
func (o CommandOutcome) Result() Result {
	return o.result(o.commandSucceeded())
}

// Snapshot renders an outcome as a report ABOUT a command rather than as its
// result. `process read` answers "here is where that pid stands", and it answers
// that correctly whether the command exited 0, exited 3, or was killed -- so its
// Success describes the read, not the exit code.
//
// Keeping these apart matters: folding them together made `process read` report
// failure for every non-zero process, so a caller polling a killed pid saw a
// tool error instead of the exit code it was waiting for.
//
// Always structured, never bare stdout: the answer to "is it still running, and
// with what code" cannot be read off the output, and a caller polling a pid
// parses every reply.
func (o CommandOutcome) Snapshot() Result {
	return Result{Success: true, Output: o.encode()}
}

func (o CommandOutcome) result(success bool) Result {
	if success && o.plain() {
		return Ok(o.Stdout)
	}
	return Result{Success: success, Output: o.encode()}
}

func (o CommandOutcome) encode() string {
	payload, err := json.Marshal(o)
	if err != nil {
		// Unreachable for this struct; degrade to the merged text rather than
		// losing the output entirely.
		return o.Stdout + o.Stderr
	}
	return string(payload)
}

// commandSucceeded reports whether the command itself came out clean.
func (o CommandOutcome) commandSucceeded() bool {
	return !o.TimedOut && (o.ExitCode == nil || *o.ExitCode == 0)
}

// plain reports whether the outcome holds nothing a bare stdout would omit.
func (o CommandOutcome) plain() bool {
	return !o.Running && o.Stderr == "" &&
		!o.StdoutTruncated && !o.StderrTruncated
}
