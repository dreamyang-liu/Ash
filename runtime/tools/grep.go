package tools

import (
	"bufio"
	"bytes"
	"fmt"
	"os/exec"
	goruntime "runtime"
	"strings"
	"sync"
)

const rgVersion = "14.1.1"

// rgReleaseDir is the tarball's top-level directory for this machine, or ""
// when no static release exists for it. Arch matters: a wrong-arch binary
// downloads, unpacks and lands on PATH without complaint, and only fails at
// exec time -- which is also why verification below runs rg instead of
// looking it up.
func rgReleaseDir() string {
	switch goruntime.GOARCH {
	case "amd64":
		return "ripgrep-" + rgVersion + "-x86_64-unknown-linux-musl"
	case "arm64":
		return "ripgrep-" + rgVersion + "-aarch64-unknown-linux-gnu"
	default:
		return ""
	}
}

// Provisioning ripgrep is retried until it succeeds, and only success is
// remembered. A sync.Once here cached the *failure*: one attempt in a bare
// image disabled grep_files for the runtime's whole life, even after rg was
// installed by other means moments later. The mutex is held across the install
// so concurrent callers wait for one attempt instead of racing several.
var (
	rgMu    sync.Mutex
	rgReady bool
)

// rgInstallAttempts are tried in order: the static tarball first, package
// managers as fallbacks. It used to be the other way around, and the cost
// hid in plain sight: on apt images the first grep_files ran `apt-get
// update`, whose package indexes alone are ~76 MiB of disk writes (measured
// ~89 MiB total and ~15s), charged to whatever was watching the disk -- for
// a checkpointed rollout, the episode's first snapshot. The tarball is a
// ~5 MiB static binary. Package managers remain for machines the release
// does not cover and for images without a fetcher; the apt route still
// refreshes its index first, because without `apt-get update` a slim image
// reports "Unable to locate package ripgrep" and the tool looked broken.
func rgInstallAttempts() []string {
	var attempts []string
	if dir := rgReleaseDir(); dir != "" {
		tarball := "https://github.com/BurntSushi/ripgrep/releases/download/" +
			rgVersion + "/" + dir + ".tar.gz"
		fetch := " | tar xz -C /tmp && cp /tmp/" + dir + "/rg /usr/local/bin/rg" +
			"; rm -rf /tmp/" + dir
		attempts = append(attempts,
			"curl -fsSL "+tarball+fetch,
			"wget -qO- "+tarball+fetch,
		)
	}
	// Package indexes are fetched only to find one package; dropping them
	// afterwards keeps the ~50-80 MiB of lists out of the sandbox's disk
	// state (and out of a checkpointed rollout's first snapshot). The
	// install itself stays, so a failure still reports through `err`.
	return append(attempts,
		"apt-get update -qq && apt-get install -y -qq ripgrep && rm -rf /var/lib/apt/lists/*",
		"apk add --no-cache ripgrep",
		"yum install -y -q ripgrep && yum clean all -q",
	)
}

// rgWorks reports whether an rg on PATH actually executes. LookPath alone
// accepted a wrong-arch binary (present, executable bit set, exec format
// error at run time), which would have disabled grep_files while looking
// provisioned.
func rgWorks() bool {
	return exec.Command("rg", "--version").Run() == nil
}

func ensureRipgrep() error {
	rgMu.Lock()
	defer rgMu.Unlock()

	if rgReady {
		return nil
	}
	// Re-checked on every attempt, so an rg that appeared since the last
	// failure is picked up instead of being ignored for good.
	if rgWorks() {
		rgReady = true
		return nil
	}

	attempts := rgInstallAttempts()
	var failures []string
	for _, attempt := range attempts {
		out, err := exec.Command("sh", "-c", attempt).CombinedOutput()
		if err == nil {
			if rgWorks() {
				rgReady = true
				return nil
			}
			err = fmt.Errorf("completed but rg still not runnable")
		}
		failures = append(failures, fmt.Sprintf("%s: %v (%s)",
			strings.Fields(attempt)[0], err, lastLine(out)))
	}
	// Say what was tried: "failed to install ripgrep" alone gives an operator
	// nothing to act on.
	return fmt.Errorf("failed to install ripgrep; tried %d methods: %s",
		len(attempts), strings.Join(failures, "; "))
}

// lastLine is the most informative part of a package manager's noise.
func lastLine(out []byte) string {
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	last := lines[len(lines)-1]
	if len(last) > 200 {
		return last[:200]
	}
	return last
}

// GrepTool searches files using ripgrep.
type GrepTool struct{}

func (g *GrepTool) Name() string { return "grep_files" }

func (g *GrepTool) Description() string {
	return "Search files using ripgrep with a regex pattern"
}

func (g *GrepTool) Schema() map[string]any {
	props := map[string]any{
		"pattern": map[string]any{"type": "string", "description": "Regex pattern"},
		"path":    map[string]any{"type": "string", "default": ".", "description": "Search path"},
		"include": map[string]any{"type": "string", "description": "File glob (e.g. *.py)"},
		"limit":   map[string]any{"type": "integer", "default": 100, "description": "Maximum number of matching lines to return globally"},
	}
	// Matches are bounded by `limit` (count) and by the shared byte budget.
	for k, v := range outputBoundSchema() {
		props[k] = v
	}
	return map[string]any{
		"type":       "object",
		"properties": props,
		"required":   []string{"pattern"},
	}
}

func (g *GrepTool) Execute(args map[string]any) Result {
	if err := ensureRipgrep(); err != nil {
		return Err(err.Error())
	}

	pattern, _ := args["pattern"].(string)
	if pattern == "" {
		return Err("pattern is required")
	}

	path := "."
	if p, ok := args["path"].(string); ok && p != "" {
		path = p
	}

	limit := 100
	if l, ok := args["limit"].(float64); ok && int(l) > 0 {
		limit = int(l)
	}

	cmdArgs := []string{
		"--line-number", "--no-heading", "--color=never",
		"--regexp", pattern,
	}
	if include, ok := args["include"].(string); ok && include != "" {
		cmdArgs = append(cmdArgs, "--glob", include)
	}
	cmdArgs = append(cmdArgs, "--", path)

	cmd := exec.Command("rg", cmdArgs...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return Err("failed to create stdout pipe: " + err.Error())
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Start(); err != nil {
		return Err("failed to start ripgrep: " + err.Error())
	}

	var lines []string
	truncated := false
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		if len(lines) >= limit {
			truncated = true
			_ = cmd.Process.Kill()
			break
		}
		lines = append(lines, scanner.Text())
	}
	scanErr := scanner.Err()
	err = cmd.Wait()

	if scanErr != nil {
		return Err("read ripgrep output: " + scanErr.Error())
	}
	if len(lines) > 0 {
		output := strings.Join(lines, "\n")
		if truncated {
			output += fmt.Sprintf("\n... (truncated at %d matches)", limit)
		}
		// `limit` bounds the number of matches; the shared byte bound also
		// applies, since a few very long lines can still be huge.
		return Ok(boundToolOutput(output+"\n", args))
	}
	if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 1 {
		return Ok("No matches found.")
	}
	if truncated {
		return Ok(fmt.Sprintf("... (truncated at %d matches)\n", limit))
	}
	if err == nil {
		return Ok("")
	}
	return Err(stderr.String())
}
