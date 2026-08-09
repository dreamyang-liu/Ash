package tools

// Tests for the configurable truncate mode in boundedlog.go (run by
// `go test ./tools`; shell_test.go covers command execution, not the
// head/end split policy). Mode is "H<n>E<n>": weights for the head and end
// sections, with the total budget given separately by max_output_bytes.
// User instruction: mode 表示 开始:结束 的比例，byte 数量是 total byte 数量;
// "那我觉得要不就没有middle ？" (no middle section).

import (
	"strings"
	"testing"
)

func TestParseTruncateMode(t *testing.T) {
	def := truncateMode{headWeight: 2, endWeight: 3}
	cases := []struct {
		in        string
		head, end int
	}{
		{"H2E3", 2, 3},
		{"H1E1", 1, 1},
		{"h1e1", 1, 1},   // case-insensitive
		{" H1E1 ", 1, 1}, // trimmed
		{"E1", 0, 1},     // tail only
		{"H1", 1, 0},     // head only
		{"H50E50", 50, 50},
		{"", 2, 3},        // empty -> default
		{"garbage", 2, 3}, // unparseable -> default
		{"H0E0", 2, 3},    // zero total weight -> default
		{"H1M2E3", 2, 3},  // middle is not supported -> default
	}
	for _, c := range cases {
		got := parseTruncateMode(c.in, def)
		if got.headWeight != c.head || got.endWeight != c.end {
			t.Errorf("parseTruncateMode(%q) = H%dE%d, want H%dE%d",
				c.in, got.headWeight, got.endWeight, c.head, c.end)
		}
	}
}

// renderWith writes n bytes through a BoundedLog and returns the rendered
// output plus the truncated flag. The first half is "A", the second "Z", so
// tests can tell which side survived.
func renderWith(mode string, max, n int) (string, bool) {
	log := NewBoundedLogMode(max, parseTruncateMode(mode, truncateMode{headWeight: 2, endWeight: 3}))
	body := strings.Repeat("A", n/2) + strings.Repeat("Z", n-n/2)
	log.Write([]byte(body))
	return log.Render()
}

func TestTruncateModeSplitsBudget(t *testing.T) {
	const max = 1000
	const total = 4000 // well over the budget

	// H1E1: both halves present.
	out, truncated := renderWith("H1E1", max, total)
	if !truncated {
		t.Fatal("expected truncation")
	}
	if !strings.Contains(out, "A") || !strings.Contains(out, "Z") {
		t.Errorf("H1E1 should keep both ends, got %d bytes", len(out))
	}

	// E1: tail only -- no head content at all.
	out, _ = renderWith("E1", max, total)
	if strings.Contains(out, "A") {
		t.Error("E1 should keep only the tail, but head content survived")
	}
	if !strings.Contains(out, "Z") {
		t.Error("E1 should keep tail content")
	}

	// H1: head only -- no tail content.
	out, _ = renderWith("H1", max, total)
	if strings.Contains(out, "Z") {
		t.Error("H1 should keep only the head, but tail content survived")
	}
	if !strings.Contains(out, "A") {
		t.Error("H1 should keep head content")
	}
}

func TestTruncateModeWeightsAreProportional(t *testing.T) {
	// 4000 is above minMaxOutputBytes, so the budget is used verbatim.
	log := NewBoundedLogMode(4000, truncateMode{headWeight: 1, endWeight: 3})
	if got := log.headLimit(); got != 1000 {
		t.Errorf("H1E3 head limit = %d, want 1000", got)
	}
	if got := log.tailLimit(); got != 3000 {
		t.Errorf("H1E3 tail limit = %d, want 3000", got)
	}

	// H1E1 and H50E50 are equivalent: weights, not percentages.
	a := NewBoundedLogMode(4000, truncateMode{headWeight: 1, endWeight: 1})
	b := NewBoundedLogMode(4000, truncateMode{headWeight: 50, endWeight: 50})
	if a.headLimit() != b.headLimit() {
		t.Errorf("H1E1 (%d) and H50E50 (%d) should be equivalent", a.headLimit(), b.headLimit())
	}
}

func TestTruncateModeRespectsTotalBudget(t *testing.T) {
	// The mode divides the budget; it never changes the total.
	for _, mode := range []string{"H2E3", "H1E1", "E1", "H1"} {
		out, _ := renderWith(mode, 2000, 10000)
		// Rendered output = kept bytes + a truncation marker line.
		if len(out) > 2000+300 {
			t.Errorf("mode %s exceeded the total budget: %d bytes", mode, len(out))
		}
	}
}

func TestTruncationIsSelfDescribing(t *testing.T) {
	out, truncated := renderWith("H2E3", 1000, 5000)
	if !truncated {
		t.Fatal("expected truncation")
	}
	if !strings.Contains(out, "truncated") {
		t.Errorf("truncated output must say so; got: %.120s", out)
	}
}

func TestShortOutputIsNotTruncated(t *testing.T) {
	out, truncated := renderWith("H2E3", 1000, 100)
	if truncated {
		t.Error("output within budget should not be marked truncated")
	}
	if strings.Contains(out, "truncated") {
		t.Error("output within budget should carry no truncation marker")
	}
}

func TestTruncateModeArgOverridesDefault(t *testing.T) {
	got := truncateModeArg(map[string]any{"truncate_mode": "E1"})
	if got.headWeight != 0 || got.endWeight != 1 {
		t.Errorf("per-call override = H%dE%d, want H0E1", got.headWeight, got.endWeight)
	}
	// Absent argument falls back to the process default.
	got = truncateModeArg(map[string]any{})
	if got != envTruncateMode {
		t.Errorf("absent argument should use the process default, got H%dE%d", got.headWeight, got.endWeight)
	}
}
