package tools

import (
	"fmt"
	"strings"
	"sync"
)

const (
	defaultMaxOutputBytes = 1024 * 1024
	minMaxOutputBytes     = 1024
	maxMaxOutputBytes     = 8 * 1024 * 1024
)

// BoundedLog stores command output with a fixed memory ceiling. Once the
// output exceeds the limit, Render keeps the first 40% and last 60%.
type BoundedLog struct {
	mu    sync.Mutex
	max   int
	head  []byte
	tail  []byte
	total int64
}

func NewBoundedLog(max int) *BoundedLog {
	max = normalizeMaxOutputBytes(max)
	return &BoundedLog{max: max}
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

func (b *BoundedLog) headLimit() int {
	return b.max * 40 / 100
}

func (b *BoundedLog) tailLimit() int {
	return b.max - b.headLimit()
}

func renderCommandOutput(stdout, stderr *BoundedLog, tailLines int) string {
	stdoutText, stdoutTruncated := stdout.Render()
	stderrText, stderrTruncated := stderr.Render()

	if tailLines > 0 {
		stdoutText = lastNLines(stdoutText, tailLines)
		stderrText = lastNLines(stderrText, tailLines)
	}

	var b strings.Builder
	if stdoutText != "" {
		b.WriteString(stdoutText)
	}
	if stderrText != "" {
		if b.Len() > 0 && !strings.HasSuffix(b.String(), "\n") {
			b.WriteString("\n")
		}
		b.WriteString("[stderr]\n")
		b.WriteString(stderrText)
	}
	if stdoutTruncated || stderrTruncated {
		if b.Len() > 0 && !strings.HasSuffix(b.String(), "\n") {
			b.WriteString("\n")
		}
		b.WriteString(fmt.Sprintf(
			"[output_limit] stdout=%d bytes stderr=%d bytes max_per_stream=%d bytes\n",
			stdout.TotalBytes(), stderr.TotalBytes(), stdout.max,
		))
	}
	return b.String()
}
