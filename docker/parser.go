package docker

import (
	"strings"

	"github.com/hinshun/vt10x"
)

type ScrollBufferScreen struct {
	width            int
	buffer           [][]rune
	cursorX, cursorY int
}

func NewScrollBufferScreen(width int) *ScrollBufferScreen {
	return &ScrollBufferScreen{
		width:  width,
		buffer: make([][]rune, 0),
	}
}

func (s *ScrollBufferScreen) ensureHeight(h int) {
	for len(s.buffer) <= h {
		row := make([]rune, s.width)
		for i := range row {
			row[i] = ' '
		}
		s.buffer = append(s.buffer, row)
	}
}

func (s *ScrollBufferScreen) Write(data []byte) (int, error) {
	for _, b := range data {
		if b == '\n' {
			s.cursorY++
			s.cursorX = 0
		} else if b == '\r' {
			s.cursorX = 0
		} else if b == 0x08 { // backspace
			if s.cursorX > 0 {
				s.cursorX--
			}
		} else {
			s.ensureHeight(s.cursorY)
			if s.cursorX >= s.width {
				s.cursorY++
				s.cursorX = 0
				s.ensureHeight(s.cursorY)
			}
			s.buffer[s.cursorY][s.cursorX] = rune(b)
			s.cursorX++
		}
	}
	return len(data), nil
}

func (s *ScrollBufferScreen) String() string {
	var sb strings.Builder
	for _, row := range s.buffer {
		sb.WriteString(strings.TrimRight(string(row), " "))
		sb.WriteByte('\n')
	}
	return sb.String()
}

func CleanUseEmulator(buf []byte) string {
	screen := vt10x.New(vt10x.WithSize(300, 1000))
	_, err := screen.Write(buf)
	if err != nil {
		panic(err)
	}

	return cleanTerminalOutput(screen.String())
}

func cleanTerminalOutput(raw string) string {
	var sb strings.Builder
	for _, line := range strings.Split(raw, "\n") {
		trimmed := strings.TrimRight(line, " ")
		if len(strings.TrimSpace(trimmed)) == 0 {
			continue // skip full empty lines
		}
		sb.WriteString(trimmed)
		sb.WriteByte('\n')
	}
	return sb.String()
}
