package utils

import "regexp"

func StripAnsi(str string) string {
	ansiRegex := regexp.MustCompile(`\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])`)
	return ansiRegex.ReplaceAllString(str, "")
}
