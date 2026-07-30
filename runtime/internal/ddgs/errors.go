// Package ddgs is a Go port of the Python "ddgs" library
// (https://github.com/deedy5/ddgs) — DDGS | Dux Distributed Global Search.
// A metasearch library that aggregates results from diverse web search services.
package ddgs

import "errors"

// ErrDDGS is the base error for the ddgs package.
// All errors returned by this package wrap ErrDDGS.
var ErrDDGS = errors.New("ddgs")

// ErrRatelimit is returned for rate limit exceeded errors during requests.
var ErrRatelimit = errors.New("ddgs: ratelimit")

// ErrTimeout is returned for timeout errors during requests.
var ErrTimeout = errors.New("ddgs: timeout")

// ErrNoResults is returned when no engine produced any results.
var ErrNoResults = errors.New("ddgs: no results found")
