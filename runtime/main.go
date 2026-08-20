package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	// Embedded CA roots: TLS (artifact downloads, web_search/web_fetch)
	// works even in images without ca-certificates (debian-slim,
	// distroless). Only used when no system roots are present.
	_ "golang.org/x/crypto/x509roots/fallback"

	"github.com/dreamyang-liu/ash/runtime/tools"
)

type RPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type RPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
}

type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func main() {
	port := flag.Int("port", 3000, "port to listen on")
	mode := flag.String("mode", "http", "run mode: http or stdio")
	flag.Parse()

	allTools := tools.All()

	if *mode == "stdio" {
		runStdio(allTools)
		return
	}

	addr := fmt.Sprintf("0.0.0.0:%d", *port)
	fmt.Printf("ash-runtime v0.1.0\n  tools: %d loaded\n  listening on %s\n", len(allTools), addr)

	mux := http.NewServeMux()

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"status":"ok"}`))
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req RPCRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeRPC(w, nil, nil, &RPCError{Code: -32700, Message: "parse error"})
			return
		}
		if req.JSONRPC != "2.0" {
			writeRPC(w, req.ID, nil, &RPCError{Code: -32600, Message: "invalid request"})
			return
		}

		switch req.Method {
		case "tools/list":
			list := make([]map[string]any, 0, len(allTools))
			for _, t := range allTools {
				list = append(list, map[string]any{
					"name":        t.Name(),
					"description": t.Description(),
					"inputSchema": t.Schema(),
				})
			}
			writeRPC(w, req.ID, list, nil)

		case "tools/call":
			var params struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
				AgentID   string         `json:"agent_id"`
			}
			if err := json.Unmarshal(req.Params, &params); err != nil {
				writeRPC(w, req.ID, nil, &RPCError{Code: -32602, Message: "invalid params"})
				return
			}

			var target tools.Tool
			for _, t := range allTools {
				if t.Name() == params.Name {
					target = t
					break
				}
			}
			if target == nil {
				writeRPC(w, req.ID, nil, &RPCError{Code: -32602, Message: "unknown tool: " + params.Name})
				return
			}

			// Headers first: the tool is about to take as long as its command
			// does, and a proxy watching for a response gives up long before.
			stop, _ := beginResponse(w)
			result, notifications := executeTool(target, params.Arguments, params.AgentID)
			stop()

			text := result.Output
			isError := !result.Success
			if !result.Success && result.Error != "" {
				text = result.Error
			}

			callResult := map[string]any{
				"content":       []map[string]any{{"type": "text", "text": text}},
				"isError":       isError,
				"notifications": notifications,
			}
			writeRPC(w, req.ID, callResult, nil)

		default:
			writeRPC(w, req.ID, nil, &RPCError{Code: -32601, Message: "method not found"})
		}
	})

	// MCP Streamable HTTP endpoint — compatible with FastMCP, Claude Desktop, etc.
	mux.HandleFunc("/mcp", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			// SSE stream not supported — return 405
			http.Error(w, "SSE not supported, use POST", http.StatusMethodNotAllowed)
			return
		}
		if r.Method == http.MethodDelete {
			// Session termination — just acknowledge
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req RPCRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeMCP(w, nil, nil, &RPCError{Code: -32700, Message: "parse error"})
			return
		}

		switch req.Method {
		case "initialize":
			writeMCP(w, req.ID, map[string]any{
				"protocolVersion": "2025-03-26",
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]any{"name": "ash-runtime", "version": "0.1.0"},
			}, nil)

		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)

		case "tools/list":
			list := make([]map[string]any, 0, len(allTools))
			for _, t := range allTools {
				list = append(list, map[string]any{
					"name":        t.Name(),
					"description": t.Description(),
					"inputSchema": t.Schema(),
				})
			}
			writeMCP(w, req.ID, map[string]any{"tools": list}, nil)

		case "tools/call":
			var params struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
				AgentID   string         `json:"agent_id"`
			}
			if err := json.Unmarshal(req.Params, &params); err != nil {
				writeMCP(w, req.ID, nil, &RPCError{Code: -32602, Message: "invalid params"})
				return
			}
			var target tools.Tool
			for _, t := range allTools {
				if t.Name() == params.Name {
					target = t
					break
				}
			}
			if target == nil {
				writeMCP(w, req.ID, nil, &RPCError{Code: -32602, Message: "unknown tool: " + params.Name})
				return
			}
			stop, _ := beginResponse(w)
			result, notifications := executeTool(target, params.Arguments, params.AgentID)
			stop()
			text := result.Output
			if !result.Success && result.Error != "" {
				text = result.Error
			}
			writeMCP(w, req.ID, map[string]any{
				"content":       []map[string]any{{"type": "text", "text": text}},
				"isError":       !result.Success,
				"notifications": notifications,
			}, nil)

		default:
			writeMCP(w, req.ID, nil, &RPCError{Code: -32601, Message: "method not found: " + req.Method})
		}
	})

	srv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		// No WriteTimeout: it is a deadline on the whole response, and a tool
		// call's response is only written once its command finishes, so any
		// value here is really a cap on how long a command may run -- one that
		// silently disagrees with the caller's own `timeout` argument (up to
		// maxTimeoutSeconds). Killing the command is the shell tool's job, where
		// the caller's timeout is known; the write deadline is set per request in
		// beginResponse instead.
		IdleTimeout: 120 * time.Second,
	}
	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("server error: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("shutting down...")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("forced shutdown: %v", err)
	}
	log.Println("server stopped")
}

func writeRPC(w http.ResponseWriter, id json.RawMessage, result any, rpcErr *RPCError) {
	resp := RPCResponse{JSONRPC: "2.0", ID: id, Result: result, Error: rpcErr}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func writeMCP(w http.ResponseWriter, id json.RawMessage, result any, rpcErr *RPCError) {
	resp := RPCResponse{JSONRPC: "2.0", ID: id, Result: result, Error: rpcErr}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// keepaliveInterval is how often a long-running call writes a byte to prove the
// connection is alive. Comfortably under the 30 s that reverse proxies commonly
// allow for response headers, with room for one lost tick.
var keepaliveInterval = 10 * time.Second

// beginResponse sends the response headers before the tool runs, then keeps the
// connection warm until stop() is called.
//
// A tool call takes as long as its command does -- minutes, for a test suite or
// a compile. The runtime answers in one shot, so nothing at all reaches the
// client until the command finishes, and a proxy in between reads that silence
// as a dead upstream: AgentENV cuts the connection at 30 s
// (response_header_timeout_ms=30000), and any gateway-style backend may do the
// same. The command keeps running in the sandbox, so the caller loses the result
// of work that actually happened.
//
// Sending the headers early answers the only question a proxy is asking. The
// heartbeat is a newline, which JSON ignores as leading whitespace, so a client
// that knows nothing about any of this parses the response exactly as before --
// no protocol change, no client change, no version negotiation.
//
// Returns a stop function; the caller must invoke it before writing the body.
// Also reports whether headers were actually sent, since a ResponseWriter that
// cannot flush (a test recorder, a wrapped writer) must fall back to writing the
// whole response at the end.
func beginResponse(w http.ResponseWriter) (stop func(), started bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return func() {}, false
	}

	// Clear the write deadline for this response. A tool call is not done until
	// its command is, and the caller states how long that may be (`timeout`, up
	// to maxTimeoutSeconds); a transport-level deadline would cut the connection
	// while the command still runs, losing the result of work that happened. The
	// shell tool remains the thing that stops a command.
	if rc := http.NewResponseController(w); rc != nil {
		_ = rc.SetWriteDeadline(time.Time{})
	}

	w.Header().Set("Content-Type", "application/json")
	// A tool call answers 200 even when the tool fails: the failure travels in
	// the JSON-RPC body (or isError), not the HTTP status. That is what makes it
	// safe to commit to a status code before knowing the outcome.
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	done := make(chan struct{})
	exited := make(chan struct{})
	go func() {
		defer close(exited)
		ticker := time.NewTicker(keepaliveInterval)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				// Writing after the handler returns panics, so the write must
				// happen only while the handler is still running -- guaranteed
				// because stop() runs before the handler writes the body.
				if _, err := w.Write([]byte("\n")); err != nil {
					return // client hung up; the body write will fail too
				}
				flusher.Flush()
			}
		}
	}()

	var once sync.Once
	return func() {
		// Wait for the heartbeat to exit, not just signal it: a ResponseWriter
		// is not safe for concurrent use, and the caller writes the body next.
		once.Do(func() {
			close(done)
			<-exited
		})
	}, true
}

func runStdio(allTools []tools.Tool) {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var req RPCRequest
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			writeStdio(nil, nil, &RPCError{Code: -32700, Message: "parse error"})
			continue
		}

		switch req.Method {
		case "initialize":
			writeStdio(req.ID, map[string]any{
				"protocolVersion": "2025-03-26",
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]any{"name": "ash-runtime", "version": "0.1.0"},
			}, nil)

		case "notifications/initialized":
			// no response needed

		case "tools/list":
			list := make([]map[string]any, 0, len(allTools))
			for _, t := range allTools {
				list = append(list, map[string]any{
					"name":        t.Name(),
					"description": t.Description(),
					"inputSchema": t.Schema(),
				})
			}
			writeStdio(req.ID, map[string]any{"tools": list}, nil)

		case "tools/call":
			var params struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
				AgentID   string         `json:"agent_id"`
			}
			if err := json.Unmarshal(req.Params, &params); err != nil {
				writeStdio(req.ID, nil, &RPCError{Code: -32602, Message: "invalid params"})
				continue
			}
			var target tools.Tool
			for _, t := range allTools {
				if t.Name() == params.Name {
					target = t
					break
				}
			}
			if target == nil {
				writeStdio(req.ID, nil, &RPCError{Code: -32602, Message: "unknown tool: " + params.Name})
				continue
			}
			result, notifications := executeTool(target, params.Arguments, params.AgentID)
			text := result.Output
			if !result.Success && result.Error != "" {
				text = result.Error
			}
			writeStdio(req.ID, map[string]any{
				"content":       []map[string]any{{"type": "text", "text": text}},
				"isError":       !result.Success,
				"notifications": notifications,
			}, nil)

		default:
			writeStdio(req.ID, nil, &RPCError{Code: -32601, Message: "method not found: " + req.Method})
		}
	}
}

func writeStdio(id json.RawMessage, result any, rpcErr *RPCError) {
	resp := RPCResponse{JSONRPC: "2.0", ID: id, Result: result, Error: rpcErr}
	data, _ := json.Marshal(resp)
	fmt.Println(string(data))
}
