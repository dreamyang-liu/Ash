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

			result, notifications := executeTool(target, params.Arguments, params.AgentID)

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
			result, notifications := executeTool(target, params.Arguments, params.AgentID)
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
		WriteTimeout:      6 * time.Minute, // slightly over shell max (5min) to allow response write
		IdleTimeout:       120 * time.Second,
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
