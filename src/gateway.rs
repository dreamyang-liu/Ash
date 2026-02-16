//! Gateway - routes tool calls to ash-mcp endpoints
//!
//! Architecture:
//! - Listens on Unix socket (~/.ash/gateway.sock) for JSON-RPC requests from CLI
//! - Manages a local ash-mcp subprocess for host execution
//! - Routes tool calls to the correct ash-mcp endpoint:
//!   - No session_id → local ash-mcp
//!   - Docker session → container's ash-mcp (HTTP)
//!   - K8s session → K8s gateway → pod's ash-mcp (HTTP)
//! - Session management tools (create/destroy/list) execute in the gateway process
//!   because they manage infrastructure (Docker containers, K8s pods)

use crate::backend::{BackendType, Session};
use crate::style;
use crate::tools::session::BackendManager;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::RwLock;

/// MCP server port inside Docker containers
const DOCKER_MCP_PORT: u16 = 3000;

/// Session tools that run in the gateway process (not forwarded to ash-mcp)
const SESSION_TOOLS: &[&str] = &[
    "session_create",
    "session_destroy",
    "session_list",
    "session_info",
    "backend_switch",
    "backend_status",
];

// ==================== JSON-RPC Types ====================

#[derive(Deserialize)]
pub struct JsonRpcRequest {
    #[allow(dead_code)]
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
}

impl JsonRpcResponse {
    fn success(id: Value, result: Value) -> Self {
        Self { jsonrpc: "2.0".into(), id, result: Some(result), error: None }
    }

    fn error(id: Value, code: i32, message: String) -> Self {
        Self { jsonrpc: "2.0".into(), id, result: None, error: Some(JsonRpcError { code, message }) }
    }
}

// ==================== Local ash-mcp Process ====================

struct LocalMcpProcess {
    child: tokio::process::Child,
    port: u16,
    url: String,
}

// ==================== Gateway ====================

pub struct Gateway {
    /// Session routing table: session_id -> MCP endpoint URL
    routes: Arc<RwLock<HashMap<String, String>>>,
    /// Backend manager for session lifecycle (Docker/K8s container management)
    backend_manager: Arc<RwLock<BackendManager>>,
    /// Local ash-mcp subprocess
    local_mcp: Arc<RwLock<Option<LocalMcpProcess>>>,
    /// HTTP client for forwarding requests
    http_client: Client,
    /// Start time for uptime tracking
    start_time: Instant,
}

impl Gateway {
    pub fn new() -> Self {
        Self {
            routes: Arc::new(RwLock::new(HashMap::new())),
            backend_manager: Arc::new(RwLock::new(BackendManager::new())),
            local_mcp: Arc::new(RwLock::new(None)),
            http_client: Client::new(),
            start_time: Instant::now(),
        }
    }

    // ==================== Local ash-mcp Management ====================

    /// Start local ash-mcp subprocess if not already running
    async fn ensure_local_mcp(&self) -> Result<(), String> {
        // Fast path: already running
        {
            let guard = self.local_mcp.read().await;
            if guard.is_some() {
                return Ok(());
            }
        }

        // Slow path: start it
        let mut guard = self.local_mcp.write().await;
        if guard.is_some() {
            return Ok(()); // double-check after lock
        }

        // Find ash-mcp binary (same directory as current exe)
        let exe = std::env::current_exe()
            .map_err(|e| format!("Failed to get current exe: {e}"))?;
        let mcp_exe = exe.parent()
            .ok_or("No parent directory")?
            .join("ash-mcp");

        if !mcp_exe.exists() {
            return Err(format!("ash-mcp binary not found at {}", mcp_exe.display()));
        }

        // Spawn with port 0 (auto-assign)
        let mut child = tokio::process::Command::new(&mcp_exe)
            .args(["--transport", "http", "--port", "0"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("Failed to spawn ash-mcp: {e}"))?;

        // Read stderr to discover assigned port (LISTENING:{port})
        let stderr = child.stderr.take()
            .ok_or("Failed to capture ash-mcp stderr")?;
        let mut reader = BufReader::new(stderr);
        let port = tokio::time::timeout(Duration::from_secs(10), async {
            let mut line = String::new();
            loop {
                line.clear();
                let n = reader.read_line(&mut line).await
                    .map_err(|e| format!("Failed to read ash-mcp stderr: {e}"))?;
                if n == 0 {
                    return Err("ash-mcp exited before reporting port".to_string());
                }
                if let Some(port_str) = line.trim().strip_prefix("LISTENING:") {
                    return port_str.parse::<u16>()
                        .map_err(|e| format!("Invalid port from ash-mcp: {e}"));
                }
            }
        })
        .await
        .map_err(|_| "Timeout waiting for ash-mcp to start".to_string())??;

        let url = format!("http://127.0.0.1:{}", port);

        // Verify it's actually ready by hitting the health endpoint
        self.wait_for_mcp_ready(&url, 10).await?;

        eprintln!("  {} ash-mcp started on port {}",
            style::ecolor(style::check(), style::GREEN),
            style::ecolor(&port.to_string(), style::CYAN));
        *guard = Some(LocalMcpProcess { child, port, url });
        Ok(())
    }

    /// Wait for an MCP HTTP endpoint to become ready
    async fn wait_for_mcp_ready(&self, base_url: &str, timeout_secs: u64) -> Result<(), String> {
        let start = Instant::now();
        let timeout = Duration::from_secs(timeout_secs);

        loop {
            if start.elapsed() > timeout {
                return Err(format!("MCP server not ready after {}s at {}", timeout_secs, base_url));
            }

            match self.http_client
                .get(base_url)
                .timeout(Duration::from_secs(2))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => return Ok(()),
                _ => tokio::time::sleep(Duration::from_millis(100)).await,
            }
        }
    }

    /// Get the local ash-mcp URL
    async fn local_mcp_url(&self) -> Result<String, String> {
        self.ensure_local_mcp().await?;
        let guard = self.local_mcp.read().await;
        Ok(guard.as_ref().unwrap().url.clone())
    }

    // ==================== HTTP Forwarding ====================

    /// Forward a JSON-RPC tools/call to an MCP endpoint
    async fn forward_tool_call(
        &self,
        url: &str,
        tool_name: &str,
        args: Value,
    ) -> Result<Value, String> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": chrono::Utc::now().timestamp_millis(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": args
            }
        });

        let response = self.http_client
            .post(url)
            .header("Content-Type", "application/json")
            .json(&request)
            .timeout(Duration::from_secs(300))
            .send()
            .await
            .map_err(|e| format!("HTTP forward failed: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(format!("MCP call failed ({}): {}", status, text));
        }

        let data: Value = response.json().await
            .map_err(|e| format!("Invalid MCP response: {e}"))?;

        // Extract result from JSON-RPC response
        if let Some(error) = data.get("error") {
            return Err(format!("MCP error: {}", error));
        }

        Ok(data.get("result").cloned().unwrap_or(Value::Null))
    }

    /// Forward a generic JSON-RPC method to an MCP endpoint
    async fn forward_method(
        &self,
        url: &str,
        method: &str,
        params: Value,
    ) -> Result<Value, String> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": chrono::Utc::now().timestamp_millis(),
            "method": method,
            "params": params
        });

        let response = self.http_client
            .post(url)
            .header("Content-Type", "application/json")
            .json(&request)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| format!("HTTP forward failed: {e}"))?;

        let data: Value = response.json().await
            .map_err(|e| format!("Invalid response: {e}"))?;

        if let Some(error) = data.get("error") {
            return Err(format!("MCP error: {}", error));
        }

        Ok(data.get("result").cloned().unwrap_or(Value::Null))
    }

    // ==================== Route Management ====================

    /// Get the MCP endpoint URL for a session, or local if no session
    async fn resolve_endpoint(&self, session_id: Option<&str>) -> Result<String, String> {
        match session_id {
            Some(sid) if sid != "local" => {
                let routes = self.routes.read().await;
                routes.get(sid).cloned()
                    .ok_or_else(|| format!("Session not found: {}", sid))
            }
            _ => self.local_mcp_url().await,
        }
    }

    /// Register a route for a newly created session
    async fn register_route(&self, session: &Session) -> Result<(), String> {
        let url = match session.backend {
            BackendType::Docker => {
                // Find the host port mapped to the MCP port inside the container
                session.ports.iter()
                    .find(|p| p.container_port == DOCKER_MCP_PORT)
                    .and_then(|p| p.host_port)
                    .map(|port| format!("http://localhost:{}", port))
                    .ok_or("No MCP port mapped for Docker session")?
            }
            BackendType::K8s => {
                // K8s routes through the external K8s gateway
                // Use the K8s gateway URL from environment
                let gw_url = std::env::var("ASH_GATEWAY_URL")
                    .unwrap_or_else(|_| "http://localhost:8081".to_string());
                format!("{}/mcp", gw_url)
            }
            BackendType::Local => {
                // Local sessions route to local ash-mcp
                self.local_mcp_url().await?
            }
        };

        self.routes.write().await.insert(session.id.clone(), url);
        Ok(())
    }

    /// Remove a session route
    async fn remove_route(&self, session_id: &str) {
        self.routes.write().await.remove(session_id);
    }

    // ==================== Request Handling ====================

    /// Handle a JSON-RPC request
    pub async fn handle_request(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let id = request.id.unwrap_or(Value::Null);

        match request.method.as_str() {
            "ping" => {
                let uptime = self.start_time.elapsed().as_secs();
                JsonRpcResponse::success(id, json!({
                    "status": "ok",
                    "uptime_secs": uptime
                }))
            }

            "gateway/info" => {
                let uptime = self.start_time.elapsed().as_secs();
                let routes = self.routes.read().await;
                let route_count = routes.len();
                drop(routes);

                let local_mcp = self.local_mcp.read().await;
                let local_mcp_port = local_mcp.as_ref().map(|p| p.port);
                drop(local_mcp);

                let manager = self.backend_manager.read().await;
                let default_backend = manager.default_backend();
                let local_ok = manager.health_check(BackendType::Local).await.is_ok();
                let docker_ok = manager.health_check(BackendType::Docker).await.is_ok();
                let k8s_ok = manager.health_check(BackendType::K8s).await.is_ok();
                let sessions = manager.list().await.unwrap_or_default();
                drop(manager);

                JsonRpcResponse::success(id, json!({
                    "uptime_secs": uptime,
                    "default_backend": default_backend.to_string(),
                    "backends": {
                        "local": local_ok,
                        "docker": docker_ok,
                        "k8s": k8s_ok
                    },
                    "sessions": sessions.len(),
                    "routes": route_count,
                    "local_mcp_port": local_mcp_port,
                }))
            }

            "tools/call" => {
                let name = request.params.get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("");
                let arguments = request.params.get("arguments")
                    .cloned()
                    .unwrap_or(json!({}));
                let session_id = request.params.get("session_id")
                    .and_then(|s| s.as_str());

                // Session management tools execute in the gateway process
                if SESSION_TOOLS.contains(&name) {
                    return self.handle_session_tool(&id, name, arguments).await;
                }

                // All other tools: forward to the correct ash-mcp endpoint
                let url = match self.resolve_endpoint(session_id).await {
                    Ok(u) => u,
                    Err(e) => return JsonRpcResponse::success(id, json!({
                        "content": [{"type": "text", "text": format!("Gateway routing error: {e}")}],
                        "isError": true
                    })),
                };

                match self.forward_tool_call(&url, name, arguments).await {
                    Ok(result) => JsonRpcResponse::success(id, result),
                    Err(e) => JsonRpcResponse::success(id, json!({
                        "content": [{"type": "text", "text": format!("Forward error: {e}")}],
                        "isError": true
                    })),
                }
            }

            "tools/list" => {
                let url = match self.local_mcp_url().await {
                    Ok(u) => u,
                    Err(e) => return JsonRpcResponse::error(id, -32000, e),
                };

                match self.forward_method(&url, "tools/list", json!({})).await {
                    Ok(result) => JsonRpcResponse::success(id, result),
                    Err(e) => JsonRpcResponse::error(id, -32000, e),
                }
            }

            _ => JsonRpcResponse::error(id, -32601, format!("Method not found: {}", request.method)),
        }
    }

    /// Handle session management tools (execute in gateway, not forwarded)
    async fn handle_session_tool(&self, id: &Value, name: &str, args: Value) -> JsonRpcResponse {
        let manager = self.backend_manager.read().await;

        match name {
            "session_create" => {
                let backend_str = args.get("backend").and_then(|b| b.as_str());
                let backend = backend_str.and_then(|s| s.parse::<BackendType>().ok());

                let options = crate::backend::CreateOptions {
                    name: args.get("name").and_then(|n| n.as_str()).map(String::from),
                    image: args.get("image").and_then(|i| i.as_str()).map(String::from),
                    env: args.get("env")
                        .and_then(|e| serde_json::from_value::<HashMap<String, String>>(e.clone()).ok())
                        .unwrap_or_default(),
                    ports: args.get("ports")
                        .and_then(|p| serde_json::from_value::<Vec<u16>>(p.clone()).ok())
                        .unwrap_or_default(),
                    working_dir: args.get("working_dir").and_then(|w| w.as_str()).map(String::from),
                    command: None,
                    resources: args.get("resources").and_then(|r| {
                        Some(crate::backend::ResourceSpec {
                            cpu_request: r.get("cpu").and_then(|v| v.as_str()).map(String::from),
                            memory_request: r.get("memory").and_then(|v| v.as_str()).map(String::from),
                            cpu_limit: r.get("cpu_limit").and_then(|v| v.as_str()).map(String::from),
                            memory_limit: r.get("memory_limit").and_then(|v| v.as_str()).map(String::from),
                        })
                    }),
                    labels: args.get("labels")
                        .and_then(|l| serde_json::from_value::<HashMap<String, String>>(l.clone()).ok())
                        .unwrap_or_default(),
                };

                match manager.create(backend, options).await {
                    Ok(session) => {
                        // Register route for the new session
                        drop(manager);
                        if let Err(e) = self.register_route(&session).await {
                            eprintln!("  {} route registration failed for {}: {}",
                                style::ecolor("!", style::BRIGHT_YELLOW), session.id, e);
                        }
                        let text = json!({
                            "session_id": session.id,
                            "name": session.name,
                            "backend": session.backend.to_string(),
                            "status": session.status.to_string(),
                            "host": session.host,
                            "image": session.image,
                        }).to_string();
                        JsonRpcResponse::success(id.clone(), json!({
                            "content": [{"type": "text", "text": text}],
                            "isError": false
                        }))
                    }
                    Err(e) => JsonRpcResponse::success(id.clone(), json!({
                        "content": [{"type": "text", "text": format!("Create failed: {e}")}],
                        "isError": true
                    })),
                }
            }

            "session_destroy" => {
                let session_id = args.get("session_id")
                    .and_then(|s| s.as_str())
                    .unwrap_or("");

                match manager.destroy(session_id).await {
                    Ok(()) => {
                        drop(manager);
                        self.remove_route(session_id).await;
                        JsonRpcResponse::success(id.clone(), json!({
                            "content": [{"type": "text", "text": format!("Destroyed: {}", session_id)}],
                            "isError": false
                        }))
                    }
                    Err(e) => JsonRpcResponse::success(id.clone(), json!({
                        "content": [{"type": "text", "text": format!("Destroy failed: {e}")}],
                        "isError": true
                    })),
                }
            }

            "session_list" => {
                match manager.list().await {
                    Ok(sessions) => {
                        let list: Vec<Value> = sessions.iter().map(|s| json!({
                            "session_id": s.id,
                            "name": s.name,
                            "backend": s.backend.to_string(),
                            "status": s.status.to_string(),
                            "host": s.host,
                            "image": s.image,
                        })).collect();
                        JsonRpcResponse::success(id.clone(), json!({
                            "content": [{"type": "text", "text": serde_json::to_string_pretty(&list).unwrap_or_default()}],
                            "isError": false
                        }))
                    }
                    Err(e) => JsonRpcResponse::success(id.clone(), json!({
                        "content": [{"type": "text", "text": format!("List failed: {e}")}],
                        "isError": true
                    })),
                }
            }

            "session_info" => {
                let session_id = args.get("session_id")
                    .and_then(|s| s.as_str())
                    .unwrap_or("");

                match manager.get(session_id).await {
                    Ok(Some(session)) => {
                        let info = json!({
                            "session_id": session.id,
                            "name": session.name,
                            "backend": session.backend.to_string(),
                            "status": session.status.to_string(),
                            "host": session.host,
                            "image": session.image,
                            "ports": session.ports,
                            "created_at": session.created_at.to_rfc3339(),
                        });
                        JsonRpcResponse::success(id.clone(), json!({
                            "content": [{"type": "text", "text": serde_json::to_string_pretty(&info).unwrap_or_default()}],
                            "isError": false
                        }))
                    }
                    Ok(None) => JsonRpcResponse::success(id.clone(), json!({
                        "content": [{"type": "text", "text": format!("Session not found: {}", session_id)}],
                        "isError": true
                    })),
                    Err(e) => JsonRpcResponse::success(id.clone(), json!({
                        "content": [{"type": "text", "text": format!("Failed: {e}")}],
                        "isError": true
                    })),
                }
            }

            "backend_switch" => {
                let backend_str = args.get("backend")
                    .and_then(|b| b.as_str())
                    .unwrap_or("");

                match backend_str.parse::<BackendType>() {
                    Ok(backend) => {
                        drop(manager);
                        let mut manager = self.backend_manager.write().await;
                        manager.set_default(backend);
                        JsonRpcResponse::success(id.clone(), json!({
                            "content": [{"type": "text", "text": format!("Default backend set to: {}", backend)}],
                            "isError": false
                        }))
                    }
                    Err(e) => JsonRpcResponse::success(id.clone(), json!({
                        "content": [{"type": "text", "text": format!("Invalid backend: {e}")}],
                        "isError": true
                    })),
                }
            }

            "backend_status" => {
                let local_ok = manager.health_check(BackendType::Local).await.is_ok();
                let docker_ok = manager.health_check(BackendType::Docker).await.is_ok();
                let k8s_ok = manager.health_check(BackendType::K8s).await.is_ok();
                let status = json!({
                    "default": manager.default_backend().to_string(),
                    "backends": {
                        "local": if local_ok { "available" } else { "unavailable" },
                        "docker": if docker_ok { "available" } else { "unavailable" },
                        "k8s": if k8s_ok { "available" } else { "unavailable" },
                    }
                });
                JsonRpcResponse::success(id.clone(), json!({
                    "content": [{"type": "text", "text": status.to_string()}],
                    "isError": false
                }))
            }

            _ => JsonRpcResponse::error(id.clone(), -32601, format!("Unknown session tool: {name}")),
        }
    }

    /// Graceful shutdown
    pub async fn shutdown(&self) {
        // Kill local ash-mcp subprocess
        let mut guard = self.local_mcp.write().await;
        if let Some(mut proc) = guard.take() {
            eprintln!("  {} stopping ash-mcp {}",
            style::ecolor(style::cross(), style::GRAY),
            style::ecolor(&format!("(port {})", proc.port), style::GRAY));
            let _ = proc.child.kill().await;
        }
    }
}

// ==================== Gateway Server (Unix Socket) ====================

/// Run the gateway server on a Unix socket
pub async fn run_gateway() -> anyhow::Result<()> {
    let socket = crate::daemon::socket_path();
    let pid_file = crate::daemon::pid_path();

    let _ = std::fs::create_dir_all(crate::daemon::ash_dir());
    let _ = std::fs::remove_file(&socket);

    // Write PID
    std::fs::write(&pid_file, std::process::id().to_string())?;

    let gateway = Arc::new(Gateway::new());

    let ver = env!("CARGO_PKG_VERSION");
    eprintln!("{}", style::gateway_banner(ver));

    // Pre-start local ash-mcp
    if let Err(e) = gateway.ensure_local_mcp().await {
        eprintln!("  {} failed to start ash-mcp: {e}",
            style::ecolor("!", style::BRIGHT_YELLOW));
        eprintln!("  {} local tool calls will fail until ash-mcp is available",
            style::ecolor("!", style::BRIGHT_YELLOW));
    }

    let listener = tokio::net::UnixListener::bind(&socket)?;
    eprintln!("  {} listening on {}",
        style::ecolor(style::check(), style::GREEN),
        style::ecolor(&socket.display().to_string(), style::CYAN));
    eprintln!();

    // Graceful shutdown on SIGTERM/SIGINT
    let socket_cleanup = socket.clone();
    let pid_cleanup = pid_file.clone();
    let gateway_shutdown = gateway.clone();

    tokio::spawn(async move {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to register SIGTERM");
        let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .expect("failed to register SIGINT");

        tokio::select! {
            _ = sigterm.recv() => {}
            _ = sigint.recv() => {}
        }

        eprintln!("\n  {} gateway shutting down",
            style::ecolor(style::cross(), style::GRAY));
        gateway_shutdown.shutdown().await;
        let _ = std::fs::remove_file(&socket_cleanup);
        let _ = std::fs::remove_file(&pid_cleanup);
        std::process::exit(0);
    });

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let gw = gateway.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_connection(gw, stream).await {
                        eprintln!("  {} connection error: {e}",
                            style::ecolor("!", style::BRIGHT_YELLOW));
                    }
                });
            }
            Err(e) => {
                eprintln!("  {} accept error: {e}",
                    style::ecolor("!", style::BRIGHT_YELLOW));
            }
        }
    }
}

/// Handle a single Unix socket connection
async fn handle_connection(gateway: Arc<Gateway>, stream: tokio::net::UnixStream) -> anyhow::Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    reader.read_line(&mut line).await?;

    if line.trim().is_empty() {
        return Ok(());
    }

    let request: JsonRpcRequest = match serde_json::from_str(&line) {
        Ok(r) => r,
        Err(e) => {
            let response = JsonRpcResponse::error(Value::Null, -32700, format!("Parse error: {e}"));
            let mut out = serde_json::to_string(&response)?;
            out.push('\n');
            writer.write_all(out.as_bytes()).await?;
            return Ok(());
        }
    };

    let response = gateway.handle_request(request).await;
    let mut out = serde_json::to_string(&response)?;
    out.push('\n');
    writer.write_all(out.as_bytes()).await?;

    Ok(())
}
