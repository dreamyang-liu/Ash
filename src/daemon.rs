//! Gateway client - shared constants and client helpers
//!
//! The gateway is a long-lived process that:
//! - Routes tool calls to ash-mcp endpoints (local, Docker, K8s)
//! - Manages session lifecycle (create/destroy containers/pods)
//! - Auto-starts a local ash-mcp subprocess for host execution
//!
//! CLI talks to gateway via JSON-RPC over Unix domain socket.

use serde_json::Value;
use std::path::PathBuf;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Ash data directory (~/.ash/)
pub fn ash_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".ash")
}

/// Path to the gateway Unix socket
pub fn socket_path() -> PathBuf {
    ash_dir().join("gateway.sock")
}

/// Path to the gateway PID file
pub fn pid_path() -> PathBuf {
    ash_dir().join("gateway.pid")
}

/// Check if the gateway process is alive by reading PID file and checking /proc
pub fn is_gateway_running() -> bool {
    let pid_file = pid_path();
    if let Ok(contents) = std::fs::read_to_string(&pid_file) {
        if let Ok(pid) = contents.trim().parse::<u32>() {
            return std::path::Path::new(&format!("/proc/{}", pid)).exists();
        }
    }
    false
}

/// Ensure gateway is running, auto-start if needed.
/// Returns true if gateway is reachable after this call.
pub async fn ensure_gateway() -> bool {
    // Quick check: already reachable?
    if gateway_call("ping", serde_json::json!({})).await.is_some() {
        return true;
    }

    // Auto-start gateway
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(_) => return false,
    };

    let _ = std::fs::create_dir_all(ash_dir());

    match std::process::Command::new(&exe)
        .args(["gateway", "start", "--foreground"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(_child) => {}
        Err(_) => return false,
    }

    // Wait for gateway to be ready (poll ping)
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if gateway_call("ping", serde_json::json!({})).await.is_some() {
            return true;
        }
    }

    false
}

/// Send a JSON-RPC request to the gateway via Unix socket.
/// Returns None if gateway is not reachable.
pub async fn gateway_call(method: &str, params: Value) -> Option<Value> {
    let path = socket_path();
    let stream = tokio::net::UnixStream::connect(&path).await.ok()?;
    let (reader, mut writer) = stream.into_split();

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    });

    let mut line = serde_json::to_string(&request).ok()?;
    line.push('\n');
    writer.write_all(line.as_bytes()).await.ok()?;
    writer.shutdown().await.ok()?;

    let mut reader = BufReader::new(reader);
    let mut response_line = String::new();
    reader.read_line(&mut response_line).await.ok()?;

    serde_json::from_str(&response_line).ok()
}

/// Send a tools/call request to gateway and parse into ToolResult.
/// If session_id is provided, gateway routes the call to the correct ash-mcp endpoint.
/// Returns None if gateway is not reachable.
pub async fn gateway_tool_call(
    tool_name: &str,
    args: Value,
    session_id: &Option<String>,
) -> Option<crate::ToolResult> {
    let mut params = serde_json::json!({
        "name": tool_name,
        "arguments": args
    });
    if let Some(ref sid) = session_id {
        params["session_id"] = serde_json::json!(sid);
    }
    let response = gateway_call("tools/call", params).await?;

    // Parse JSON-RPC response
    if let Some(error) = response.get("error") {
        let msg = error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("gateway error");
        return Some(crate::ToolResult::err(msg.to_string()));
    }

    let result = response.get("result")?;
    let is_error = result
        .get("isError")
        .and_then(|e| e.as_bool())
        .unwrap_or(false);
    let text = result
        .get("content")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_string();

    if is_error {
        Some(crate::ToolResult {
            success: false,
            output: text.clone(),
            error: Some(text),
        })
    } else {
        Some(crate::ToolResult::ok(text))
    }
}
