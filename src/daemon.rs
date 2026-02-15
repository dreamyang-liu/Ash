//! Daemon support - shared constants and client helper
//!
//! The daemon is a long-lived process that holds stateful resources:
//! - BackendManager (persistent Docker/K8s connections)
//! - ProcessRegistry (async process tracking survives CLI exit)
//! - EventSystem (event queue persists across invocations)
//!
//! CLI talks to daemon via JSON-RPC over Unix domain socket.

use serde_json::Value;
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Ash data directory (~/.ash/)
pub fn ash_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".ash")
}

/// Path to the daemon Unix socket
pub fn socket_path() -> PathBuf {
    ash_dir().join("daemon.sock")
}

/// Path to the daemon PID file
pub fn pid_path() -> PathBuf {
    ash_dir().join("daemon.pid")
}

/// Check if the daemon process is alive by reading PID file and checking /proc
pub fn is_daemon_running() -> bool {
    let pid_file = pid_path();
    if let Ok(contents) = std::fs::read_to_string(&pid_file) {
        if let Ok(pid) = contents.trim().parse::<u32>() {
            // Check if process exists
            return std::path::Path::new(&format!("/proc/{}", pid)).exists();
        }
    }
    false
}

/// Send a JSON-RPC request to the daemon via Unix socket.
/// Returns None if daemon is not reachable.
pub async fn daemon_call(method: &str, params: Value) -> Option<Value> {
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

/// Send a tools/call request to daemon and parse into ToolResult.
/// If session_id is provided, daemon routes the call through BackendManager.
/// Returns None if daemon is not reachable.
pub async fn daemon_tool_call(
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
    let response = daemon_call("tools/call", params).await?;

    // Parse JSON-RPC response
    if let Some(error) = response.get("error") {
        let msg = error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("daemon error");
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
