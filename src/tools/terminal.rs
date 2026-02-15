//! Async terminal/process management tools

use crate::{BoxFuture, Tool, ToolResult};
use crate::tools::session;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use lazy_static::lazy_static;
use uuid::Uuid;

// ==================== Persistent handle → session_id mapping ====================

/// Path to the handle store file
fn handle_store_path() -> std::path::PathBuf {
    let dir = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join(".ash");
    let _ = std::fs::create_dir_all(&dir);
    dir.join("handles.json")
}

/// Load handle → session_id mapping from disk
pub fn load_handle_map() -> HashMap<String, String> {
    let path = handle_store_path();
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Save handle → session_id mapping to disk
fn save_handle_map(map: &HashMap<String, String>) {
    let path = handle_store_path();
    let _ = std::fs::write(&path, serde_json::to_string(map).unwrap_or_default());
}

/// Record that a handle belongs to a session
fn record_handle(handle: &str, session_id: &str) {
    let mut map = load_handle_map();
    map.insert(handle.to_string(), session_id.to_string());
    save_handle_map(&map);
}

/// Look up which session owns a handle
fn lookup_handle_session(handle: &str) -> Option<String> {
    let map = load_handle_map();
    map.get(handle).cloned()
}

/// Remove a handle from the store
fn remove_handle(handle: &str) {
    let mut map = load_handle_map();
    map.remove(handle);
    save_handle_map(&map);
}


/// Extract text from MCP call_tool result
fn mcp_result_to_tool_result(result: Value) -> ToolResult {
    let is_error = result.get("isError").and_then(|e| e.as_bool()).unwrap_or(false);
    let text = result.get("content")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or("");
    if is_error {
        ToolResult::err(text.to_string())
    } else {
        ToolResult::ok(text.to_string())
    }
}

/// Process output information
#[derive(Debug, Clone, Serialize)]
pub struct ProcessOutput {
    pub stdout: String,
    pub stderr: String,
    pub complete: bool,
    pub exit_code: Option<i32>,
}

/// Async process handle stored in the registry
struct AsyncProcess {
    stdout_lines: Arc<Mutex<Vec<String>>>,
    stderr_lines: Arc<Mutex<Vec<String>>>,
    exit_code: Arc<Mutex<Option<i32>>>,
    command: String,
    revert_command: Option<String>, // None = cannot revert, Some("") = no state change
    started_at: chrono::DateTime<chrono::Utc>,
    // Keep child for kill
    child: Arc<Mutex<Option<Child>>>,
}

/// Global process registry
struct ProcessRegistry {
    processes: HashMap<String, AsyncProcess>,
}

impl ProcessRegistry {
    fn new() -> Self {
        Self { processes: HashMap::new() }
    }

    async fn start(&mut self, command: &str, working_dir: Option<&str>, env: Option<HashMap<String, String>>, revert_command: Option<String>) -> anyhow::Result<String> {
        let id = Uuid::new_v4().to_string();
        
        let mut cmd = Command::new("sh");
        cmd.args(["-c", command])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        if let Some(dir) = working_dir {
            cmd.current_dir(dir);
        }
        if let Some(env_vars) = env {
            for (k, v) in env_vars {
                cmd.env(k, v);
            }
        }

        let mut child = cmd.spawn()?;
        
        let stdout_lines = Arc::new(Mutex::new(Vec::new()));
        let stderr_lines = Arc::new(Mutex::new(Vec::new()));
        let exit_code: Arc<Mutex<Option<i32>>> = Arc::new(Mutex::new(None));

        // Spawn stdout reader
        if let Some(stdout) = child.stdout.take() {
            let lines = stdout_lines.clone();
            tokio::spawn(async move {
                let reader = BufReader::new(stdout);
                let mut stream = reader.lines();
                while let Ok(Some(line)) = stream.next_line().await {
                    lines.lock().await.push(line);
                }
            });
        }

        // Spawn stderr reader
        if let Some(stderr) = child.stderr.take() {
            let lines = stderr_lines.clone();
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut stream = reader.lines();
                while let Ok(Some(line)) = stream.next_line().await {
                    lines.lock().await.push(line);
                }
            });
        }

        // Spawn exit watcher - push event when done
        let exit_clone = exit_code.clone();
        let child_arc = Arc::new(Mutex::new(Some(child)));
        let child_clone = child_arc.clone();
        let id_for_event = id.clone();
        let cmd_for_event = command.to_string();
        tokio::spawn(async move {
            if let Some(ref mut c) = *child_clone.lock().await {
                if let Ok(status) = c.wait().await {
                    let code = status.code().or(Some(-1));
                    *exit_clone.lock().await = code;
                    
                    // Push event when process completes
                    crate::tools::events::push_event(
                        "process_complete",
                        &id_for_event,
                        serde_json::json!({
                            "handle": id_for_event,
                            "command": cmd_for_event,
                            "exit_code": code,
                            "success": code == Some(0)
                        })
                    ).await;
                }
            }
        });

        let process = AsyncProcess {
            stdout_lines,
            stderr_lines,
            exit_code,
            command: command.to_string(),
            revert_command,
            started_at: chrono::Utc::now(),
            child: child_arc,
        };

        self.processes.insert(id.clone(), process);
        Ok(id)
    }

    async fn get_output(&self, id: &str, tail: Option<usize>) -> Option<ProcessOutput> {
        let process = self.processes.get(id)?;
        let stdout_lines = process.stdout_lines.lock().await;
        let stderr_lines = process.stderr_lines.lock().await;
        let exit = *process.exit_code.lock().await;
        
        let (stdout, stderr) = if let Some(n) = tail {
            let skip_stdout = stdout_lines.len().saturating_sub(n);
            let skip_stderr = stderr_lines.len().saturating_sub(n);
            let stdout = if skip_stdout > 0 {
                format!("... ({} lines skipped)\n{}", skip_stdout, stdout_lines[skip_stdout..].join("\n"))
            } else {
                stdout_lines.join("\n")
            };
            let stderr = if skip_stderr > 0 {
                format!("... ({} lines skipped)\n{}", skip_stderr, stderr_lines[skip_stderr..].join("\n"))
            } else {
                stderr_lines.join("\n")
            };
            (stdout, stderr)
        } else {
            (stdout_lines.join("\n"), stderr_lines.join("\n"))
        };
        
        Some(ProcessOutput {
            stdout,
            stderr,
            complete: exit.is_some(),
            exit_code: exit,
        })
    }

    async fn kill(&self, id: &str) -> bool {
        if let Some(process) = self.processes.get(id) {
            if let Some(ref mut child) = *process.child.lock().await {
                let _ = child.kill().await;
                *process.exit_code.lock().await = Some(-9);
                return true;
            }
        }
        false
    }

    fn list(&self) -> Vec<(String, String, String, bool, Option<String>)> {
        self.processes.iter().map(|(id, p)| {
            let running = p.exit_code.try_lock().map(|e| e.is_none()).unwrap_or(true);
            (id.clone(), p.command.clone(), p.started_at.to_rfc3339(), running, p.revert_command.clone())
        }).collect()
    }

    fn remove(&mut self, id: &str) -> bool {
        self.processes.remove(id).is_some()
    }

    fn get_revert_command(&self, id: &str) -> Option<Option<String>> {
        self.processes.get(id).map(|p| p.revert_command.clone())
    }
}

lazy_static! {
    static ref REGISTRY: Mutex<ProcessRegistry> = Mutex::new(ProcessRegistry::new());
}

/// Get (running, completed) counts of local async processes
pub async fn local_process_counts() -> (usize, usize) {
    let registry = REGISTRY.lock().await;
    let list = registry.list();
    let running = list.iter().filter(|(_, _, _, r, _)| *r).count();
    (running, list.len() - running)
}

// ==================== Tools ====================

pub struct TerminalRunAsyncTool;

impl Tool for TerminalRunAsyncTool {
    fn name(&self) -> &'static str { "terminal_run_async" }
    fn description(&self) -> &'static str { "Start a background process, returns handle ID" }
    
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "working_dir": {"type": "string", "description": "Working directory"},
                "env": {"type": "object", "description": "Environment variables"},
                "revert_command": {
                    "type": ["string", "null"],
                    "description": "Command to revert this command's changes. Empty string = no state change, null = cannot revert"
                }
            },
            "required": ["command"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                command: String,
                working_dir: Option<String>,
                env: Option<HashMap<String, String>>,
                revert_command: Option<String>,
                session_id: Option<String>,
            }

            let args: Args = match serde_json::from_value(args.clone()) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };

            if let Some(ref sid) = args.session_id {
                let mut remote_args = serde_json::json!({"command": args.command});
                if let Some(ref dir) = args.working_dir { remote_args["working_dir"] = serde_json::json!(dir); }
                if let Some(ref env) = args.env { remote_args["env"] = serde_json::json!(env); }
                if let Some(ref revert) = args.revert_command { remote_args["revert_command"] = serde_json::json!(revert); }
                return match session::call_tool_in_session(sid, "terminal_run_async", remote_args).await {
                    Ok(result) => {
                        // Persist handle → session_id so later commands auto-resolve
                        let text = result.get("content")
                            .and_then(|c| c.as_array())
                            .and_then(|arr| arr.first())
                            .and_then(|c| c.get("text"))
                            .and_then(|t| t.as_str())
                            .unwrap_or("");
                        if let Ok(obj) = serde_json::from_str::<Value>(text) {
                            if let Some(handle) = obj.get("handle").and_then(|h| h.as_str()) {
                                record_handle(handle, sid);
                            }
                        }
                        mcp_result_to_tool_result(result)
                    }
                    Err(e) => ToolResult::err(format!("Session error: {e}")),
                };
            }

            let mut registry = REGISTRY.lock().await;
            match registry.start(&args.command, args.working_dir.as_deref(), args.env, args.revert_command).await {
                Ok(id) => ToolResult::ok(serde_json::json!({"handle": id}).to_string()),
                Err(e) => ToolResult::err(e.to_string()),
            }
        })
    }
}

pub struct TerminalGetOutputTool;

impl Tool for TerminalGetOutputTool {
    fn name(&self) -> &'static str { "terminal_get_output" }
    fn description(&self) -> &'static str { "Get output from async process by handle" }
    
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "handle": {"type": "string", "description": "Process handle from terminal_run_async"},
                "tail": {"type": "integer", "description": "Only return last N lines"}
            },
            "required": ["handle"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                handle: String,
                tail: Option<usize>,
                session_id: Option<String>,
            }

            let args: Args = match serde_json::from_value(args.clone()) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };

            // Auto-resolve session from persisted handle map
            let sid = args.session_id.or_else(|| lookup_handle_session(&args.handle));

            if let Some(ref sid) = sid {
                let mut remote_args = serde_json::json!({"handle": args.handle});
                if let Some(tail) = args.tail { remote_args["tail"] = serde_json::json!(tail); }
                return match session::call_tool_in_session(sid, "terminal_get_output", remote_args).await {
                    Ok(result) => mcp_result_to_tool_result(result),
                    Err(e) => ToolResult::err(format!("Session error: {e}")),
                };
            }

            let registry = REGISTRY.lock().await;
            match registry.get_output(&args.handle, args.tail).await {
                Some(output) => ToolResult::ok(serde_json::to_string_pretty(&output).unwrap()),
                None => ToolResult::err(format!("Handle not found: {}", args.handle)),
            }
        })
    }
}

pub struct TerminalKillTool;

impl Tool for TerminalKillTool {
    fn name(&self) -> &'static str { "terminal_kill" }
    fn description(&self) -> &'static str { "Kill an async process by handle" }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "handle": {"type": "string", "description": "Process handle"},
                "session_id": {"type": "string", "description": "Route to session sandbox"}
            },
            "required": ["handle"]
        })
    }

    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                handle: String,
                session_id: Option<String>,
            }

            let args: Args = match serde_json::from_value(args.clone()) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };

            let sid = args.session_id.or_else(|| lookup_handle_session(&args.handle));

            if let Some(ref sid) = sid {
                let remote_args = serde_json::json!({"handle": args.handle});
                return match session::call_tool_in_session(sid, "terminal_kill", remote_args).await {
                    Ok(result) => mcp_result_to_tool_result(result),
                    Err(e) => ToolResult::err(format!("Session error: {e}")),
                };
            }

            let registry = REGISTRY.lock().await;
            if registry.kill(&args.handle).await {
                ToolResult::ok("Process killed".to_string())
            } else {
                ToolResult::err("Handle not found or already dead".to_string())
            }
        })
    }
}

pub struct TerminalListTool;

impl Tool for TerminalListTool {
    fn name(&self) -> &'static str { "terminal_list" }
    fn description(&self) -> &'static str { "List all tracked async processes" }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Route to session sandbox"}
            }
        })
    }

    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let session_id = args.get("session_id").and_then(|v| v.as_str()).map(|s| s.to_string());

            // If explicit session, only list that session
            if let Some(ref sid) = session_id {
                return match session::call_tool_in_session(sid, "terminal_list", serde_json::json!({})).await {
                    Ok(result) => mcp_result_to_tool_result(result),
                    Err(e) => ToolResult::err(format!("Session error: {e}")),
                };
            }

            let mut out = String::new();

            // Local processes
            {
                let registry = REGISTRY.lock().await;
                let list = registry.list();
                if !list.is_empty() {
                    out.push_str("Local:\n");
                    for (id, cmd, started, running, revert) in list {
                        let status = if running { "running" } else { "complete" };
                        let revert_info = match &revert {
                            Some(s) if s.is_empty() => " [no state change]",
                            Some(_) => " [revertible]",
                            None => " [non-revertible]",
                        };
                        out.push_str(&format!("  {} [{}]{} {} ({})\n", id, status, revert_info, cmd, started));
                    }
                }
            }

            // Tracked handles (from persisted handle map)
            let handle_map = load_handle_map();
            if !handle_map.is_empty() {
                // Group handles by session
                let mut by_session: HashMap<String, Vec<String>> = HashMap::new();
                for (handle, sid) in &handle_map {
                    by_session.entry(sid.clone()).or_default().push(handle.clone());
                }

                for (sid, handles) in &by_session {
                    let short_id = if sid.len() > 12 { &sid[..12] } else { sid };
                    out.push_str(&format!("Session {}:\n", short_id));

                    // Try to get live status from session
                    let live_info = session::call_tool_in_session(sid, "terminal_list", serde_json::json!({})).await.ok()
                        .and_then(|result| {
                            result.get("content")
                                .and_then(|c| c.as_array())
                                .and_then(|arr| arr.first())
                                .and_then(|c| c.get("text"))
                                .and_then(|t| t.as_str())
                                .map(|s| s.to_string())
                        });

                    for handle in handles {
                        let short_handle = if handle.len() > 8 { &handle[..8] } else { handle };
                        // Check if live info contains this handle
                        let status = match &live_info {
                            Some(text) if text.contains(handle) => {
                                if text.contains(&format!("{} [running]", handle)) {
                                    "running"
                                } else if text.contains(&format!("{} [complete]", handle)) {
                                    "complete"
                                } else {
                                    "tracked"
                                }
                            }
                            _ => "tracked",
                        };
                        out.push_str(&format!("  {} [{}]\n", short_handle, status));
                    }
                }
            }

            if out.is_empty() {
                return ToolResult::ok("No async processes".to_string());
            }
            ToolResult::ok(out)
        })
    }
}

pub struct TerminalRemoveTool;

impl Tool for TerminalRemoveTool {
    fn name(&self) -> &'static str { "terminal_remove" }
    fn description(&self) -> &'static str { "Remove a completed process from tracking" }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "handle": {"type": "string", "description": "Process handle"},
                "session_id": {"type": "string", "description": "Route to session sandbox"}
            },
            "required": ["handle"]
        })
    }

    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                handle: String,
                session_id: Option<String>,
            }

            let args: Args = match serde_json::from_value(args.clone()) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };

            let sid = args.session_id.or_else(|| lookup_handle_session(&args.handle));

            if let Some(ref sid) = sid {
                let remote_args = serde_json::json!({"handle": args.handle});
                let result = match session::call_tool_in_session(sid, "terminal_remove", remote_args).await {
                    Ok(result) => mcp_result_to_tool_result(result),
                    Err(e) => ToolResult::err(format!("Session error: {e}")),
                };
                // Clean up persisted mapping
                remove_handle(&args.handle);
                return result;
            }

            let mut registry = REGISTRY.lock().await;
            if registry.remove(&args.handle) {
                ToolResult::ok("Removed".to_string())
            } else {
                ToolResult::err("Handle not found".to_string())
            }
        })
    }
}
