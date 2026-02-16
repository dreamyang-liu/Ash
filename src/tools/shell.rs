//! Shell tool - local or via session backend

use crate::{BoxFuture, Tool, ToolResult};
use crate::backend::ExecOptions;
use crate::tools::session;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::process::Command;
use std::time::Duration;
use std::collections::VecDeque;

/// Run history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunHistoryEntry {
    pub id: String,
    pub command: String,
    pub revert_command: Option<String>,
    pub exit_code: i32,
    pub timestamp: String,
}

/// Get path to run history file
fn history_path() -> std::path::PathBuf {
    let dir = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join(".ash");
    let _ = std::fs::create_dir_all(&dir);
    dir.join("run_history.json")
}

/// Load run history (last 100 entries)
pub fn load_run_history() -> VecDeque<RunHistoryEntry> {
    let path = history_path();
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Save run history entry
fn save_run_history(entry: RunHistoryEntry) {
    let mut history = load_run_history();
    history.push_front(entry);
    // Keep only last 100
    while history.len() > 100 {
        history.pop_back();
    }
    let path = history_path();
    let _ = std::fs::write(&path, serde_json::to_string_pretty(&history).unwrap_or_default());
}

/// Remove entry from history by id
fn remove_from_history(id: &str) {
    let mut history = load_run_history();
    history.retain(|e| e.id != id);
    let path = history_path();
    let _ = std::fs::write(&path, serde_json::to_string_pretty(&history).unwrap_or_default());
}

/// Get last run with a revert command
pub fn get_last_revertible() -> Option<RunHistoryEntry> {
    load_run_history().into_iter()
        .find(|e| e.revert_command.as_ref().map(|s| !s.is_empty()).unwrap_or(false))
}

/// Get run by id
pub fn get_run_by_id(id: &str) -> Option<RunHistoryEntry> {
    load_run_history().into_iter().find(|e| e.id == id)
}

#[derive(Debug, Clone, Deserialize)]
pub struct ShellArgs {
    pub command: String,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Session ID - if provided, executes in that sandbox
    #[serde(default)]
    pub session_id: Option<String>,
    /// Only return the last N lines of output
    #[serde(default)]
    pub tail_lines: Option<usize>,
    /// Working directory
    #[serde(default)]
    pub working_dir: Option<String>,
    /// Command to revert changes (empty = no state change, omit = cannot revert)
    #[serde(default)]
    pub revert_command: Option<String>,
}

fn default_timeout() -> u64 { 300 }

/// Tail the output to last N lines
fn tail_output(output: &str, lines: usize) -> String {
    let all_lines: Vec<&str> = output.lines().collect();
    if all_lines.len() <= lines {
        output.to_string()
    } else {
        let skipped = all_lines.len() - lines;
        format!(
            "... ({} lines skipped)\n{}",
            skipped,
            all_lines[all_lines.len() - lines..].join("\n")
        )
    }
}

pub struct ShellTool;

impl Tool for ShellTool {
    fn name(&self) -> &'static str { "shell" }
    fn description(&self) -> &'static str { "Execute shell command (locally or in session sandbox)" }
    
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command"},
                "timeout_secs": {"type": "integer", "default": 300},
                "session_id": {"type": "string", "description": "Execute in this session's sandbox"},
                "tail_lines": {"type": "integer", "description": "Only return the last N lines of output"},
                "working_dir": {"type": "string", "description": "Working directory"},
                "revert_command": {
                    "type": ["string", "null"],
                    "description": "Command to revert changes. Empty string = no state change, null/omit = cannot revert. Returned in output for tracking."
                }
            },
            "required": ["command"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: ShellArgs = match serde_json::from_value(args.clone()) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            // If session_id provided, route through backend
            if let Some(session_id) = &args.session_id {
                let options = ExecOptions {
                    working_dir: args.working_dir,
                    timeout_secs: Some(args.timeout_secs),
                    ..Default::default()
                };
                
                match session::exec_in_session(session_id, &args.command, options).await {
                    Ok(result) => {
                        let mut output = result.output();
                        
                        // Apply tail if requested
                        if let Some(n) = args.tail_lines {
                            output = tail_output(&output, n);
                        }
                        
                        if result.success() {
                            ToolResult::ok(output)
                        } else {
                            ToolResult { 
                                success: false, 
                                output, 
                                error: Some(format!("Exit: {}", result.exit_code)) 
                            }
                        }
                    }
                    Err(e) => ToolResult::err(format!("Exec failed: {e}")),
                }
            } else {
                // Local execution
                let timeout = Duration::from_secs(args.timeout_secs);
                
                let mut cmd = Command::new("sh");
                cmd.arg("-c").arg(&args.command);
                
                if let Some(ref dir) = args.working_dir {
                    cmd.current_dir(dir);
                }
                
                let result = tokio::time::timeout(timeout, cmd.output()).await;
                
                match result {
                    Ok(Ok(output)) => {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        let mut out = stdout.to_string();
                        if !stderr.is_empty() {
                            if !out.is_empty() { out.push('\n'); }
                            out.push_str("[stderr]\n");
                            out.push_str(&stderr);
                        }
                        
                        // Apply tail if requested
                        let out = if let Some(n) = args.tail_lines {
                            tail_output(&out, n)
                        } else {
                            out
                        };
                        
                        let exit_code = output.status.code().unwrap_or(-1);
                        
                        // Save to history if revert_command provided
                        if args.revert_command.is_some() {
                            let entry = RunHistoryEntry {
                                id: uuid::Uuid::new_v4().to_string(),
                                command: args.command.clone(),
                                revert_command: args.revert_command.clone(),
                                exit_code,
                                timestamp: chrono::Utc::now().to_rfc3339(),
                            };
                            save_run_history(entry);
                        }
                        
                        if output.status.success() {
                            ToolResult::ok(out)
                        } else {
                            ToolResult { 
                                success: false, 
                                output: out, 
                                error: Some(format!("Exit: {}", exit_code)) 
                            }
                        }
                    }
                    Ok(Err(e)) => ToolResult::err(format!("Exec failed: {e}")),
                    Err(_) => ToolResult::err(format!("Timeout after {}s", args.timeout_secs)),
                }
            }
        })
    }
}

pub struct ShellRevertTool;

impl Tool for ShellRevertTool {
    fn name(&self) -> &'static str { "shell_revert" }
    fn description(&self) -> &'static str { "Revert last shell command (or by ID) that had a revert_command" }
    
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Specific run ID to revert (optional, defaults to last revertible)"}
            }
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let id = args.get("id").and_then(|v| v.as_str());
            
            let entry = if let Some(id) = id {
                get_run_by_id(id)
            } else {
                get_last_revertible()
            };
            
            let entry = match entry {
                Some(e) => e,
                None => return ToolResult::err("No revertible command found in history".to_string()),
            };
            
            let revert_cmd = match &entry.revert_command {
                Some(cmd) if !cmd.is_empty() => cmd,
                Some(_) => return ToolResult::ok(format!("Command '{}' had no state changes to revert", entry.command)),
                None => return ToolResult::err(format!("Command '{}' cannot be reverted", entry.command)),
            };
            
            // Execute revert
            match Command::new("sh").args(["-c", revert_cmd]).output().await {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    if output.status.success() {
                        // Remove from history after successful revert
                        remove_from_history(&entry.id);
                        ToolResult::ok(format!(
                            "Reverted '{}' with '{}'\n{}{}",
                            entry.command, revert_cmd, stdout, stderr
                        ))
                    } else {
                        ToolResult::err(format!(
                            "Revert failed (exit {}): {}\n{}{}",
                            output.status.code().unwrap_or(-1), revert_cmd, stdout, stderr
                        ))
                    }
                }
                Err(e) => ToolResult::err(format!("Failed to execute revert: {}", e)),
            }
        })
    }
}

pub struct ShellHistoryTool;

impl Tool for ShellHistoryTool {
    fn name(&self) -> &'static str { "shell_history" }
    fn description(&self) -> &'static str { "Show recent shell commands with revert info" }
    
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 10, "description": "Number of entries to show"}
            }
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
            let history = load_run_history();
            
            if history.is_empty() {
                return ToolResult::ok("No run history".to_string());
            }
            
            let mut out = String::new();
            for (i, entry) in history.iter().take(limit).enumerate() {
                let revert_status = match &entry.revert_command {
                    Some(s) if s.is_empty() => "[no state change]",
                    Some(_) => "[revertible]",
                    None => "[non-revertible]",
                };
                out.push_str(&format!(
                    "{}. {} {} (exit {})\n   ID: {}\n",
                    i + 1, revert_status, entry.command, entry.exit_code, entry.id
                ));
            }
            
            ToolResult::ok(out)
        })
    }
}
