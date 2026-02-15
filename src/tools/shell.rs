//! Shell tool - local or via MCP Gateway

use crate::{BoxFuture, Tool, ToolResult};
use crate::tools::session;
use serde::Deserialize;
use serde_json::Value;
use tokio::process::Command;
use std::time::Duration;

#[derive(Debug, Clone, Deserialize)]
pub struct ShellArgs {
    pub command: String,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Session ID - if provided, executes via MCP Gateway in that sandbox
    #[serde(default)]
    pub session_id: Option<String>,
}

fn default_timeout() -> u64 { 300 }

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
                "session_id": {"type": "string", "description": "Execute in this session's sandbox via MCP"}
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
            
            // If session_id provided, route through MCP Gateway
            if let Some(session_id) = &args.session_id {
                // Call shell tool in remote sandbox
                let mcp_args = serde_json::json!({
                    "command": args.command,
                    "timeout_secs": args.timeout_secs,
                });
                
                match session::call_tool_in_session(session_id, "shell", mcp_args).await {
                    Ok(result) => {
                        // Parse MCP result
                        let content = result.get("content")
                            .and_then(|c| c.as_array())
                            .and_then(|arr| arr.first())
                            .and_then(|c| c.get("text"))
                            .and_then(|t| t.as_str())
                            .unwrap_or("");
                        let is_error = result.get("isError").and_then(|e| e.as_bool()).unwrap_or(false);
                        
                        if is_error {
                            ToolResult::err(content.to_string())
                        } else {
                            ToolResult::ok(content.to_string())
                        }
                    }
                    Err(e) => ToolResult::err(e),
                }
            } else {
                // Local execution
                let timeout = Duration::from_secs(args.timeout_secs);
                let result = tokio::time::timeout(timeout, 
                    Command::new("sh").arg("-c").arg(&args.command).output()
                ).await;
                
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
                        if output.status.success() {
                            ToolResult::ok(out)
                        } else {
                            ToolResult { 
                                success: false, 
                                output: out, 
                                error: Some(format!("Exit: {}", output.status.code().unwrap_or(-1))) 
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
