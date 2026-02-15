//! Grep tool - Search with ripgrep

use crate::{BoxFuture, Tool, ToolResult};
use crate::tools::session;
use serde::Deserialize;
use serde_json::Value;
use tokio::process::Command;

#[derive(Debug, Clone, Deserialize)]
pub struct GrepArgs {
    pub pattern: String,
    #[serde(default = "default_path")]
    pub path: String,
    #[serde(default)]
    pub include: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Execute in session sandbox via MCP
    #[serde(default)]
    pub session_id: Option<String>,
}

fn default_path() -> String { ".".to_string() }
fn default_limit() -> usize { 100 }

pub struct GrepTool;

impl Tool for GrepTool {
    fn name(&self) -> &'static str { "grep_files" }
    fn description(&self) -> &'static str { "Search for pattern in files" }
    
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern"},
                "path": {"type": "string", "description": "Search path", "default": "."},
                "include": {"type": "string", "description": "File glob (e.g., *.py)"},
                "limit": {"type": "integer", "description": "Max results", "default": 100},
                "session_id": {"type": "string", "description": "Execute in session sandbox"}
            },
            "required": ["pattern"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: GrepArgs = match serde_json::from_value(args.clone()) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            // Route to session if provided
            if let Some(session_id) = &args.session_id {
                let mcp_args = serde_json::json!({
                    "pattern": args.pattern,
                    "path": args.path,
                    "include": args.include,
                    "limit": args.limit,
                });
                return call_mcp_tool(session_id, "grep_files", mcp_args).await;
            }
            
            // Local execution
            let mut cmd = Command::new("rg");
            cmd.arg("--line-number").arg("--no-heading").arg("--color=never")
               .arg("--max-count").arg(args.limit.to_string())
               .arg("--regexp").arg(&args.pattern);
            if let Some(ref glob) = args.include { cmd.arg("--glob").arg(glob); }
            cmd.arg("--").arg(&args.path);
            
            match cmd.output().await {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    match output.status.code() {
                        Some(0) => ToolResult::ok(stdout.to_string()),
                        Some(1) => ToolResult::ok("No matches found.".to_string()),
                        _ => ToolResult::err(String::from_utf8_lossy(&output.stderr).to_string()),
                    }
                }
                Err(e) => ToolResult::err(format!("Failed to run rg: {e}")),
            }
        })
    }
}

async fn call_mcp_tool(session_id: &str, tool_name: &str, args: Value) -> ToolResult {
    match session::call_tool_in_session(session_id, tool_name, args).await {
        Ok(result) => {
            let content = result.get("content")
                .and_then(|c| c.as_array())
                .and_then(|arr| arr.first())
                .and_then(|c| c.get("text"))
                .and_then(|t| t.as_str())
                .unwrap_or("");
            let is_error = result.get("isError").and_then(|e| e.as_bool()).unwrap_or(false);
            if is_error { ToolResult::err(content.to_string()) } else { ToolResult::ok(content.to_string()) }
        }
        Err(e) => ToolResult::err(e),
    }
}
