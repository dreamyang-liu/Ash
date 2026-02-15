//! Git tools

use crate::{BoxFuture, Tool, ToolResult};
use crate::tools::session;
use serde::Deserialize;
use serde_json::Value;
use tokio::process::Command;

// Helper for MCP routing
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

// --- Git Status ---
#[derive(Debug, Clone, Deserialize, Default)]
pub struct GitStatusArgs {
    #[serde(default)]
    pub short: bool,
    #[serde(default)]
    pub session_id: Option<String>,
}

pub struct GitStatusTool;

impl Tool for GitStatusTool {
    fn name(&self) -> &'static str { "git_status" }
    fn description(&self) -> &'static str { "Git status" }
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "short": {"type": "boolean", "description": "Short format"},
                "session_id": {"type": "string", "description": "Execute in session sandbox"}
            }
        })
    }
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: GitStatusArgs = serde_json::from_value(args.clone()).unwrap_or_default();
            
            if let Some(session_id) = &args.session_id {
                return call_mcp_tool(session_id, "git_status", serde_json::json!({"short": args.short})).await;
            }
            
            let mut cmd = Command::new("git");
            cmd.arg("status");
            if args.short { cmd.arg("-s"); }
            run_git(cmd).await
        })
    }
}

// --- Git Diff ---
#[derive(Debug, Clone, Deserialize, Default)]
pub struct GitDiffArgs {
    #[serde(default)]
    pub staged: bool,
    #[serde(default)]
    pub paths: Vec<String>,
    #[serde(default)]
    pub session_id: Option<String>,
}

pub struct GitDiffTool;

impl Tool for GitDiffTool {
    fn name(&self) -> &'static str { "git_diff" }
    fn description(&self) -> &'static str { "Git diff" }
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "staged": {"type": "boolean", "description": "Compare staged changes"},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Specific paths"},
                "session_id": {"type": "string", "description": "Execute in session sandbox"}
            }
        })
    }
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: GitDiffArgs = serde_json::from_value(args.clone()).unwrap_or_default();
            
            if let Some(session_id) = &args.session_id {
                return call_mcp_tool(session_id, "git_diff", serde_json::json!({
                    "staged": args.staged,
                    "paths": args.paths
                })).await;
            }
            
            let mut cmd = Command::new("git");
            cmd.arg("diff");
            if args.staged { cmd.arg("--staged"); }
            for p in &args.paths { cmd.arg(p); }
            run_git(cmd).await
        })
    }
}

// --- Git Log ---
#[derive(Debug, Clone, Deserialize)]
pub struct GitLogArgs {
    #[serde(default = "default_count")]
    pub count: usize,
    #[serde(default)]
    pub oneline: bool,
    #[serde(default)]
    pub session_id: Option<String>,
}

fn default_count() -> usize { 10 }

pub struct GitLogTool;

impl Tool for GitLogTool {
    fn name(&self) -> &'static str { "git_log" }
    fn description(&self) -> &'static str { "Git log" }
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "Number of commits", "default": 10},
                "oneline": {"type": "boolean", "description": "One line format"},
                "session_id": {"type": "string", "description": "Execute in session sandbox"}
            }
        })
    }
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: GitLogArgs = serde_json::from_value(args.clone())
                .unwrap_or(GitLogArgs { count: 10, oneline: false, session_id: None });
            
            if let Some(session_id) = &args.session_id {
                return call_mcp_tool(session_id, "git_log", serde_json::json!({
                    "count": args.count,
                    "oneline": args.oneline
                })).await;
            }
            
            let mut cmd = Command::new("git");
            cmd.arg("log").arg(format!("-{}", args.count));
            if args.oneline { cmd.arg("--oneline"); }
            run_git(cmd).await
        })
    }
}

async fn run_git(mut cmd: Command) -> ToolResult {
    match cmd.output().await {
        Ok(o) if o.status.success() => ToolResult::ok(String::from_utf8_lossy(&o.stdout).to_string()),
        Ok(o) => ToolResult::err(format!("{}\n{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))),
        Err(e) => ToolResult::err(format!("git failed: {e}")),
    }
}
