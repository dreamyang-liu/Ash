//! Grep tool - Search with ripgrep

use crate::{BoxFuture, Tool, ToolResult};
use crate::backend::ExecOptions;
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
    /// Execute in session sandbox
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
            
            // Build grep command
            let mut cmd_str = format!(
                "rg --line-number --no-heading --color=never --max-count {} --regexp {} ",
                args.limit,
                shell_escape(&args.pattern)
            );
            if let Some(ref glob) = args.include {
                cmd_str.push_str(&format!("--glob {} ", shell_escape(glob)));
            }
            cmd_str.push_str(&format!("-- {}", shell_escape(&args.path)));
            
            // Route to session if provided
            if let Some(session_id) = &args.session_id {
                match session::exec_in_session(&session_id, &cmd_str, ExecOptions::default()).await {
                    Ok(result) => {
                        if result.exit_code == 1 {
                            // No matches
                            ToolResult::ok("No matches found.".to_string())
                        } else if result.success() {
                            ToolResult::ok(result.stdout)
                        } else {
                            ToolResult::err(result.stderr)
                        }
                    }
                    Err(e) => ToolResult::err(format!("{e}")),
                }
            } else {
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
            }
        })
    }
}

/// Basic shell escaping
fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}
