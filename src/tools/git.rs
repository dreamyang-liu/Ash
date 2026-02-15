//! Git tools

use crate::{BoxFuture, Tool, ToolResult};
use crate::backend::ExecOptions;
use crate::tools::session;
use serde::Deserialize;
use serde_json::Value;
use tokio::process::Command;

// Helper for session execution
async fn exec_git_in_session(session_id: &str, cmd_args: &str) -> ToolResult {
    let cmd = format!("git {}", cmd_args);
    match session::exec_in_session(session_id, &cmd, ExecOptions::default()).await {
        Ok(result) => {
            if result.success() {
                ToolResult::ok(result.stdout)
            } else {
                ToolResult::err(format!("{}\n{}", result.stdout, result.stderr))
            }
        }
        Err(e) => ToolResult::err(format!("{e}")),
    }
}

async fn run_git(mut cmd: Command) -> ToolResult {
    match cmd.output().await {
        Ok(o) if o.status.success() => ToolResult::ok(String::from_utf8_lossy(&o.stdout).to_string()),
        Ok(o) => ToolResult::err(format!("{}\n{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))),
        Err(e) => ToolResult::err(format!("git failed: {e}")),
    }
}

// ==================== git_status ====================

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
            #[derive(Deserialize, Default)]
            struct Args {
                #[serde(default)]
                short: bool,
                session_id: Option<String>,
            }
            let args: Args = serde_json::from_value(args.clone()).unwrap_or_default();
            
            if let Some(sid) = &args.session_id {
                let cmd = if args.short { "status -s" } else { "status" };
                return exec_git_in_session(&sid, cmd).await;
            }
            
            let mut cmd = Command::new("git");
            cmd.arg("status");
            if args.short { cmd.arg("-s"); }
            run_git(cmd).await
        })
    }
}

// ==================== git_diff ====================

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
            #[derive(Deserialize, Default)]
            struct Args {
                #[serde(default)]
                staged: bool,
                #[serde(default)]
                paths: Vec<String>,
                session_id: Option<String>,
            }
            let args: Args = serde_json::from_value(args.clone()).unwrap_or_default();
            
            if let Some(sid) = &args.session_id {
                let mut cmd = String::from("diff");
                if args.staged { cmd.push_str(" --staged"); }
                for p in &args.paths { cmd.push_str(&format!(" '{}'", p.replace('\'', "'\\''"))); }
                return exec_git_in_session(&sid, &cmd).await;
            }
            
            let mut cmd = Command::new("git");
            cmd.arg("diff");
            if args.staged { cmd.arg("--staged"); }
            for p in &args.paths { cmd.arg(p); }
            run_git(cmd).await
        })
    }
}

// ==================== git_log ====================

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
            #[derive(Deserialize)]
            struct Args {
                #[serde(default = "default_count")]
                count: usize,
                #[serde(default)]
                oneline: bool,
                session_id: Option<String>,
            }
            fn default_count() -> usize { 10 }
            
            let args: Args = serde_json::from_value(args.clone())
                .unwrap_or(Args { count: 10, oneline: false, session_id: None });
            
            if let Some(sid) = &args.session_id {
                let mut cmd = format!("log -{}", args.count);
                if args.oneline { cmd.push_str(" --oneline"); }
                return exec_git_in_session(&sid, &cmd).await;
            }
            
            let mut cmd = Command::new("git");
            cmd.arg("log").arg(format!("-{}", args.count));
            if args.oneline { cmd.arg("--oneline"); }
            run_git(cmd).await
        })
    }
}

// ==================== git_add ====================

pub struct GitAddTool;

impl Tool for GitAddTool {
    fn name(&self) -> &'static str { "git_add" }
    fn description(&self) -> &'static str { "Git add (stage files)" }
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Files to stage"},
                "all": {"type": "boolean", "default": false, "description": "Stage all changes (-A)"},
                "session_id": {"type": "string", "description": "Execute in session sandbox"}
            }
        })
    }
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize, Default)]
            struct Args {
                #[serde(default)]
                paths: Vec<String>,
                #[serde(default)]
                all: bool,
                session_id: Option<String>,
            }
            let args: Args = serde_json::from_value(args.clone()).unwrap_or_default();
            
            if let Some(sid) = &args.session_id {
                let cmd = if args.all {
                    "add -A".to_string()
                } else if args.paths.is_empty() {
                    return ToolResult::err("Specify paths or use all=true".to_string());
                } else {
                    format!("add {}", args.paths.iter().map(|p| format!("'{}'", p.replace('\'', "'\\''"))).collect::<Vec<_>>().join(" "))
                };
                return exec_git_in_session(&sid, &cmd).await;
            }
            
            let mut cmd = Command::new("git");
            cmd.arg("add");
            
            if args.all {
                cmd.arg("-A");
            } else if args.paths.is_empty() {
                return ToolResult::err("Specify paths or use all=true".to_string());
            } else {
                for p in &args.paths { cmd.arg(p); }
            }
            
            run_git(cmd).await
        })
    }
}

// ==================== git_commit ====================

pub struct GitCommitTool;

impl Tool for GitCommitTool {
    fn name(&self) -> &'static str { "git_commit" }
    fn description(&self) -> &'static str { "Git commit" }
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Commit message"},
                "all": {"type": "boolean", "default": false, "description": "Stage all and commit (-a)"},
                "session_id": {"type": "string", "description": "Execute in session sandbox"}
            },
            "required": ["message"]
        })
    }
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                message: String,
                #[serde(default)]
                all: bool,
                session_id: Option<String>,
            }
            let args: Args = match serde_json::from_value(args.clone()) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            if let Some(sid) = &args.session_id {
                let mut cmd = String::from("commit");
                if args.all { cmd.push_str(" -a"); }
                cmd.push_str(&format!(" -m '{}'", args.message.replace('\'', "'\\''")));
                return exec_git_in_session(&sid, &cmd).await;
            }
            
            let mut cmd = Command::new("git");
            cmd.arg("commit");
            if args.all { cmd.arg("-a"); }
            cmd.arg("-m").arg(&args.message);
            
            run_git(cmd).await
        })
    }
}
