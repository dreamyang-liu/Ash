//! Async terminal/process management tools

use crate::{BoxFuture, Tool, ToolResult};
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

    async fn start(&mut self, command: &str, working_dir: Option<&str>, env: Option<HashMap<String, String>>) -> anyhow::Result<String> {
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

        // Spawn exit watcher
        let exit_clone = exit_code.clone();
        let child_arc = Arc::new(Mutex::new(Some(child)));
        let child_clone = child_arc.clone();
        tokio::spawn(async move {
            if let Some(ref mut c) = *child_clone.lock().await {
                if let Ok(status) = c.wait().await {
                    *exit_clone.lock().await = status.code().or(Some(-1));
                }
            }
        });

        let process = AsyncProcess {
            stdout_lines,
            stderr_lines,
            exit_code,
            command: command.to_string(),
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

    fn list(&self) -> Vec<(String, String, String, bool)> {
        self.processes.iter().map(|(id, p)| {
            let running = p.exit_code.try_lock().map(|e| e.is_none()).unwrap_or(true);
            (id.clone(), p.command.clone(), p.started_at.to_rfc3339(), running)
        }).collect()
    }

    fn remove(&mut self, id: &str) -> bool {
        self.processes.remove(id).is_some()
    }
}

lazy_static! {
    static ref REGISTRY: Mutex<ProcessRegistry> = Mutex::new(ProcessRegistry::new());
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
                "env": {"type": "object", "description": "Environment variables"}
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
            }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let mut registry = REGISTRY.lock().await;
            match registry.start(&args.command, args.working_dir.as_deref(), args.env).await {
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
            }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
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
                "handle": {"type": "string", "description": "Process handle"}
            },
            "required": ["handle"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args { handle: String }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
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
        serde_json::json!({"type": "object", "properties": {}})
    }
    
    fn execute(&self, _args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let registry = REGISTRY.lock().await;
            let list = registry.list();
            
            if list.is_empty() {
                return ToolResult::ok("No async processes".to_string());
            }
            
            let mut out = String::new();
            for (id, cmd, started, running) in list {
                let status = if running { "running" } else { "complete" };
                out.push_str(&format!("{} [{}] {} ({})\n", id, status, cmd, started));
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
                "handle": {"type": "string", "description": "Process handle"}
            },
            "required": ["handle"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args { handle: String }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let mut registry = REGISTRY.lock().await;
            if registry.remove(&args.handle) {
                ToolResult::ok("Removed".to_string())
            } else {
                ToolResult::err("Handle not found".to_string())
            }
        })
    }
}
