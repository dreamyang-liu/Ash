//! Utility tools - find, tree, diff, patch, http, file_info, undo

use crate::{BoxFuture, Tool, ToolResult};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::process::Command;
use tokio::fs;

use lazy_static::lazy_static;
use tokio::sync::Mutex;
use std::collections::VecDeque;

// ==================== Edit History for Undo ====================

struct EditRecord {
    path: String,
    content: String,
    timestamp: chrono::DateTime<chrono::Utc>,
}

lazy_static! {
    static ref EDIT_HISTORY: Mutex<VecDeque<EditRecord>> = Mutex::new(VecDeque::new());
}

const MAX_UNDO_HISTORY: usize = 50;

/// Call this before any edit operation to save undo state
pub async fn save_undo_state(path: &str) -> anyhow::Result<()> {
    if let Ok(content) = fs::read_to_string(path).await {
        let mut history = EDIT_HISTORY.lock().await;
        history.push_back(EditRecord {
            path: path.to_string(),
            content,
            timestamp: chrono::Utc::now(),
        });
        // Keep bounded
        while history.len() > MAX_UNDO_HISTORY {
            history.pop_front();
        }
    }
    Ok(())
}

// ==================== find_files ====================

pub struct FindFilesTool;

impl Tool for FindFilesTool {
    fn name(&self) -> &'static str { "find_files" }
    fn description(&self) -> &'static str { "Find files by name pattern (glob)" }
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "File name pattern (glob, e.g., *.py, test_*)"},
                "path": {"type": "string", "description": "Search directory", "default": "."},
                "max_depth": {"type": "integer", "description": "Max directory depth"},
                "limit": {"type": "integer", "default": 100}
            },
            "required": ["pattern"]
        })
    }
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                pattern: String,
                #[serde(default = "default_path")]
                path: String,
                max_depth: Option<usize>,
                #[serde(default = "default_limit")]
                limit: usize,
            }
            fn default_path() -> String { ".".to_string() }
            fn default_limit() -> usize { 100 }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            // Use fd if available, fallback to find
            let mut cmd = Command::new("fd");
            cmd.arg("--glob").arg(&args.pattern).arg(&args.path);
            if let Some(d) = args.max_depth { cmd.arg("--max-depth").arg(d.to_string()); }
            
            let output = match cmd.output().await {
                Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
                _ => {
                    // Fallback to find
                    let mut cmd = Command::new("find");
                    cmd.arg(&args.path).arg("-name").arg(&args.pattern);
                    if let Some(d) = args.max_depth { cmd.arg("-maxdepth").arg(d.to_string()); }
                    
                    match cmd.output().await {
                        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
                        Err(e) => return ToolResult::err(format!("find failed: {e}")),
                    }
                }
            };
            
            let lines: Vec<&str> = output.lines().take(args.limit).collect();
            let truncated = output.lines().count() > args.limit;
            let mut result = lines.join("\n");
            if truncated {
                result.push_str(&format!("\n... (truncated to {} results)", args.limit));
            }
            if result.is_empty() {
                result = "No files found".to_string();
            }
            
            ToolResult::ok(result)
        })
    }
}

// ==================== tree ====================

pub struct TreeTool;

impl Tool for TreeTool {
    fn name(&self) -> &'static str { "tree" }
    fn description(&self) -> &'static str { "Show directory tree structure" }
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "max_depth": {"type": "integer", "default": 3},
                "show_hidden": {"type": "boolean", "default": false}
            }
        })
    }
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                #[serde(default = "default_path")]
                path: String,
                #[serde(default = "default_depth")]
                max_depth: usize,
                #[serde(default)]
                show_hidden: bool,
            }
            fn default_path() -> String { ".".to_string() }
            fn default_depth() -> usize { 3 }
            
            let args: Args = serde_json::from_value(args).unwrap_or(Args {
                path: ".".to_string(), max_depth: 3, show_hidden: false
            });
            
            // Try tree command first
            let mut cmd = Command::new("tree");
            cmd.arg(&args.path)
                .arg("-L").arg(args.max_depth.to_string())
                .arg("--noreport");
            if !args.show_hidden { cmd.arg("-I").arg(".*"); }
            
            match cmd.output().await {
                Ok(o) if o.status.success() => {
                    ToolResult::ok(String::from_utf8_lossy(&o.stdout).to_string())
                }
                _ => {
                    // Fallback: simple recursive listing
                    let result = build_tree(&args.path, args.max_depth, args.show_hidden, 0).await;
                    ToolResult::ok(result)
                }
            }
        })
    }
}

fn build_tree(path: &str, max_depth: usize, show_hidden: bool, depth: usize) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send + '_>> {
    Box::pin(async move {
        if depth >= max_depth {
            return String::new();
        }
        
        let mut result = String::new();
        let indent = "  ".repeat(depth);
        
        if let Ok(mut entries) = fs::read_dir(path).await {
            let mut items = Vec::new();
            while let Ok(Some(entry)) = entries.next_entry().await {
                let name = entry.file_name().to_string_lossy().to_string();
                if !show_hidden && name.starts_with('.') { continue; }
                items.push((name, entry.path(), entry.file_type().await.ok()));
            }
            items.sort_by(|a, b| a.0.cmp(&b.0));
            
            for (name, entry_path, file_type) in items {
                let is_dir = file_type.map(|ft| ft.is_dir()).unwrap_or(false);
                let suffix = if is_dir { "/" } else { "" };
                result.push_str(&format!("{}├── {}{}\n", indent, name, suffix));
                
                if is_dir {
                    let sub = build_tree(entry_path.to_str().unwrap_or(""), max_depth, show_hidden, depth + 1).await;
                    result.push_str(&sub);
                }
            }
        }
        result
    })
}

// ==================== diff_files ====================

pub struct DiffFilesTool;

impl Tool for DiffFilesTool {
    fn name(&self) -> &'static str { "diff_files" }
    fn description(&self) -> &'static str { "Compare two files (unified diff)" }
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "file1": {"type": "string", "description": "First file"},
                "file2": {"type": "string", "description": "Second file"},
                "context": {"type": "integer", "default": 3, "description": "Context lines"}
            },
            "required": ["file1", "file2"]
        })
    }
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                file1: String,
                file2: String,
                #[serde(default = "default_context")]
                context: usize,
            }
            fn default_context() -> usize { 3 }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let output = Command::new("diff")
                .arg("-u")
                .arg(format!("-U{}", args.context))
                .arg(&args.file1)
                .arg(&args.file2)
                .output()
                .await;
            
            match output {
                Ok(o) => {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    if stdout.is_empty() {
                        ToolResult::ok("Files are identical".to_string())
                    } else {
                        ToolResult::ok(stdout.to_string())
                    }
                }
                Err(e) => ToolResult::err(format!("diff failed: {e}")),
            }
        })
    }
}

// ==================== patch_apply ====================

pub struct PatchApplyTool;

impl Tool for PatchApplyTool {
    fn name(&self) -> &'static str { "patch_apply" }
    fn description(&self) -> &'static str { "Apply a unified diff patch" }
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "Unified diff content"},
                "path": {"type": "string", "description": "Base directory to apply patch"},
                "dry_run": {"type": "boolean", "default": false, "description": "Check without applying"}
            },
            "required": ["patch"]
        })
    }
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                patch: String,
                path: Option<String>,
                #[serde(default)]
                dry_run: bool,
            }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let mut cmd = Command::new("patch");
            cmd.arg("-p1"); // Strip one leading path component
            if args.dry_run { cmd.arg("--dry-run"); }
            if let Some(p) = &args.path { cmd.current_dir(p); }
            
            cmd.stdin(std::process::Stdio::piped());
            
            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(e) => return ToolResult::err(format!("Failed to spawn patch: {e}")),
            };
            
            if let Some(mut stdin) = child.stdin.take() {
                use tokio::io::AsyncWriteExt;
                let _ = stdin.write_all(args.patch.as_bytes()).await;
            }
            
            match child.wait_with_output().await {
                Ok(o) => {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    if o.status.success() {
                        ToolResult::ok(format!("{}\n{}", stdout, stderr))
                    } else {
                        ToolResult::err(format!("{}\n{}", stdout, stderr))
                    }
                }
                Err(e) => ToolResult::err(format!("patch failed: {e}")),
            }
        })
    }
}

// ==================== http_fetch ====================

pub struct HttpFetchTool;

impl Tool for HttpFetchTool {
    fn name(&self) -> &'static str { "http_fetch" }
    fn description(&self) -> &'static str { "HTTP GET request" }
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
                "headers": {"type": "object", "description": "Request headers"},
                "timeout_secs": {"type": "integer", "default": 30}
            },
            "required": ["url"]
        })
    }
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                url: String,
                #[serde(default)]
                headers: std::collections::HashMap<String, String>,
                #[serde(default = "default_timeout")]
                timeout_secs: u64,
            }
            fn default_timeout() -> u64 { 30 }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(args.timeout_secs))
                .build()
                .unwrap_or_default();
            
            let mut req = client.get(&args.url);
            for (k, v) in &args.headers {
                req = req.header(k.as_str(), v.as_str());
            }
            
            match req.send().await {
                Ok(resp) => {
                    let status = resp.status();
                    let headers: Vec<String> = resp.headers()
                        .iter()
                        .take(10)
                        .map(|(k, v)| format!("{}: {}", k, v.to_str().unwrap_or("")))
                        .collect();
                    
                    match resp.text().await {
                        Ok(body) => {
                            let truncated = if body.len() > 50000 {
                                format!("{}... (truncated)", &body[..50000])
                            } else {
                                body
                            };
                            ToolResult::ok(format!("Status: {}\nHeaders:\n{}\n\nBody:\n{}", 
                                status, headers.join("\n"), truncated))
                        }
                        Err(e) => ToolResult::err(format!("Failed to read body: {e}")),
                    }
                }
                Err(e) => ToolResult::err(format!("Request failed: {e}")),
            }
        })
    }
}

// ==================== file_info ====================

pub struct FileInfoTool;

impl Tool for FileInfoTool {
    fn name(&self) -> &'static str { "file_info" }
    fn description(&self) -> &'static str { "Get file type and encoding info" }
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"}
            },
            "required": ["path"]
        })
    }
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args { path: String }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            // Use file command
            let output = Command::new("file")
                .arg("-b") // Brief
                .arg(&args.path)
                .output()
                .await;
            
            let file_type = match output {
                Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
                Err(_) => "unknown".to_string(),
            };
            
            // Get size
            let size = fs::metadata(&args.path).await.map(|m| m.len()).unwrap_or(0);
            
            // Check if text
            let is_text = file_type.contains("text") || 
                          file_type.contains("ASCII") || 
                          file_type.contains("UTF-8");
            
            let info = json!({
                "path": args.path,
                "type": file_type,
                "size": size,
                "is_text": is_text,
            });
            
            ToolResult::ok(serde_json::to_string_pretty(&info).unwrap())
        })
    }
}

// ==================== undo ====================

pub struct UndoTool;

impl Tool for UndoTool {
    fn name(&self) -> &'static str { "undo" }
    fn description(&self) -> &'static str { "Undo last file edit" }
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Specific file to undo (optional, defaults to last edited)"},
                "list": {"type": "boolean", "description": "List undo history instead of undoing"}
            }
        })
    }
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize, Default)]
            struct Args {
                path: Option<String>,
                #[serde(default)]
                list: bool,
            }
            let args: Args = serde_json::from_value(args).unwrap_or_default();
            
            let mut history = EDIT_HISTORY.lock().await;
            
            if args.list {
                if history.is_empty() {
                    return ToolResult::ok("No edit history".to_string());
                }
                let mut out = String::from("Edit history (newest first):\n");
                for (i, record) in history.iter().rev().enumerate().take(20) {
                    out.push_str(&format!("  {}. {} ({})\n", i + 1, record.path, record.timestamp.format("%H:%M:%S")));
                }
                return ToolResult::ok(out);
            }
            
            // Find record to undo
            let record_idx = if let Some(path) = &args.path {
                history.iter().rposition(|r| r.path == *path)
            } else {
                if history.is_empty() { None } else { Some(history.len() - 1) }
            };
            
            match record_idx {
                Some(idx) => {
                    let record = history.remove(idx).unwrap();
                    match fs::write(&record.path, &record.content).await {
                        Ok(_) => ToolResult::ok(format!("Restored {} to state from {}", record.path, record.timestamp.format("%H:%M:%S"))),
                        Err(e) => ToolResult::err(format!("Failed to restore: {e}")),
                    }
                }
                None => ToolResult::err("No matching edit to undo".to_string()),
            }
        })
    }
}
