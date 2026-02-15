//! Edit tool - str_replace, insert, create

use crate::{BoxFuture, Tool, ToolResult};
use crate::tools::session;
use serde::Deserialize;
use serde_json::Value;
use tokio::fs;

#[derive(Debug, Clone, Deserialize)]
pub struct EditArgs {
    pub command: String,  // view, str_replace, insert, create
    pub path: String,
    // view
    #[serde(default)]
    pub view_range: Option<Vec<i64>>,
    // str_replace
    #[serde(default)]
    pub old_str: Option<String>,
    #[serde(default)]
    pub new_str: Option<String>,
    // insert
    #[serde(default)]
    pub insert_line: Option<i64>,
    #[serde(default)]
    pub insert_text: Option<String>,
    // create
    #[serde(default)]
    pub file_text: Option<String>,
    /// Execute in session sandbox via MCP
    #[serde(default)]
    pub session_id: Option<String>,
}

pub struct EditTool;

impl Tool for EditTool {
    fn name(&self) -> &'static str { "text_editor" }
    fn description(&self) -> &'static str { "Edit files with view/str_replace/insert/create" }
    
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "enum": ["view", "str_replace", "insert", "create"], "description": "Edit command"},
                "path": {"type": "string", "description": "File path"},
                "view_range": {"type": "array", "items": {"type": "integer"}, "description": "[start, end] lines for view"},
                "old_str": {"type": "string", "description": "Text to find (str_replace)"},
                "new_str": {"type": "string", "description": "Replacement text (str_replace)"},
                "insert_line": {"type": "integer", "description": "Line to insert after"},
                "insert_text": {"type": "string", "description": "Text to insert"},
                "file_text": {"type": "string", "description": "File content (create)"},
                "session_id": {"type": "string", "description": "Execute in session sandbox"}
            },
            "required": ["command", "path"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: EditArgs = match serde_json::from_value(args.clone()) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            // Route to session if provided
            if let Some(session_id) = &args.session_id {
                let mut mcp_args = serde_json::json!({
                    "command": args.command,
                    "path": args.path,
                });
                if let Some(v) = &args.view_range { mcp_args["view_range"] = serde_json::json!(v); }
                if let Some(v) = &args.old_str { mcp_args["old_str"] = serde_json::json!(v); }
                if let Some(v) = &args.new_str { mcp_args["new_str"] = serde_json::json!(v); }
                if let Some(v) = &args.insert_line { mcp_args["insert_line"] = serde_json::json!(v); }
                if let Some(v) = &args.insert_text { mcp_args["insert_text"] = serde_json::json!(v); }
                if let Some(v) = &args.file_text { mcp_args["file_text"] = serde_json::json!(v); }
                return call_mcp_tool(session_id, "text_editor", mcp_args).await;
            }
            
            // Local execution
            match args.command.as_str() {
                "view" => view_file(&args.path, args.view_range).await,
                "str_replace" => str_replace(&args.path, args.old_str.as_deref().unwrap_or(""), args.new_str.as_deref().unwrap_or("")).await,
                "insert" => insert_at(&args.path, args.insert_line.unwrap_or(0), args.insert_text.as_deref().unwrap_or("")).await,
                "create" => create_file(&args.path, args.file_text.as_deref().unwrap_or("")).await,
                _ => ToolResult::err(format!("Unknown command: {}", args.command)),
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

async fn view_file(path: &str, range: Option<Vec<i64>>) -> ToolResult {
    let content = match fs::read_to_string(path).await {
        Ok(c) => c,
        Err(e) => return ToolResult::err(format!("Read failed: {e}")),
    };
    let lines: Vec<&str> = content.lines().collect();
    let (start, end) = match range {
        Some(r) if r.len() >= 2 => ((r[0].max(1) - 1) as usize, if r[1] == -1 { lines.len() } else { r[1] as usize }),
        Some(r) if r.len() == 1 => ((r[0].max(1) - 1) as usize, lines.len()),
        _ => (0, lines.len()),
    };
    let result: Vec<String> = lines[start..end.min(lines.len())]
        .iter().enumerate()
        .map(|(i, l)| format!("{:>6} | {}", start + i + 1, l))
        .collect();
    ToolResult::ok(result.join("\n"))
}

async fn str_replace(path: &str, old: &str, new: &str) -> ToolResult {
    let content = match fs::read_to_string(path).await {
        Ok(c) => c,
        Err(e) => return ToolResult::err(format!("Read failed: {e}")),
    };
    let count = content.matches(old).count();
    if count == 0 { return ToolResult::err("No match found for old_str"); }
    if count > 1 { return ToolResult::err(format!("Multiple matches ({count}). old_str must be unique.")); }
    let new_content = content.replace(old, new);
    if let Err(e) = fs::write(path, &new_content).await {
        return ToolResult::err(format!("Write failed: {e}"));
    }
    ToolResult::ok("Replaced successfully")
}

async fn insert_at(path: &str, line: i64, text: &str) -> ToolResult {
    let content = match fs::read_to_string(path).await {
        Ok(c) => c,
        Err(e) => return ToolResult::err(format!("Read failed: {e}")),
    };
    let mut lines: Vec<&str> = content.lines().collect();
    let idx = if line <= 0 { 0 } else { (line as usize).min(lines.len()) };
    for (i, new_line) in text.lines().enumerate() {
        lines.insert(idx + i, new_line);
    }
    if let Err(e) = fs::write(path, lines.join("\n")).await {
        return ToolResult::err(format!("Write failed: {e}"));
    }
    ToolResult::ok(format!("Inserted at line {}", line))
}

async fn create_file(path: &str, content: &str) -> ToolResult {
    if fs::metadata(path).await.is_ok() {
        return ToolResult::err("File exists. Use str_replace to modify.");
    }
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = fs::create_dir_all(parent).await;
    }
    if let Err(e) = fs::write(path, content).await {
        return ToolResult::err(format!("Create failed: {e}"));
    }
    ToolResult::ok(format!("Created: {path}"))
}
