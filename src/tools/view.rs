//! View tool - Read file with line numbers

use crate::{BoxFuture, Tool, ToolResult};
use crate::tools::session;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
pub struct ViewArgs {
    pub file_path: String,
    #[serde(default = "default_offset")]
    pub offset: usize,
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Execute in session sandbox
    #[serde(default)]
    pub session_id: Option<String>,
}

fn default_offset() -> usize { 1 }
fn default_limit() -> usize { 100 }

pub struct ViewTool;

impl Tool for ViewTool {
    fn name(&self) -> &'static str { "read_file" }
    
    fn description(&self) -> &'static str { 
        "Read file contents with line numbers"
    }
    
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path"},
                "offset": {"type": "integer", "description": "Start line (1-indexed)", "default": 1},
                "limit": {"type": "integer", "description": "Max lines", "default": 100},
                "session_id": {"type": "string", "description": "Execute in session sandbox"}
            },
            "required": ["file_path"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: ViewArgs = match serde_json::from_value(args.clone()) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            // Route to session if provided
            if let Some(session_id) = &args.session_id {
                match session::read_file_in_session(&session_id, &args.file_path).await {
                    Ok(content) => {
                        let formatted = format_with_lines(&content, args.offset, args.limit);
                        ToolResult::ok(formatted)
                    }
                    Err(e) => ToolResult::err(format!("{e}")),
                }
            } else {
                // Local execution
                match read_file_with_lines(&args.file_path, args.offset, args.limit).await {
                    Ok(content) => ToolResult::ok(content),
                    Err(e) => ToolResult::err(e),
                }
            }
        })
    }
}

/// Format content with line numbers, applying offset and limit
fn format_with_lines(content: &str, offset: usize, limit: usize) -> String {
    content.lines()
        .enumerate()
        .skip(offset.saturating_sub(1))
        .take(limit)
        .map(|(i, line)| format!("{:>6} | {}", i + 1, &line[..line.len().min(500)]))
        .collect::<Vec<_>>()
        .join("\n")
}

async fn read_file_with_lines(path: &str, offset: usize, limit: usize) -> Result<String, String> {
    use tokio::fs::File;
    use tokio::io::{AsyncBufReadExt, BufReader};
    
    let file = File::open(path).await.map_err(|e| format!("Failed to open: {e}"))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut result = Vec::new();
    let mut line_num = 0usize;
    
    while let Some(line) = lines.next_line().await.map_err(|e| format!("Read error: {e}"))? {
        line_num += 1;
        if line_num < offset { continue; }
        if result.len() >= limit { break; }
        result.push(format!("{:>6} | {}", line_num, &line[..line.len().min(500)]));
    }
    
    if result.is_empty() {
        return Err(format!("No content at offset {offset}"));
    }
    Ok(result.join("\n"))
}
