//! Buffer tool - Named text buffers for agent workspace
//!
//! Provides line-based read/write operations on named buffers.

use crate::{BoxFuture, Tool, ToolResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;

// ========== Storage ==========

/// Buffer storage format
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct BufferStore {
    buffers: HashMap<String, Buffer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Buffer {
    lines: Vec<String>,
    created_at: String,
    modified_at: String,
}

impl Buffer {
    fn new() -> Self {
        Self {
            lines: Vec::new(),
            created_at: chrono::Utc::now().to_rfc3339(),
            modified_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    fn touch(&mut self) {
        self.modified_at = chrono::Utc::now().to_rfc3339();
    }

    fn line_count(&self) -> usize {
        self.lines.len()
    }

    fn char_count(&self) -> usize {
        self.lines.iter().map(|l| l.len() + 1).sum::<usize>().saturating_sub(1)
    }
}

fn buffer_store_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ash")
        .join("buffers.json")
}

async fn load_store() -> BufferStore {
    let path = buffer_store_path();
    match fs::read_to_string(&path).await {
        Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
        Err(_) => BufferStore::default(),
    }
}

async fn save_store(store: &BufferStore) -> Result<(), String> {
    let path = buffer_store_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await.map_err(|e| e.to_string())?;
    }
    let content = serde_json::to_string_pretty(store).map_err(|e| e.to_string())?;
    fs::write(&path, content).await.map_err(|e| e.to_string())
}

const DEFAULT_BUFFER: &str = "main";

// ========== buffer_read ==========

#[derive(Debug, Clone, Deserialize, Default)]
pub struct BufferReadArgs {
    /// Buffer name (default: "main")
    #[serde(default)]
    pub name: Option<String>,
    /// Start line (1-indexed, default: 1)
    #[serde(default)]
    pub start_line: Option<usize>,
    /// End line (inclusive, default: end of buffer)
    #[serde(default)]
    pub end_line: Option<usize>,
}

pub struct BufferReadTool;

impl Tool for BufferReadTool {
    fn name(&self) -> &'static str { "buffer_read" }
    fn description(&self) -> &'static str { "Read lines from a named buffer" }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Buffer name (default: 'main')"},
                "start_line": {"type": "integer", "description": "Start line, 1-indexed (default: 1)"},
                "end_line": {"type": "integer", "description": "End line, inclusive (default: end)"}
            }
        })
    }

    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: BufferReadArgs = serde_json::from_value(args).unwrap_or_default();
            let name = args.name.as_deref().unwrap_or(DEFAULT_BUFFER);
            let store = load_store().await;

            let buffer = match store.buffers.get(name) {
                Some(b) => b,
                None => return ToolResult::err(format!("Buffer '{}' not found. Use buffer_list to see available buffers.", name)),
            };

            if buffer.lines.is_empty() {
                return ToolResult::ok(format!("Buffer '{}' is empty (0 lines)", name));
            }

            let start = args.start_line.unwrap_or(1).max(1);
            let end = args.end_line.unwrap_or(buffer.lines.len()).min(buffer.lines.len());

            if start > buffer.lines.len() {
                return ToolResult::err(format!("Start line {} exceeds buffer length {}", start, buffer.lines.len()));
            }

            let mut output = String::new();
            for (i, line) in buffer.lines.iter().enumerate() {
                let line_num = i + 1;
                if line_num < start { continue; }
                if line_num > end { break; }
                output.push_str(&format!("{:6} | {}\n", line_num, line));
            }

            output.push_str(&format!("\n[{} lines shown, buffer '{}' has {} total]", end - start + 1, name, buffer.lines.len()));
            ToolResult::ok(output)
        })
    }
}

// ========== buffer_write ==========

#[derive(Debug, Clone, Deserialize, Default)]
pub struct BufferWriteArgs {
    /// Buffer name (default: "main")
    #[serde(default)]
    pub name: Option<String>,
    /// Content to write (will be split into lines)
    pub content: String,
    /// Line number to insert at (1-indexed). If omitted, replaces entire buffer.
    #[serde(default)]
    pub at_line: Option<usize>,
    /// If true, append to end instead of inserting
    #[serde(default)]
    pub append: bool,
}

pub struct BufferWriteTool;

impl Tool for BufferWriteTool {
    fn name(&self) -> &'static str { "buffer_write" }
    fn description(&self) -> &'static str { "Write content to a named buffer (create if needed)" }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "required": ["content"],
            "properties": {
                "name": {"type": "string", "description": "Buffer name (default: 'main')"},
                "content": {"type": "string", "description": "Content to write"},
                "at_line": {"type": "integer", "description": "Insert before this line (1-indexed). Omit to replace entire buffer."},
                "append": {"type": "boolean", "description": "Append to end (default: false)"}
            }
        })
    }

    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: BufferWriteArgs = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };

            let name = args.name.as_deref().unwrap_or(DEFAULT_BUFFER).to_string();
            let new_lines: Vec<String> = args.content.lines().map(String::from).collect();
            let new_count = new_lines.len();

            let mut store = load_store().await;
            let buffer = store.buffers.entry(name.clone()).or_insert_with(Buffer::new);

            let action = if args.append {
                buffer.lines.extend(new_lines);
                "appended"
            } else if let Some(at) = args.at_line {
                let idx = (at.saturating_sub(1)).min(buffer.lines.len());
                for (i, line) in new_lines.into_iter().enumerate() {
                    buffer.lines.insert(idx + i, line);
                }
                "inserted"
            } else {
                buffer.lines = new_lines;
                "replaced"
            };

            buffer.touch();
            let final_len = buffer.lines.len();

            if let Err(e) = save_store(&store).await {
                return ToolResult::err(format!("Failed to save: {e}"));
            }

            ToolResult::ok(format!(
                "Buffer '{}': {} {} lines (now {} total)",
                name, action, new_count, final_len
            ))
        })
    }
}

// ========== buffer_delete ==========

#[derive(Debug, Clone, Deserialize, Default)]
pub struct BufferDeleteArgs {
    /// Buffer name (default: "main")
    #[serde(default)]
    pub name: Option<String>,
    /// Start line to delete (1-indexed)
    pub start_line: usize,
    /// End line to delete (inclusive)
    pub end_line: usize,
}

pub struct BufferDeleteTool;

impl Tool for BufferDeleteTool {
    fn name(&self) -> &'static str { "buffer_delete" }
    fn description(&self) -> &'static str { "Delete a range of lines from buffer" }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "required": ["start_line", "end_line"],
            "properties": {
                "name": {"type": "string", "description": "Buffer name (default: 'main')"},
                "start_line": {"type": "integer", "description": "First line to delete (1-indexed)"},
                "end_line": {"type": "integer", "description": "Last line to delete (inclusive)"}
            }
        })
    }

    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: BufferDeleteArgs = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };

            let name = args.name.as_deref().unwrap_or(DEFAULT_BUFFER);
            let mut store = load_store().await;

            let buffer = match store.buffers.get_mut(name) {
                Some(b) => b,
                None => return ToolResult::err(format!("Buffer '{}' not found", name)),
            };

            if args.start_line < 1 || args.start_line > buffer.lines.len() {
                return ToolResult::err(format!("Invalid start_line {}", args.start_line));
            }

            let start_idx = args.start_line - 1;
            let end_idx = args.end_line.min(buffer.lines.len());
            let count = end_idx - start_idx;

            buffer.lines.drain(start_idx..end_idx);
            buffer.touch();
            let final_len = buffer.lines.len();

            if let Err(e) = save_store(&store).await {
                return ToolResult::err(format!("Failed to save: {e}"));
            }

            ToolResult::ok(format!(
                "Buffer '{}': deleted {} lines (now {} total)",
                name, count, final_len
            ))
        })
    }
}

// ========== buffer_replace ==========

#[derive(Debug, Clone, Deserialize, Default)]
pub struct BufferReplaceArgs {
    /// Buffer name (default: "main")
    #[serde(default)]
    pub name: Option<String>,
    /// Start line to replace (1-indexed)
    pub start_line: usize,
    /// End line to replace (inclusive)
    pub end_line: usize,
    /// Replacement content
    pub content: String,
}

pub struct BufferReplaceTool;

impl Tool for BufferReplaceTool {
    fn name(&self) -> &'static str { "buffer_replace" }
    fn description(&self) -> &'static str { "Replace a range of lines in buffer" }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "required": ["start_line", "end_line", "content"],
            "properties": {
                "name": {"type": "string", "description": "Buffer name (default: 'main')"},
                "start_line": {"type": "integer", "description": "First line to replace (1-indexed)"},
                "end_line": {"type": "integer", "description": "Last line to replace (inclusive)"},
                "content": {"type": "string", "description": "Replacement content"}
            }
        })
    }

    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: BufferReplaceArgs = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };

            let name = args.name.as_deref().unwrap_or(DEFAULT_BUFFER);
            let new_lines: Vec<String> = args.content.lines().map(String::from).collect();

            let mut store = load_store().await;

            let buffer = match store.buffers.get_mut(name) {
                Some(b) => b,
                None => return ToolResult::err(format!("Buffer '{}' not found", name)),
            };

            if args.start_line < 1 || args.start_line > buffer.lines.len() {
                return ToolResult::err(format!("Invalid start_line {}", args.start_line));
            }

            let start_idx = args.start_line - 1;
            let end_idx = args.end_line.min(buffer.lines.len());
            let old_count = end_idx - start_idx;

            // Remove old range
            buffer.lines.drain(start_idx..end_idx);
            // Insert new content
            let new_count = new_lines.len();
            for (i, line) in new_lines.into_iter().enumerate() {
                buffer.lines.insert(start_idx + i, line);
            }
            buffer.touch();
            let final_len = buffer.lines.len();

            if let Err(e) = save_store(&store).await {
                return ToolResult::err(format!("Failed to save: {e}"));
            }

            ToolResult::ok(format!(
                "Buffer '{}': replaced {} lines with {} lines (now {} total)",
                name, old_count, new_count, final_len
            ))
        })
    }
}

// ========== buffer_list ==========

pub struct BufferListTool;

impl Tool for BufferListTool {
    fn name(&self) -> &'static str { "buffer_list" }
    fn description(&self) -> &'static str { "List all named buffers" }

    fn schema(&self) -> Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    fn execute(&self, _args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let store = load_store().await;

            if store.buffers.is_empty() {
                return ToolResult::ok("No buffers. Use buffer_write to create one.");
            }

            let mut entries: Vec<_> = store.buffers.iter().collect();
            entries.sort_by(|a, b| b.1.modified_at.cmp(&a.1.modified_at));

            let mut output = String::from("NAME             | LINES  | CHARS   | MODIFIED\n");
            output.push_str(&"-".repeat(60));
            output.push('\n');

            for (name, buf) in entries {
                let modified = &buf.modified_at[..19]; // trim timezone
                output.push_str(&format!(
                    "{:16} | {:6} | {:7} | {}\n",
                    name, buf.line_count(), buf.char_count(), modified
                ));
            }

            ToolResult::ok(output)
        })
    }
}

// ========== buffer_clear ==========

#[derive(Debug, Clone, Deserialize, Default)]
pub struct BufferClearArgs {
    /// Buffer name to clear (if omitted, clears ALL buffers)
    #[serde(default)]
    pub name: Option<String>,
}

pub struct BufferClearTool;

impl Tool for BufferClearTool {
    fn name(&self) -> &'static str { "buffer_clear" }
    fn description(&self) -> &'static str { "Clear buffer content or delete buffer entirely" }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Buffer to clear (omit to clear ALL buffers)"}
            }
        })
    }

    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: BufferClearArgs = serde_json::from_value(args).unwrap_or_default();
            let mut store = load_store().await;

            let msg = match args.name {
                Some(name) => {
                    if store.buffers.remove(&name).is_some() {
                        format!("Deleted buffer '{}'", name)
                    } else {
                        return ToolResult::err(format!("Buffer '{}' not found", name));
                    }
                }
                None => {
                    let count = store.buffers.len();
                    store.buffers.clear();
                    format!("Cleared all {} buffers", count)
                }
            };

            if let Err(e) = save_store(&store).await {
                return ToolResult::err(format!("Failed to save: {e}"));
            }

            ToolResult::ok(msg)
        })
    }
}

