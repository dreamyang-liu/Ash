//! Clipboard tool - Named clips for working memory

use crate::{BoxFuture, Tool, ToolResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, BufReader};

/// Clipboard storage format
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Clipboard {
    clips: HashMap<String, ClipEntry>,
    counter: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClipEntry {
    content: String,
    source: Option<String>,
    created_at: String,
}

fn clipboard_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ash")
        .join("clipboard.json")
}

async fn load_clipboard() -> Clipboard {
    let path = clipboard_path();
    match fs::read_to_string(&path).await {
        Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
        Err(_) => Clipboard::default(),
    }
}

async fn save_clipboard(cb: &Clipboard) -> Result<(), String> {
    let path = clipboard_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await.map_err(|e| e.to_string())?;
    }
    let content = serde_json::to_string_pretty(cb).map_err(|e| e.to_string())?;
    fs::write(&path, content).await.map_err(|e| e.to_string())
}

// ========== Public helpers for buffer integration ==========

/// Save content to clipboard (for buffer_to_clip)
pub async fn save_clip_entry(name: &str, content: &str, source: Option<&str>) -> Result<(), String> {
    let mut cb = load_clipboard().await;
    
    let entry = ClipEntry {
        content: content.to_string(),
        source: source.map(String::from),
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    
    cb.clips.insert(name.to_string(), entry);
    save_clipboard(&cb).await
}

/// Load content from clipboard (for clip_to_buffer)
pub async fn load_clip_entry(name: &str) -> Result<String, String> {
    let cb = load_clipboard().await;
    
    match cb.clips.get(name) {
        Some(entry) => Ok(entry.content.clone()),
        None => Err(format!("Clip '{}' not found", name)),
    }
}

// ========== Clip (Copy) ==========

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ClipArgs {
    /// Direct content to clip
    #[serde(default)]
    pub content: Option<String>,
    /// File path with optional :start-end range
    #[serde(default)]
    pub file: Option<String>,
    /// Clip name (auto if omitted)
    #[serde(default)]
    pub name: Option<String>,
    /// Source reference override
    #[serde(default)]
    pub source: Option<String>,
}

pub struct ClipTool;

impl Tool for ClipTool {
    fn name(&self) -> &'static str { "clip" }
    fn description(&self) -> &'static str { "Save content or file range to named clipboard" }
    
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Direct content to clip"},
                "file": {"type": "string", "description": "File path, optionally with :start-end (e.g., src/lib.rs:10-20)"},
                "name": {"type": "string", "description": "Clip name (auto-generated if omitted)"},
                "source": {"type": "string", "description": "Source reference override"}
            }
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: ClipArgs = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            // Get content and source
            let (content, auto_source) = if let Some(file_spec) = &args.file {
                match read_file_range(file_spec).await {
                    Ok((c, s)) => (c, Some(s)),
                    Err(e) => return ToolResult::err(e),
                }
            } else if let Some(c) = &args.content {
                (c.clone(), None)
            } else {
                return ToolResult::err("Provide 'content' or 'file'");
            };
            
            let source = args.source.or(auto_source);
            
            let mut cb = load_clipboard().await;
            let name = args.name.unwrap_or_else(|| {
                cb.counter += 1;
                format!("_{}", cb.counter)
            });
            
            let entry = ClipEntry {
                content: content.clone(),
                source,
                created_at: chrono::Utc::now().to_rfc3339(),
            };
            
            cb.clips.insert(name.clone(), entry);
            
            if let Err(e) = save_clipboard(&cb).await {
                return ToolResult::err(format!("Failed to save: {e}"));
            }
            
            let lines = content.lines().count();
            let preview = if content.len() > 60 {
                format!("{}...", &content[..60].replace('\n', "\\n"))
            } else {
                content.replace('\n', "\\n")
            };
            
            ToolResult::ok(format!("Clipped '{}' ({} lines): {}", name, lines, preview))
        })
    }
}

/// Parse file:start-end and read content
async fn read_file_range(spec: &str) -> Result<(String, String), String> {
    // Parse path:start-end
    let (path, range) = if let Some(colon_idx) = spec.rfind(':') {
        let maybe_range = &spec[colon_idx + 1..];
        if maybe_range.contains('-') || maybe_range.chars().all(|c| c.is_ascii_digit()) {
            (&spec[..colon_idx], Some(maybe_range))
        } else {
            (spec, None)
        }
    } else {
        (spec, None)
    };
    
    let (start, end) = match range {
        Some(r) => {
            let parts: Vec<&str> = r.split('-').collect();
            let s: usize = parts.first().and_then(|x| x.parse().ok()).unwrap_or(1);
            let e: Option<usize> = parts.get(1).and_then(|x| x.parse().ok());
            (s, e)
        }
        None => (1, None),
    };
    
    // Read file
    let file = fs::File::open(path).await.map_err(|e| format!("Failed to open {path}: {e}"))?;
    let reader = BufReader::new(file);
    let mut lines_iter = reader.lines();
    let mut result = Vec::new();
    let mut line_num = 0usize;
    
    while let Some(line) = lines_iter.next_line().await.map_err(|e| format!("Read error: {e}"))? {
        line_num += 1;
        if line_num < start { continue; }
        if let Some(e) = end {
            if line_num > e { break; }
        }
        result.push(line);
        // Default limit if no end specified
        if end.is_none() && result.len() >= 100 { break; }
    }
    
    if result.is_empty() {
        return Err(format!("No content in range"));
    }
    
    let actual_end = start + result.len() - 1;
    let source = if start == 1 && end.is_none() {
        path.to_string()
    } else {
        format!("{}:{}-{}", path, start, actual_end)
    };
    
    Ok((result.join("\n"), source))
}

// ========== Paste ==========

#[derive(Debug, Clone, Deserialize, Default)]
pub struct PasteArgs {
    #[serde(default)]
    pub name: Option<String>,
}

pub struct PasteTool;

impl Tool for PasteTool {
    fn name(&self) -> &'static str { "paste" }
    fn description(&self) -> &'static str { "Retrieve content from clipboard" }
    
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Clip name (latest if omitted)"}
            }
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: PasteArgs = serde_json::from_value(args).unwrap_or_default();
            let cb = load_clipboard().await;
            
            if cb.clips.is_empty() {
                return ToolResult::err("Clipboard is empty");
            }
            
            let entry = match &args.name {
                Some(name) => cb.clips.get(name),
                None => cb.clips.values().max_by_key(|e| &e.created_at),
            };
            
            match entry {
                Some(e) => {
                    let mut result = e.content.clone();
                    if let Some(ref src) = e.source {
                        result = format!("# source: {}\n{}", src, result);
                    }
                    ToolResult::ok(result)
                }
                None => ToolResult::err(format!("Clip '{}' not found", args.name.unwrap_or_default())),
            }
        })
    }
}

// ========== Clips (List) ==========

pub struct ClipsTool;

impl Tool for ClipsTool {
    fn name(&self) -> &'static str { "clips" }
    fn description(&self) -> &'static str { "List clipboard entries" }
    
    fn schema(&self) -> Value {
        serde_json::json!({"type": "object", "properties": {}})
    }
    
    fn execute(&self, _args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let cb = load_clipboard().await;
            
            if cb.clips.is_empty() {
                return ToolResult::ok("Clipboard is empty");
            }
            
            let mut entries: Vec<_> = cb.clips.iter().collect();
            entries.sort_by(|a, b| b.1.created_at.cmp(&a.1.created_at));
            
            let lines: Vec<String> = entries.iter().map(|(name, entry)| {
                let preview = if entry.content.len() > 40 {
                    format!("{}...", &entry.content[..40].replace('\n', "\\n"))
                } else {
                    entry.content.replace('\n', "\\n")
                };
                let src = entry.source.as_deref().unwrap_or("-");
                format!("{:12} | {:45} | {}", name, preview, src)
            }).collect();
            
            ToolResult::ok(format!("NAME         | CONTENT                                       | SOURCE\n{}", lines.join("\n")))
        })
    }
}

// ========== Clear ==========

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ClearClipsArgs {
    #[serde(default)]
    pub name: Option<String>,
}

pub struct ClearClipsTool;

impl Tool for ClearClipsTool {
    fn name(&self) -> &'static str { "clips_clear" }
    fn description(&self) -> &'static str { "Clear clipboard" }
    
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Specific clip to remove (all if omitted)"}
            }
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: ClearClipsArgs = serde_json::from_value(args).unwrap_or_default();
            let mut cb = load_clipboard().await;
            
            let msg = match args.name {
                Some(name) => {
                    if cb.clips.remove(&name).is_some() {
                        format!("Removed '{}'", name)
                    } else {
                        return ToolResult::err(format!("Clip '{}' not found", name));
                    }
                }
                None => {
                    let count = cb.clips.len();
                    cb.clips.clear();
                    cb.counter = 0;
                    format!("Cleared {} clips", count)
                }
            };
            
            if let Err(e) = save_clipboard(&cb).await {
                return ToolResult::err(format!("Failed to save: {e}"));
            }
            
            ToolResult::ok(msg)
        })
    }
}
