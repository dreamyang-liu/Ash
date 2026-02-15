//! Filesystem tools - list_dir, mkdir, remove, move, copy, stat

use crate::{BoxFuture, Tool, ToolResult};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::fs;
use std::path::Path;

// ==================== fs_list_dir ====================

pub struct FsListDirTool;

impl Tool for FsListDirTool {
    fn name(&self) -> &'static str { "fs_list_dir" }
    fn description(&self) -> &'static str { "List directory contents" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path"}
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
            
            let mut entries = match fs::read_dir(&args.path).await {
                Ok(d) => d,
                Err(e) => return ToolResult::err(format!("Failed to read dir: {e}")),
            };
            
            let mut out = String::new();
            while let Ok(Some(entry)) = entries.next_entry().await {
                let name = entry.file_name().to_string_lossy().to_string();
                let meta = entry.metadata().await.ok();
                let suffix = if meta.as_ref().map(|m| m.is_dir()).unwrap_or(false) { "/" } else { "" };
                let size = meta.map(|m| m.len()).unwrap_or(0);
                out.push_str(&format!("{}{} ({})\n", name, suffix, size));
            }
            
            if out.is_empty() {
                out = "(empty directory)\n".to_string();
            }
            
            ToolResult::ok(out)
        })
    }
}

// ==================== fs_mkdir ====================

pub struct FsMkdirTool;

impl Tool for FsMkdirTool {
    fn name(&self) -> &'static str { "fs_mkdir" }
    fn description(&self) -> &'static str { "Create directory (recursive by default)" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path"},
                "recursive": {"type": "boolean", "default": true}
            },
            "required": ["path"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                path: String,
                #[serde(default = "default_true")]
                recursive: bool,
            }
            fn default_true() -> bool { true }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let result = if args.recursive {
                fs::create_dir_all(&args.path).await
            } else {
                fs::create_dir(&args.path).await
            };
            
            match result {
                Ok(_) => ToolResult::ok(format!("Created {}", args.path)),
                Err(e) => ToolResult::err(format!("Failed: {e}")),
            }
        })
    }
}

// ==================== fs_remove ====================

pub struct FsRemoveTool;

impl Tool for FsRemoveTool {
    fn name(&self) -> &'static str { "fs_remove" }
    fn description(&self) -> &'static str { "Remove file or directory" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to remove"},
                "recursive": {"type": "boolean", "default": false, "description": "Remove directories recursively"}
            },
            "required": ["path"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                path: String,
                #[serde(default)]
                recursive: bool,
            }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let meta = match fs::metadata(&args.path).await {
                Ok(m) => m,
                Err(e) => return ToolResult::err(format!("Path error: {e}")),
            };
            
            let result = if meta.is_dir() {
                if args.recursive {
                    fs::remove_dir_all(&args.path).await
                } else {
                    fs::remove_dir(&args.path).await
                }
            } else {
                fs::remove_file(&args.path).await
            };
            
            match result {
                Ok(_) => ToolResult::ok(format!("Removed {}", args.path)),
                Err(e) => ToolResult::err(format!("Failed: {e}")),
            }
        })
    }
}

// ==================== fs_move ====================

pub struct FsMoveTool;

impl Tool for FsMoveTool {
    fn name(&self) -> &'static str { "fs_move" }
    fn description(&self) -> &'static str { "Move/rename file or directory" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "from": {"type": "string", "description": "Source path"},
                "to": {"type": "string", "description": "Destination path"}
            },
            "required": ["from", "to"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args { from: String, to: String }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            match fs::rename(&args.from, &args.to).await {
                Ok(_) => ToolResult::ok(format!("Moved {} -> {}", args.from, args.to)),
                Err(e) => ToolResult::err(format!("Failed: {e}")),
            }
        })
    }
}

// ==================== fs_copy ====================

pub struct FsCopyTool;

impl Tool for FsCopyTool {
    fn name(&self) -> &'static str { "fs_copy" }
    fn description(&self) -> &'static str { "Copy file or directory" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "from": {"type": "string", "description": "Source path"},
                "to": {"type": "string", "description": "Destination path"},
                "recursive": {"type": "boolean", "default": false}
            },
            "required": ["from", "to"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                from: String,
                to: String,
                #[serde(default)]
                recursive: bool,
            }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let meta = match fs::metadata(&args.from).await {
                Ok(m) => m,
                Err(e) => return ToolResult::err(format!("Source error: {e}")),
            };
            
            if meta.is_file() {
                // Create parent dirs
                if let Some(parent) = Path::new(&args.to).parent() {
                    let _ = fs::create_dir_all(parent).await;
                }
                match fs::copy(&args.from, &args.to).await {
                    Ok(_) => ToolResult::ok(format!("Copied {} -> {}", args.from, args.to)),
                    Err(e) => ToolResult::err(format!("Failed: {e}")),
                }
            } else if meta.is_dir() {
                if !args.recursive {
                    return ToolResult::err("Directory copy requires recursive=true".to_string());
                }
                match copy_dir_recursive(&args.from, &args.to).await {
                    Ok(_) => ToolResult::ok(format!("Copied {} -> {}", args.from, args.to)),
                    Err(e) => ToolResult::err(format!("Failed: {e}")),
                }
            } else {
                ToolResult::err("Unknown file type".to_string())
            }
        })
    }
}

async fn copy_dir_recursive(from: &str, to: &str) -> anyhow::Result<()> {
    fs::create_dir_all(to).await?;
    let mut dir = fs::read_dir(from).await?;
    
    while let Some(entry) = dir.next_entry().await? {
        let src = entry.path();
        let dst = Path::new(to).join(entry.file_name());
        
        if entry.file_type().await?.is_dir() {
            Box::pin(copy_dir_recursive(
                src.to_str().unwrap(),
                dst.to_str().unwrap(),
            )).await?;
        } else {
            fs::copy(&src, &dst).await?;
        }
    }
    Ok(())
}

// ==================== fs_stat ====================

pub struct FsStatTool;

impl Tool for FsStatTool {
    fn name(&self) -> &'static str { "fs_stat" }
    fn description(&self) -> &'static str { "Get file/directory metadata" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to stat"}
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
            
            let meta = match fs::metadata(&args.path).await {
                Ok(m) => m,
                Err(e) => return ToolResult::err(format!("Stat error: {e}")),
            };
            
            let symlink = fs::symlink_metadata(&args.path).await
                .map(|m| m.file_type().is_symlink())
                .unwrap_or(false);
            
            let modified = meta.modified().ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs());
            
            let stat = json!({
                "path": args.path,
                "is_file": meta.is_file(),
                "is_dir": meta.is_dir(),
                "is_symlink": symlink,
                "size": meta.len(),
                "readonly": meta.permissions().readonly(),
                "modified_epoch": modified,
            });
            
            ToolResult::ok(serde_json::to_string_pretty(&stat).unwrap())
        })
    }
}

// ==================== fs_write ====================

pub struct FsWriteTool;

impl Tool for FsWriteTool {
    fn name(&self) -> &'static str { "fs_write" }
    fn description(&self) -> &'static str { "Write content to file (creates parent dirs)" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "Content to write"}
            },
            "required": ["path", "content"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args { path: String, content: String }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            // Create parent dirs
            if let Some(parent) = Path::new(&args.path).parent() {
                let _ = fs::create_dir_all(parent).await;
            }
            
            match fs::write(&args.path, &args.content).await {
                Ok(_) => ToolResult::ok(format!("Wrote {} bytes to {}", args.content.len(), args.path)),
                Err(e) => ToolResult::err(format!("Failed: {e}")),
            }
        })
    }
}
