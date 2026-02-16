//! Filesystem tools - ls

use crate::{BoxFuture, Tool, ToolResult};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::fs;

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
