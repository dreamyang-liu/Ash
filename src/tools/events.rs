//! Events system - bidirectional LLM-environment interaction
//! Custom tools - simple scripts in a folder (CLI-style)

use crate::{BoxFuture, Tool, ToolResult};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tokio::fs;
use lazy_static::lazy_static;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use chrono::{DateTime, Utc};
use tokio::process::Command;

// ==================== Events System ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: String,
    pub kind: String,
    pub source: String,
    pub data: Value,
    pub timestamp: DateTime<Utc>,
}

struct EventSystem {
    queue: VecDeque<Event>,
    subscriptions: HashSet<String>,
    counter: u64,
}

impl EventSystem {
    fn new() -> Self {
        Self { queue: VecDeque::new(), subscriptions: HashSet::new(), counter: 0 }
    }

    fn subscribe(&mut self, kinds: Vec<String>) {
        for k in kinds { self.subscriptions.insert(k); }
    }

    fn unsubscribe(&mut self, kinds: Vec<String>) {
        for k in kinds { self.subscriptions.remove(&k); }
    }

    fn push(&mut self, kind: &str, source: &str, data: Value) {
        if !self.subscriptions.is_empty() && !self.subscriptions.contains(kind) { return; }
        self.counter += 1;
        self.queue.push_back(Event {
            id: format!("evt_{}", self.counter),
            kind: kind.to_string(),
            source: source.to_string(),
            data,
            timestamp: Utc::now(),
        });
        while self.queue.len() > 100 { self.queue.pop_front(); }
    }

    fn poll(&mut self, limit: usize) -> Vec<Event> {
        let mut events = Vec::new();
        for _ in 0..limit {
            if let Some(e) = self.queue.pop_front() { events.push(e); } else { break; }
        }
        events
    }

    fn peek(&self, limit: usize) -> Vec<Event> {
        self.queue.iter().take(limit).cloned().collect()
    }
}

lazy_static! {
    static ref EVENTS: Mutex<EventSystem> = Mutex::new(EventSystem::new());
}

pub async fn push_event(kind: &str, source: &str, data: Value) {
    EVENTS.lock().await.push(kind, source, data);
}

// ==================== Custom Tools (Simple Scripts) ====================
// Just scripts in a folder. Description from first comment line.
// Format: # DESC: description here

fn tools_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ash")
        .join("tools")  // simpler name
}

#[derive(Debug, Clone)]
struct ScriptTool {
    name: String,
    description: String,
    lang: String,  // "sh" or "python"
    path: PathBuf,
}

/// Extract description from script's first comment line
fn extract_description(content: &str) -> String {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("# DESC:") {
            return trimmed.strip_prefix("# DESC:").unwrap_or("").trim().to_string();
        }
        if trimmed.starts_with("#!") { continue; }  // skip shebang
        if trimmed.starts_with('#') {
            // First comment line is description
            return trimmed.strip_prefix('#').unwrap_or("").trim().to_string();
        }
        if !trimmed.is_empty() { break; }  // non-comment, non-empty line
    }
    "No description".to_string()
}

fn detect_lang_from_ext(path: &PathBuf) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("py") => "python",
        Some("sh") | Some("bash") => "sh",
        _ => "sh",
    }
}

async fn ensure_tools_dir() -> anyhow::Result<PathBuf> {
    let dir = tools_dir();
    fs::create_dir_all(&dir).await?;
    Ok(dir)
}

async fn list_scripts() -> anyhow::Result<Vec<ScriptTool>> {
    let dir = tools_dir();
    if !dir.exists() { return Ok(vec![]); }
    
    let mut tools = Vec::new();
    let mut entries = fs::read_dir(&dir).await?;
    
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if !path.is_file() { continue; }
        
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext != "sh" && ext != "py" && ext != "bash" { continue; }
        
        let name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        
        if let Ok(content) = fs::read_to_string(&path).await {
            tools.push(ScriptTool {
                name,
                description: extract_description(&content),
                lang: detect_lang_from_ext(&path).to_string(),
                path,
            });
        }
    }
    
    tools.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(tools)
}

async fn find_script(name: &str) -> anyhow::Result<ScriptTool> {
    let dir = tools_dir();
    
    // Try .sh first, then .py
    for ext in &["sh", "py"] {
        let path = dir.join(format!("{}.{}", name, ext));
        if path.exists() {
            let content = fs::read_to_string(&path).await?;
            return Ok(ScriptTool {
                name: name.to_string(),
                description: extract_description(&content),
                lang: detect_lang_from_ext(&path).to_string(),
                path,
            });
        }
    }
    
    anyhow::bail!("Tool not found: {}", name)
}

// ==================== Event Tools ====================

pub struct EventsSubscribeTool;

impl Tool for EventsSubscribeTool {
    fn name(&self) -> &'static str { "events_subscribe" }
    fn description(&self) -> &'static str { "Subscribe to event types (process_complete, file_change, error, custom)" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "events": {"type": "array", "items": {"type": "string"}},
                "unsubscribe": {"type": "boolean", "default": false}
            },
            "required": ["events"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args { events: Vec<String>, #[serde(default)] unsubscribe: bool }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let mut sys = EVENTS.lock().await;
            if args.unsubscribe {
                sys.unsubscribe(args.events.clone());
                ToolResult::ok(format!("Unsubscribed from: {:?}", args.events))
            } else {
                sys.subscribe(args.events.clone());
                ToolResult::ok(format!("Subscribed to: {:?}", args.events))
            }
        })
    }
}

pub struct EventsPollTool;

impl Tool for EventsPollTool {
    fn name(&self) -> &'static str { "events_poll" }
    fn description(&self) -> &'static str { "Poll pending events (removes from queue)" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 10},
                "peek": {"type": "boolean", "default": false}
            }
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args { #[serde(default = "default_limit")] limit: usize, #[serde(default)] peek: bool }
            fn default_limit() -> usize { 10 }
            
            let args: Args = serde_json::from_value(args).unwrap_or(Args { limit: 10, peek: false });
            
            let mut sys = EVENTS.lock().await;
            let events = if args.peek { sys.peek(args.limit) } else { sys.poll(args.limit) };
            
            if events.is_empty() {
                ToolResult::ok("No pending events".to_string())
            } else {
                ToolResult::ok(serde_json::to_string_pretty(&events).unwrap())
            }
        })
    }
}

pub struct EventsPushTool;

impl Tool for EventsPushTool {
    fn name(&self) -> &'static str { "events_push" }
    fn description(&self) -> &'static str { "Push a custom event" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "kind": {"type": "string"},
                "source": {"type": "string", "default": "llm"},
                "data": {"type": "object"}
            },
            "required": ["kind"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args { kind: String, #[serde(default = "default_src")] source: String, #[serde(default)] data: Value }
            fn default_src() -> String { "llm".to_string() }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            push_event(&args.kind, &args.source, args.data).await;
            ToolResult::ok("Event pushed".to_string())
        })
    }
}

// ==================== Custom Tool Tools (Simple CLI Style) ====================

pub struct ToolRegisterTool;

impl Tool for ToolRegisterTool {
    fn name(&self) -> &'static str { "tool_create" }  // renamed: create script
    fn description(&self) -> &'static str { "Create a custom tool script (.sh or .py)" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Tool name (becomes <name>.sh or <name>.py)"},
                "script": {"type": "string", "description": "Script content (first # comment = description)"},
                "lang": {"type": "string", "enum": ["sh", "python"], "default": "sh"}
            },
            "required": ["name", "script"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args { 
                name: String, 
                script: String, 
                #[serde(default = "default_lang")] lang: String,
            }
            fn default_lang() -> String { "sh".to_string() }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let dir = match ensure_tools_dir().await {
                Ok(d) => d,
                Err(e) => return ToolResult::err(format!("Failed to create dir: {e}")),
            };
            
            let ext = if args.lang == "python" || args.lang == "py" { "py" } else { "sh" };
            let path = dir.join(format!("{}.{}", args.name, ext));
            
            if let Err(e) = fs::write(&path, &args.script).await {
                return ToolResult::err(format!("Write failed: {e}"));
            }
            
            // Make executable
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if let Ok(meta) = fs::metadata(&path).await {
                    let mut perms = meta.permissions();
                    perms.set_mode(0o755);
                    let _ = fs::set_permissions(&path, perms).await;
                }
            }
            
            let desc = extract_description(&args.script);
            ToolResult::ok(format!("Created: {} [{}]\n{}\nPath: {}", args.name, ext, desc, path.display()))
        })
    }
}

pub struct ToolListCustomTool;

impl Tool for ToolListCustomTool {
    fn name(&self) -> &'static str { "tool_list" }  // renamed
    fn description(&self) -> &'static str { "List custom tool scripts" }
    
    fn schema(&self) -> Value { json!({"type": "object", "properties": {}}) }
    
    fn execute(&self, _args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            match list_scripts().await {
                Ok(tools) if tools.is_empty() => {
                    ToolResult::ok(format!("No tools. Dir: {}", tools_dir().display()))
                }
                Ok(tools) => {
                    let mut out = format!("Tools ({}):\n", tools_dir().display());
                    for t in tools {
                        out.push_str(&format!("\n  {} [{}] - {}", t.name, t.lang, t.description));
                    }
                    ToolResult::ok(out)
                }
                Err(e) => ToolResult::err(format!("Failed: {e}")),
            }
        })
    }
}

pub struct ToolCallCustomTool;

impl Tool for ToolCallCustomTool {
    fn name(&self) -> &'static str { "tool_run" }  // renamed
    fn description(&self) -> &'static str { "Run a custom tool script" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Tool name"},
                "args": {"type": "array", "items": {"type": "string"}, "description": "Positional arguments"},
                "env": {"type": "object", "description": "Environment variables"}
            },
            "required": ["name"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args { 
                name: String, 
                #[serde(default)] args: Vec<String>,
                #[serde(default)] env: HashMap<String, String>,
            }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let tool = match find_script(&args.name).await {
                Ok(t) => t,
                Err(e) => return ToolResult::err(e.to_string()),
            };
            
            // Build command
            let mut cmd = if tool.lang == "python" {
                let mut c = Command::new("python3");
                c.arg(&tool.path);
                c
            } else {
                let mut c = Command::new("sh");
                c.arg(&tool.path);
                c
            };
            
            // Add positional args
            for arg in &args.args {
                cmd.arg(arg);
            }
            
            // Add env vars
            for (k, v) in &args.env {
                cmd.env(k, v);
            }
            
            // Execute
            match cmd.output().await {
                Ok(o) => {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    
                    push_event("tool_complete", &args.name, json!({
                        "tool": args.name,
                        "success": o.status.success(),
                        "exit_code": o.status.code()
                    })).await;
                    
                    if o.status.success() {
                        ToolResult::ok(stdout.to_string())
                    } else {
                        let mut out = stdout.to_string();
                        if !stderr.is_empty() {
                            out.push_str("\n[stderr]\n");
                            out.push_str(&stderr);
                        }
                        ToolResult::err(out)
                    }
                }
                Err(e) => ToolResult::err(format!("Exec failed: {e}")),
            }
        })
    }
}

pub struct ToolRemoveCustomTool;

impl Tool for ToolRemoveCustomTool {
    fn name(&self) -> &'static str { "tool_remove" }  // renamed
    fn description(&self) -> &'static str { "Remove a custom tool script" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "name": {"type": "string"} },
            "required": ["name"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args { name: String }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let tool = match find_script(&args.name).await {
                Ok(t) => t,
                Err(e) => return ToolResult::err(e.to_string()),
            };
            
            if let Err(e) = fs::remove_file(&tool.path).await {
                return ToolResult::err(format!("Remove failed: {e}"));
            }
            
            ToolResult::ok(format!("Removed: {}", args.name))
        })
    }
}

pub struct ToolViewCustomTool;

impl Tool for ToolViewCustomTool {
    fn name(&self) -> &'static str { "tool_view" }  // renamed
    fn description(&self) -> &'static str { "View a custom tool's script" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "name": {"type": "string"} },
            "required": ["name"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args { name: String }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let tool = match find_script(&args.name).await {
                Ok(t) => t,
                Err(e) => return ToolResult::err(e.to_string()),
            };
            
            match fs::read_to_string(&tool.path).await {
                Ok(content) => {
                    let header = format!("# {} [{}]\n# {}\n\n", tool.name, tool.lang, tool.path.display());
                    ToolResult::ok(header + &content)
                }
                Err(e) => ToolResult::err(format!("Read failed: {e}")),
            }
        })
    }
}
