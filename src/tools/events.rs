//! Events system - bidirectional LLM-environment interaction
//! Custom tools - LLM can register and call custom tools (file-based, multi-language)

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

// ==================== Custom Tools (File-Based, Multi-Language) ====================

fn tools_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ash")
        .join("custom_tools")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomToolMeta {
    pub name: String,
    pub description: String,
    pub schema: Value,
    pub lang: String,  // "sh", "python", "node", etc.
    pub created_at: DateTime<Utc>,
}

/// Detect language from script content or explicit lang parameter
fn detect_lang(script: &str, explicit_lang: Option<&str>) -> String {
    if let Some(lang) = explicit_lang {
        return lang.to_string();
    }
    
    // Check shebang
    if let Some(first_line) = script.lines().next() {
        if first_line.starts_with("#!") {
            if first_line.contains("python") { return "python".to_string(); }
            if first_line.contains("node") { return "node".to_string(); }
            if first_line.contains("ruby") { return "ruby".to_string(); }
            if first_line.contains("perl") { return "perl".to_string(); }
        }
    }
    
    // Check content patterns
    if script.contains("def ") && script.contains(":") { return "python".to_string(); }
    if script.contains("import ") && script.contains("from ") { return "python".to_string(); }
    if script.contains("const ") || script.contains("function ") { return "node".to_string(); }
    
    "sh".to_string()
}

fn get_extension(lang: &str) -> &'static str {
    match lang {
        "python" | "py" => "py",
        "node" | "js" | "javascript" => "js",
        "ruby" | "rb" => "rb",
        "perl" | "pl" => "pl",
        _ => "sh",
    }
}

fn get_interpreter(lang: &str) -> (&'static str, Vec<&'static str>) {
    match lang {
        "python" | "py" => ("python3", vec![]),
        "node" | "js" | "javascript" => ("node", vec![]),
        "ruby" | "rb" => ("ruby", vec![]),
        "perl" | "pl" => ("perl", vec![]),
        _ => ("sh", vec!["-c"]),
    }
}

async fn ensure_tools_dir() -> anyhow::Result<PathBuf> {
    let dir = tools_dir();
    fs::create_dir_all(&dir).await?;
    Ok(dir)
}

async fn save_custom_tool(name: &str, description: &str, script: &str, schema: &Value, lang: Option<&str>) -> anyhow::Result<String> {
    let dir = ensure_tools_dir().await?;
    
    let detected_lang = detect_lang(script, lang);
    let ext = get_extension(&detected_lang);
    
    // Save script file
    let script_path = dir.join(format!("{}.{}", name, ext));
    fs::write(&script_path, script).await?;
    
    // Make executable on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&script_path).await?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&script_path, perms).await?;
    }
    
    // Save metadata
    let meta = CustomToolMeta {
        name: name.to_string(),
        description: description.to_string(),
        schema: schema.clone(),
        lang: detected_lang.clone(),
        created_at: Utc::now(),
    };
    let meta_path = dir.join(format!("{}.json", name));
    fs::write(&meta_path, serde_json::to_string_pretty(&meta)?).await?;
    
    Ok(detected_lang)
}

async fn load_custom_tool(name: &str) -> anyhow::Result<(CustomToolMeta, String, PathBuf)> {
    let dir = tools_dir();
    
    // Load metadata
    let meta_path = dir.join(format!("{}.json", name));
    let meta_content = fs::read_to_string(&meta_path).await?;
    let meta: CustomToolMeta = serde_json::from_str(&meta_content)?;
    
    // Find script file
    let ext = get_extension(&meta.lang);
    let script_path = dir.join(format!("{}.{}", name, ext));
    let script = fs::read_to_string(&script_path).await?;
    
    Ok((meta, script, script_path))
}

async fn list_custom_tools() -> anyhow::Result<Vec<CustomToolMeta>> {
    let dir = tools_dir();
    if !dir.exists() { return Ok(vec![]); }
    
    let mut tools = Vec::new();
    let mut entries = fs::read_dir(&dir).await?;
    
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().map(|e| e == "json").unwrap_or(false) {
            if let Ok(content) = fs::read_to_string(&path).await {
                if let Ok(meta) = serde_json::from_str::<CustomToolMeta>(&content) {
                    tools.push(meta);
                }
            }
        }
    }
    
    Ok(tools)
}

async fn remove_custom_tool(name: &str) -> anyhow::Result<bool> {
    let dir = tools_dir();
    
    // Load meta to get extension
    let meta_path = dir.join(format!("{}.json", name));
    let mut removed = false;
    
    if let Ok(content) = fs::read_to_string(&meta_path).await {
        if let Ok(meta) = serde_json::from_str::<CustomToolMeta>(&content) {
            let ext = get_extension(&meta.lang);
            let script_path = dir.join(format!("{}.{}", name, ext));
            if fs::metadata(&script_path).await.is_ok() {
                fs::remove_file(&script_path).await?;
                removed = true;
            }
        }
        fs::remove_file(&meta_path).await?;
        removed = true;
    }
    
    Ok(removed)
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
                "events": {"type": "array", "items": {"type": "string"}, "description": "Event types to subscribe to"},
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

// ==================== Custom Tool Tools ====================

pub struct ToolRegisterTool;

impl Tool for ToolRegisterTool {
    fn name(&self) -> &'static str { "tool_register" }
    fn description(&self) -> &'static str { "Register a custom tool (sh/python/node/ruby - auto-detected or specify lang)" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Tool name"},
                "description": {"type": "string", "description": "What the tool does"},
                "script": {"type": "string", "description": "Script content (use $ARG_xxx or os.environ['ARG_xxx'] for arguments)"},
                "lang": {"type": "string", "description": "Language: sh, python, node, ruby (auto-detected if omitted)"},
                "schema": {"type": "object", "description": "JSON schema for arguments", "default": {}}
            },
            "required": ["name", "description", "script"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args { 
                name: String, 
                description: String, 
                script: String, 
                lang: Option<String>,
                #[serde(default)] schema: Value 
            }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            match save_custom_tool(&args.name, &args.description, &args.script, &args.schema, args.lang.as_deref()).await {
                Ok(lang) => {
                    let ext = get_extension(&lang);
                    let path = tools_dir().join(format!("{}.{}", args.name, ext));
                    ToolResult::ok(format!("Registered: {} [{}] ({})", args.name, lang, path.display()))
                }
                Err(e) => ToolResult::err(format!("Failed: {e}")),
            }
        })
    }
}

pub struct ToolListCustomTool;

impl Tool for ToolListCustomTool {
    fn name(&self) -> &'static str { "tool_list_custom" }
    fn description(&self) -> &'static str { "List all registered custom tools" }
    
    fn schema(&self) -> Value { json!({"type": "object", "properties": {}}) }
    
    fn execute(&self, _args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            match list_custom_tools().await {
                Ok(tools) if tools.is_empty() => {
                    ToolResult::ok(format!("No custom tools. Dir: {}", tools_dir().display()))
                }
                Ok(tools) => {
                    let mut out = format!("Custom tools ({}):\n", tools_dir().display());
                    for tool in tools {
                        let ext = get_extension(&tool.lang);
                        out.push_str(&format!("\n## {} [{}]\n", tool.name, tool.lang));
                        out.push_str(&format!("Description: {}\n", tool.description));
                        out.push_str(&format!("Script: {}.{}\n", tool.name, ext));
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
    fn name(&self) -> &'static str { "tool_call_custom" }
    fn description(&self) -> &'static str { "Call a registered custom tool" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "arguments": {"type": "object", "description": "Arguments (available as ARG_xxx env vars)"}
            },
            "required": ["name"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args { name: String, #[serde(default)] arguments: HashMap<String, Value> }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            // Load tool
            let (meta, _script, script_path) = match load_custom_tool(&args.name).await {
                Ok(t) => t,
                Err(e) => return ToolResult::err(format!("Tool '{}' not found: {e}", args.name)),
            };
            
            // Build command based on language
            let (interpreter, interp_args) = get_interpreter(&meta.lang);
            let mut cmd = Command::new(interpreter);
            
            for arg in interp_args {
                cmd.arg(arg);
            }
            
            // For sh -c, pass script content; for others, pass script path
            if meta.lang == "sh" {
                let script_content = fs::read_to_string(&script_path).await.unwrap_or_default();
                cmd.arg(&script_content);
            } else {
                cmd.arg(&script_path);
            }
            
            // Set ARG_xxx environment variables
            for (key, value) in &args.arguments {
                let value_str = match value {
                    Value::String(s) => s.clone(),
                    v => v.to_string(),
                };
                cmd.env(format!("ARG_{}", key), &value_str);
            }
            
            // Execute
            let output = cmd.output().await;
            
            match output {
                Ok(o) => {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    
                    push_event("custom_tool_complete", &args.name, json!({
                        "tool": args.name,
                        "lang": meta.lang,
                        "success": o.status.success(),
                        "exit_code": o.status.code()
                    })).await;
                    
                    if o.status.success() {
                        ToolResult::ok(stdout.to_string())
                    } else {
                        ToolResult::err(format!("{}\n{}", stdout, stderr))
                    }
                }
                Err(e) => ToolResult::err(format!("Failed to execute: {e}")),
            }
        })
    }
}

pub struct ToolRemoveCustomTool;

impl Tool for ToolRemoveCustomTool {
    fn name(&self) -> &'static str { "tool_remove_custom" }
    fn description(&self) -> &'static str { "Remove a registered custom tool" }
    
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
            
            match remove_custom_tool(&args.name).await {
                Ok(true) => ToolResult::ok(format!("Removed: {}", args.name)),
                Ok(false) => ToolResult::err(format!("Tool '{}' not found", args.name)),
                Err(e) => ToolResult::err(format!("Failed: {e}")),
            }
        })
    }
}

pub struct ToolViewCustomTool;

impl Tool for ToolViewCustomTool {
    fn name(&self) -> &'static str { "tool_view_custom" }
    fn description(&self) -> &'static str { "View a custom tool's script content" }
    
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
            
            match load_custom_tool(&args.name).await {
                Ok((meta, script, path)) => {
                    let out = format!(
                        "# {} [{}]\n# {}\n# Path: {}\n# Schema: {}\n\n{}",
                        meta.name, meta.lang, meta.description,
                        path.display(),
                        serde_json::to_string(&meta.schema).unwrap_or_default(),
                        script
                    );
                    ToolResult::ok(out)
                }
                Err(e) => ToolResult::err(format!("Tool '{}' not found: {e}", args.name)),
            }
        })
    }
}
