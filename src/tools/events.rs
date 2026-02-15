//! Events system - bidirectional LLM-environment interaction
//! Custom tools - LLM can register and call custom tools

use crate::{BoxFuture, Tool, ToolResult};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::Mutex;
use lazy_static::lazy_static;
use std::collections::{HashMap, HashSet, VecDeque};
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
    /// Pending events queue
    queue: VecDeque<Event>,
    /// Subscribed event kinds
    subscriptions: HashSet<String>,
    /// Event counter for IDs
    counter: u64,
}

impl EventSystem {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            subscriptions: HashSet::new(),
            counter: 0,
        }
    }

    fn subscribe(&mut self, kinds: Vec<String>) {
        for k in kinds {
            self.subscriptions.insert(k);
        }
    }

    fn unsubscribe(&mut self, kinds: Vec<String>) {
        for k in kinds {
            self.subscriptions.remove(&k);
        }
    }

    fn push(&mut self, kind: &str, source: &str, data: Value) {
        // Only push if subscribed (or if subscriptions is empty = subscribe all)
        if !self.subscriptions.is_empty() && !self.subscriptions.contains(kind) {
            return;
        }
        
        self.counter += 1;
        self.queue.push_back(Event {
            id: format!("evt_{}", self.counter),
            kind: kind.to_string(),
            source: source.to_string(),
            data,
            timestamp: Utc::now(),
        });
        
        // Keep bounded
        while self.queue.len() > 100 {
            self.queue.pop_front();
        }
    }

    fn poll(&mut self, limit: usize) -> Vec<Event> {
        let mut events = Vec::new();
        for _ in 0..limit {
            if let Some(e) = self.queue.pop_front() {
                events.push(e);
            } else {
                break;
            }
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

/// Push an event (called by other tools)
pub async fn push_event(kind: &str, source: &str, data: Value) {
    EVENTS.lock().await.push(kind, source, data);
}

// ==================== Custom Tools System ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomTool {
    pub name: String,
    pub description: String,
    pub script: String,
    pub schema: Value,
    pub created_at: DateTime<Utc>,
}

struct CustomToolRegistry {
    tools: HashMap<String, CustomTool>,
}

impl CustomToolRegistry {
    fn new() -> Self {
        Self { tools: HashMap::new() }
    }

    fn register(&mut self, tool: CustomTool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    fn remove(&mut self, name: &str) -> bool {
        self.tools.remove(name).is_some()
    }

    fn get(&self, name: &str) -> Option<&CustomTool> {
        self.tools.get(name)
    }

    fn list(&self) -> Vec<&CustomTool> {
        self.tools.values().collect()
    }
}

lazy_static! {
    static ref CUSTOM_TOOLS: Mutex<CustomToolRegistry> = Mutex::new(CustomToolRegistry::new());
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
                "events": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Event types to subscribe to"
                },
                "unsubscribe": {
                    "type": "boolean",
                    "default": false,
                    "description": "Unsubscribe instead of subscribe"
                }
            },
            "required": ["events"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                events: Vec<String>,
                #[serde(default)]
                unsubscribe: bool,
            }
            
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
                "limit": {"type": "integer", "default": 10, "description": "Max events to return"},
                "peek": {"type": "boolean", "default": false, "description": "Peek without removing"}
            }
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                #[serde(default = "default_limit")]
                limit: usize,
                #[serde(default)]
                peek: bool,
            }
            fn default_limit() -> usize { 10 }
            
            let args: Args = serde_json::from_value(args).unwrap_or(Args { limit: 10, peek: false });
            
            let mut sys = EVENTS.lock().await;
            let events = if args.peek {
                sys.peek(args.limit)
            } else {
                sys.poll(args.limit)
            };
            
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
                "kind": {"type": "string", "description": "Event type"},
                "source": {"type": "string", "description": "Event source"},
                "data": {"type": "object", "description": "Event data"}
            },
            "required": ["kind"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                kind: String,
                #[serde(default = "default_source")]
                source: String,
                #[serde(default)]
                data: Value,
            }
            fn default_source() -> String { "llm".to_string() }
            
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
    fn description(&self) -> &'static str { "Register a custom tool with script and manual" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Tool name (unique identifier)"},
                "description": {"type": "string", "description": "What the tool does"},
                "script": {"type": "string", "description": "Shell script to execute (use $ARG_name for arguments)"},
                "schema": {
                    "type": "object",
                    "description": "JSON schema for arguments",
                    "default": {}
                }
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
                #[serde(default)]
                schema: Value,
            }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let tool = CustomTool {
                name: args.name.clone(),
                description: args.description,
                script: args.script,
                schema: args.schema,
                created_at: Utc::now(),
            };
            
            let mut registry = CUSTOM_TOOLS.lock().await;
            registry.register(tool);
            
            ToolResult::ok(format!("Registered custom tool: {}", args.name))
        })
    }
}

pub struct ToolListCustomTool;

impl Tool for ToolListCustomTool {
    fn name(&self) -> &'static str { "tool_list_custom" }
    fn description(&self) -> &'static str { "List all registered custom tools" }
    
    fn schema(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    
    fn execute(&self, _args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let registry = CUSTOM_TOOLS.lock().await;
            let tools = registry.list();
            
            if tools.is_empty() {
                return ToolResult::ok("No custom tools registered".to_string());
            }
            
            let mut out = String::from("Custom tools:\n");
            for tool in tools {
                out.push_str(&format!("\n## {}\n", tool.name));
                out.push_str(&format!("Description: {}\n", tool.description));
                out.push_str(&format!("Script: {}\n", tool.script));
                if !tool.schema.is_null() {
                    out.push_str(&format!("Schema: {}\n", serde_json::to_string(&tool.schema).unwrap()));
                }
            }
            
            ToolResult::ok(out)
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
                "name": {"type": "string", "description": "Custom tool name"},
                "arguments": {"type": "object", "description": "Arguments to pass (available as $ARG_key in script)"}
            },
            "required": ["name"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                name: String,
                #[serde(default)]
                arguments: HashMap<String, Value>,
            }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let registry = CUSTOM_TOOLS.lock().await;
            let tool = match registry.get(&args.name) {
                Some(t) => t.clone(),
                None => return ToolResult::err(format!("Custom tool '{}' not found", args.name)),
            };
            drop(registry);
            
            // Build script with argument substitution
            let mut script = tool.script.clone();
            for (key, value) in &args.arguments {
                let value_str = match value {
                    Value::String(s) => s.clone(),
                    v => v.to_string(),
                };
                script = script.replace(&format!("$ARG_{}", key), &value_str);
            }
            
            // Execute script
            let output = Command::new("sh")
                .arg("-c")
                .arg(&script)
                .output()
                .await;
            
            match output {
                Ok(o) => {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    
                    // Push event for completion
                    push_event("custom_tool_complete", &args.name, json!({
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
            "properties": {
                "name": {"type": "string", "description": "Tool name to remove"}
            },
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
            
            let mut registry = CUSTOM_TOOLS.lock().await;
            if registry.remove(&args.name) {
                ToolResult::ok(format!("Removed custom tool: {}", args.name))
            } else {
                ToolResult::err(format!("Custom tool '{}' not found", args.name))
            }
        })
    }
}
