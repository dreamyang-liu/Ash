//! Session management via remote Control Plane + MCP Gateway
//!
//! Architecture:
//! - Control Plane: spawn/destroy sandbox pods (POST /spawn, DELETE /deprovision/:uuid)
//! - MCP Gateway: route tool calls to sandbox by X-Session-ID header

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::{BoxFuture, Tool, ToolResult};

/// Session representing a remote sandbox
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub uuid: String,
    pub name: String,
    pub namespace: String,
    pub status: String,
    pub host: String,
    pub ports: Vec<i32>,
    pub message: String,
}

/// Client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    pub control_plane_url: String,
    pub gateway_url: String,
    pub image: String,
    pub timeout: u64,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            control_plane_url: std::env::var("ASH_CONTROL_PLANE_URL")
                .unwrap_or_else(|_| "http://localhost:8080".to_string()),
            gateway_url: std::env::var("ASH_GATEWAY_URL")
                .unwrap_or_else(|_| "http://localhost:8081".to_string()),
            image: "timemagic/rl-mcp:general-1.7".to_string(),
            timeout: 300,
        }
    }
}

lazy_static::lazy_static! {
    static ref CONFIG: Arc<RwLock<ClientConfig>> = Arc::new(RwLock::new(ClientConfig::default()));
    static ref SESSIONS: Arc<RwLock<HashMap<String, Session>>> = Arc::new(RwLock::new(HashMap::new()));
}

/// Set client configuration
pub async fn set_config(config: ClientConfig) {
    let mut cfg = CONFIG.write().await;
    *cfg = config;
}

/// Get current configuration
pub async fn get_config() -> ClientConfig {
    CONFIG.read().await.clone()
}

// ========== Spawn ==========

#[derive(Debug, Clone, Deserialize, Default)]
pub struct SessionCreateArgs {
    /// Custom name (auto-generated if not provided)
    #[serde(default)]
    pub name: Option<String>,
    /// Docker image override
    #[serde(default)]
    pub image: Option<String>,
    /// Environment variables
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Container ports
    #[serde(default)]
    pub ports: Vec<i32>,
    /// Resource requests
    #[serde(default)]
    pub resources: Option<ResourceSpec>,
    /// Node selector
    #[serde(default)]
    pub node_selector: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ResourceSpec {
    #[serde(default)]
    pub requests: ResourceValues,
    #[serde(default)]
    pub limits: ResourceValues,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ResourceValues {
    #[serde(default)]
    pub cpu: Option<String>,
    #[serde(default)]
    pub memory: Option<String>,
}

pub struct SessionCreateTool;

impl Tool for SessionCreateTool {
    fn name(&self) -> &'static str { "session_create" }
    fn description(&self) -> &'static str { "Spawn a new sandbox, returns session_id (uuid)" }
    
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Custom name"},
                "image": {"type": "string", "description": "Docker image"},
                "env": {"type": "object", "description": "Environment variables"},
                "ports": {"type": "array", "items": {"type": "integer"}},
                "resources": {
                    "type": "object",
                    "properties": {
                        "requests": {"type": "object", "properties": {"cpu": {"type": "string"}, "memory": {"type": "string"}}},
                        "limits": {"type": "object", "properties": {"cpu": {"type": "string"}, "memory": {"type": "string"}}}
                    }
                },
                "node_selector": {"type": "object"}
            }
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: SessionCreateArgs = serde_json::from_value(args).unwrap_or_default();
            let config = get_config().await;
            
            // Build spawn request
            let image = args.image.unwrap_or(config.image);
            let ports: Vec<Value> = if args.ports.is_empty() {
                vec![serde_json::json!({"container_port": 3000})]
            } else {
                args.ports.iter().map(|p| serde_json::json!({"container_port": p})).collect()
            };
            
            let mut body = serde_json::json!({
                "image": image,
                "ports": ports,
            });
            
            if let Some(name) = args.name {
                body["name"] = serde_json::json!(name);
            }
            if !args.env.is_empty() {
                body["env"] = serde_json::json!(args.env);
            }
            if !args.node_selector.is_empty() {
                body["node_selector"] = serde_json::json!(args.node_selector);
            }
            if let Some(resources) = args.resources {
                body["resources"] = serde_json::json!(resources);
            }
            
            // Call control plane
            let client = reqwest::Client::new();
            let url = format!("{}/spawn", config.control_plane_url);
            
            let response = match client
                .post(&url)
                .json(&body)
                .timeout(std::time::Duration::from_secs(config.timeout))
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => return ToolResult::err(format!("Failed to connect to control plane: {e}")),
            };
            
            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                return ToolResult::err(format!("Spawn failed ({}): {}", status, text));
            }
            
            let data: Value = match response.json().await {
                Ok(d) => d,
                Err(e) => return ToolResult::err(format!("Invalid response: {e}")),
            };
            
            let session = Session {
                uuid: data["uuid"].as_str().unwrap_or_default().to_string(),
                name: data["name"].as_str().unwrap_or_default().to_string(),
                namespace: data["namespace"].as_str().unwrap_or_default().to_string(),
                status: data["status"].as_str().unwrap_or_default().to_string(),
                host: data["host"].as_str().unwrap_or_default().to_string(),
                ports: data["ports"].as_array()
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64().map(|n| n as i32)).collect())
                    .unwrap_or_default(),
                message: data["message"].as_str().unwrap_or_default().to_string(),
            };
            
            let uuid = session.uuid.clone();
            
            // Store session
            let mut sessions = SESSIONS.write().await;
            sessions.insert(uuid.clone(), session);
            
            ToolResult::ok(serde_json::json!({
                "session_id": uuid,
                "status": data["status"],
                "host": data["host"],
            }).to_string())
        })
    }
}

// ========== Destroy ==========

#[derive(Debug, Clone, Deserialize)]
pub struct SessionDestroyArgs {
    pub session_id: String,
}

pub struct SessionDestroyTool;

impl Tool for SessionDestroyTool {
    fn name(&self) -> &'static str { "session_destroy" }
    fn description(&self) -> &'static str { "Destroy a sandbox by session_id" }
    
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "session_id": {"type": "string"}
            },
            "required": ["session_id"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: SessionDestroyArgs = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let config = get_config().await;
            let client = reqwest::Client::new();
            let url = format!("{}/deprovision/{}", config.control_plane_url, args.session_id);
            
            let response = match client
                .delete(&url)
                .timeout(std::time::Duration::from_secs(30))
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => return ToolResult::err(format!("Failed to connect: {e}")),
            };
            
            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                return ToolResult::err(format!("Destroy failed ({}): {}", status, text));
            }
            
            // Remove from local cache
            let mut sessions = SESSIONS.write().await;
            sessions.remove(&args.session_id);
            
            ToolResult::ok(format!("Destroyed: {}", args.session_id))
        })
    }
}

// ========== List ==========

pub struct SessionListTool;

impl Tool for SessionListTool {
    fn name(&self) -> &'static str { "session_list" }
    fn description(&self) -> &'static str { "List active sessions" }
    fn schema(&self) -> Value { serde_json::json!({"type": "object"}) }
    
    fn execute(&self, _args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let sessions = SESSIONS.read().await;
            let list: Vec<Value> = sessions.values().map(|s| {
                serde_json::json!({
                    "session_id": s.uuid,
                    "name": s.name,
                    "status": s.status,
                    "host": s.host,
                })
            }).collect();
            ToolResult::ok(serde_json::to_string_pretty(&list).unwrap())
        })
    }
}

// ========== MCP Call Helper ==========

/// Call a tool in a specific session via MCP Gateway
pub async fn call_tool_in_session(
    session_id: &str,
    tool_name: &str,
    args: Value,
) -> Result<Value, String> {
    let config = get_config().await;
    let client = reqwest::Client::new();
    
    let url = format!("{}/mcp", config.gateway_url);
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": chrono::Utc::now().timestamp_millis(),
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": args
        }
    });
    
    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .header("X-Session-ID", session_id)
        .json(&request)
        .timeout(std::time::Duration::from_secs(300))
        .send()
        .await
        .map_err(|e| format!("MCP request failed: {e}"))?;
    
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!("MCP call failed ({}): {}", status, text));
    }
    
    let data: Value = response.json().await
        .map_err(|e| format!("Invalid MCP response: {e}"))?;
    
    if let Some(error) = data.get("error") {
        return Err(format!("MCP error: {}", error));
    }
    
    Ok(data.get("result").cloned().unwrap_or(Value::Null))
}

// ========== Get session ==========

pub async fn get_session(session_id: &str) -> Option<Session> {
    let sessions = SESSIONS.read().await;
    sessions.get(session_id).cloned()
}
