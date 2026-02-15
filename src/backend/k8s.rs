//! K8s backend - remote sandbox management via Control Plane + MCP Gateway
//!
//! Architecture:
//! - Control Plane: spawn/destroy sandbox pods (POST /spawn, DELETE /deprovision/:uuid)
//! - MCP Gateway: route tool calls to sandbox by X-Session-ID header

use async_trait::async_trait;
use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use super::{
    Backend, BackendError, BackendType, CreateOptions, ExecOptions, ExecResult,
    PortMapping, Session, SessionStatus,
};

/// K8s backend configuration
#[derive(Debug, Clone)]
pub struct K8sConfig {
    /// Control plane URL (e.g., http://control-plane:8080)
    pub control_plane_url: String,
    /// MCP Gateway URL (e.g., http://gateway:8081)
    pub gateway_url: String,
    /// Default image for pods
    pub default_image: String,
    /// Default timeout for operations (seconds)
    pub timeout_secs: u64,
}

impl Default for K8sConfig {
    fn default() -> Self {
        Self {
            control_plane_url: std::env::var("ASH_CONTROL_PLANE_URL")
                .unwrap_or_else(|_| "http://localhost:8080".to_string()),
            gateway_url: std::env::var("ASH_GATEWAY_URL")
                .unwrap_or_else(|_| "http://localhost:8081".to_string()),
            default_image: "timemagic/rl-mcp:general-1.7".to_string(),
            timeout_secs: 300,
        }
    }
}

/// K8s backend implementation
pub struct K8sBackend {
    client: Client,
    config: K8sConfig,
    /// Cache of sessions we've created
    sessions: Arc<RwLock<HashMap<String, Session>>>,
}

impl K8sBackend {
    /// Create new K8s backend with default config
    pub fn new() -> Self {
        Self::with_config(K8sConfig::default())
    }
    
    /// Create with custom config
    pub fn with_config(config: K8sConfig) -> Self {
        Self {
            client: Client::new(),
            config,
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Update configuration
    pub fn set_config(&mut self, config: K8sConfig) {
        self.config = config;
    }
    
    /// Get current configuration
    pub fn config(&self) -> &K8sConfig {
        &self.config
    }
    
    /// Call a tool via MCP Gateway
    async fn mcp_call(&self, session_id: &str, tool_name: &str, args: Value) -> Result<Value, BackendError> {
        let url = format!("{}/mcp", self.config.gateway_url);
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": chrono::Utc::now().timestamp_millis(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": args
            }
        });
        
        let response = self.client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("X-Session-ID", session_id)
            .json(&request)
            .timeout(Duration::from_secs(self.config.timeout_secs))
            .send()
            .await
            .map_err(|e| BackendError::Connection(format!("MCP request failed: {e}")))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(BackendError::ExecFailed(format!("MCP call failed ({}): {}", status, text)));
        }
        
        let data: Value = response.json().await
            .map_err(|e| BackendError::ExecFailed(format!("Invalid MCP response: {e}")))?;
        
        if let Some(error) = data.get("error") {
            return Err(BackendError::ExecFailed(format!("MCP error: {}", error)));
        }
        
        Ok(data.get("result").cloned().unwrap_or(Value::Null))
    }
}

#[async_trait]
impl Backend for K8sBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::K8s
    }
    
    async fn create(&self, options: CreateOptions) -> Result<Session, BackendError> {
        let image = options.image.as_deref().unwrap_or(&self.config.default_image);
        
        // Build port specs
        let ports: Vec<Value> = if options.ports.is_empty() {
            vec![serde_json::json!({"container_port": 3000})]
        } else {
            options.ports.iter().map(|p| serde_json::json!({"container_port": p})).collect()
        };
        
        // Build request body
        let mut body = serde_json::json!({
            "image": image,
            "ports": ports,
        });
        
        if let Some(ref name) = options.name {
            body["name"] = serde_json::json!(name);
        }
        if !options.env.is_empty() {
            body["env"] = serde_json::json!(options.env);
        }
        if !options.labels.is_empty() {
            body["node_selector"] = serde_json::json!(options.labels);
        }
        if let Some(ref resources) = options.resources {
            let mut res = serde_json::json!({});
            let mut requests = serde_json::json!({});
            let mut limits = serde_json::json!({});
            
            if let Some(ref cpu) = resources.cpu_request {
                requests["cpu"] = serde_json::json!(cpu);
            }
            if let Some(ref mem) = resources.memory_request {
                requests["memory"] = serde_json::json!(mem);
            }
            if let Some(ref cpu) = resources.cpu_limit {
                limits["cpu"] = serde_json::json!(cpu);
            }
            if let Some(ref mem) = resources.memory_limit {
                limits["memory"] = serde_json::json!(mem);
            }
            
            res["requests"] = requests;
            res["limits"] = limits;
            body["resources"] = res;
        }
        
        // Call control plane
        let url = format!("{}/spawn", self.config.control_plane_url);
        let response = self.client
            .post(&url)
            .json(&body)
            .timeout(Duration::from_secs(self.config.timeout_secs))
            .send()
            .await
            .map_err(|e| BackendError::Connection(format!("Control plane: {e}")))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(BackendError::CreateFailed(format!("Spawn failed ({}): {}", status, text)));
        }
        
        let data: Value = response.json().await
            .map_err(|e| BackendError::CreateFailed(format!("Invalid response: {e}")))?;
        
        let session = Session {
            id: data["uuid"].as_str().unwrap_or_default().to_string(),
            name: data["name"].as_str().unwrap_or_default().to_string(),
            backend: BackendType::K8s,
            status: match data["status"].as_str() {
                Some("Ready") | Some("running") => SessionStatus::Running,
                Some("Pending") | Some("creating") => SessionStatus::Creating,
                _ => SessionStatus::Unknown,
            },
            host: data["host"].as_str().unwrap_or_default().to_string(),
            ports: data["ports"].as_array()
                .map(|arr| arr.iter().filter_map(|v| {
                    Some(PortMapping {
                        container_port: v.as_i64()? as u16,
                        host_port: None,
                        protocol: "tcp".to_string(),
                    })
                }).collect())
                .unwrap_or_default(),
            image: image.to_string(),
            created_at: chrono::Utc::now(),
        };
        
        // Cache session
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(session.id.clone(), session.clone());
        }
        
        Ok(session)
    }
    
    async fn destroy(&self, session_id: &str) -> Result<(), BackendError> {
        let url = format!("{}/deprovision/{}", self.config.control_plane_url, session_id);
        
        let response = self.client
            .delete(&url)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| BackendError::Connection(format!("Control plane: {e}")))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(BackendError::Other(format!("Destroy failed ({}): {}", status, text)));
        }
        
        // Remove from cache
        {
            let mut sessions = self.sessions.write().await;
            sessions.remove(session_id);
        }
        
        Ok(())
    }
    
    async fn list(&self) -> Result<Vec<Session>, BackendError> {
        // Return cached sessions (K8s backend doesn't have a list API currently)
        let sessions = self.sessions.read().await;
        Ok(sessions.values().cloned().collect())
    }
    
    async fn get(&self, session_id: &str) -> Result<Option<Session>, BackendError> {
        let sessions = self.sessions.read().await;
        Ok(sessions.get(session_id).cloned())
    }
    
    async fn exec(&self, session_id: &str, command: &str, options: ExecOptions) -> Result<ExecResult, BackendError> {
        let mut args = serde_json::json!({
            "command": command,
        });
        
        if let Some(timeout) = options.timeout_secs {
            args["timeout_secs"] = serde_json::json!(timeout);
        }
        if let Some(ref dir) = options.working_dir {
            args["working_dir"] = serde_json::json!(dir);
        }
        if !options.env.is_empty() {
            args["env"] = serde_json::json!(options.env);
        }
        
        let result = self.mcp_call(session_id, "shell", args).await?;
        
        // Parse result
        let exit_code = result["exit_code"].as_i64().unwrap_or(-1) as i32;
        let stdout = result["stdout"].as_str().unwrap_or("").to_string();
        let stderr = result["stderr"].as_str().unwrap_or("").to_string();
        
        Ok(ExecResult {
            exit_code,
            stdout,
            stderr,
        })
    }
    
    async fn read_file(&self, session_id: &str, path: &str) -> Result<String, BackendError> {
        let args = serde_json::json!({
            "file_path": path,
        });
        
        let result = self.mcp_call(session_id, "read_file", args).await?;
        
        if let Some(content) = result.get("content").and_then(|c| c.as_str()) {
            Ok(content.to_string())
        } else if let Some(text) = result.as_str() {
            Ok(text.to_string())
        } else {
            Err(BackendError::FileError(format!("Read {}: unexpected response", path)))
        }
    }
    
    async fn write_file(&self, session_id: &str, path: &str, content: &str) -> Result<(), BackendError> {
        let args = serde_json::json!({
            "command": "create",
            "path": path,
            "file_text": content,
        });
        
        let result = self.mcp_call(session_id, "text_editor", args).await?;
        
        if result.get("error").is_some() {
            Err(BackendError::FileError(format!("Write {}: {:?}", path, result)))
        } else {
            Ok(())
        }
    }
    
    async fn call_tool(&self, session_id: &str, tool_name: &str, args: Value) -> Result<Value, BackendError> {
        self.mcp_call(session_id, tool_name, args).await
    }

    async fn health_check(&self) -> Result<(), BackendError> {
        let url = format!("{}/health", self.config.control_plane_url);
        
        let response = self.client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| BackendError::Unavailable(format!("Control plane: {e}")))?;
        
        if response.status().is_success() {
            Ok(())
        } else {
            Err(BackendError::Unavailable(format!(
                "Control plane returned {}",
                response.status()
            )))
        }
    }
}
