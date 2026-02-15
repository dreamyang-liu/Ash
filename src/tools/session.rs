//! Session management with pluggable backends (Docker / K8s)
//!
//! Architecture:
//! - BackendManager: manages multiple backends (docker, k8s)
//! - Sessions are identified by UUID, backend is tracked per session
//! - Operations route to the correct backend automatically

use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::backend::{
    Backend, BackendError, BackendType, CreateOptions, DockerBackend, ExecOptions,
    ExecResult, K8sBackend, K8sConfig, ResourceSpec, Session,
};
use crate::{BoxFuture, Tool, ToolResult};

// ========== Backend Manager ==========

/// Manages multiple backends and routes operations
pub struct BackendManager {
    docker: Option<DockerBackend>,
    k8s: Option<K8sBackend>,
    /// Default backend for new sessions
    default_backend: BackendType,
    /// Session -> Backend mapping
    session_backends: Arc<RwLock<HashMap<String, BackendType>>>,
}

impl BackendManager {
    /// Create with both backends available
    pub fn new() -> Self {
        let docker = DockerBackend::new().ok();
        let k8s = Some(K8sBackend::new());
        
        // Default to Docker if available, otherwise K8s
        let default_backend = if docker.is_some() {
            BackendType::Docker
        } else {
            BackendType::K8s
        };
        
        Self {
            docker,
            k8s,
            default_backend,
            session_backends: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Create with specific default backend
    pub fn with_default(backend: BackendType) -> Self {
        let mut manager = Self::new();
        manager.default_backend = backend;
        manager
    }
    
    /// Set the default backend
    pub fn set_default(&mut self, backend: BackendType) {
        self.default_backend = backend;
    }
    
    /// Get the default backend type
    pub fn default_backend(&self) -> BackendType {
        self.default_backend
    }
    
    /// Configure K8s backend
    pub fn configure_k8s(&mut self, config: K8sConfig) {
        self.k8s = Some(K8sBackend::with_config(config));
    }
    
    /// Get backend for a session
    async fn get_backend(&self, session_id: &str) -> Result<&dyn Backend, BackendError> {
        let backends = self.session_backends.read().await;
        let backend_type = backends.get(session_id).copied().unwrap_or(self.default_backend);
        self.get_backend_by_type(backend_type)
    }
    
    /// Get backend by type
    fn get_backend_by_type(&self, backend_type: BackendType) -> Result<&dyn Backend, BackendError> {
        match backend_type {
            BackendType::Docker => self.docker.as_ref()
                .map(|b| b as &dyn Backend)
                .ok_or_else(|| BackendError::Unavailable("Docker backend not available".into())),
            BackendType::K8s => self.k8s.as_ref()
                .map(|b| b as &dyn Backend)
                .ok_or_else(|| BackendError::Unavailable("K8s backend not available".into())),
        }
    }
    
    /// Create a session using specified or default backend
    pub async fn create(&self, backend: Option<BackendType>, options: CreateOptions) -> Result<Session, BackendError> {
        let backend_type = backend.unwrap_or(self.default_backend);
        let backend = self.get_backend_by_type(backend_type)?;
        
        let session = backend.create(options).await?;
        
        // Track which backend owns this session
        {
            let mut backends = self.session_backends.write().await;
            backends.insert(session.id.clone(), backend_type);
        }
        
        Ok(session)
    }
    
    /// Destroy a session
    pub async fn destroy(&self, session_id: &str) -> Result<(), BackendError> {
        let backend = self.get_backend(session_id).await?;
        backend.destroy(session_id).await?;
        
        // Remove tracking
        {
            let mut backends = self.session_backends.write().await;
            backends.remove(session_id);
        }
        
        Ok(())
    }
    
    /// List all sessions across backends
    pub async fn list(&self) -> Result<Vec<Session>, BackendError> {
        let mut sessions = Vec::new();
        
        if let Some(ref docker) = self.docker {
            if let Ok(docker_sessions) = docker.list().await {
                sessions.extend(docker_sessions);
            }
        }
        
        if let Some(ref k8s) = self.k8s {
            if let Ok(k8s_sessions) = k8s.list().await {
                sessions.extend(k8s_sessions);
            }
        }
        
        Ok(sessions)
    }
    
    /// Get session info
    pub async fn get(&self, session_id: &str) -> Result<Option<Session>, BackendError> {
        // Try the tracked backend first
        let backends = self.session_backends.read().await;
        if let Some(&backend_type) = backends.get(session_id) {
            drop(backends);
            let backend = self.get_backend_by_type(backend_type)?;
            return backend.get(session_id).await;
        }
        drop(backends);
        
        // Search all backends
        if let Some(ref docker) = self.docker {
            if let Ok(Some(session)) = docker.get(session_id).await {
                return Ok(Some(session));
            }
        }
        
        if let Some(ref k8s) = self.k8s {
            if let Ok(Some(session)) = k8s.get(session_id).await {
                return Ok(Some(session));
            }
        }
        
        Ok(None)
    }
    
    /// Execute command in session
    pub async fn exec(&self, session_id: &str, command: &str, options: ExecOptions) -> Result<ExecResult, BackendError> {
        let backend = self.get_backend(session_id).await?;
        backend.exec(session_id, command, options).await
    }
    
    /// Read file from session
    pub async fn read_file(&self, session_id: &str, path: &str) -> Result<String, BackendError> {
        let backend = self.get_backend(session_id).await?;
        backend.read_file(session_id, path).await
    }
    
    /// Write file to session
    pub async fn write_file(&self, session_id: &str, path: &str, content: &str) -> Result<(), BackendError> {
        let backend = self.get_backend(session_id).await?;
        backend.write_file(session_id, path, content).await
    }
    
    /// Call any MCP tool in session (generic pass-through)
    pub async fn call_tool(&self, session_id: &str, tool_name: &str, args: Value) -> Result<Value, BackendError> {
        let backend = self.get_backend(session_id).await?;
        backend.call_tool(session_id, tool_name, args).await
    }

    /// Check backend health
    pub async fn health_check(&self, backend: BackendType) -> Result<(), BackendError> {
        let backend = self.get_backend_by_type(backend)?;
        backend.health_check().await
    }
}

// Global manager instance
lazy_static::lazy_static! {
    pub static ref BACKEND_MANAGER: Arc<RwLock<BackendManager>> = Arc::new(RwLock::new(BackendManager::new()));
}

/// Set the default backend
pub async fn set_default_backend(backend: BackendType) {
    let mut manager = BACKEND_MANAGER.write().await;
    manager.set_default(backend);
}

/// Configure K8s backend
pub async fn configure_k8s(config: K8sConfig) {
    let mut manager = BACKEND_MANAGER.write().await;
    manager.configure_k8s(config);
}

// ========== Session Create Tool ==========

#[derive(Debug, Clone, Deserialize, Default)]
pub struct SessionCreateArgs {
    /// Backend to use: "docker" or "k8s" (default: auto-detect)
    #[serde(default)]
    pub backend: Option<String>,
    /// Custom name
    #[serde(default)]
    pub name: Option<String>,
    /// Docker image
    #[serde(default)]
    pub image: Option<String>,
    /// Environment variables
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Ports to expose
    #[serde(default)]
    pub ports: Vec<u16>,
    /// Working directory
    #[serde(default)]
    pub working_dir: Option<String>,
    /// Resource limits
    #[serde(default)]
    pub resources: Option<ResourceSpecArgs>,
    /// Labels / node selector
    #[serde(default)]
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ResourceSpecArgs {
    pub cpu: Option<String>,
    pub memory: Option<String>,
    pub cpu_limit: Option<String>,
    pub memory_limit: Option<String>,
}

pub struct SessionCreateTool;

impl Tool for SessionCreateTool {
    fn name(&self) -> &'static str { "session_create" }
    fn description(&self) -> &'static str { 
        "Create a new sandbox session. Backend: 'docker' (local) or 'k8s' (remote). Returns session_id." 
    }
    
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "backend": {
                    "type": "string",
                    "enum": ["docker", "k8s"],
                    "description": "Backend: 'docker' for local containers, 'k8s' for remote sandboxes"
                },
                "name": {"type": "string", "description": "Custom session name"},
                "image": {"type": "string", "description": "Docker image"},
                "env": {"type": "object", "description": "Environment variables"},
                "ports": {"type": "array", "items": {"type": "integer"}, "description": "Ports to expose"},
                "working_dir": {"type": "string", "description": "Working directory"},
                "resources": {
                    "type": "object",
                    "properties": {
                        "cpu": {"type": "string", "description": "CPU request"},
                        "memory": {"type": "string", "description": "Memory request"},
                        "cpu_limit": {"type": "string", "description": "CPU limit"},
                        "memory_limit": {"type": "string", "description": "Memory limit"}
                    }
                },
                "labels": {"type": "object", "description": "Labels / node selector"}
            }
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: SessionCreateArgs = serde_json::from_value(args).unwrap_or_default();
            
            // Parse backend
            let backend = args.backend.as_ref().and_then(|s| s.parse::<BackendType>().ok());
            
            // Build options
            let options = CreateOptions {
                name: args.name,
                image: args.image,
                env: args.env,
                ports: args.ports,
                working_dir: args.working_dir,
                command: None,
                resources: args.resources.map(|r| ResourceSpec {
                    cpu_request: r.cpu.clone(),
                    memory_request: r.memory.clone(),
                    cpu_limit: r.cpu_limit.or(r.cpu),
                    memory_limit: r.memory_limit.or(r.memory),
                }),
                labels: args.labels,
            };
            
            let manager = BACKEND_MANAGER.read().await;
            match manager.create(backend, options).await {
                Ok(session) => ToolResult::ok(serde_json::json!({
                    "session_id": session.id,
                    "name": session.name,
                    "backend": session.backend.to_string(),
                    "status": session.status.to_string(),
                    "host": session.host,
                    "image": session.image,
                }).to_string()),
                Err(e) => ToolResult::err(format!("Create failed: {e}")),
            }
        })
    }
}

// ========== Session Destroy Tool ==========

#[derive(Debug, Clone, Deserialize)]
pub struct SessionDestroyArgs {
    pub session_id: String,
}

pub struct SessionDestroyTool;

impl Tool for SessionDestroyTool {
    fn name(&self) -> &'static str { "session_destroy" }
    fn description(&self) -> &'static str { "Destroy a sandbox session" }
    
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID to destroy"}
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
            
            let manager = BACKEND_MANAGER.read().await;
            match manager.destroy(&args.session_id).await {
                Ok(()) => ToolResult::ok(format!("Destroyed: {}", args.session_id)),
                Err(e) => ToolResult::err(format!("Destroy failed: {e}")),
            }
        })
    }
}

// ========== Session List Tool ==========

pub struct SessionListTool;

impl Tool for SessionListTool {
    fn name(&self) -> &'static str { "session_list" }
    fn description(&self) -> &'static str { "List all active sessions (across backends)" }
    fn schema(&self) -> Value { serde_json::json!({"type": "object"}) }
    
    fn execute(&self, _args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let manager = BACKEND_MANAGER.read().await;
            match manager.list().await {
                Ok(sessions) => {
                    let list: Vec<Value> = sessions.iter().map(|s| {
                        serde_json::json!({
                            "session_id": s.id,
                            "name": s.name,
                            "backend": s.backend.to_string(),
                            "status": s.status.to_string(),
                            "host": s.host,
                            "image": s.image,
                        })
                    }).collect();
                    ToolResult::ok(serde_json::to_string_pretty(&list).unwrap())
                }
                Err(e) => ToolResult::err(format!("List failed: {e}")),
            }
        })
    }
}

// ========== Session Info Tool ==========

#[derive(Debug, Clone, Deserialize)]
pub struct SessionInfoArgs {
    pub session_id: String,
}

pub struct SessionInfoTool;

impl Tool for SessionInfoTool {
    fn name(&self) -> &'static str { "session_info" }
    fn description(&self) -> &'static str { "Get detailed info about a session" }
    
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
            let args: SessionInfoArgs = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let manager = BACKEND_MANAGER.read().await;
            match manager.get(&args.session_id).await {
                Ok(Some(session)) => {
                    ToolResult::ok(serde_json::to_string_pretty(&serde_json::json!({
                        "session_id": session.id,
                        "name": session.name,
                        "backend": session.backend.to_string(),
                        "status": session.status.to_string(),
                        "host": session.host,
                        "image": session.image,
                        "ports": session.ports,
                        "created_at": session.created_at.to_rfc3339(),
                    })).unwrap())
                }
                Ok(None) => ToolResult::err(format!("Session not found: {}", args.session_id)),
                Err(e) => ToolResult::err(format!("Failed: {e}")),
            }
        })
    }
}

// ========== Backend Switch Tool ==========

#[derive(Debug, Clone, Deserialize)]
pub struct BackendSwitchArgs {
    pub backend: String,
}

pub struct BackendSwitchTool;

impl Tool for BackendSwitchTool {
    fn name(&self) -> &'static str { "backend_switch" }
    fn description(&self) -> &'static str { "Switch default backend: 'docker' (local) or 'k8s' (remote)" }
    
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "backend": {
                    "type": "string",
                    "enum": ["docker", "k8s"],
                    "description": "Backend to switch to"
                }
            },
            "required": ["backend"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: BackendSwitchArgs = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let backend = match args.backend.parse::<BackendType>() {
                Ok(b) => b,
                Err(e) => return ToolResult::err(e),
            };
            
            set_default_backend(backend).await;
            ToolResult::ok(format!("Default backend set to: {}", backend))
        })
    }
}

// ========== Backend Status Tool ==========

pub struct BackendStatusTool;

impl Tool for BackendStatusTool {
    fn name(&self) -> &'static str { "backend_status" }
    fn description(&self) -> &'static str { "Check status of backends (docker/k8s availability)" }
    fn schema(&self) -> Value { serde_json::json!({"type": "object"}) }
    
    fn execute(&self, _args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let manager = BACKEND_MANAGER.read().await;
            
            let docker_status = match manager.health_check(BackendType::Docker).await {
                Ok(()) => "available",
                Err(e) => {
                    tracing::debug!("Docker health check failed: {e}");
                    "unavailable"
                }
            };
            
            let k8s_status = match manager.health_check(BackendType::K8s).await {
                Ok(()) => "available",
                Err(e) => {
                    tracing::debug!("K8s health check failed: {e}");
                    "unavailable"
                }
            };
            
            ToolResult::ok(serde_json::json!({
                "default": manager.default_backend().to_string(),
                "backends": {
                    "docker": docker_status,
                    "k8s": k8s_status,
                }
            }).to_string())
        })
    }
}

// ========== Exec in Session Helper ==========

/// Execute a command in a session (helper for other tools)
pub async fn exec_in_session(session_id: &str, command: &str, options: ExecOptions) -> Result<ExecResult, BackendError> {
    let manager = BACKEND_MANAGER.read().await;
    manager.exec(session_id, command, options).await
}

/// Read file from session (helper for other tools)
pub async fn read_file_in_session(session_id: &str, path: &str) -> Result<String, BackendError> {
    let manager = BACKEND_MANAGER.read().await;
    manager.read_file(session_id, path).await
}

/// Write file to session (helper for other tools)
pub async fn write_file_in_session(session_id: &str, path: &str, content: &str) -> Result<(), BackendError> {
    let manager = BACKEND_MANAGER.read().await;
    manager.write_file(session_id, path, content).await
}

/// Call any MCP tool in session (generic pass-through for terminal, etc.)
pub async fn call_tool_in_session(session_id: &str, tool_name: &str, args: Value) -> Result<Value, BackendError> {
    let manager = BACKEND_MANAGER.read().await;
    manager.call_tool(session_id, tool_name, args).await
}

/// Get session by ID
pub async fn get_session(session_id: &str) -> Option<Session> {
    let manager = BACKEND_MANAGER.read().await;
    manager.get(session_id).await.ok().flatten()
}
