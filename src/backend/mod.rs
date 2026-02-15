//! Backend abstraction for session management
//!
//! Two backends:
//! - Docker: Local containers via Docker Engine API
//! - K8s: Remote sandboxes via Control Plane

mod docker;
mod k8s;

pub use docker::DockerBackend;
pub use k8s::{K8sBackend, K8sConfig};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Session representing a running sandbox (container or pod)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub name: String,
    pub backend: BackendType,
    pub status: SessionStatus,
    pub host: String,
    pub ports: Vec<PortMapping>,
    pub image: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendType {
    Docker,
    K8s,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::Docker => write!(f, "docker"),
            BackendType::K8s => write!(f, "k8s"),
        }
    }
}

impl std::str::FromStr for BackendType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "docker" | "local" => Ok(BackendType::Docker),
            "k8s" | "kubernetes" | "remote" => Ok(BackendType::K8s),
            _ => Err(format!("Unknown backend: {s}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionStatus {
    Creating,
    Running,
    Stopped,
    Failed,
    Unknown,
}

impl std::fmt::Display for SessionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionStatus::Creating => write!(f, "creating"),
            SessionStatus::Running => write!(f, "running"),
            SessionStatus::Stopped => write!(f, "stopped"),
            SessionStatus::Failed => write!(f, "failed"),
            SessionStatus::Unknown => write!(f, "unknown"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortMapping {
    pub container_port: u16,
    pub host_port: Option<u16>,
    pub protocol: String,
}

/// Options for creating a session
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CreateOptions {
    pub name: Option<String>,
    pub image: Option<String>,
    pub env: HashMap<String, String>,
    pub ports: Vec<u16>,
    pub working_dir: Option<String>,
    pub command: Option<Vec<String>>,
    pub resources: Option<ResourceSpec>,
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceSpec {
    pub cpu_limit: Option<String>,
    pub memory_limit: Option<String>,
    pub cpu_request: Option<String>,
    pub memory_request: Option<String>,
}

/// Options for executing commands
#[derive(Debug, Clone, Default)]
pub struct ExecOptions {
    pub working_dir: Option<String>,
    pub env: HashMap<String, String>,
    pub timeout_secs: Option<u64>,
}

/// Result of command execution
#[derive(Debug, Clone)]
pub struct ExecResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl ExecResult {
    pub fn success(&self) -> bool {
        self.exit_code == 0
    }
    
    pub fn output(&self) -> String {
        if self.stderr.is_empty() {
            self.stdout.clone()
        } else if self.stdout.is_empty() {
            self.stderr.clone()
        } else {
            format!("{}\n--- stderr ---\n{}", self.stdout, self.stderr)
        }
    }
}

/// Backend trait - abstraction over Docker/K8s
#[async_trait]
pub trait Backend: Send + Sync {
    /// Backend type identifier
    fn backend_type(&self) -> BackendType;
    
    /// Create a new session (container/pod)
    async fn create(&self, options: CreateOptions) -> Result<Session, BackendError>;
    
    /// Destroy a session
    async fn destroy(&self, session_id: &str) -> Result<(), BackendError>;
    
    /// List all sessions managed by this backend
    async fn list(&self) -> Result<Vec<Session>, BackendError>;
    
    /// Get session info
    async fn get(&self, session_id: &str) -> Result<Option<Session>, BackendError>;
    
    /// Execute a command in session
    async fn exec(&self, session_id: &str, command: &str, options: ExecOptions) -> Result<ExecResult, BackendError>;
    
    /// Read file from session
    async fn read_file(&self, session_id: &str, path: &str) -> Result<String, BackendError>;
    
    /// Write file to session
    async fn write_file(&self, session_id: &str, path: &str, content: &str) -> Result<(), BackendError>;
    
    /// Call any MCP tool in session (generic pass-through)
    async fn call_tool(&self, session_id: &str, tool_name: &str, args: serde_json::Value) -> Result<serde_json::Value, BackendError>;

    /// Check if backend is available
    async fn health_check(&self) -> Result<(), BackendError>;
}

/// Backend errors
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("Connection failed: {0}")]
    Connection(String),
    
    #[error("Session not found: {0}")]
    NotFound(String),
    
    #[error("Session creation failed: {0}")]
    CreateFailed(String),
    
    #[error("Command execution failed: {0}")]
    ExecFailed(String),
    
    #[error("File operation failed: {0}")]
    FileError(String),
    
    #[error("Backend not available: {0}")]
    Unavailable(String),
    
    #[error("Timeout: {0}")]
    Timeout(String),
    
    #[error("{0}")]
    Other(String),
}

impl BackendError {
    pub fn connection(msg: impl Into<String>) -> Self {
        BackendError::Connection(msg.into())
    }
    
    pub fn not_found(id: impl Into<String>) -> Self {
        BackendError::NotFound(id.into())
    }
}
