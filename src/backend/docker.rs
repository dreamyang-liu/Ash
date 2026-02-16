//! Docker backend - local container management via Docker Engine API
//!
//! Architecture:
//! - Create container with ash MCP server running inside
//! - Connect to container's MCP server via HTTP (similar to K8s backend)
//! - Port 3000 inside container is mapped to a random host port

use async_trait::async_trait;
use bollard::container::{
    Config, CreateContainerOptions, ListContainersOptions, 
    RemoveContainerOptions, StartContainerOptions,
};
use bollard::image::CreateImageOptions;
use bollard::Docker;
use futures_util::StreamExt;
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

/// MCP server port inside container
const MCP_PORT: u16 = 3000;

/// ash-mcp release download URL
const ASH_RELEASE_URL: &str = "https://github.com/dreamyang-liu/Ash/releases/download/dev/ash-linux-x86_64-ubuntu2204.tar.gz";

/// Docker backend configuration
#[derive(Debug, Clone)]
pub struct DockerConfig {
    /// Docker socket path (default: auto-detect)
    pub socket_path: Option<String>,
    /// Default image for containers (ash-mcp is auto-installed at startup)
    pub default_image: String,
    /// Container name prefix
    pub name_prefix: String,
    /// Labels to apply to all containers
    pub labels: HashMap<String, String>,
    /// Timeout for MCP calls (seconds)
    pub timeout_secs: u64,
}

impl Default for DockerConfig {
    fn default() -> Self {
        // Try to find Docker socket
        let socket_path = if std::path::Path::new("/var/run/docker.sock").exists() {
            None // Use default
        } else {
            // Docker Desktop on macOS
            let home = std::env::var("HOME").unwrap_or_default();
            let desktop_sock = format!("{}/.docker/run/docker.sock", home);
            if std::path::Path::new(&desktop_sock).exists() {
                Some(desktop_sock)
            } else {
                None
            }
        };
        
        Self {
            socket_path,
            default_image: "ubuntu:24.04".to_string(),
            name_prefix: "ash-".to_string(),
            labels: {
                let mut m = HashMap::new();
                m.insert("managed-by".to_string(), "ash-cli".to_string());
                m
            },
            timeout_secs: 300,
        }
    }
}

/// Docker backend implementation
pub struct DockerBackend {
    docker: Docker,
    http_client: Client,
    config: DockerConfig,
    /// Track sessions we've created (session_id -> Session with host port)
    sessions: Arc<RwLock<HashMap<String, Session>>>,
}

impl DockerBackend {
    /// Create new Docker backend with default config
    pub fn new() -> Result<Self, BackendError> {
        Self::with_config(DockerConfig::default())
    }
    
    /// Create with custom config
    pub fn with_config(config: DockerConfig) -> Result<Self, BackendError> {
        let docker = if let Some(ref path) = config.socket_path {
            Docker::connect_with_socket(path, 120, bollard::API_DEFAULT_VERSION)
                .map_err(|e| BackendError::connection(format!("Docker socket {path}: {e}")))?
        } else {
            Docker::connect_with_socket_defaults()
                .map_err(|e| BackendError::connection(format!("Docker: {e}")))?
        };
        
        Ok(Self {
            docker,
            http_client: Client::new(),
            config,
            sessions: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Generate container name
    fn generate_name(&self, custom: Option<&str>) -> String {
        match custom {
            Some(name) => format!("{}{}", self.config.name_prefix, name),
            None => format!("{}{}", self.config.name_prefix, uuid::Uuid::new_v4().to_string()[..8].to_string()),
        }
    }
    
    /// Pull image if not present
    async fn ensure_image(&self, image: &str) -> Result<(), BackendError> {
        // Check if image exists locally
        if self.docker.inspect_image(image).await.is_ok() {
            return Ok(());
        }
        
        // Pull the image
        let options = CreateImageOptions {
            from_image: image,
            ..Default::default()
        };
        
        let mut stream = self.docker.create_image(Some(options), None, None);
        while let Some(result) = stream.next().await {
            match result {
                Ok(info) => {
                    if let Some(status) = info.status {
                        tracing::debug!("Pull {}: {}", image, status);
                    }
                }
                Err(e) => {
                    return Err(BackendError::CreateFailed(format!("Failed to pull image {image}: {e}")));
                }
            }
        }
        
        Ok(())
    }
    
    /// Get MCP endpoint URL for a session
    fn get_mcp_url(&self, session: &Session) -> Option<String> {
        // Find the mapped host port for MCP_PORT
        session.ports.iter()
            .find(|p| p.container_port == MCP_PORT)
            .and_then(|p| p.host_port)
            .map(|port| format!("http://localhost:{}/mcp", port))
    }
    
    /// Call a tool via MCP on the container
    async fn mcp_call(&self, session: &Session, tool_name: &str, args: Value) -> Result<Value, BackendError> {
        let url = self.get_mcp_url(session)
            .ok_or_else(|| BackendError::Connection("No MCP port mapped".into()))?;
        
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": chrono::Utc::now().timestamp_millis(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": args
            }
        });
        
        let response = self.http_client
            .post(&url)
            .header("Content-Type", "application/json")
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
    
    /// Wait for MCP server to be ready
    async fn wait_for_mcp(&self, session: &Session, timeout_secs: u64) -> Result<(), BackendError> {
        let url = self.get_mcp_url(session)
            .ok_or_else(|| BackendError::Connection("No MCP port mapped".into()))?;
        
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(timeout_secs);
        
        loop {
            if start.elapsed() > timeout {
                return Err(BackendError::Timeout(format!(
                    "MCP server not ready after {}s", timeout_secs
                )));
            }
            
            // Try to call tools/list to check if server is ready
            let request = serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            });
            
            match self.http_client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&request)
                .timeout(Duration::from_secs(2))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    tracing::debug!("MCP server ready at {}", url);
                    return Ok(());
                }
                _ => {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            }
        }
    }
    
    /// Convert container info to Session
    fn container_to_session(&self, id: &str, name: &str, image: &str, status: &str, ports: Vec<PortMapping>) -> Session {
        let status = match status.to_lowercase().as_str() {
            s if s.contains("running") => SessionStatus::Running,
            s if s.contains("created") => SessionStatus::Creating,
            s if s.contains("exited") | s.contains("stopped") => SessionStatus::Stopped,
            s if s.contains("dead") | s.contains("error") => SessionStatus::Failed,
            _ => SessionStatus::Unknown,
        };
        
        Session {
            id: id.to_string(),
            name: name.trim_start_matches('/').to_string(),
            backend: BackendType::Docker,
            status,
            host: "localhost".to_string(),
            ports,
            image: image.to_string(),
            created_at: chrono::Utc::now(),
        }
    }
}

#[async_trait]
impl Backend for DockerBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Docker
    }
    
    async fn create(&self, options: CreateOptions) -> Result<Session, BackendError> {
        let image = options.image.as_deref().unwrap_or(&self.config.default_image);
        let name = self.generate_name(options.name.as_deref());
        
        // Ensure image is available
        self.ensure_image(image).await?;
        
        // Build labels
        let mut labels = self.config.labels.clone();
        labels.extend(options.labels);
        
        // Build environment variables
        let env: Vec<String> = options.env.iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        
        // Always expose MCP port, plus any user-requested ports
        let mut all_ports = vec![MCP_PORT];
        all_ports.extend(options.ports.iter().copied());
        
        // Build port bindings
        let mut exposed_ports = HashMap::new();
        let mut port_bindings = HashMap::new();
        
        for port in &all_ports {
            let port_key = format!("{}/tcp", port);
            exposed_ports.insert(port_key.clone(), HashMap::new());
            port_bindings.insert(
                port_key,
                Some(vec![bollard::service::PortBinding {
                    host_ip: Some("0.0.0.0".to_string()),
                    host_port: None, // Auto-assign
                }]),
            );
        }
        
        // Container config - download ash-mcp and run it
        let bootstrap_script = format!(
            "export DEBIAN_FRONTEND=noninteractive; \
             apt-get update -qq && apt-get install -y -qq curl > /dev/null 2>&1; \
             curl -fsSL {} | tar xz -C /tmp && \
             mv /tmp/ash-linux-x86_64-ubuntu2204 /usr/local/bin/ash && \
             mv /tmp/ash-linux-x86_64-ubuntu2204-mcp /usr/local/bin/ash-mcp && \
             chmod +x /usr/local/bin/ash /usr/local/bin/ash-mcp && \
             ash-mcp --transport http --port {}",
            ASH_RELEASE_URL, MCP_PORT,
        );
        let config = Config {
            image: Some(image.to_string()),
            labels: Some(labels),
            env: if env.is_empty() { None } else { Some(env) },
            working_dir: options.working_dir.clone(),
            cmd: Some(vec!["sh".to_string(), "-c".to_string(), bootstrap_script]),
            exposed_ports: Some(exposed_ports),
            host_config: Some(bollard::service::HostConfig {
                port_bindings: Some(port_bindings),
                ..Default::default()
            }),
            ..Default::default()
        };
        
        // Create container
        let create_options = CreateContainerOptions { name: name.clone(), ..Default::default() };
        let response = self.docker
            .create_container(Some(create_options), config)
            .await
            .map_err(|e| BackendError::CreateFailed(format!("Create container: {e}")))?;
        
        let container_id = response.id;
        
        // Start container
        self.docker
            .start_container(&container_id, None::<StartContainerOptions<String>>)
            .await
            .map_err(|e| BackendError::CreateFailed(format!("Start container: {e}")))?;
        
        // Get container info for port mappings
        let info = self.docker
            .inspect_container(&container_id, None)
            .await
            .map_err(|e| BackendError::CreateFailed(format!("Inspect container: {e}")))?;
        
        // Extract actual port mappings
        let mut port_mappings = Vec::new();
        if let Some(network) = info.network_settings {
            if let Some(ports) = network.ports {
                for (container_port, bindings) in ports {
                    if let Some(bindings) = bindings {
                        for binding in bindings {
                            if let Some(host_port) = binding.host_port {
                                let port: u16 = container_port
                                    .split('/')
                                    .next()
                                    .and_then(|s| s.parse().ok())
                                    .unwrap_or(0);
                                port_mappings.push(PortMapping {
                                    container_port: port,
                                    host_port: host_port.parse().ok(),
                                    protocol: "tcp".to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }
        
        let session = Session {
            id: container_id.clone(),
            name: name.clone(),
            backend: BackendType::Docker,
            status: SessionStatus::Running,
            host: "localhost".to_string(),
            ports: port_mappings,
            image: image.to_string(),
            created_at: chrono::Utc::now(),
        };
        
        // Wait for MCP server to be ready (longer timeout for bootstrap download)
        self.wait_for_mcp(&session, 120).await?;
        
        // Cache session
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(container_id.clone(), session.clone());
        }
        
        Ok(session)
    }
    
    async fn destroy(&self, session_id: &str) -> Result<(), BackendError> {
        let options = RemoveContainerOptions {
            force: true,
            v: true, // Remove volumes
            ..Default::default()
        };
        
        self.docker
            .remove_container(session_id, Some(options))
            .await
            .map_err(|e| BackendError::Other(format!("Remove container: {e}")))?;
        
        // Remove from cache
        {
            let mut sessions = self.sessions.write().await;
            sessions.remove(session_id);
        }
        
        Ok(())
    }
    
    async fn list(&self) -> Result<Vec<Session>, BackendError> {
        let mut filters = HashMap::new();
        filters.insert("label", vec!["managed-by=ash-cli"]);
        
        let options = ListContainersOptions {
            all: true,
            filters,
            ..Default::default()
        };
        
        let containers = self.docker
            .list_containers(Some(options))
            .await
            .map_err(|e| BackendError::Connection(format!("List containers: {e}")))?;
        
        let sessions: Vec<Session> = containers
            .into_iter()
            .map(|c| {
                let id = c.id.unwrap_or_default();
                let name = c.names.and_then(|n| n.first().cloned()).unwrap_or_default();
                let image = c.image.unwrap_or_default();
                let status = c.state.unwrap_or_default();
                
                // Extract port mappings
                let ports: Vec<PortMapping> = c.ports
                    .unwrap_or_default()
                    .into_iter()
                    .map(|p| PortMapping {
                        container_port: p.private_port as u16,
                        host_port: p.public_port.map(|p| p as u16),
                        protocol: p.typ.map(|t| t.to_string()).unwrap_or_else(|| "tcp".to_string()),
                    })
                    .collect();
                
                self.container_to_session(&id, &name, &image, &status, ports)
            })
            .collect();
        
        Ok(sessions)
    }
    
    async fn get(&self, session_id: &str) -> Result<Option<Session>, BackendError> {
        // Check cache first
        {
            let sessions = self.sessions.read().await;
            if let Some(session) = sessions.get(session_id) {
                return Ok(Some(session.clone()));
            }
        }
        
        // Query Docker
        match self.docker.inspect_container(session_id, None).await {
            Ok(info) => {
                let id = info.id.unwrap_or_default();
                let name = info.name.unwrap_or_default();
                let image = info.config.and_then(|c| c.image).unwrap_or_default();
                let status = info.state.and_then(|s| s.status).map(|s| s.to_string()).unwrap_or_default();
                
                // Extract port mappings
                let ports = info.network_settings
                    .and_then(|n| n.ports)
                    .map(|ports| {
                        ports.into_iter()
                            .filter_map(|(container_port, bindings)| {
                                let port: u16 = container_port.split('/').next()?.parse().ok()?;
                                let host_port = bindings?.first()?.host_port.as_ref()?.parse().ok();
                                Some(PortMapping {
                                    container_port: port,
                                    host_port,
                                    protocol: "tcp".to_string(),
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                
                Ok(Some(self.container_to_session(&id, &name, &image, &status, ports)))
            }
            Err(bollard::errors::Error::DockerResponseServerError { status_code: 404, .. }) => {
                Ok(None)
            }
            Err(e) => Err(BackendError::Connection(format!("Inspect container: {e}"))),
        }
    }
    
    async fn exec(&self, session_id: &str, command: &str, options: ExecOptions) -> Result<ExecResult, BackendError> {
        let session = match {
            let sessions = self.sessions.read().await;
            sessions.get(session_id).cloned()
        } {
            Some(s) => s,
            None => self.get(session_id).await?
                .ok_or_else(|| BackendError::NotFound(session_id.to_string()))?,
        };
        
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
        
        let result = self.mcp_call(&session, "shell", args).await?;
        
        // Parse MCP result
        let content = result.get("content")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or("");
        
        let is_error = result.get("isError")
            .and_then(|e| e.as_bool())
            .unwrap_or(false);
        
        Ok(ExecResult {
            exit_code: if is_error { 1 } else { 0 },
            stdout: content.to_string(),
            stderr: String::new(),
        })
    }
    
    async fn read_file(&self, session_id: &str, path: &str) -> Result<String, BackendError> {
        let session = match {
            let sessions = self.sessions.read().await;
            sessions.get(session_id).cloned()
        } {
            Some(s) => s,
            None => self.get(session_id).await?
                .ok_or_else(|| BackendError::NotFound(session_id.to_string()))?,
        };
        
        let args = serde_json::json!({
            "file_path": path,
        });
        
        let result = self.mcp_call(&session, "read_file", args).await?;
        
        // Parse MCP result
        let content = result.get("content")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or("");
        
        Ok(content.to_string())
    }
    
    async fn write_file(&self, session_id: &str, path: &str, content: &str) -> Result<(), BackendError> {
        let session = match {
            let sessions = self.sessions.read().await;
            sessions.get(session_id).cloned()
        } {
            Some(s) => s,
            None => self.get(session_id).await?
                .ok_or_else(|| BackendError::NotFound(session_id.to_string()))?,
        };
        
        let args = serde_json::json!({
            "command": "create",
            "path": path,
            "file_text": content,
        });
        
        let result = self.mcp_call(&session, "text_editor", args).await?;
        
        if result.get("isError").and_then(|e| e.as_bool()).unwrap_or(false) {
            let msg = result.get("content")
                .and_then(|c| c.as_array())
                .and_then(|arr| arr.first())
                .and_then(|c| c.get("text"))
                .and_then(|t| t.as_str())
                .unwrap_or("Write failed");
            Err(BackendError::FileError(msg.to_string()))
        } else {
            Ok(())
        }
    }
    
    async fn call_tool(&self, session_id: &str, tool_name: &str, args: Value) -> Result<Value, BackendError> {
        let session = match {
            let sessions = self.sessions.read().await;
            sessions.get(session_id).cloned()
        } {
            Some(s) => s,
            None => self.get(session_id).await?
                .ok_or_else(|| BackendError::NotFound(session_id.to_string()))?,
        };
        self.mcp_call(&session, tool_name, args).await
    }

    async fn health_check(&self) -> Result<(), BackendError> {
        self.docker
            .ping()
            .await
            .map_err(|e| BackendError::Unavailable(format!("Docker ping failed: {e}")))?;
        Ok(())
    }
}
