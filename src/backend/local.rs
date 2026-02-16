//! Local backend - direct host execution
//!
//! This is the simplest backend - executes commands directly on the host machine.
//! No containerization, no isolation. Used as the default "session".

use super::{Backend, BackendError, BackendType, CreateOptions, ExecOptions, ExecResult, Session, SessionStatus};
use async_trait::async_trait;
use std::process::Stdio;
use tokio::process::Command;

/// Local backend configuration
#[derive(Debug, Clone, Default)]
pub struct LocalConfig {
    /// Default working directory
    pub default_workdir: Option<String>,
}

/// Local backend - executes on host
pub struct LocalBackend {
    config: LocalConfig,
}

impl LocalBackend {
    pub fn new(config: LocalConfig) -> Self {
        Self { config }
    }

    /// Create with default config
    pub fn default() -> Self {
        Self::new(LocalConfig::default())
    }
    
    /// Create the implicit local session
    fn local_session() -> Session {
        Session {
            id: "local".to_string(),
            name: "local".to_string(),
            backend: BackendType::Local,
            status: SessionStatus::Running,
            host: "localhost".to_string(),
            ports: vec![],
            image: "host".to_string(),
            created_at: chrono::Utc::now(),
        }
    }
}

#[async_trait]
impl Backend for LocalBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Local
    }

    async fn create(&self, _options: CreateOptions) -> Result<Session, BackendError> {
        // Local backend has a single implicit session
        Ok(Self::local_session())
    }

    async fn destroy(&self, id: &str) -> Result<(), BackendError> {
        if id == "local" {
            Err(BackendError::operation("Cannot destroy local session"))
        } else {
            Err(BackendError::NotFound(id.to_string()))
        }
    }

    async fn list(&self) -> Result<Vec<Session>, BackendError> {
        Ok(vec![Self::local_session()])
    }

    async fn get(&self, id: &str) -> Result<Option<Session>, BackendError> {
        if id == "local" {
            Ok(Some(Self::local_session()))
        } else {
            Ok(None)
        }
    }

    async fn exec(&self, _session_id: &str, command: &str, options: ExecOptions) -> Result<ExecResult, BackendError> {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", command]);

        if let Some(ref dir) = options.working_dir.as_ref().or(self.config.default_workdir.as_ref()) {
            cmd.current_dir(dir);
        }

        for (k, v) in &options.env {
            cmd.env(k, v);
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let timeout = std::time::Duration::from_secs(options.timeout_secs.unwrap_or(300));

        let output = match tokio::time::timeout(timeout, cmd.output()).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => return Err(BackendError::exec(format!("Failed to execute: {e}"))),
            Err(_) => return Err(BackendError::exec("Command timed out")),
        };

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        Ok(ExecResult {
            stdout,
            stderr,
            exit_code,
        })
    }

    async fn read_file(&self, _session_id: &str, path: &str) -> Result<String, BackendError> {
        tokio::fs::read_to_string(path)
            .await
            .map_err(|e| BackendError::file(format!("Failed to read {path}: {e}")))
    }

    async fn write_file(&self, _session_id: &str, path: &str, content: &str) -> Result<(), BackendError> {
        // Create parent directories
        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        tokio::fs::write(path, content)
            .await
            .map_err(|e| BackendError::file(format!("Failed to write {path}: {e}")))
    }

    async fn call_tool(&self, _session_id: &str, tool_name: &str, args: serde_json::Value) -> Result<serde_json::Value, BackendError> {
        // For local backend, we execute tools directly
        // This is handled by the BackendManager, not here
        Err(BackendError::operation(format!(
            "Local backend does not support remote tool calls. Use direct execution for tool '{}'",
            tool_name
        )))
    }

    async fn health_check(&self) -> Result<(), BackendError> {
        // Local is always healthy
        Ok(())
    }
}
