//! MCP Mount System - install and proxy other MCP servers

use crate::{BoxFuture, Tool, ToolResult};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use lazy_static::lazy_static;

/// Installed MCP server info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledMcp {
    pub name: String,
    pub source: String,
    pub command: String,
    pub args: Vec<String>,
    pub installed_at: chrono::DateTime<chrono::Utc>,
}

/// Tool info from mounted MCP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    pub description: Option<String>,
}

/// Mounted MCP info (serializable)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MountedMcpInfo {
    pub name: String,
    pub tools: Vec<McpToolInfo>,
    pub mounted_at: chrono::DateTime<chrono::Utc>,
}

/// Active MCP process
struct MountedMcp {
    info: MountedMcpInfo,
    child: Child,
    stdin: tokio::process::ChildStdin,
    stdout: BufReader<tokio::process::ChildStdout>,
    request_id: u64,
}

impl MountedMcp {
    async fn send_request(&mut self, method: &str, params: Value) -> anyhow::Result<Value> {
        self.request_id += 1;
        let request = json!({
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        });

        let mut request_str = serde_json::to_string(&request)?;
        request_str.push('\n');
        
        self.stdin.write_all(request_str.as_bytes()).await?;
        self.stdin.flush().await?;

        let mut response_line = String::new();
        self.stdout.read_line(&mut response_line).await?;
        
        let response: Value = serde_json::from_str(&response_line)?;
        
        if let Some(error) = response.get("error") {
            anyhow::bail!("MCP error: {}", error);
        }
        
        Ok(response.get("result").cloned().unwrap_or(Value::Null))
    }

    async fn initialize(&mut self) -> anyhow::Result<Value> {
        let params = json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "ash", "version": "0.1.0" }
        });
        
        let result = self.send_request("initialize", params).await?;
        
        // Send initialized notification
        let notification = json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });
        let mut notif_str = serde_json::to_string(&notification)?;
        notif_str.push('\n');
        self.stdin.write_all(notif_str.as_bytes()).await?;
        self.stdin.flush().await?;
        
        Ok(result)
    }

    async fn list_tools(&mut self) -> anyhow::Result<Vec<McpToolInfo>> {
        let result = self.send_request("tools/list", json!({})).await?;
        
        let tools = result.get("tools")
            .and_then(|t| t.as_array())
            .map(|arr| {
                arr.iter().filter_map(|t| {
                    Some(McpToolInfo {
                        name: t.get("name")?.as_str()?.to_string(),
                        description: t.get("description").and_then(|d| d.as_str().map(String::from)),
                    })
                }).collect()
            })
            .unwrap_or_default();
        
        Ok(tools)
    }

    async fn call_tool(&mut self, tool_name: &str, arguments: Value) -> anyhow::Result<Value> {
        self.send_request("tools/call", json!({
            "name": tool_name,
            "arguments": arguments
        })).await
    }
}

/// MCP Mount manager
struct McpManager {
    install_dir: PathBuf,
    installed: HashMap<String, InstalledMcp>,
    mounted: HashMap<String, Mutex<MountedMcp>>,
}

impl McpManager {
    fn new() -> Self {
        let install_dir = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("ash")
            .join("mcp");
        Self {
            install_dir,
            installed: HashMap::new(),
            mounted: HashMap::new(),
        }
    }

    async fn install(&mut self, name: &str, source: &str) -> anyhow::Result<InstalledMcp> {
        tokio::fs::create_dir_all(&self.install_dir).await?;

        let (command, args) = if source.starts_with("npm:") {
            let package = &source[4..];
            ("npx".to_string(), vec!["-y".to_string(), package.to_string()])
        } else if source.starts_with("pip:") {
            let package = &source[4..];
            // Install if needed
            let _ = Command::new("pip").args(["install", "-q", package]).status().await;
            (package.to_string(), vec![])
        } else if source.starts_with("uvx:") {
            let package = &source[4..];
            ("uvx".to_string(), vec![package.to_string()])
        } else if source.starts_with("command:") {
            let cmd = &source[8..];
            let parts: Vec<&str> = cmd.split_whitespace().collect();
            let command = parts.first().map(|s| s.to_string()).unwrap_or_default();
            let args: Vec<String> = parts.iter().skip(1).map(|s| s.to_string()).collect();
            (command, args)
        } else {
            anyhow::bail!("Unknown source. Use npm:, pip:, uvx:, or command:");
        };

        let installed = InstalledMcp {
            name: name.to_string(),
            source: source.to_string(),
            command,
            args,
            installed_at: chrono::Utc::now(),
        };

        self.installed.insert(name.to_string(), installed.clone());
        Ok(installed)
    }

    async fn mount(&mut self, name: &str) -> anyhow::Result<MountedMcpInfo> {
        let installed = self.installed.get(name)
            .ok_or_else(|| anyhow::anyhow!("MCP '{}' not installed", name))?
            .clone();

        let mut child = Command::new(&installed.command)
            .args(&installed.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()?;

        let stdin = child.stdin.take().ok_or_else(|| anyhow::anyhow!("No stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow::anyhow!("No stdout"))?;

        let mut mounted = MountedMcp {
            info: MountedMcpInfo {
                name: name.to_string(),
                tools: vec![],
                mounted_at: chrono::Utc::now(),
            },
            child,
            stdin,
            stdout: BufReader::new(stdout),
            request_id: 0,
        };

        mounted.initialize().await?;
        let tools = mounted.list_tools().await?;
        mounted.info.tools = tools;

        let info = mounted.info.clone();
        self.mounted.insert(name.to_string(), Mutex::new(mounted));

        Ok(info)
    }

    async fn unmount(&mut self, name: &str) -> bool {
        if let Some(mounted) = self.mounted.remove(name) {
            let mut m = mounted.lock().await;
            let _ = m.child.kill().await;
            true
        } else {
            false
        }
    }

    async fn call_tool(&self, mcp_name: &str, tool_name: &str, arguments: Value) -> anyhow::Result<Value> {
        let mounted = self.mounted.get(mcp_name)
            .ok_or_else(|| anyhow::anyhow!("MCP '{}' not mounted", mcp_name))?;
        
        let mut m = mounted.lock().await;
        m.call_tool(tool_name, arguments).await
    }

    fn list(&self) -> (Vec<InstalledMcp>, Vec<String>) {
        (self.installed.values().cloned().collect(), self.mounted.keys().cloned().collect())
    }
}

lazy_static! {
    static ref MANAGER: Mutex<McpManager> = Mutex::new(McpManager::new());
}

// ==================== Tools ====================

pub struct McpInstallTool;

impl Tool for McpInstallTool {
    fn name(&self) -> &'static str { "mcp_install" }
    fn description(&self) -> &'static str { "Install an MCP server (npm:, pip:, uvx:, command:)" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name for this MCP"},
                "source": {"type": "string", "description": "Source: npm:pkg, pip:pkg, uvx:pkg, command:..."}
            },
            "required": ["name", "source"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args { name: String, source: String }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let mut manager = MANAGER.lock().await;
            match manager.install(&args.name, &args.source).await {
                Ok(info) => ToolResult::ok(format!("Installed '{}' from {}", info.name, info.source)),
                Err(e) => ToolResult::err(e.to_string()),
            }
        })
    }
}

pub struct McpMountTool;

impl Tool for McpMountTool {
    fn name(&self) -> &'static str { "mcp_mount" }
    fn description(&self) -> &'static str { "Mount an installed MCP, enabling its tools" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "MCP name to mount"}
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
            
            let mut manager = MANAGER.lock().await;
            match manager.mount(&args.name).await {
                Ok(info) => {
                    let tools: Vec<_> = info.tools.iter().map(|t| &t.name).collect();
                    ToolResult::ok(format!("Mounted '{}' with {} tools: {:?}", info.name, info.tools.len(), tools))
                }
                Err(e) => ToolResult::err(e.to_string()),
            }
        })
    }
}

pub struct McpUnmountTool;

impl Tool for McpUnmountTool {
    fn name(&self) -> &'static str { "mcp_unmount" }
    fn description(&self) -> &'static str { "Unmount a running MCP" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "MCP name"}
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
            
            let mut manager = MANAGER.lock().await;
            if manager.unmount(&args.name).await {
                ToolResult::ok(format!("Unmounted '{}'", args.name))
            } else {
                ToolResult::err(format!("MCP '{}' not mounted", args.name))
            }
        })
    }
}

pub struct McpListTool;

impl Tool for McpListTool {
    fn name(&self) -> &'static str { "mcp_list" }
    fn description(&self) -> &'static str { "List installed and mounted MCPs" }
    
    fn schema(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    
    fn execute(&self, _args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let manager = MANAGER.lock().await;
            let (installed, mounted) = manager.list();
            
            let mut out = String::new();
            out.push_str("Installed:\n");
            for mcp in &installed {
                let status = if mounted.contains(&mcp.name) { " [mounted]" } else { "" };
                out.push_str(&format!("  {} ({}){}\n", mcp.name, mcp.source, status));
            }
            if installed.is_empty() {
                out.push_str("  (none)\n");
            }
            
            ToolResult::ok(out)
        })
    }
}

pub struct McpCallTool;

impl Tool for McpCallTool {
    fn name(&self) -> &'static str { "mcp_call" }
    fn description(&self) -> &'static str { "Call a tool on a mounted MCP" }
    
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "mcp": {"type": "string", "description": "MCP name"},
                "tool": {"type": "string", "description": "Tool name"},
                "arguments": {"type": "object", "description": "Tool arguments"}
            },
            "required": ["mcp", "tool"]
        })
    }
    
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct Args {
                mcp: String,
                tool: String,
                #[serde(default)]
                arguments: Value,
            }
            
            let args: Args = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };
            
            let manager = MANAGER.lock().await;
            match manager.call_tool(&args.mcp, &args.tool, args.arguments).await {
                Ok(result) => ToolResult::ok(serde_json::to_string_pretty(&result).unwrap()),
                Err(e) => ToolResult::err(e.to_string()),
            }
        })
    }
}
