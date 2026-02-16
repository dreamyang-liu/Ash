//! MCP Server implementation
//!
//! Speaks MCP protocol over stdio. All tools are executed directly.
//! This is the simplest way to expose ash tools to any MCP client.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

use crate::Tool;

/// MCP JSON-RPC request
#[derive(Debug, Deserialize)]
struct McpRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

/// MCP JSON-RPC response
#[derive(Debug, Serialize)]
struct McpResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<McpError>,
}

#[derive(Debug, Serialize)]
struct McpError {
    code: i32,
    message: String,
}

impl McpResponse {
    fn ok(id: Value, result: Value) -> Self {
        Self { jsonrpc: "2.0".into(), id, result: Some(result), error: None }
    }
    
    fn err(id: Value, code: i32, message: impl Into<String>) -> Self {
        Self { 
            jsonrpc: "2.0".into(), 
            id, 
            result: None, 
            error: Some(McpError { code, message: message.into() }) 
        }
    }
}

/// MCP Server - handles stdio communication
pub struct McpServer {
    tools: Vec<Box<dyn Tool>>,
}

impl McpServer {
    pub fn new(tools: Vec<Box<dyn Tool>>) -> Self {
        Self { tools }
    }
    
    /// Run the server (blocking, reads from stdin)
    pub async fn run(&self) -> io::Result<()> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        
        for line in stdin.lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            
            let response = self.handle_request(&line).await;
            let output = serde_json::to_string(&response)?;
            writeln!(stdout, "{}", output)?;
            stdout.flush()?;
        }
        
        Ok(())
    }
    
    async fn handle_request(&self, line: &str) -> McpResponse {
        let req: McpRequest = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => return McpResponse::err(Value::Null, -32700, format!("Parse error: {}", e)),
        };
        
        let id = req.id.unwrap_or(Value::Null);
        
        match req.method.as_str() {
            "initialize" => self.handle_initialize(id, req.params),
            "tools/list" => self.handle_tools_list(id),
            "tools/call" => self.handle_tools_call(id, req.params).await,
            "notifications/initialized" => McpResponse::ok(id, json!({})),
            _ => McpResponse::err(id, -32601, format!("Method not found: {}", req.method)),
        }
    }
    
    fn handle_initialize(&self, id: Value, _params: Value) -> McpResponse {
        McpResponse::ok(id, json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "ash",
                "version": env!("CARGO_PKG_VERSION")
            }
        }))
    }
    
    fn handle_tools_list(&self, id: Value) -> McpResponse {
        let tools: Vec<Value> = self.tools.iter().map(|t| {
            json!({
                "name": t.name(),
                "description": t.description(),
                "inputSchema": t.schema()
            })
        }).collect();
        
        McpResponse::ok(id, json!({ "tools": tools }))
    }
    
    async fn handle_tools_call(&self, id: Value, params: Value) -> McpResponse {
        let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
        let args = params.get("arguments").cloned().unwrap_or(json!({}));
        
        // Find tool
        let tool = match self.tools.iter().find(|t| t.name() == name) {
            Some(t) => t,
            None => return McpResponse::err(id, -32602, format!("Unknown tool: {}", name)),
        };
        
        // Execute
        let result = tool.execute(args).await;
        
        // Convert to MCP format
        let content = vec![json!({
            "type": "text",
            "text": if result.success { result.output } else { result.error.unwrap_or_default() }
        })];
        
        McpResponse::ok(id, json!({
            "content": content,
            "isError": !result.success
        }))
    }
}
