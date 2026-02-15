//! ash-mcp - MCP Server using the same tools as ash CLI
//!
//! Provides tools over MCP protocol (JSON-RPC over stdio or HTTP)

use ash::{Tool, ToolResult};
use ash::tools;
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, Write};

#[derive(Parser)]
#[command(name = "ash-mcp")]
#[command(about = "MCP Server - same tools as ash CLI")]
struct Cli {
    /// Transport type
    #[arg(long, default_value = "stdio")]
    transport: Transport,
    
    /// Port for HTTP transport
    #[arg(long, default_value = "8080")]
    port: u16,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum Transport {
    Stdio,
    Http,
}

#[derive(Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    
    match cli.transport {
        Transport::Stdio => run_stdio(),
        Transport::Http => {
            eprintln!("HTTP transport not yet implemented");
            std::process::exit(1);
        }
    }
}

fn run_stdio() -> anyhow::Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    
    let rt = tokio::runtime::Runtime::new()?;
    
    for line in stdin.lock().lines() {
        let line = line?;
        if line.is_empty() { continue; }
        
        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let response = JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Value::Null,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("Parse error: {e}"),
                    }),
                };
                writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
                continue;
            }
        };
        
        let response = rt.block_on(handle_request(request));
        writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
        stdout.flush()?;
    }
    
    Ok(())
}

async fn handle_request(request: JsonRpcRequest) -> JsonRpcResponse {
    let id = request.id.unwrap_or(Value::Null);
    
    match request.method.as_str() {
        "initialize" => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "ash-mcp",
                    "version": env!("CARGO_PKG_VERSION")
                },
                "capabilities": {
                    "tools": {}
                }
            })),
            error: None,
        },
        
        "notifications/initialized" => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(json!({})),
            error: None,
        },
        
        "tools/list" => {
            let tools_list: Vec<Value> = tools::all_tools()
                .iter()
                .map(|t| json!({
                    "name": t.name(),
                    "description": t.description(),
                    "inputSchema": t.schema()
                }))
                .collect();
            
            JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: Some(json!({"tools": tools_list})),
                error: None,
            }
        }
        
        "tools/call" => {
            let name = request.params.get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let arguments = request.params.get("arguments")
                .cloned()
                .unwrap_or(json!({}));
            
            match tools::find_tool(name) {
                Some(tool) => {
                    let result = tool.execute(arguments).await;
                    
                    JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id,
                        result: Some(json!({
                            "content": [{
                                "type": "text",
                                "text": if result.success { 
                                    result.output 
                                } else { 
                                    result.error.unwrap_or_default() 
                                }
                            }],
                            "isError": !result.success
                        })),
                        error: None,
                    }
                }
                None => JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32601,
                        message: format!("Unknown tool: {name}"),
                    }),
                },
            }
        }
        
        _ => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: format!("Method not found: {}", request.method),
            }),
        },
    }
}
