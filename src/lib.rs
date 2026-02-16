//! Core tool definitions - shared between CLI and MCP server
//!
//! Each tool is defined once, used in both binaries.

use serde::Serialize;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;

pub mod backend;
pub mod daemon;
pub mod gateway;
pub mod mcp;
pub mod tools;

/// Tool execution result
#[derive(Debug, Serialize)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ToolResult {
    pub fn ok(output: impl Into<String>) -> Self {
        Self { success: true, output: output.into(), error: None }
    }
    
    pub fn err(error: impl Into<String>) -> Self {
        Self { success: false, output: String::new(), error: Some(error.into()) }
    }
}

/// Boxed future for dyn compatibility
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Trait for tools - dyn-compatible using BoxFuture
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn schema(&self) -> Value;
    fn execute(&self, args: Value) -> BoxFuture<'_, ToolResult>;
}
