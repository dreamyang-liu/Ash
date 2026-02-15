//! ash CLI

use ash::{Tool, ToolResult};
use ash::tools;
use clap::{Parser, Subcommand};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Parser)]
#[command(name = "ash")]
#[command(about = "Code Agent CLI & MCP Server")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Output format
    #[arg(short, long, default_value = "text", global = true)]
    output: OutputFormat,
    
    /// Session ID - all tool calls will execute in this sandbox
    #[arg(long, global = true)]
    session: Option<String>,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum OutputFormat { Text, Json }

#[derive(Subcommand)]
enum Commands {
    // ==================== File Operations ====================
    
    /// Read file with line numbers
    View {
        file_path: String,
        #[arg(short = 'n', long, default_value = "1")]
        offset: usize,
        #[arg(short, long, default_value = "100")]
        limit: usize,
    },
    
    /// Search for pattern in files (ripgrep)
    Grep {
        pattern: String,
        #[arg(default_value = ".")]
        path: String,
        #[arg(short, long)]
        include: Option<String>,
        #[arg(short, long, default_value = "100")]
        limit: usize,
    },
    
    /// Edit file
    Edit {
        #[command(subcommand)]
        op: EditOp,
    },
    
    /// Find files by name pattern
    Find {
        /// Glob pattern (e.g., *.py, test_*)
        pattern: String,
        #[arg(default_value = ".")]
        path: String,
        #[arg(short, long)]
        max_depth: Option<usize>,
        #[arg(short, long, default_value = "100")]
        limit: usize,
    },
    
    /// Show directory tree
    Tree {
        #[arg(default_value = ".")]
        path: String,
        #[arg(short, long, default_value = "3")]
        max_depth: usize,
        #[arg(long)]
        show_hidden: bool,
    },
    
    /// Compare two files
    Diff {
        file1: String,
        file2: String,
        #[arg(short, long, default_value = "3")]
        context: usize,
    },
    
    /// Apply unified diff patch
    Patch {
        /// Patch content (or use stdin)
        patch: String,
        #[arg(short, long)]
        path: Option<String>,
        #[arg(long)]
        dry_run: bool,
    },
    
    /// Get file type and info
    FileInfo {
        path: String,
    },
    
    /// HTTP GET request
    Fetch {
        url: String,
        #[arg(short, long, default_value = "30")]
        timeout: u64,
    },
    
    /// Undo last file edit
    Undo {
        /// Specific file to undo
        path: Option<String>,
        /// List undo history
        #[arg(long)]
        list: bool,
    },
    
    // ==================== Shell ====================
    
    /// Execute shell command (sync)
    Run {
        command: String,
        #[arg(short, long, default_value = "300")]
        timeout: u64,
        #[arg(long)]
        tail: Option<usize>,
    },
    
    /// Async terminal management
    Terminal {
        #[command(subcommand)]
        op: TerminalOp,
    },
    
    // ==================== Git ====================
    
    /// Git status
    #[command(name = "git-status")]
    GitStatus {
        #[arg(long)]
        short: bool,
    },
    
    /// Git diff
    #[command(name = "git-diff")]
    GitDiff {
        #[arg(long)]
        staged: bool,
        paths: Vec<String>,
    },
    
    /// Git log
    #[command(name = "git-log")]
    GitLog {
        #[arg(short = 'n', long, default_value = "10")]
        count: usize,
        #[arg(long)]
        oneline: bool,
    },
    
    /// Git add (stage files)
    #[command(name = "git-add")]
    GitAdd {
        /// Files to stage
        paths: Vec<String>,
        /// Stage all changes (-A)
        #[arg(short, long)]
        all: bool,
    },
    
    /// Git commit
    #[command(name = "git-commit")]
    GitCommit {
        /// Commit message
        #[arg(short, long)]
        message: String,
        /// Stage all and commit (-a)
        #[arg(short, long)]
        all: bool,
    },
    
    // ==================== Clipboard ====================
    
    /// Save to clipboard
    Clip {
        content: Option<String>,
        #[arg(short, long)]
        file: Option<String>,
        #[arg(short, long)]
        name: Option<String>,
        #[arg(short, long)]
        source: Option<String>,
    },
    
    /// Retrieve from clipboard
    Paste { name: Option<String> },
    
    /// List clipboard entries
    Clips,
    
    /// Clear clipboard
    #[command(name = "clips-clear")]
    ClipsClear { name: Option<String> },
    
    // ==================== Session/Sandbox ====================
    
    /// Session/sandbox management
    Session {
        #[command(subcommand)]
        op: SessionOp,
    },
    
    // ==================== MCP ====================
    
    /// MCP server management
    Mcp {
        #[command(subcommand)]
        op: McpOp,
    },
    
    // ==================== Config ====================
    
    /// Configure endpoints
    Config {
        #[arg(long)]
        control_plane_url: Option<String>,
        #[arg(long)]
        gateway_url: Option<String>,
    },
    
    // ==================== Events ====================
    
    /// Events management
    Events {
        #[command(subcommand)]
        op: EventsOp,
    },
    
    // ==================== Custom Tools ====================
    
    /// Custom tools management
    #[command(name = "custom-tool")]
    CustomTool {
        #[command(subcommand)]
        op: CustomToolOp,
    },
    
    /// List all available tools
    Tools,
}

#[derive(Subcommand)]
enum EditOp {
    View {
        path: String,
        #[arg(long, default_value = "1")]
        start: i64,
        #[arg(long, default_value = "-1")]
        end: i64,
    },
    Replace {
        path: String,
        #[arg(long)]
        old: String,
        #[arg(long)]
        new: String,
    },
    Insert {
        path: String,
        #[arg(long)]
        line: i64,
        #[arg(long)]
        text: String,
    },
    Create {
        path: String,
        content: String,
    },
}

#[derive(Subcommand)]
enum TerminalOp {
    /// Start async process
    Start {
        command: String,
        #[arg(short, long)]
        workdir: Option<String>,
        #[arg(short, long)]
        env: Vec<String>,
    },
    /// Get output from handle
    Output {
        handle: String,
        #[arg(long)]
        tail: Option<usize>,
    },
    /// Kill process
    Kill { handle: String },
    /// List processes
    List,
    /// Remove completed process
    Remove { handle: String },
}

#[derive(Subcommand)]
enum SessionOp {
    Create {
        #[arg(short, long)]
        name: Option<String>,
        #[arg(short, long)]
        image: Option<String>,
        #[arg(short, long)]
        port: Vec<i32>,
        #[arg(short, long)]
        env: Vec<String>,
        #[arg(long)]
        cpu_request: Option<String>,
        #[arg(long)]
        cpu_limit: Option<String>,
        #[arg(long)]
        memory_request: Option<String>,
        #[arg(long)]
        memory_limit: Option<String>,
        #[arg(long)]
        node_selector: Vec<String>,
    },
    List,
    Destroy { session_id: String },
}

#[derive(Subcommand)]
enum McpOp {
    /// Install MCP (npm:, pip:, uvx:, command:)
    Install { name: String, source: String },
    /// Mount installed MCP
    Mount { name: String },
    /// Unmount MCP
    Unmount { name: String },
    /// List MCPs
    List,
    /// Call tool on mounted MCP
    Call {
        mcp: String,
        tool: String,
        #[arg(long, default_value = "{}")]
        args: String,
    },
}

#[derive(Subcommand)]
enum EventsOp {
    /// Subscribe to event types
    Subscribe {
        /// Event types (process_complete, file_change, error, custom)
        events: Vec<String>,
        /// Unsubscribe instead
        #[arg(short, long)]
        unsubscribe: bool,
    },
    /// Poll pending events
    Poll {
        #[arg(short, long, default_value = "10")]
        limit: usize,
        /// Peek without removing
        #[arg(long)]
        peek: bool,
    },
    /// Push a custom event
    Push {
        kind: String,
        #[arg(short, long, default_value = "llm")]
        source: String,
        #[arg(short, long, default_value = "{}")]
        data: String,
    },
}

#[derive(Subcommand)]
enum CustomToolOp {
    /// Register a custom tool
    Register {
        name: String,
        #[arg(short, long)]
        description: String,
        #[arg(short, long)]
        script: String,
        #[arg(long, default_value = "{}")]
        schema: String,
    },
    /// List custom tools
    List,
    /// Call a custom tool
    Call {
        name: String,
        #[arg(short, long, default_value = "{}")]
        args: String,
    },
    /// Remove a custom tool
    Remove { name: String },
}

fn parse_key_value(items: &[String]) -> HashMap<String, String> {
    items.iter()
        .filter_map(|s| {
            let parts: Vec<&str> = s.splitn(2, '=').collect();
            if parts.len() == 2 { Some((parts[0].to_string(), parts[1].to_string())) } else { None }
        })
        .collect()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let session_id = cli.session.clone();
    
    let result = match cli.command {
        // ==================== File Operations ====================
        
        Commands::View { file_path, offset, limit } => {
            tools::ViewTool.execute(serde_json::json!({
                "file_path": file_path, "offset": offset, "limit": limit, "session_id": session_id
            })).await
        }
        
        Commands::Grep { pattern, path, include, limit } => {
            tools::GrepTool.execute(serde_json::json!({
                "pattern": pattern, "path": path, "include": include, "limit": limit, "session_id": session_id
            })).await
        }
        
        Commands::Edit { op } => {
            let mut args = match op {
                EditOp::View { path, start, end } => serde_json::json!({
                    "command": "view", "path": path, "view_range": [start, end]
                }),
                EditOp::Replace { path, old, new } => serde_json::json!({
                    "command": "str_replace", "path": path, "old_str": old, "new_str": new
                }),
                EditOp::Insert { path, line, text } => serde_json::json!({
                    "command": "insert", "path": path, "insert_line": line, "insert_text": text
                }),
                EditOp::Create { path, content } => serde_json::json!({
                    "command": "create", "path": path, "file_text": content
                }),
            };
            if let Some(sid) = &session_id { args["session_id"] = serde_json::json!(sid); }
            tools::EditTool.execute(args).await
        }
        
        Commands::Find { pattern, path, max_depth, limit } => {
            tools::FindFilesTool.execute(serde_json::json!({
                "pattern": pattern, "path": path, "max_depth": max_depth, "limit": limit
            })).await
        }
        
        Commands::Tree { path, max_depth, show_hidden } => {
            tools::TreeTool.execute(serde_json::json!({
                "path": path, "max_depth": max_depth, "show_hidden": show_hidden
            })).await
        }
        
        Commands::Diff { file1, file2, context } => {
            tools::DiffFilesTool.execute(serde_json::json!({
                "file1": file1, "file2": file2, "context": context
            })).await
        }
        
        Commands::Patch { patch, path, dry_run } => {
            tools::PatchApplyTool.execute(serde_json::json!({
                "patch": patch, "path": path, "dry_run": dry_run
            })).await
        }
        
        Commands::FileInfo { path } => {
            tools::FileInfoTool.execute(serde_json::json!({"path": path})).await
        }
        
        Commands::Fetch { url, timeout } => {
            tools::HttpFetchTool.execute(serde_json::json!({
                "url": url, "timeout_secs": timeout
            })).await
        }
        
        Commands::Undo { path, list } => {
            tools::UndoTool.execute(serde_json::json!({"path": path, "list": list})).await
        }
        
        // ==================== Shell ====================
        
        Commands::Run { command, timeout, tail } => {
            tools::ShellTool.execute(serde_json::json!({
                "command": command, "timeout_secs": timeout, "session_id": session_id, "tail_lines": tail
            })).await
        }
        
        Commands::Terminal { op } => {
            match op {
                TerminalOp::Start { command, workdir, env } => {
                    let env_map = parse_key_value(&env);
                    tools::TerminalRunAsyncTool.execute(serde_json::json!({
                        "command": command, "working_dir": workdir, "env": env_map
                    })).await
                }
                TerminalOp::Output { handle, tail } => {
                    tools::TerminalGetOutputTool.execute(serde_json::json!({"handle": handle, "tail": tail})).await
                }
                TerminalOp::Kill { handle } => {
                    tools::TerminalKillTool.execute(serde_json::json!({"handle": handle})).await
                }
                TerminalOp::List => {
                    tools::TerminalListTool.execute(serde_json::json!({})).await
                }
                TerminalOp::Remove { handle } => {
                    tools::TerminalRemoveTool.execute(serde_json::json!({"handle": handle})).await
                }
            }
        }
        
        // ==================== Git ====================
        
        Commands::GitStatus { short } => {
            tools::GitStatusTool.execute(serde_json::json!({"short": short, "session_id": session_id})).await
        }
        
        Commands::GitDiff { staged, paths } => {
            tools::GitDiffTool.execute(serde_json::json!({"staged": staged, "paths": paths, "session_id": session_id})).await
        }
        
        Commands::GitLog { count, oneline } => {
            tools::GitLogTool.execute(serde_json::json!({"count": count, "oneline": oneline, "session_id": session_id})).await
        }
        
        Commands::GitAdd { paths, all } => {
            tools::GitAddTool.execute(serde_json::json!({"paths": paths, "all": all, "session_id": session_id})).await
        }
        
        Commands::GitCommit { message, all } => {
            tools::GitCommitTool.execute(serde_json::json!({"message": message, "all": all, "session_id": session_id})).await
        }
        
        // ==================== Clipboard ====================
        
        Commands::Clip { content, file, name, source } => {
            tools::ClipTool.execute(serde_json::json!({"content": content, "file": file, "name": name, "source": source})).await
        }
        
        Commands::Paste { name } => {
            tools::PasteTool.execute(serde_json::json!({"name": name})).await
        }
        
        Commands::Clips => {
            tools::ClipsTool.execute(serde_json::json!({})).await
        }
        
        Commands::ClipsClear { name } => {
            tools::ClearClipsTool.execute(serde_json::json!({"name": name})).await
        }
        
        // ==================== Session ====================
        
        Commands::Session { op } => {
            match op {
                SessionOp::Create { name, image, port, env, cpu_request, cpu_limit, memory_request, memory_limit, node_selector } => {
                    let env_map = parse_key_value(&env);
                    let node_sel = parse_key_value(&node_selector);
                    
                    let mut args = serde_json::json!({
                        "name": name, "image": image, "ports": port, "env": env_map, "node_selector": node_sel,
                    });
                    
                    let mut resources = serde_json::json!({});
                    if cpu_request.is_some() || memory_request.is_some() {
                        let mut req = serde_json::json!({});
                        if let Some(v) = cpu_request { req["cpu"] = serde_json::json!(v); }
                        if let Some(v) = memory_request { req["memory"] = serde_json::json!(v); }
                        resources["requests"] = req;
                    }
                    if cpu_limit.is_some() || memory_limit.is_some() {
                        let mut lim = serde_json::json!({});
                        if let Some(v) = cpu_limit { lim["cpu"] = serde_json::json!(v); }
                        if let Some(v) = memory_limit { lim["memory"] = serde_json::json!(v); }
                        resources["limits"] = lim;
                    }
                    if !resources.as_object().unwrap().is_empty() { args["resources"] = resources; }
                    
                    tools::SessionCreateTool.execute(args).await
                }
                SessionOp::List => tools::SessionListTool.execute(serde_json::json!({})).await,
                SessionOp::Destroy { session_id } => {
                    tools::SessionDestroyTool.execute(serde_json::json!({"session_id": session_id})).await
                }
            }
        }
        
        // ==================== MCP ====================
        
        Commands::Mcp { op } => {
            match op {
                McpOp::Install { name, source } => {
                    tools::McpInstallTool.execute(serde_json::json!({"name": name, "source": source})).await
                }
                McpOp::Mount { name } => {
                    tools::McpMountTool.execute(serde_json::json!({"name": name})).await
                }
                McpOp::Unmount { name } => {
                    tools::McpUnmountTool.execute(serde_json::json!({"name": name})).await
                }
                McpOp::List => {
                    tools::McpListTool.execute(serde_json::json!({})).await
                }
                McpOp::Call { mcp, tool, args } => {
                    let arguments: serde_json::Value = serde_json::from_str(&args).unwrap_or(serde_json::json!({}));
                    tools::McpCallTool.execute(serde_json::json!({"mcp": mcp, "tool": tool, "arguments": arguments})).await
                }
            }
        }
        
        // ==================== Config ====================
        
        Commands::Config { control_plane_url, gateway_url } => {
            let config = tools::session::get_config().await;
            if control_plane_url.is_none() && gateway_url.is_none() {
                return Ok(println!("control_plane_url: {}\ngateway_url: {}", config.control_plane_url, config.gateway_url));
            }
            let new_config = tools::session::ClientConfig {
                control_plane_url: control_plane_url.unwrap_or(config.control_plane_url),
                gateway_url: gateway_url.unwrap_or(config.gateway_url),
                ..config
            };
            tools::session::set_config(new_config.clone()).await;
            return Ok(println!("Updated config:\ncontrol_plane_url: {}\ngateway_url: {}", 
                new_config.control_plane_url, new_config.gateway_url));
        }
        
        // ==================== Events ====================
        
        Commands::Events { op } => {
            match op {
                EventsOp::Subscribe { events, unsubscribe } => {
                    tools::EventsSubscribeTool.execute(serde_json::json!({
                        "events": events, "unsubscribe": unsubscribe
                    })).await
                }
                EventsOp::Poll { limit, peek } => {
                    tools::EventsPollTool.execute(serde_json::json!({
                        "limit": limit, "peek": peek
                    })).await
                }
                EventsOp::Push { kind, source, data } => {
                    let data_val: serde_json::Value = serde_json::from_str(&data).unwrap_or(serde_json::json!({}));
                    tools::EventsPushTool.execute(serde_json::json!({
                        "kind": kind, "source": source, "data": data_val
                    })).await
                }
            }
        }
        
        // ==================== Custom Tools ====================
        
        Commands::CustomTool { op } => {
            match op {
                CustomToolOp::Register { name, description, script, schema } => {
                    let schema_val: serde_json::Value = serde_json::from_str(&schema).unwrap_or(serde_json::json!({}));
                    tools::ToolRegisterTool.execute(serde_json::json!({
                        "name": name, "description": description, "script": script, "schema": schema_val
                    })).await
                }
                CustomToolOp::List => {
                    tools::ToolListCustomTool.execute(serde_json::json!({})).await
                }
                CustomToolOp::Call { name, args } => {
                    let arguments: serde_json::Value = serde_json::from_str(&args).unwrap_or(serde_json::json!({}));
                    tools::ToolCallCustomTool.execute(serde_json::json!({
                        "name": name, "arguments": arguments
                    })).await
                }
                CustomToolOp::Remove { name } => {
                    tools::ToolRemoveCustomTool.execute(serde_json::json!({"name": name})).await
                }
            }
        }
        
        Commands::Tools => {
            for tool in tools::all_tools() {
                println!("{}: {}", tool.name(), tool.description());
            }
            return Ok(());
        }
    };
    
    match cli.output {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&result)?),
        OutputFormat::Text => {
            if result.success {
                print!("{}", result.output);
                if !result.output.ends_with('\n') { println!(); }
            } else {
                eprintln!("Error: {}", result.error.unwrap_or_default());
                std::process::exit(1);
            }
        }
    }
    
    Ok(())
}
