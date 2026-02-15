//! ash CLI

use ash::{Tool, ToolResult};
use ash::tools;
use clap::{Parser, Subcommand};
use serde_json::Value;

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
    /// Read file with line numbers
    View {
        /// File path
        file_path: String,
        /// Start line (1-indexed)
        #[arg(short = 'n', long, default_value = "1")]
        offset: usize,
        /// Max lines to return
        #[arg(short, long, default_value = "100")]
        limit: usize,
    },
    
    /// Search for pattern in files (ripgrep)
    Grep {
        /// Regex pattern
        pattern: String,
        /// Search path
        #[arg(default_value = ".")]
        path: String,
        /// File glob (e.g., *.py)
        #[arg(short, long)]
        include: Option<String>,
        /// Max results
        #[arg(short, long, default_value = "100")]
        limit: usize,
    },
    
    /// Edit file
    Edit {
        #[command(subcommand)]
        op: EditOp,
    },
    
    /// Execute shell command
    Run {
        /// Shell command
        command: String,
        /// Timeout in seconds
        #[arg(short, long, default_value = "300")]
        timeout: u64,
    },
    
    /// Git status
    #[command(name = "git-status")]
    GitStatus {
        /// Short format
        #[arg(long)]
        short: bool,
    },
    
    /// Git diff
    #[command(name = "git-diff")]
    GitDiff {
        /// Compare staged changes
        #[arg(long)]
        staged: bool,
        /// Specific paths
        paths: Vec<String>,
    },
    
    /// Git log
    #[command(name = "git-log")]
    GitLog {
        /// Number of commits
        #[arg(short = 'n', long, default_value = "10")]
        count: usize,
        /// One line format
        #[arg(long)]
        oneline: bool,
    },
    
    /// Save to clipboard
    Clip {
        /// Content to save
        content: Option<String>,
        /// File path with optional :start-end
        #[arg(short, long)]
        file: Option<String>,
        /// Clip name
        #[arg(short, long)]
        name: Option<String>,
        /// Source reference
        #[arg(short, long)]
        source: Option<String>,
    },
    
    /// Retrieve from clipboard
    Paste {
        /// Clip name (latest if omitted)
        name: Option<String>,
    },
    
    /// List clipboard entries
    Clips,
    
    /// Clear clipboard
    #[command(name = "clips-clear")]
    ClipsClear {
        /// Specific clip to remove
        name: Option<String>,
    },
    
    /// Session management
    Session {
        #[command(subcommand)]
        op: SessionOp,
    },
    
    /// Configure endpoints
    Config {
        /// Control plane URL
        #[arg(long)]
        control_plane_url: Option<String>,
        /// MCP gateway URL
        #[arg(long)]
        gateway_url: Option<String>,
    },
    
    /// List all available tools
    Tools,
}

#[derive(Subcommand)]
enum EditOp {
    /// View file with line range
    View {
        path: String,
        #[arg(long, default_value = "1")]
        start: i64,
        #[arg(long, default_value = "-1")]
        end: i64,
    },
    /// Replace text (must be unique)
    Replace {
        path: String,
        #[arg(long)]
        old: String,
        #[arg(long)]
        new: String,
    },
    /// Insert text after line
    Insert {
        path: String,
        #[arg(long)]
        line: i64,
        #[arg(long)]
        text: String,
    },
    /// Create new file
    Create {
        path: String,
        content: String,
    },
}

#[derive(Subcommand)]
enum SessionOp {
    /// Spawn a new sandbox
    Create {
        /// Custom name
        #[arg(short, long)]
        name: Option<String>,
        /// Docker image
        #[arg(short, long)]
        image: Option<String>,
        /// Container ports (can repeat)
        #[arg(short, long)]
        port: Vec<i32>,
        /// Environment variables (KEY=VALUE, can repeat)
        #[arg(short, long)]
        env: Vec<String>,
        /// CPU request (e.g., 100m)
        #[arg(long)]
        cpu_request: Option<String>,
        /// CPU limit (e.g., 1)
        #[arg(long)]
        cpu_limit: Option<String>,
        /// Memory request (e.g., 256Mi)
        #[arg(long)]
        memory_request: Option<String>,
        /// Memory limit (e.g., 1Gi)
        #[arg(long)]
        memory_limit: Option<String>,
        /// Node selector (KEY=VALUE, can repeat)
        #[arg(long)]
        node_selector: Vec<String>,
    },
    /// List active sessions
    List,
    /// Destroy a sandbox
    Destroy {
        /// Session ID (uuid)
        session_id: String,
    },
}

fn parse_key_value(items: &[String]) -> std::collections::HashMap<String, String> {
    items.iter()
        .filter_map(|s| {
            let parts: Vec<&str> = s.splitn(2, '=').collect();
            if parts.len() == 2 {
                Some((parts[0].to_string(), parts[1].to_string()))
            } else {
                None
            }
        })
        .collect()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let session_id = cli.session.clone();
    
    let result = match cli.command {
        Commands::View { file_path, offset, limit } => {
            tools::ViewTool.execute(serde_json::json!({
                "file_path": file_path,
                "offset": offset,
                "limit": limit,
                "session_id": session_id
            })).await
        }
        
        Commands::Grep { pattern, path, include, limit } => {
            tools::GrepTool.execute(serde_json::json!({
                "pattern": pattern,
                "path": path,
                "include": include,
                "limit": limit,
                "session_id": session_id
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
            if let Some(sid) = &session_id {
                args["session_id"] = serde_json::json!(sid);
            }
            tools::EditTool.execute(args).await
        }
        
        Commands::Run { command, timeout } => {
            tools::ShellTool.execute(serde_json::json!({
                "command": command,
                "timeout_secs": timeout,
                "session_id": session_id
            })).await
        }
        
        Commands::GitStatus { short } => {
            tools::GitStatusTool.execute(serde_json::json!({
                "short": short,
                "session_id": session_id
            })).await
        }
        
        Commands::GitDiff { staged, paths } => {
            tools::GitDiffTool.execute(serde_json::json!({
                "staged": staged,
                "paths": paths,
                "session_id": session_id
            })).await
        }
        
        Commands::GitLog { count, oneline } => {
            tools::GitLogTool.execute(serde_json::json!({
                "count": count,
                "oneline": oneline,
                "session_id": session_id
            })).await
        }
        
        Commands::Clip { content, file, name, source } => {
            tools::ClipTool.execute(serde_json::json!({
                "content": content, "file": file, "name": name, "source": source
            })).await
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
        
        Commands::Session { op } => {
            match op {
                SessionOp::Create { name, image, port, env, cpu_request, cpu_limit, memory_request, memory_limit, node_selector } => {
                    let env_map = parse_key_value(&env);
                    let node_sel = parse_key_value(&node_selector);
                    
                    let mut args = serde_json::json!({
                        "name": name,
                        "image": image,
                        "ports": port,
                        "env": env_map,
                        "node_selector": node_sel,
                    });
                    
                    // Build resources
                    let mut resources = serde_json::json!({});
                    if cpu_request.is_some() || memory_request.is_some() {
                        let mut requests = serde_json::json!({});
                        if let Some(v) = cpu_request { requests["cpu"] = serde_json::json!(v); }
                        if let Some(v) = memory_request { requests["memory"] = serde_json::json!(v); }
                        resources["requests"] = requests;
                    }
                    if cpu_limit.is_some() || memory_limit.is_some() {
                        let mut limits = serde_json::json!({});
                        if let Some(v) = cpu_limit { limits["cpu"] = serde_json::json!(v); }
                        if let Some(v) = memory_limit { limits["memory"] = serde_json::json!(v); }
                        resources["limits"] = limits;
                    }
                    if !resources.as_object().unwrap().is_empty() {
                        args["resources"] = resources;
                    }
                    
                    tools::SessionCreateTool.execute(args).await
                }
                SessionOp::List => {
                    tools::SessionListTool.execute(serde_json::json!({})).await
                }
                SessionOp::Destroy { session_id } => {
                    tools::SessionDestroyTool.execute(serde_json::json!({
                        "session_id": session_id
                    })).await
                }
            }
        }
        
        Commands::Config { control_plane_url, gateway_url } => {
            let config = tools::session::get_config().await;
            if control_plane_url.is_none() && gateway_url.is_none() {
                // Show current config
                return Ok(println!("control_plane_url: {}\ngateway_url: {}", 
                    config.control_plane_url, config.gateway_url));
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
        
        Commands::Tools => {
            for tool in tools::all_tools() {
                println!("{}: {}", tool.name(), tool.description());
            }
            return Ok(());
        }
    };
    
    match cli.output {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
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
