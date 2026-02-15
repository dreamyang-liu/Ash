//! ash - Sandbox Management & MCP CLI
//!
//! Code agent tools inspired by Claude Code, Codex, SWE-Agent, Aider.
//! MCP tool arguments directly mapped to CLI flags.

use clap::{Parser, Subcommand, Args};
use serde_json::Value;

mod client;
mod config;

use client::AshClient;
use crate::config::Config;

#[derive(Parser)]
#[command(name = "ash")]
#[command(about = "Sandbox Management & MCP CLI for AI Agents")]
#[command(version)]
#[command(long_about = r#"
ash - Autonomous Sandbox Host CLI

CODE AGENT WORKFLOW:
  ash create                         # Create sandbox
  ash view /path/file.py             # View with line numbers  
  ash view /path/file.py:100-150     # View specific lines
  ash grep "pattern" src/            # Search code
  ash edit /path replace ...         # Make edits
  ash apply-patch "..."              # Apply unified diff
  ash run "pytest"                   # Run commands
  ash git-diff                       # See changes
  ash destroy                        # Cleanup

ENVIRONMENT:
  ASH_CONTROL_PLANE_URL   Control plane URL
  ASH_GATEWAY_URL         MCP gateway URL  
  ASH_SESSION             Active session ID
"#)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(long, env = "ASH_CONTROL_PLANE_URL", global = true)]
    control_plane: Option<String>,

    #[arg(long, env = "ASH_GATEWAY_URL", global = true)]
    gateway: Option<String>,

    #[arg(short, long, env = "ASH_SESSION", global = true)]
    session: Option<String>,

    #[arg(short, long, default_value = "text", global = true)]
    output: OutputFormat,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum OutputFormat { Text, Json }

// =============================================================================
// Tool Argument Structs - Direct CLI to MCP mapping
// =============================================================================

// --- Sandbox ---
#[derive(Args, Debug)]
struct CreateArgs {
    #[arg(short, long)]
    name: Option<String>,
    #[arg(short, long, default_value = "timemagic/rl-mcp:general-1.7")]
    image: String,
    #[arg(short, long, value_parser = parse_env_var)]
    env: Vec<(String, String)>,
    #[arg(long, default_value = "true")]
    activate: bool,
}

// --- View (like cat -n, with line range) ---
// Inspired by Codex read_file and SWE-Agent view
#[derive(Args, Debug)]
struct ViewArgs {
    /// File path. Supports :START-END suffix (e.g., file.py:100-150)
    path: String,
    /// Start line (1-indexed). Overrides path suffix.
    #[arg(short = 'n', long)]
    offset: Option<usize>,
    /// Max lines to show
    #[arg(short, long, default_value = "100")]
    limit: usize,
}

impl ViewArgs {
    fn to_json(&self) -> Value {
        // Parse path:start-end syntax
        let (path, offset) = if let Some(idx) = self.path.rfind(':') {
            let (p, range) = self.path.split_at(idx);
            let range = &range[1..];
            if let Some(start) = range.split('-').next().and_then(|s| s.parse().ok()) {
                (p.to_string(), Some(start))
            } else {
                (self.path.clone(), None)
            }
        } else {
            (self.path.clone(), None)
        };
        
        let offset = self.offset.or(offset).unwrap_or(1);
        serde_json::json!({
            "file_path": path,
            "offset": offset,
            "limit": self.limit
        })
    }
}

// --- Grep (ripgrep-style search) ---
// Inspired by Codex grep_files
#[derive(Args, Debug)]
struct GrepArgs {
    /// Search pattern (regex)
    pattern: String,
    /// Path to search in
    #[arg(default_value = ".")]
    path: String,
    /// File pattern to include (e.g., "*.py")
    #[arg(short, long)]
    include: Option<String>,
    /// Max results
    #[arg(short, long, default_value = "100")]
    limit: usize,
}

impl GrepArgs {
    fn to_json(&self) -> Value {
        serde_json::json!({
            "pattern": self.pattern,
            "path": self.path,
            "include": self.include,
            "limit": self.limit
        })
    }
}

// --- Edit (str_replace, insert, create) ---
// Inspired by Anthropic text_editor
#[derive(Args, Debug)]
struct EditArgs {
    /// File path
    path: String,
    #[command(subcommand)]
    op: EditOp,
}

#[derive(Subcommand, Debug)]
enum EditOp {
    /// Replace text (must be unique)
    Replace {
        /// Text to find (exact match, must be unique)
        #[arg(long)]
        old: String,
        /// Replacement text
        #[arg(long)]
        new: String,
    },
    /// Insert text after a line
    Insert {
        /// Line number to insert after (0 = beginning)
        #[arg(long)]
        line: i64,
        /// Text to insert
        #[arg(long)]
        text: String,
    },
    /// Create new file
    Create {
        /// File content
        content: String,
    },
    /// View file at line range
    View {
        /// Start line
        #[arg(long, default_value = "1")]
        start: i64,
        /// End line (-1 for EOF)
        #[arg(long, default_value = "-1")]
        end: i64,
    },
}

impl EditArgs {
    fn to_json(&self) -> Value {
        match &self.op {
            EditOp::Replace { old, new } => serde_json::json!({
                "command": "str_replace",
                "path": self.path,
                "old_str": old,
                "new_str": new
            }),
            EditOp::Insert { line, text } => serde_json::json!({
                "command": "insert",
                "path": self.path,
                "insert_line": line,
                "insert_text": text
            }),
            EditOp::Create { content } => serde_json::json!({
                "command": "create",
                "path": self.path,
                "file_text": content
            }),
            EditOp::View { start, end } => serde_json::json!({
                "command": "view",
                "path": self.path,
                "view_range": [start, end]
            }),
        }
    }
}

// --- Apply Patch (unified diff format) ---
// Inspired by Codex apply_patch
#[derive(Args, Debug)]
struct ApplyPatchArgs {
    /// Unified diff content (or - for stdin)
    patch: String,
}

impl ApplyPatchArgs {
    fn to_json(&self) -> Value {
        serde_json::json!({"input": self.patch})
    }
}

// --- List Dir (tree-like) ---
// Inspired by Codex list_dir
#[derive(Args, Debug)]
struct LsArgs {
    /// Directory path
    #[arg(default_value = ".")]
    path: String,
    /// Max depth
    #[arg(short, long, default_value = "2")]
    depth: usize,
    /// Max entries
    #[arg(short, long, default_value = "50")]
    limit: usize,
}

impl LsArgs {
    fn to_json(&self) -> Value {
        serde_json::json!({
            "dir_path": self.path,
            "depth": self.depth,
            "limit": self.limit
        })
    }
}

// --- Find (glob search for files) ---
#[derive(Args, Debug)]
struct FindArgs {
    /// File name pattern (glob)
    pattern: String,
    /// Search directory
    #[arg(default_value = ".")]
    path: String,
    /// Max depth
    #[arg(short, long)]
    depth: Option<usize>,
}

impl FindArgs {
    fn to_json(&self) -> Value {
        serde_json::json!({
            "pattern": self.pattern,
            "path": self.path,
            "depth": self.depth
        })
    }
}

// --- Run (shell command) ---
#[derive(Args, Debug)]
struct RunArgs {
    /// Command to execute
    command: String,
    /// Timeout in seconds
    #[arg(short, long, default_value = "300")]
    timeout: u64,
}

impl RunArgs {
    fn to_json(&self) -> Value {
        serde_json::json!({
            "command": self.command,
            "timeout_secs": self.timeout
        })
    }
}

// --- Background Run ---
#[derive(Args, Debug)]
struct RunBgArgs {
    /// Command to execute
    command: String,
}

impl RunBgArgs {
    fn to_json(&self) -> Value {
        serde_json::json!({"command": self.command})
    }
}

// --- Git Operations ---
#[derive(Args, Debug)]
struct GitDiffArgs {
    /// Compare with staged
    #[arg(long)]
    staged: bool,
    /// Specific files
    paths: Vec<String>,
}

impl GitDiffArgs {
    fn to_json(&self) -> Value {
        serde_json::json!({
            "staged": self.staged,
            "paths": self.paths
        })
    }
}

#[derive(Args, Debug)]
struct GitLogArgs {
    /// Number of commits
    #[arg(short, long, default_value = "10")]
    count: usize,
    /// One line format
    #[arg(long)]
    oneline: bool,
}

impl GitLogArgs {
    fn to_json(&self) -> Value {
        serde_json::json!({
            "count": self.count,
            "oneline": self.oneline
        })
    }
}

// --- Simple Args ---
#[derive(Args, Debug)]
struct PathArg {
    path: String,
}

#[derive(Args, Debug)]
struct PidArg {
    pid: String,
}

#[derive(Args, Debug)]
struct WriteArgs {
    path: String,
    content: String,
}

// =============================================================================
// Commands
// =============================================================================

#[derive(Subcommand)]
enum Commands {
    // --- Sandbox Lifecycle ---
    /// Create a new sandbox
    Create(#[command(flatten)] CreateArgs),
    /// Destroy sandbox
    Destroy { session_id: Option<String>, #[arg(long)] all: bool },
    /// Health check
    Health,
    /// Ready check
    Ready,

    // --- Session ---
    /// Set active session
    Use { session_id: String },
    /// Show current session
    Current,
    /// List available tools
    Tools,

    // --- Code Agent: View & Navigate ---
    /// View file with line numbers (supports path:start-end syntax)
    View(#[command(flatten)] ViewArgs),
    /// List directory (tree-like)
    Ls(#[command(flatten)] LsArgs),
    /// Find files by pattern
    Find(#[command(flatten)] FindArgs),

    // --- Code Agent: Search ---
    /// Search for pattern in files (ripgrep)
    Grep(#[command(flatten)] GrepArgs),

    // --- Code Agent: Edit ---
    /// Edit file (replace, insert, create, view)
    Edit(#[command(flatten)] EditArgs),
    /// Apply unified diff patch
    #[command(name = "apply-patch")]
    ApplyPatch(#[command(flatten)] ApplyPatchArgs),
    /// Write file (overwrite)
    Write(#[command(flatten)] WriteArgs),

    // --- Terminal ---
    /// Run command synchronously
    Run(#[command(flatten)] RunArgs),
    /// Run command in background
    RunBg(#[command(flatten)] RunBgArgs),
    /// Kill background process
    Kill(#[command(flatten)] PidArg),
    /// Get process output
    Output(#[command(flatten)] PidArg),

    // --- Working Directory ---
    /// Print working directory
    Pwd,
    /// Change working directory
    Cd(#[command(flatten)] PathArg),

    // --- Git ---
    /// Git status
    #[command(name = "git-status")]
    GitStatus { #[arg(long)] short: bool },
    /// Git diff
    #[command(name = "git-diff")]
    GitDiff(#[command(flatten)] GitDiffArgs),
    /// Git log
    #[command(name = "git-log")]
    GitLog(#[command(flatten)] GitLogArgs),
    /// Git add
    #[command(name = "git-add")]
    GitAdd { paths: Vec<String> },
    /// Git commit
    #[command(name = "git-commit")]
    GitCommit { #[arg(short, long)] message: String },
    /// Git reset
    #[command(name = "git-reset")]
    GitReset { #[arg(long)] hard: bool, target: Option<String> },

    // --- Raw MCP ---
    /// Call any MCP tool directly
    Call { tool: String, #[arg(default_value = "{}")] args: String },
}

fn parse_env_var(s: &str) -> Result<(String, String), String> {
    let parts: Vec<&str> = s.splitn(2, '=').collect();
    if parts.len() != 2 { return Err("Use KEY=VALUE".into()); }
    Ok((parts[0].to_string(), parts[1].to_string()))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let mut config = Config::load()?;

    if let Some(url) = cli.control_plane { config.control_plane_url = url; }
    if let Some(url) = cli.gateway { config.gateway_url = url; }
    if let Some(s) = cli.session { config.active_session = Some(s); }

    let client = AshClient::new(config);

    match cli.command {
        // --- Sandbox ---
        Commands::Create(args) => {
            let sandbox = client.create(args.name, args.image, args.env).await?;
            if args.activate { client.set_active_session(&sandbox.uuid)?; }
            output(&serde_json::to_value(&sandbox)?, cli.output);
        }
        Commands::Destroy { session_id, all } => {
            if all {
                output(&client.destroy_all().await?, cli.output);
            } else {
                let id = session_id.or(client.config.active_session.clone())
                    .ok_or_else(|| anyhow::anyhow!("No session"))?;
                output(&client.destroy(&id).await?, cli.output);
            }
        }
        Commands::Health => {
            let ok = client.health().await?;
            println!("{}", if ok { "healthy" } else { "unhealthy" });
            if !ok { std::process::exit(1); }
        }
        Commands::Ready => {
            let ok = client.ready().await?;
            println!("{}", if ok { "ready" } else { "not ready" });
            if !ok { std::process::exit(1); }
        }

        // --- Session ---
        Commands::Use { session_id } => {
            client.set_active_session(&session_id)?;
            println!("{session_id}");
        }
        Commands::Current => {
            if let Some(ref id) = client.config.active_session {
                println!("{id}");
            } else if Config::session_path().exists() {
                print!("{}", std::fs::read_to_string(Config::session_path())?);
            } else {
                eprintln!("No active session");
                std::process::exit(1);
            }
        }
        Commands::Tools => {
            for tool in client.list_tools().await? {
                println!("{}: {}", tool.name, tool.description.unwrap_or_default());
            }
        }

        // --- Code Agent: View & Navigate ---
        Commands::View(args) => {
            output(&client.call_tool("read_file", args.to_json()).await?, cli.output);
        }
        Commands::Ls(args) => {
            output(&client.call_tool("list_dir", args.to_json()).await?, cli.output);
        }
        Commands::Find(args) => {
            output(&client.call_tool("find_files", args.to_json()).await?, cli.output);
        }

        // --- Code Agent: Search ---
        Commands::Grep(args) => {
            output(&client.call_tool("grep_files", args.to_json()).await?, cli.output);
        }

        // --- Code Agent: Edit ---
        Commands::Edit(args) => {
            output(&client.call_tool("text_editor", args.to_json()).await?, cli.output);
        }
        Commands::ApplyPatch(args) => {
            output(&client.call_tool("apply_patch", args.to_json()).await?, cli.output);
        }
        Commands::Write(args) => {
            output(&client.call_tool("write_file", serde_json::json!({
                "file_path": args.path,
                "content": args.content
            })).await?, cli.output);
        }

        // --- Terminal ---
        Commands::Run(args) => {
            output(&client.call_tool("shell", args.to_json()).await?, cli.output);
        }
        Commands::RunBg(args) => {
            output(&client.call_tool("shell_bg", args.to_json()).await?, cli.output);
        }
        Commands::Kill(args) => {
            output(&client.call_tool("shell_kill", serde_json::json!({"pid": args.pid})).await?, cli.output);
        }
        Commands::Output(args) => {
            output(&client.call_tool("shell_output", serde_json::json!({"pid": args.pid})).await?, cli.output);
        }

        // --- Working Directory ---
        Commands::Pwd => {
            output(&client.call_tool("pwd", serde_json::json!({})).await?, cli.output);
        }
        Commands::Cd(args) => {
            output(&client.call_tool("cd", serde_json::json!({"path": args.path})).await?, cli.output);
        }

        // --- Git ---
        Commands::GitStatus { short } => {
            output(&client.call_tool("git_status", serde_json::json!({"short": short})).await?, cli.output);
        }
        Commands::GitDiff(args) => {
            output(&client.call_tool("git_diff", args.to_json()).await?, cli.output);
        }
        Commands::GitLog(args) => {
            output(&client.call_tool("git_log", args.to_json()).await?, cli.output);
        }
        Commands::GitAdd { paths } => {
            output(&client.call_tool("git_add", serde_json::json!({"paths": paths})).await?, cli.output);
        }
        Commands::GitCommit { message } => {
            output(&client.call_tool("git_commit", serde_json::json!({"message": message})).await?, cli.output);
        }
        Commands::GitReset { hard, target } => {
            output(&client.call_tool("git_reset", serde_json::json!({
                "hard": hard,
                "target": target
            })).await?, cli.output);
        }

        // --- Raw ---
        Commands::Call { tool, args } => {
            let args: Value = serde_json::from_str(&args)?;
            output(&client.call_tool(&tool, args).await?, cli.output);
        }
    }

    Ok(())
}

fn output(value: &Value, format: OutputFormat) {
    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(value).unwrap()),
        OutputFormat::Text => {
            // Extract text from MCP response
            if let Some(arr) = value.get("content").and_then(|c| c.as_array()) {
                for item in arr {
                    if let Some(t) = item.get("text").and_then(|t| t.as_str()) {
                        print!("{t}");
                    }
                }
            } else if let Some(text) = value.as_str() {
                println!("{text}");
            } else {
                println!("{}", serde_json::to_string_pretty(value).unwrap());
            }
        }
    }
}
