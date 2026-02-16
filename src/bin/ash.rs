//! ash CLI

use ash::{Tool, ToolResult};
use ash::daemon;
use ash::tools;
use clap::{Parser, Subcommand};
use serde_json::Value;

use std::collections::HashMap;

/// Execute a tool: try daemon first, then fallback to direct execution.
/// Daemon handles both local and session-routed calls with persistent connections.
async fn exec_tool(tool: &dyn Tool, args: Value, session_id: &Option<String>) -> ToolResult {
    // 1. Try daemon (handles local + session routing with persistent BackendManager)
    if let Some(result) = daemon::daemon_tool_call(tool.name(), args.clone(), session_id).await {
        return result;
    }

    // 2. Fallback: direct execution (no daemon running)
    if let Some(ref sid) = session_id {
        match tools::session::call_tool_in_session(sid, tool.name(), args).await {
            Ok(result) => {
                let is_error = result.get("isError").and_then(|e| e.as_bool()).unwrap_or(false);
                let text = result.get("content")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|c| c.get("text"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                if is_error {
                    ToolResult { success: false, output: text.clone(), error: Some(text) }
                } else {
                    ToolResult::ok(text)
                }
            }
            Err(e) => ToolResult::err(format!("Session error: {e}")),
        }
    } else {
        tool.execute(args).await
    }
}

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

    /// Show code outline (classes, functions, methods)
    Outline {
        file_path: String,
    },

    // ==================== Filesystem ====================

    /// Filesystem operations
    Fs {
        #[command(subcommand)]
        op: FsOp,
    },

    // ==================== Buffer ====================

    /// Buffer management
    Buffer {
        #[command(subcommand)]
        op: BufferOp,
    },

    // ==================== Shell ====================

    /// Execute shell command (sync)
    Run {
        command: String,
        #[arg(short, long, default_value = "300")]
        timeout: u64,
        #[arg(long)]
        tail: Option<usize>,
        /// Command to revert changes (for tracking)
        #[arg(long)]
        revert: Option<String>,
    },
    
    /// Revert last shell command
    #[command(name = "run-revert")]
    RunRevert {
        /// Specific run ID to revert
        #[arg(long)]
        id: Option<String>,
    },
    
    /// Show shell command history
    #[command(name = "run-history")]
    RunHistory {
        #[arg(short, long, default_value = "10")]
        limit: usize,
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
    
    /// Daemon management (long-lived background process)
    Daemon {
        #[command(subcommand)]
        op: DaemonOp,
    },

    /// Show ash status: backends, sessions, processes, config
    Info,

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
        /// Command to revert changes (empty string = no state change, omit = cannot revert)
        #[arg(long)]
        revert: Option<String>,
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
    /// Revert a command's changes (if revert_command was provided)
    Revert { handle: String },
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
    /// Get session details
    Info { session_id: String },
    List,
    Destroy { session_id: String },
    /// Switch session backend
    Switch {
        session_id: String,
        /// Target backend (docker or k8s)
        backend: String,
    },
}

#[derive(Subcommand)]
enum FsOp {
    /// List directory contents
    Ls { path: String },
    /// Get file/directory metadata
    Stat { path: String },
    /// Write content to file
    Write {
        path: String,
        content: String,
    },
    /// Create directory
    Mkdir {
        path: String,
        #[arg(long, default_value = "true")]
        recursive: bool,
    },
    /// Remove file or directory
    Rm {
        path: String,
        #[arg(short, long)]
        recursive: bool,
    },
    /// Move/rename file or directory
    Mv {
        from: String,
        to: String,
    },
    /// Copy file or directory
    Cp {
        from: String,
        to: String,
        #[arg(short, long)]
        recursive: bool,
    },
}

#[derive(Subcommand)]
enum BufferOp {
    /// Read lines from a buffer
    Read {
        #[arg(short, long, default_value = "main")]
        name: String,
        #[arg(long)]
        start: Option<usize>,
        #[arg(long)]
        end: Option<usize>,
    },
    /// Write content to a buffer
    Write {
        #[arg(short, long, default_value = "main")]
        name: String,
        content: String,
        #[arg(long)]
        at_line: Option<usize>,
        #[arg(long)]
        append: bool,
    },
    /// Delete lines from a buffer
    Delete {
        #[arg(short, long, default_value = "main")]
        name: String,
        #[arg(long)]
        start: usize,
        #[arg(long)]
        end: usize,
    },
    /// Replace lines in a buffer
    Replace {
        #[arg(short, long, default_value = "main")]
        name: String,
        #[arg(long)]
        start: usize,
        #[arg(long)]
        end: usize,
        content: String,
    },
    /// List all buffers
    List,
    /// Clear a buffer or all buffers
    Clear {
        #[arg(short, long)]
        name: Option<String>,
    },
    /// Copy buffer range to clipboard
    ToClip {
        #[arg(short, long, default_value = "main")]
        buffer: String,
        #[arg(long)]
        start: Option<usize>,
        #[arg(long)]
        end: Option<usize>,
        /// Clip name
        clip_name: String,
    },
    /// Paste clipboard into buffer
    FromClip {
        /// Clip name to paste from
        clip_name: String,
        #[arg(short, long, default_value = "main")]
        buffer: String,
        #[arg(long)]
        at_line: Option<usize>,
        #[arg(long)]
        append: bool,
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
    /// Create a custom tool script
    Create {
        name: String,
        #[arg(short, long)]
        script: String,
        #[arg(short, long, default_value = "sh")]
        lang: String,
    },
    /// List custom tools
    List,
    /// View a custom tool's script
    View { name: String },
    /// Run a custom tool
    Run {
        name: String,
        /// Positional arguments
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
    /// Remove a custom tool
    Remove { name: String },
}

#[derive(Subcommand)]
enum DaemonOp {
    /// Start the daemon
    Start {
        /// Run in foreground (don't detach)
        #[arg(long)]
        foreground: bool,
    },
    /// Stop the daemon
    Stop,
    /// Check daemon status
    Status,
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
            exec_tool(&tools::ViewTool, serde_json::json!({
                "file_path": file_path, "offset": offset, "limit": limit
            }), &session_id).await
        }

        Commands::Grep { pattern, path, include, limit } => {
            exec_tool(&tools::GrepTool, serde_json::json!({
                "pattern": pattern, "path": path, "include": include, "limit": limit
            }), &session_id).await
        }

        Commands::Edit { op } => {
            let args = match op {
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
            exec_tool(&tools::EditTool, args, &session_id).await
        }

        Commands::Find { pattern, path, max_depth, limit } => {
            exec_tool(&tools::FindFilesTool, serde_json::json!({
                "pattern": pattern, "path": path, "max_depth": max_depth, "limit": limit
            }), &session_id).await
        }

        Commands::Tree { path, max_depth, show_hidden } => {
            exec_tool(&tools::TreeTool, serde_json::json!({
                "path": path, "max_depth": max_depth, "show_hidden": show_hidden
            }), &session_id).await
        }

        Commands::Diff { file1, file2, context } => {
            exec_tool(&tools::DiffFilesTool, serde_json::json!({
                "file1": file1, "file2": file2, "context": context
            }), &session_id).await
        }

        Commands::Patch { patch, path, dry_run } => {
            exec_tool(&tools::PatchApplyTool, serde_json::json!({
                "patch": patch, "path": path, "dry_run": dry_run
            }), &session_id).await
        }

        Commands::FileInfo { path } => {
            exec_tool(&tools::FileInfoTool, serde_json::json!({"path": path}), &session_id).await
        }

        Commands::Fetch { url, timeout } => {
            exec_tool(&tools::HttpFetchTool, serde_json::json!({
                "url": url, "timeout_secs": timeout
            }), &session_id).await
        }

        Commands::Undo { path, list } => {
            exec_tool(&tools::UndoTool, serde_json::json!({"path": path, "list": list}), &session_id).await
        }

        Commands::Outline { file_path } => {
            exec_tool(&tools::OutlineTool, serde_json::json!({"file_path": file_path}), &session_id).await
        }

        // ==================== Filesystem ====================

        Commands::Fs { op } => {
            match op {
                FsOp::Ls { path } => {
                    exec_tool(&tools::FsListDirTool, serde_json::json!({"path": path}), &session_id).await
                }
                FsOp::Stat { path } => {
                    exec_tool(&tools::FsStatTool, serde_json::json!({"path": path}), &session_id).await
                }
                FsOp::Write { path, content } => {
                    exec_tool(&tools::FsWriteTool, serde_json::json!({"path": path, "content": content}), &session_id).await
                }
                FsOp::Mkdir { path, recursive } => {
                    exec_tool(&tools::FsMkdirTool, serde_json::json!({"path": path, "recursive": recursive}), &session_id).await
                }
                FsOp::Rm { path, recursive } => {
                    exec_tool(&tools::FsRemoveTool, serde_json::json!({"path": path, "recursive": recursive}), &session_id).await
                }
                FsOp::Mv { from, to } => {
                    exec_tool(&tools::FsMoveTool, serde_json::json!({"from": from, "to": to}), &session_id).await
                }
                FsOp::Cp { from, to, recursive } => {
                    exec_tool(&tools::FsCopyTool, serde_json::json!({"from": from, "to": to, "recursive": recursive}), &session_id).await
                }
            }
        }

        // ==================== Buffer ====================

        Commands::Buffer { op } => {
            match op {
                BufferOp::Read { name, start, end } => {
                    exec_tool(&tools::BufferReadTool, serde_json::json!({
                        "name": name, "start_line": start, "end_line": end
                    }), &session_id).await
                }
                BufferOp::Write { name, content, at_line, append } => {
                    exec_tool(&tools::BufferWriteTool, serde_json::json!({
                        "name": name, "content": content, "at_line": at_line, "append": append
                    }), &session_id).await
                }
                BufferOp::Delete { name, start, end } => {
                    exec_tool(&tools::BufferDeleteTool, serde_json::json!({
                        "name": name, "start_line": start, "end_line": end
                    }), &session_id).await
                }
                BufferOp::Replace { name, start, end, content } => {
                    exec_tool(&tools::BufferReplaceTool, serde_json::json!({
                        "name": name, "start_line": start, "end_line": end, "content": content
                    }), &session_id).await
                }
                BufferOp::List => {
                    exec_tool(&tools::BufferListTool, serde_json::json!({}), &session_id).await
                }
                BufferOp::Clear { name } => {
                    exec_tool(&tools::BufferClearTool, serde_json::json!({"name": name}), &session_id).await
                }
                BufferOp::ToClip { buffer, start, end, clip_name } => {
                    exec_tool(&tools::BufferToClipTool, serde_json::json!({
                        "buffer": buffer, "start_line": start, "end_line": end, "clip_name": clip_name
                    }), &session_id).await
                }
                BufferOp::FromClip { clip_name, buffer, at_line, append } => {
                    exec_tool(&tools::ClipToBufferTool, serde_json::json!({
                        "clip_name": clip_name, "buffer": buffer, "at_line": at_line, "append": append
                    }), &session_id).await
                }
            }
        }

        // ==================== Shell ====================

        Commands::Run { command, timeout, tail, revert } => {
            exec_tool(&tools::ShellTool, serde_json::json!({
                "command": command, "timeout_secs": timeout, "tail_lines": tail, "revert_command": revert
            }), &session_id).await
        }

        Commands::RunRevert { id } => {
            exec_tool(&tools::ShellRevertTool, serde_json::json!({"id": id}), &session_id).await
        }

        Commands::RunHistory { limit } => {
            exec_tool(&tools::ShellHistoryTool, serde_json::json!({"limit": limit}), &session_id).await
        }

        Commands::Terminal { op } => {
            let (tool_name, args): (&str, Value) = match op {
                TerminalOp::Start { command, workdir, env, revert } => {
                    let env_map = parse_key_value(&env);
                    ("terminal_run_async", serde_json::json!({
                        "command": command, "working_dir": workdir, "env": env_map, 
                        "revert_command": revert, "session_id": session_id
                    }))
                }
                TerminalOp::Output { handle, tail } => {
                    ("terminal_get_output", serde_json::json!({"handle": handle, "tail": tail, "session_id": session_id}))
                }
                TerminalOp::Kill { handle } => {
                    ("terminal_kill", serde_json::json!({"handle": handle, "session_id": session_id}))
                }
                TerminalOp::List => {
                    ("terminal_list", serde_json::json!({"session_id": session_id}))
                }
                TerminalOp::Remove { handle } => {
                    ("terminal_remove", serde_json::json!({"handle": handle, "session_id": session_id}))
                }
                TerminalOp::Revert { handle } => {
                    ("terminal_revert", serde_json::json!({"handle": handle, "session_id": session_id}))
                }
            };
            // Try daemon first (session_id is embedded in args for terminal tools),
            // fall back to direct execution
            if let Some(result) = daemon::daemon_tool_call(tool_name, args.clone(), &None).await {
                result
            } else {
                tools::find_tool(tool_name).unwrap().execute(args).await
            }
        }

        // ==================== Git ====================

        Commands::GitStatus { short } => {
            exec_tool(&tools::GitStatusTool, serde_json::json!({"short": short}), &session_id).await
        }

        Commands::GitDiff { staged, paths } => {
            exec_tool(&tools::GitDiffTool, serde_json::json!({"staged": staged, "paths": paths}), &session_id).await
        }

        Commands::GitLog { count, oneline } => {
            exec_tool(&tools::GitLogTool, serde_json::json!({"count": count, "oneline": oneline}), &session_id).await
        }

        Commands::GitAdd { paths, all } => {
            exec_tool(&tools::GitAddTool, serde_json::json!({"paths": paths, "all": all}), &session_id).await
        }

        Commands::GitCommit { message, all } => {
            exec_tool(&tools::GitCommitTool, serde_json::json!({"message": message, "all": all}), &session_id).await
        }

        // ==================== Clipboard ====================

        Commands::Clip { content, file, name, source } => {
            exec_tool(&tools::ClipTool, serde_json::json!({"content": content, "file": file, "name": name, "source": source}), &session_id).await
        }

        Commands::Paste { name } => {
            exec_tool(&tools::PasteTool, serde_json::json!({"name": name}), &session_id).await
        }

        Commands::Clips => {
            exec_tool(&tools::ClipsTool, serde_json::json!({}), &session_id).await
        }

        Commands::ClipsClear { name } => {
            exec_tool(&tools::ClearClipsTool, serde_json::json!({"name": name}), &session_id).await
        }

        // ==================== Session ====================

        Commands::Session { op } => {
            let (tool_name, args): (&str, Value) = match op {
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

                    ("session_create", args)
                }
                SessionOp::Info { session_id } => {
                    ("session_info", serde_json::json!({"session_id": session_id}))
                }
                SessionOp::List => ("session_list", serde_json::json!({})),
                SessionOp::Destroy { session_id } => {
                    ("session_destroy", serde_json::json!({"session_id": session_id}))
                }
                SessionOp::Switch { session_id, backend } => {
                    ("backend_switch", serde_json::json!({"session_id": session_id, "backend": backend}))
                }
            };
            // Try daemon first (session management tools, no outer session routing),
            // fall back to direct execution
            if let Some(result) = daemon::daemon_tool_call(tool_name, args.clone(), &None).await {
                result
            } else {
                tools::find_tool(tool_name).unwrap().execute(args).await
            }
        }

        // ==================== Config ====================
        
        Commands::Config { control_plane_url, gateway_url } => {
            // Show current backend status instead of old config
            if control_plane_url.is_none() && gateway_url.is_none() {
                tools::BackendStatusTool.execute(serde_json::json!({})).await
            } else {
                // Configure K8s backend with new URLs
                use ash::backend::K8sConfig;
                let config = K8sConfig {
                    control_plane_url: control_plane_url.unwrap_or_else(|| {
                        std::env::var("ASH_CONTROL_PLANE_URL").unwrap_or_else(|_| "http://localhost:8080".to_string())
                    }),
                    gateway_url: gateway_url.unwrap_or_else(|| {
                        std::env::var("ASH_GATEWAY_URL").unwrap_or_else(|_| "http://localhost:8081".to_string())
                    }),
                    ..Default::default()
                };
                tools::session::configure_k8s(config.clone()).await;
                ToolResult::ok(format!("Updated K8s config:\ncontrol_plane_url: {}\ngateway_url: {}", 
                    config.control_plane_url, config.gateway_url))
            }
        }
        
        // ==================== Events ====================

        Commands::Events { op } => {
            let (tool_name, args): (&str, Value) = match op {
                EventsOp::Subscribe { events, unsubscribe } => {
                    ("events_subscribe", serde_json::json!({"events": events, "unsubscribe": unsubscribe}))
                }
                EventsOp::Poll { limit, peek } => {
                    ("events_poll", serde_json::json!({"limit": limit, "peek": peek}))
                }
                EventsOp::Push { kind, source, data } => {
                    let data_val: serde_json::Value = serde_json::from_str(&data).unwrap_or(serde_json::json!({}));
                    ("events_push", serde_json::json!({"kind": kind, "source": source, "data": data_val}))
                }
            };
            // Try daemon first (events are daemon-managed state),
            // fall back to direct execution
            if let Some(result) = daemon::daemon_tool_call(tool_name, args.clone(), &None).await {
                result
            } else {
                tools::find_tool(tool_name).unwrap().execute(args).await
            }
        }

        // ==================== Custom Tools ====================
        
        Commands::CustomTool { op } => {
            match op {
                CustomToolOp::Create { name, script, lang } => {
                    tools::ToolRegisterTool.execute(serde_json::json!({
                        "name": name, "script": script, "lang": lang
                    })).await
                }
                CustomToolOp::List => {
                    tools::ToolListCustomTool.execute(serde_json::json!({})).await
                }
                CustomToolOp::View { name } => {
                    tools::ToolViewCustomTool.execute(serde_json::json!({"name": name})).await
                }
                CustomToolOp::Run { name, args } => {
                    tools::ToolCallCustomTool.execute(serde_json::json!({
                        "name": name, "args": args
                    })).await
                }
                CustomToolOp::Remove { name } => {
                    tools::ToolRemoveCustomTool.execute(serde_json::json!({"name": name})).await
                }
            }
        }
        
        Commands::Daemon { op } => {
            match op {
                DaemonOp::Start { foreground } => {
                    if daemon::is_daemon_running() {
                        println!("Daemon is already running");
                        return Ok(());
                    }

                    if foreground {
                        run_daemon().await?;
                    } else {
                        // Spawn detached child with --foreground
                        let exe = std::env::current_exe()?;
                        let child = std::process::Command::new(exe)
                            .args(["daemon", "start", "--foreground"])
                            .stdin(std::process::Stdio::null())
                            .stdout(std::process::Stdio::null())
                            .stderr(std::process::Stdio::null())
                            .spawn()?;
                        println!("Daemon started (pid {})", child.id());
                    }
                }
                DaemonOp::Stop => {
                    let pid_file = daemon::pid_path();
                    match std::fs::read_to_string(&pid_file) {
                        Ok(contents) => {
                            if let Ok(pid) = contents.trim().parse::<u32>() {
                                // Send SIGTERM via kill command
                                let _ = std::process::Command::new("kill")
                                    .arg(pid.to_string())
                                    .status();
                                // Wait briefly for cleanup
                                tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                                let _ = std::fs::remove_file(daemon::pid_path());
                                let _ = std::fs::remove_file(daemon::socket_path());
                                println!("Daemon stopped (pid {})", pid);
                            } else {
                                eprintln!("Invalid PID file");
                            }
                        }
                        Err(_) => {
                            eprintln!("Daemon is not running");
                        }
                    }
                }
                DaemonOp::Status => {
                    match daemon::daemon_call("ping", serde_json::json!({})).await {
                        Some(resp) => {
                            let uptime = resp.get("result")
                                .and_then(|r| r.get("uptime_secs"))
                                .and_then(|u| u.as_u64())
                                .unwrap_or(0);
                            println!("Daemon running (uptime: {}s)", uptime);
                        }
                        None => {
                            println!("Daemon is not running");
                        }
                    }
                }
            }
            return Ok(());
        }

        Commands::Info => {
            let mut out = String::new();

            // Version
            out.push_str(&format!("ash {}\n\n", env!("CARGO_PKG_VERSION")));

            // Try daemon for cached info
            if let Some(resp) = daemon::daemon_call("daemon/info", serde_json::json!({})).await {
                if let Some(info) = resp.get("result") {
                    let uptime = info.get("uptime_secs").and_then(|u| u.as_u64()).unwrap_or(0);
                    out.push_str(&format!("Daemon: running (uptime: {}s)\n\n", uptime));

                    let docker_ok = info.pointer("/backends/docker").and_then(|v| v.as_bool()).unwrap_or(false);
                    let k8s_ok = info.pointer("/backends/k8s").and_then(|v| v.as_bool()).unwrap_or(false);
                    let default_backend = info.get("default_backend").and_then(|v| v.as_str()).unwrap_or("docker");

                    out.push_str("Backends:\n");
                    let docker_mark = if docker_ok { "ok" } else { "--" };
                    let k8s_mark = if k8s_ok { "ok" } else { "--" };
                    let docker_default = if default_backend == "docker" { " (default)" } else { "" };
                    let k8s_default = if default_backend == "k8s" { " (default)" } else { "" };
                    out.push_str(&format!("  docker  {}{}\n", docker_mark, docker_default));
                    out.push_str(&format!("  k8s     {}{}\n", k8s_mark, k8s_default));

                    let sessions = info.get("sessions").and_then(|s| s.as_u64()).unwrap_or(0);
                    out.push_str(&format!("\nSessions: {}\n", sessions));

                    let running = info.pointer("/processes/running").and_then(|v| v.as_u64()).unwrap_or(0);
                    let done = info.pointer("/processes/completed").and_then(|v| v.as_u64()).unwrap_or(0);
                    out.push_str(&format!("Async processes: {} running, {} completed\n", running, done));

                    let tool_count = tools::all_tools().len();
                    out.push_str(&format!("\nTools: {}\n", tool_count));

                    print!("{}", out);
                    return Ok(());
                }
            }

            // Fallback: direct check (no daemon)
            use ash::backend::BackendType;

            out.push_str("Daemon: not running\n\n");

            out.push_str("Backends:\n");
            let manager = tools::session::BACKEND_MANAGER.read().await;
            let default_backend = manager.default_backend();
            let local_ok = manager.health_check(BackendType::Local).await.is_ok();
            let docker_ok = manager.health_check(BackendType::Docker).await.is_ok();
            let k8s_ok = manager.health_check(BackendType::K8s).await.is_ok();
            drop(manager);

            let local_mark = if local_ok { "ok" } else { "--" };
            let docker_mark = if docker_ok { "ok" } else { "--" };
            let k8s_mark = if k8s_ok { "ok" } else { "--" };
            let local_default = if default_backend == BackendType::Local { " (default)" } else { "" };
            let docker_default = if default_backend == BackendType::Docker { " (default)" } else { "" };
            let k8s_default = if default_backend == BackendType::K8s { " (default)" } else { "" };
            out.push_str(&format!("  local   {}{}\n", local_mark, local_default));
            out.push_str(&format!("  docker  {}{}\n", docker_mark, docker_default));
            out.push_str(&format!("  k8s     {}{}\n", k8s_mark, k8s_default));

            // Sessions
            out.push_str("\nSessions:\n");
            let manager = tools::session::BACKEND_MANAGER.read().await;
            match manager.list().await {
                Ok(sessions) if sessions.is_empty() => {
                    out.push_str("  (none)\n");
                }
                Ok(sessions) => {
                    for s in &sessions {
                        let short_id = if s.id.len() > 12 { &s.id[..12] } else { &s.id };
                        out.push_str(&format!("  {}  {}  {}  {}\n",
                            short_id, s.status, s.backend, s.image));
                    }
                }
                Err(e) => {
                    out.push_str(&format!("  error: {e}\n"));
                }
            }
            drop(manager);

            // Async processes (local)
            let (running, done) = tools::terminal::local_process_counts().await;
            out.push_str(&format!("\nAsync processes (local): {} running, {} completed\n", running, done));

            // Tracked handles
            let handle_map = tools::terminal::load_handle_map();
            if !handle_map.is_empty() {
                let unique_sessions: std::collections::HashSet<_> = handle_map.values().collect();
                out.push_str(&format!("Tracked handles: {} across {} session(s)\n",
                    handle_map.len(), unique_sessions.len()));
            }

            // Tools count
            let tool_count = tools::all_tools().len();
            out.push_str(&format!("\nTools: {}\n", tool_count));

            print!("{}", out);
            return Ok(());
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

// ==================== Daemon Server ====================

use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader};

#[derive(Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
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

static DAEMON_START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

async fn run_daemon() -> anyhow::Result<()> {
    let socket = daemon::socket_path();
    let pid_file = daemon::pid_path();

    let _ = std::fs::create_dir_all(daemon::ash_dir());
    let _ = std::fs::remove_file(&socket);

    // Write PID
    std::fs::write(&pid_file, std::process::id().to_string())?;

    DAEMON_START.get_or_init(std::time::Instant::now);

    let listener = tokio::net::UnixListener::bind(&socket)?;
    eprintln!("ash daemon listening on {}", socket.display());

    // Graceful shutdown on SIGTERM/SIGINT
    let socket_cleanup = socket.clone();
    let pid_cleanup = pid_file.clone();

    tokio::spawn(async move {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to register SIGTERM");
        let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .expect("failed to register SIGINT");

        tokio::select! {
            _ = sigterm.recv() => {}
            _ = sigint.recv() => {}
        }

        eprintln!("ash daemon shutting down");
        let _ = std::fs::remove_file(&socket_cleanup);
        let _ = std::fs::remove_file(&pid_cleanup);
        std::process::exit(0);
    });

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                tokio::spawn(async move {
                    if let Err(e) = handle_daemon_connection(stream).await {
                        eprintln!("daemon connection error: {e}");
                    }
                });
            }
            Err(e) => {
                eprintln!("daemon accept error: {e}");
            }
        }
    }
}

async fn handle_daemon_connection(stream: tokio::net::UnixStream) -> anyhow::Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    reader.read_line(&mut line).await?;

    if line.trim().is_empty() {
        return Ok(());
    }

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
            let mut out = serde_json::to_string(&response)?;
            out.push('\n');
            writer.write_all(out.as_bytes()).await?;
            return Ok(());
        }
    };

    let response = handle_daemon_request(request).await;
    let mut out = serde_json::to_string(&response)?;
    out.push('\n');
    writer.write_all(out.as_bytes()).await?;

    Ok(())
}

async fn handle_daemon_request(request: JsonRpcRequest) -> JsonRpcResponse {
    let id = request.id.unwrap_or(Value::Null);

    match request.method.as_str() {
        "ping" => {
            let uptime = DAEMON_START
                .get()
                .map(|s| s.elapsed().as_secs())
                .unwrap_or(0);
            JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: Some(json!({
                    "status": "ok",
                    "uptime_secs": uptime
                })),
                error: None,
            }
        }

        "daemon/info" => {
            use ash::backend::BackendType;

            let uptime = DAEMON_START
                .get()
                .map(|s| s.elapsed().as_secs())
                .unwrap_or(0);

            let manager = tools::session::BACKEND_MANAGER.read().await;
            let default_backend = manager.default_backend();
            let local_ok = manager.health_check(BackendType::Local).await.is_ok();
            let docker_ok = manager.health_check(BackendType::Docker).await.is_ok();
            let k8s_ok = manager.health_check(BackendType::K8s).await.is_ok();

            let sessions = manager.list().await.unwrap_or_default();
            drop(manager);

            let (running, done) = tools::terminal::local_process_counts().await;

            JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: Some(json!({
                    "uptime_secs": uptime,
                    "default_backend": format!("{}", default_backend),
                    "backends": {
                        "local": local_ok,
                        "docker": docker_ok,
                        "k8s": k8s_ok
                    },
                    "sessions": sessions.len(),
                    "processes": {
                        "running": running,
                        "completed": done
                    }
                })),
                error: None,
            }
        }

        "tools/call" => {
            let name = request
                .params
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let arguments = request
                .params
                .get("arguments")
                .cloned()
                .unwrap_or(json!({}));
            let session_id = request
                .params
                .get("session_id")
                .and_then(|s| s.as_str());

            // If session_id provided, route through BackendManager to session MCP
            if let Some(sid) = session_id {
                match tools::session::call_tool_in_session(sid, name, arguments).await {
                    Ok(result) => {
                        return JsonRpcResponse {
                            jsonrpc: "2.0".to_string(),
                            id,
                            result: Some(result),
                            error: None,
                        };
                    }
                    Err(e) => {
                        return JsonRpcResponse {
                            jsonrpc: "2.0".to_string(),
                            id,
                            result: Some(json!({
                                "content": [{"type": "text", "text": format!("Session error: {e}")}],
                                "isError": true
                            })),
                            error: None,
                        };
                    }
                }
            }

            // Local execution
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
