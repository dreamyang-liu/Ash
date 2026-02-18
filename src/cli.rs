// CLI definitions for ash — shared between binary, build.rs, and library.
//
// This module contains ONLY clap-derived types with no dependencies on
// the ash library internals. This allows build.rs to `include!()` this
// file for build-time man page and shell completion generation.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ash")]
#[command(about = "Code Agent CLI & MCP Server")]
#[command(version)]
#[command(propagate_version = true)]
#[command(after_help = "Run '<command> --help' for detailed usage of any subcommand.\n\
    Run 'ash man [dir]' to generate man pages.\n\
    Run 'ash completions <shell>' to generate shell completions.")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Output format
    #[arg(short, long, default_value = "text", global = true)]
    pub output: OutputFormat,

    /// Session ID — all tool calls will execute in this sandbox
    #[arg(long, global = true)]
    pub session: Option<String>,
}

#[derive(Clone, Copy, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
}

#[derive(Subcommand)]
pub enum Commands {
    // ==================== File Operations ====================

    /// Search for pattern in files using ripgrep
    Grep {
        /// Regex pattern to search for
        pattern: String,
        /// File or directory to search in
        #[arg(default_value = ".")]
        path: String,
        /// File glob filter (e.g. '*.py', '*.rs')
        #[arg(short, long)]
        include: Option<String>,
        /// Max number of matching lines to return
        #[arg(short, long, default_value = "100")]
        limit: usize,
    },

    /// Edit files — view, str_replace, insert, or create
    Edit {
        #[command(subcommand)]
        op: EditOp,
    },

    /// Find files by name pattern (glob matching)
    Find {
        /// Glob pattern (e.g. '*.py', 'test_*', '**/*.rs')
        pattern: String,
        /// Directory to search in
        #[arg(default_value = ".")]
        path: String,
        /// Max directory depth to recurse
        #[arg(short, long)]
        max_depth: Option<usize>,
        /// Max number of results to return
        #[arg(short, long, default_value = "100")]
        limit: usize,
    },

    /// Show directory tree structure
    Tree {
        /// Root directory to display
        #[arg(default_value = ".")]
        path: String,
        /// Max depth to display
        #[arg(short, long, default_value = "3")]
        max_depth: usize,
        /// Include hidden files and directories (dotfiles)
        #[arg(long)]
        show_hidden: bool,
    },

    /// Compare two files and output unified diff
    Diff {
        /// First file path (original)
        file1: String,
        /// Second file path (modified)
        file2: String,
        /// Number of context lines around each change
        #[arg(short, long, default_value = "3")]
        context: usize,
    },

    /// Apply a unified diff patch to files
    Patch {
        /// Patch content in unified diff format
        patch: String,
        /// Base directory to apply patch relative to
        #[arg(short, long)]
        path: Option<String>,
        /// Preview changes without actually applying them
        #[arg(long)]
        dry_run: bool,
    },

    /// Get file type, encoding, and metadata
    #[command(name = "file-info")]
    FileInfo {
        /// File path to inspect
        path: String,
    },

    /// HTTP GET request — fetch content from URL
    Fetch {
        /// URL to fetch content from
        url: String,
        /// Request timeout in seconds
        #[arg(short, long, default_value = "30")]
        timeout: u64,
    },

    /// Undo last file edit (reverts the most recent edit operation)
    Undo {
        /// Specific file to undo (default: last edited file)
        path: Option<String>,
        /// List undo history instead of performing an undo
        #[arg(long)]
        list: bool,
    },

    /// Show code structure — classes, functions, methods with line numbers (like IDE outline)
    Outline {
        /// File path to analyze
        file_path: String,
    },

    // ==================== Filesystem ====================

    /// List directory contents with file details
    Ls {
        /// Directory path to list
        path: String,
    },

    // ==================== Buffer ====================

    /// Named scratch buffer management
    Buffer {
        #[command(subcommand)]
        op: BufferOp,
    },

    // ==================== Shell ====================

    /// Execute shell command synchronously (locally or in session sandbox)
    Run {
        /// Shell command to execute
        command: String,
        /// Timeout in seconds
        #[arg(short, long, default_value = "300")]
        timeout: u64,
        /// Only return the last N lines of output
        #[arg(long)]
        tail: Option<usize>,
        /// Command to revert this command's side effects. Empty string = no state change, omit = cannot revert
        #[arg(long)]
        revert: Option<String>,
    },

    /// Revert last shell command by executing its registered revert command
    #[command(name = "run-revert")]
    RunRevert {
        /// Specific run ID to revert (default: last revertible command)
        #[arg(long)]
        id: Option<String>,
    },

    /// Show recent shell commands with revert info
    #[command(name = "run-history")]
    RunHistory {
        /// Number of entries to show
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// Background process management — start, monitor, kill async processes
    Terminal {
        #[command(subcommand)]
        op: TerminalOp,
    },

    // ==================== Git ====================

    /// Show working tree status
    #[command(name = "git-status")]
    GitStatus {
        /// Short format output (one line per file)
        #[arg(long)]
        short: bool,
    },

    /// Show changes between working tree and index
    #[command(name = "git-diff")]
    GitDiff {
        /// Compare staged (cached) changes instead of unstaged
        #[arg(long)]
        staged: bool,
        /// Limit diff to specific file paths
        paths: Vec<String>,
    },

    /// Show commit history
    #[command(name = "git-log")]
    GitLog {
        /// Number of commits to show
        #[arg(short = 'n', long, default_value = "10")]
        count: usize,
        /// One-line format per commit (hash + message)
        #[arg(long)]
        oneline: bool,
    },

    /// Stage files for commit
    #[command(name = "git-add")]
    GitAdd {
        /// File paths to stage
        paths: Vec<String>,
        /// Stage all changes including untracked files (-A)
        #[arg(short, long)]
        all: bool,
    },

    /// Create a commit with staged changes
    #[command(name = "git-commit")]
    GitCommit {
        /// Commit message
        #[arg(short, long)]
        message: String,
        /// Automatically stage all tracked modified files before committing (-a)
        #[arg(short, long)]
        all: bool,
    },

    // ==================== Session/Sandbox ====================

    /// Session/sandbox management — create, list, destroy sandboxed environments
    Session {
        #[command(subcommand)]
        op: SessionOp,
    },

    // ==================== Config ====================

    /// Configure ash endpoints and settings
    Config {
        /// Control plane URL for remote backend
        #[arg(long)]
        control_plane_url: Option<String>,
        /// Gateway URL for tool routing
        #[arg(long)]
        gateway_url: Option<String>,
    },

    // ==================== Events ====================

    /// Event system — subscribe, poll, and push events (process_complete, file_change, error, custom)
    Events {
        #[command(subcommand)]
        op: EventsOp,
    },

    // ==================== Custom Tools ====================

    /// Custom tool script management — create, run, and manage user-defined tool scripts
    #[command(name = "custom-tool")]
    CustomTool {
        #[command(subcommand)]
        op: CustomToolOp,
    },

    /// Start MCP server over stdio (for Claude Desktop, etc.)
    Mcp,

    /// Gateway daemon management — routes tool calls to ash-mcp endpoints
    Gateway {
        #[command(subcommand)]
        op: GatewayOp,
    },

    /// Show ash status: backends, sessions, processes, config
    Info,

    /// List all available tools
    Tools,

    // ==================== Meta ====================

    /// Generate man pages for ash and all subcommands
    Man {
        /// Output directory for generated man pages
        #[arg(default_value = "man")]
        output_dir: String,
    },

    /// Generate shell completions and print to stdout
    Completions {
        /// Target shell (bash, zsh, fish, elvish, powershell)
        shell: clap_complete::Shell,
    },
}

// ==================== Nested subcommand enums ====================

#[derive(Subcommand)]
pub enum EditOp {
    /// Read file contents with line numbers
    View {
        /// File path to view
        path: String,
        /// First line number to display (1-indexed, default: 1)
        #[arg(long, default_value = "1")]
        start: i64,
        /// Last line number to display (-1 = end of file)
        #[arg(long, default_value = "-1")]
        end: i64,
    },
    /// Replace exact text in file — old_str must match exactly once
    Replace {
        /// File path to edit
        path: String,
        /// Exact string to find (must appear exactly once in the file)
        #[arg(long)]
        old: String,
        /// Replacement string (can be empty to delete the matched text)
        #[arg(long)]
        new: String,
    },
    /// Insert text after a specific line number
    Insert {
        /// File path to edit
        path: String,
        /// Line number to insert after (0 = insert before first line)
        #[arg(long)]
        line: i64,
        /// Text content to insert
        #[arg(long)]
        text: String,
    },
    /// Create a new file with content
    Create {
        /// Path for the new file (parent directories created automatically)
        path: String,
        /// Full file content to write
        content: String,
    },
}

#[derive(Subcommand)]
pub enum TerminalOp {
    /// Start a background process, returns a handle ID for later reference
    Start {
        /// Shell command to run in background
        command: String,
        /// Working directory for the process
        #[arg(short, long)]
        workdir: Option<String>,
        /// Environment variables as KEY=VALUE pairs (repeatable)
        #[arg(short, long)]
        env: Vec<String>,
        /// Command to revert this process's changes. Empty string = no state change, omit = cannot revert
        #[arg(long)]
        revert: Option<String>,
    },
    /// Get stdout/stderr output from a background process
    Output {
        /// Process handle ID (returned by 'terminal start')
        handle: String,
        /// Only return the last N lines of output
        #[arg(long)]
        tail: Option<usize>,
    },
    /// Kill a running background process by handle
    Kill {
        /// Process handle ID to kill
        handle: String,
    },
    /// List all tracked background processes and their status
    List,
    /// Remove a completed process from tracking (frees the handle)
    Remove {
        /// Process handle ID to remove
        handle: String,
    },
    /// Execute the revert command for a process (if one was provided at start)
    Revert {
        /// Process handle ID to revert
        handle: String,
    },
}

#[derive(Subcommand)]
pub enum SessionOp {
    /// Create a new sandbox session. Returns session_id for use with --session flag
    Create {
        /// Backend: 'local' (host), 'docker' (container), 'k8s' (remote). Default: docker if --image set, else local
        #[arg(short, long)]
        backend: Option<String>,
        /// Custom session name for identification
        #[arg(short, long)]
        name: Option<String>,
        /// Docker image to use for the sandbox
        #[arg(short, long)]
        image: Option<String>,
        /// Port mappings to expose (repeatable)
        #[arg(short, long)]
        port: Vec<i32>,
        /// Environment variables as KEY=VALUE pairs (repeatable)
        #[arg(short, long)]
        env: Vec<String>,
        /// CPU request for K8s pods (e.g. '500m', '1')
        #[arg(long)]
        cpu_request: Option<String>,
        /// CPU limit for K8s pods (e.g. '2', '4')
        #[arg(long)]
        cpu_limit: Option<String>,
        /// Memory request for K8s pods (e.g. '256Mi', '1Gi')
        #[arg(long)]
        memory_request: Option<String>,
        /// Memory limit for K8s pods (e.g. '512Mi', '2Gi')
        #[arg(long)]
        memory_limit: Option<String>,
        /// Node selectors as KEY=VALUE for K8s scheduling (repeatable)
        #[arg(long)]
        node_selector: Vec<String>,
    },
    /// Get detailed info about a session (status, backend, resources)
    Info {
        /// Session ID to inspect
        session_id: String,
    },
    /// List all active sessions across backends
    List,
    /// Destroy a sandbox session and release resources
    Destroy {
        /// Session ID to destroy
        session_id: String,
    },
    /// Switch a session to a different backend (e.g. docker → k8s)
    Switch {
        /// Session ID to switch
        session_id: String,
        /// Target backend ('docker' or 'k8s')
        backend: String,
    },
}

#[derive(Subcommand)]
pub enum BufferOp {
    /// Read lines from a named scratch buffer
    Read {
        /// Buffer name (default: 'main')
        #[arg(short, long, default_value = "main")]
        name: String,
        /// Start line, 1-indexed (default: beginning)
        #[arg(long)]
        start: Option<usize>,
        /// End line, inclusive (default: end of buffer)
        #[arg(long)]
        end: Option<usize>,
    },
    /// Write content to a buffer (creates if it doesn't exist)
    Write {
        /// Buffer name (default: 'main')
        #[arg(short, long, default_value = "main")]
        name: String,
        /// Content to write
        content: String,
        /// Insert before this line number (1-indexed). Omit to replace entire buffer
        #[arg(long)]
        at_line: Option<usize>,
        /// Append to end of existing content instead of overwriting
        #[arg(long)]
        append: bool,
    },
    /// Delete a range of lines from a buffer
    Delete {
        /// Buffer name (default: 'main')
        #[arg(short, long, default_value = "main")]
        name: String,
        /// First line to delete (1-indexed)
        #[arg(long)]
        start: usize,
        /// Last line to delete (inclusive)
        #[arg(long)]
        end: usize,
    },
    /// Replace a range of lines in a buffer with new content
    Replace {
        /// Buffer name (default: 'main')
        #[arg(short, long, default_value = "main")]
        name: String,
        /// First line to replace (1-indexed)
        #[arg(long)]
        start: usize,
        /// Last line to replace (inclusive)
        #[arg(long)]
        end: usize,
        /// Replacement content
        content: String,
    },
    /// List all named buffers and their sizes
    List,
    /// Clear a buffer's content or delete all buffers
    Clear {
        /// Specific buffer to clear (omit to clear ALL buffers)
        #[arg(short, long)]
        name: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum EventsOp {
    /// Subscribe to event types (process_complete, file_change, error, custom)
    Subscribe {
        /// Event type names to subscribe to: process_complete, file_change, error, custom
        events: Vec<String>,
        /// Unsubscribe from the specified types instead of subscribing
        #[arg(short, long)]
        unsubscribe: bool,
    },
    /// Poll pending events from the queue
    Poll {
        /// Max number of events to return per poll
        #[arg(short, long, default_value = "10")]
        limit: usize,
        /// Preview events without removing them from the queue
        #[arg(long)]
        peek: bool,
    },
    /// Push a custom event into the event queue
    Push {
        /// Event type string (e.g. 'custom', 'notification')
        kind: String,
        /// Event source label (default: 'llm')
        #[arg(short, long, default_value = "llm")]
        source: String,
        /// JSON data payload for the event
        #[arg(short, long, default_value = "{}")]
        data: String,
    },
}

#[derive(Subcommand)]
pub enum CustomToolOp {
    /// Create a custom tool script (.sh or .py). First # comment becomes the tool description
    Create {
        /// Tool name (becomes <name>.sh or <name>.py)
        name: String,
        /// Script content (first # comment line = tool description)
        #[arg(short, long)]
        script: String,
        /// Script language: 'sh' for shell, 'py' for Python
        #[arg(short, long, default_value = "sh")]
        lang: String,
    },
    /// List all registered custom tool scripts
    List,
    /// View a custom tool's script source
    View {
        /// Tool name to view
        name: String,
    },
    /// Run a custom tool script with arguments
    Run {
        /// Tool name to execute
        name: String,
        /// Positional arguments passed to the tool script
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
    /// Remove a custom tool script
    Remove {
        /// Tool name to remove
        name: String,
    },
}

#[derive(Subcommand)]
pub enum GatewayOp {
    /// Start the gateway daemon (routes tool calls to ash-mcp endpoints)
    Start {
        /// Run in foreground instead of detaching as a daemon
        #[arg(long)]
        foreground: bool,
    },
    /// Stop the running gateway daemon
    Stop,
    /// Check gateway daemon status and connectivity
    Status,
}
