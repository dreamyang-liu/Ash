//! Tool registry

pub mod view;
pub mod grep;
pub mod edit;
pub mod shell;
pub mod git;
pub mod clip;
pub mod session;
pub mod terminal;
pub mod mcp_mount;
pub mod filesystem;

pub use view::ViewTool;
pub use grep::GrepTool;
pub use edit::EditTool;
pub use shell::ShellTool;
pub use git::{GitStatusTool, GitDiffTool, GitLogTool};
pub use clip::{ClipTool, PasteTool, ClipsTool, ClearClipsTool};
pub use session::{SessionCreateTool, SessionListTool, SessionDestroyTool};
pub use terminal::{TerminalRunAsyncTool, TerminalGetOutputTool, TerminalKillTool, TerminalListTool, TerminalRemoveTool};
pub use mcp_mount::{McpInstallTool, McpMountTool, McpUnmountTool, McpListTool, McpCallTool};
pub use filesystem::{FsListDirTool, FsMkdirTool, FsRemoveTool, FsMoveTool, FsCopyTool, FsStatTool, FsWriteTool};

use crate::Tool;

/// All available tools
pub fn all_tools() -> Vec<Box<dyn Tool>> {
    vec![
        // File operations
        Box::new(ViewTool),
        Box::new(GrepTool),
        Box::new(EditTool),
        // Shell (sync)
        Box::new(ShellTool),
        // Terminal (async)
        Box::new(TerminalRunAsyncTool),
        Box::new(TerminalGetOutputTool),
        Box::new(TerminalKillTool),
        Box::new(TerminalListTool),
        Box::new(TerminalRemoveTool),
        // Git
        Box::new(GitStatusTool),
        Box::new(GitDiffTool),
        Box::new(GitLogTool),
        // Clipboard
        Box::new(ClipTool),
        Box::new(PasteTool),
        Box::new(ClipsTool),
        Box::new(ClearClipsTool),
        // Session/sandbox
        Box::new(SessionCreateTool),
        Box::new(SessionListTool),
        Box::new(SessionDestroyTool),
        // MCP mount
        Box::new(McpInstallTool),
        Box::new(McpMountTool),
        Box::new(McpUnmountTool),
        Box::new(McpListTool),
        Box::new(McpCallTool),
        // Filesystem
        Box::new(FsListDirTool),
        Box::new(FsMkdirTool),
        Box::new(FsRemoveTool),
        Box::new(FsMoveTool),
        Box::new(FsCopyTool),
        Box::new(FsStatTool),
        Box::new(FsWriteTool),
    ]
}

/// Find tool by name
pub fn find_tool(name: &str) -> Option<Box<dyn Tool>> {
    all_tools().into_iter().find(|t| t.name() == name)
}
