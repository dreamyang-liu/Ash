//! Tool registry

pub mod grep;
pub mod edit;
pub mod shell;
pub mod git;
pub mod clip;
pub mod buffer;
pub mod session;
pub mod terminal;
pub mod outline;

pub mod filesystem;
pub mod utils;
pub mod events;

pub use grep::GrepTool;
pub use edit::EditTool;
pub use shell::{ShellTool, ShellRevertTool, ShellHistoryTool};
pub use git::{GitStatusTool, GitDiffTool, GitLogTool, GitAddTool, GitCommitTool};
pub use clip::{ClipTool, PasteTool, ClipsTool, ClearClipsTool};
pub use buffer::{BufferReadTool, BufferWriteTool, BufferDeleteTool, BufferReplaceTool, BufferListTool, BufferClearTool, BufferToClipTool, ClipToBufferTool};
pub use session::{SessionCreateTool, SessionListTool, SessionDestroyTool, SessionInfoTool, BackendSwitchTool, BackendStatusTool};
pub use terminal::{TerminalRunAsyncTool, TerminalGetOutputTool, TerminalKillTool, TerminalListTool, TerminalRemoveTool, TerminalRevertTool};
pub use outline::OutlineTool;

pub use filesystem::FsListDirTool;
pub use utils::{FindFilesTool, TreeTool, DiffFilesTool, PatchApplyTool, HttpFetchTool, FileInfoTool, UndoTool};
pub use events::{EventsSubscribeTool, EventsPollTool, EventsPushTool, ToolRegisterTool, ToolListCustomTool, ToolCallCustomTool, ToolRemoveCustomTool, ToolViewCustomTool};

use crate::Tool;

/// All available tools
pub fn all_tools() -> Vec<Box<dyn Tool>> {
    vec![
        // File operations
        Box::new(GrepTool),
        Box::new(EditTool),
        Box::new(OutlineTool),
        Box::new(FsListDirTool),
        // Shell (sync)
        Box::new(ShellTool),
        Box::new(ShellRevertTool),
        Box::new(ShellHistoryTool),
        // Terminal (async)
        Box::new(TerminalRunAsyncTool),
        Box::new(TerminalGetOutputTool),
        Box::new(TerminalKillTool),
        Box::new(TerminalListTool),
        Box::new(TerminalRemoveTool),
        Box::new(TerminalRevertTool),
        // Git
        Box::new(GitStatusTool),
        Box::new(GitDiffTool),
        Box::new(GitLogTool),
        Box::new(GitAddTool),
        Box::new(GitCommitTool),
        // Clipboard
        Box::new(ClipTool),
        Box::new(PasteTool),
        Box::new(ClipsTool),
        Box::new(ClearClipsTool),
        // Buffers
        Box::new(BufferReadTool),
        Box::new(BufferWriteTool),
        Box::new(BufferDeleteTool),
        Box::new(BufferReplaceTool),
        Box::new(BufferListTool),
        Box::new(BufferClearTool),
        Box::new(BufferToClipTool),
        Box::new(ClipToBufferTool),
        // Session/sandbox
        Box::new(SessionCreateTool),
        Box::new(SessionListTool),
        Box::new(SessionDestroyTool),
        Box::new(SessionInfoTool),
        Box::new(BackendSwitchTool),
        Box::new(BackendStatusTool),
        // Utils
        Box::new(FindFilesTool),
        Box::new(TreeTool),
        Box::new(DiffFilesTool),
        Box::new(PatchApplyTool),
        Box::new(HttpFetchTool),
        Box::new(FileInfoTool),
        Box::new(UndoTool),
        // Events
        Box::new(EventsSubscribeTool),
        Box::new(EventsPollTool),
        Box::new(EventsPushTool),
        // Custom tools
        Box::new(ToolRegisterTool),
        Box::new(ToolListCustomTool),
        Box::new(ToolCallCustomTool),
        Box::new(ToolRemoveCustomTool),
        Box::new(ToolViewCustomTool),
    ]
}

/// Find tool by name
pub fn find_tool(name: &str) -> Option<Box<dyn Tool>> {
    all_tools().into_iter().find(|t| t.name() == name)
}
