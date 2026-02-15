//! Tool registry

pub mod view;
pub mod grep;
pub mod edit;
pub mod shell;
pub mod git;
pub mod clip;
pub mod session;

pub use view::ViewTool;
pub use grep::GrepTool;
pub use edit::EditTool;
pub use shell::ShellTool;
pub use git::{GitStatusTool, GitDiffTool, GitLogTool};
pub use clip::{ClipTool, PasteTool, ClipsTool, ClearClipsTool};
pub use session::{SessionCreateTool, SessionListTool, SessionDestroyTool};

use crate::Tool;

/// All available tools
pub fn all_tools() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(ViewTool),
        Box::new(GrepTool),
        Box::new(EditTool),
        Box::new(ShellTool),
        Box::new(GitStatusTool),
        Box::new(GitDiffTool),
        Box::new(GitLogTool),
        Box::new(ClipTool),
        Box::new(PasteTool),
        Box::new(ClipsTool),
        Box::new(ClearClipsTool),
        // Session management
        Box::new(SessionCreateTool),
        Box::new(SessionListTool),
        Box::new(SessionDestroyTool),
    ]
}

/// Find tool by name
pub fn find_tool(name: &str) -> Option<Box<dyn Tool>> {
    all_tools().into_iter().find(|t| t.name() == name)
}
