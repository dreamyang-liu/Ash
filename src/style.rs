//! Terminal styling - ANSI colors & visual formatting
//!
//! Lightweight, zero-dependency styling for ash CLI output.
//! Colors are auto-disabled when stdout is not a TTY.

use std::io::IsTerminal;

// ==================== ANSI Codes ====================

pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";
pub const DIM: &str = "\x1b[2m";
pub const ITALIC: &str = "\x1b[3m";

// Colors
pub const RED: &str = "\x1b[31m";
pub const GREEN: &str = "\x1b[32m";
pub const YELLOW: &str = "\x1b[33m";
pub const BLUE: &str = "\x1b[34m";
pub const MAGENTA: &str = "\x1b[35m";
pub const CYAN: &str = "\x1b[36m";
pub const WHITE: &str = "\x1b[37m";
pub const GRAY: &str = "\x1b[90m";

// Bright colors
pub const BRIGHT_RED: &str = "\x1b[91m";
pub const BRIGHT_GREEN: &str = "\x1b[92m";
pub const BRIGHT_YELLOW: &str = "\x1b[93m";
pub const BRIGHT_BLUE: &str = "\x1b[94m";
pub const BRIGHT_MAGENTA: &str = "\x1b[95m";
pub const BRIGHT_CYAN: &str = "\x1b[96m";

// ==================== TTY Detection ====================

/// Check if stderr is a terminal (for daemon/gateway logs)
pub fn stderr_is_tty() -> bool {
    std::io::stderr().is_terminal()
}

/// Check if stdout is a terminal
pub fn stdout_is_tty() -> bool {
    std::io::stdout().is_terminal()
}

// ==================== Styled Output Helpers ====================

/// Wrap text with color, respecting TTY
pub fn color(text: &str, code: &str) -> String {
    if stdout_is_tty() {
        format!("{}{}{}", code, text, RESET)
    } else {
        text.to_string()
    }
}

/// Wrap text with color for stderr output
pub fn ecolor(text: &str, code: &str) -> String {
    if stderr_is_tty() {
        format!("{}{}{}", code, text, RESET)
    } else {
        text.to_string()
    }
}

/// Bold text
pub fn bold(text: &str) -> String {
    color(text, BOLD)
}

/// Dim text
pub fn dim(text: &str) -> String {
    color(text, DIM)
}

// ==================== Symbols ====================

pub fn check() -> &'static str { if stdout_is_tty() { "●" } else { "ok" } }
pub fn cross() -> &'static str { if stdout_is_tty() { "○" } else { "--" } }
pub fn dot() -> &'static str { "·" }
pub fn arrow() -> &'static str { if stdout_is_tty() { "▸" } else { ">" } }
pub fn bar() -> &'static str { if stdout_is_tty() { "│" } else { "|" } }

// ==================== ASCII Art ====================

/// Ash logo banner (small, clean)
pub const ASH_BANNER: &str = r#"
       _
  __ _(_)
 / _` |___ | |__
| (_| |___ | '_ \
 \__,_|___||_| |_|
"#;

/// Compact one-line banner
pub fn banner_line(version: &str) -> String {
    if stdout_is_tty() {
        format!(
            "{}ash{} {}v{}{} {} code agent cli {}",
            BOLD, RESET,
            DIM, version, RESET,
            DIM, RESET,
        )
    } else {
        format!("ash v{}", version)
    }
}

/// Gateway startup banner (for stderr)
pub fn gateway_banner(version: &str) -> String {
    if stderr_is_tty() {
        format!(
            "\n  {BOLD}{BRIGHT_CYAN}ash gateway{RESET} {DIM}v{version}{RESET}\n"
        )
    } else {
        format!("ash gateway v{}", version)
    }
}

/// MCP server startup banner (for stderr)
pub fn mcp_banner(version: &str, transport: &str) -> String {
    if stderr_is_tty() {
        format!(
            "\n  {BOLD}{BRIGHT_MAGENTA}ash mcp{RESET} {DIM}v{version}{RESET} {DIM}({transport}){RESET}\n"
        )
    } else {
        format!("ash-mcp v{} ({})", version, transport)
    }
}

// ==================== Info Display ====================

/// Format a status line: label with colored value
pub fn status_line(label: &str, value: &str, ok: bool) -> String {
    let mark = if ok {
        color(check(), GREEN)
    } else {
        color(cross(), &format!("{DIM}"))
    };
    let val = if ok {
        color(value, GREEN)
    } else {
        color(value, &format!("{DIM}"))
    };
    format!("  {} {:<8} {}", mark, label, val)
}

/// Format a key-value pair
pub fn kv(key: &str, value: &str) -> String {
    if stdout_is_tty() {
        format!("  {DIM}{key}{RESET}  {value}")
    } else {
        format!("  {}  {}", key, value)
    }
}

/// Section header
pub fn section(title: &str) -> String {
    if stdout_is_tty() {
        format!("{DIM}─── {RESET}{BOLD}{title}{RESET} {DIM}───{RESET}")
    } else {
        format!("--- {} ---", title)
    }
}

/// Format tool name and description
pub fn tool_entry(name: &str, desc: &str) -> String {
    if stdout_is_tty() {
        format!("  {CYAN}{name}{RESET}  {DIM}{desc}{RESET}")
    } else {
        format!("  {}  {}", name, desc)
    }
}

/// Format a gateway log line (for stderr)
pub fn elog(prefix: &str, msg: &str) -> String {
    if stderr_is_tty() {
        format!("  {DIM}{prefix}{RESET} {msg}")
    } else {
        format!("{} {}", prefix, msg)
    }
}

/// Format uptime in human-friendly form
pub fn format_uptime(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}
