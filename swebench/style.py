"""Terminal styling for ash SWE-bench runner.

Cyberpunk-inspired ANSI color output. Auto-disables when not a TTY.
"""

import sys

# ==================== TTY Detection ====================

_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    """Wrap text with ANSI code if TTY."""
    if _IS_TTY:
        return f"\033[{code}m{text}\033[0m"
    return text


# ==================== Colors ====================
# Cyberpunk palette: neon magenta, electric cyan, acid green, hot pink

def bold(t: str) -> str: return _c(t, "1")
def dim(t: str) -> str: return _c(t, "2")
def italic(t: str) -> str: return _c(t, "3")
def red(t: str) -> str: return _c(t, "31")
def green(t: str) -> str: return _c(t, "32")
def yellow(t: str) -> str: return _c(t, "33")
def blue(t: str) -> str: return _c(t, "34")
def magenta(t: str) -> str: return _c(t, "35")
def cyan(t: str) -> str: return _c(t, "36")
def gray(t: str) -> str: return _c(t, "90")
def bright_cyan(t: str) -> str: return _c(t, "96")
def bright_green(t: str) -> str: return _c(t, "92")
def bright_red(t: str) -> str: return _c(t, "91")
def bright_yellow(t: str) -> str: return _c(t, "93")
def bright_magenta(t: str) -> str: return _c(t, "95")
def neon_pink(t: str) -> str: return _c(t, "38;5;198")
def neon_cyan(t: str) -> str: return _c(t, "38;5;51")
def neon_green(t: str) -> str: return _c(t, "38;5;46")
def neon_purple(t: str) -> str: return _c(t, "38;5;129")
def neon_orange(t: str) -> str: return _c(t, "38;5;208")
def electric_blue(t: str) -> str: return _c(t, "38;5;33")


# ==================== Symbols ====================

CHECK = "◆" if _IS_TTY else "[ok]"
CROSS = "✘" if _IS_TTY else "[--]"
ARROW = "▶" if _IS_TTY else ">"
DOT = "·"
BAR = "│" if _IS_TTY else "|"
BOLT = "⚡" if _IS_TTY else "*"
GEAR = "⟡" if _IS_TTY else "o"


# ==================== Formatters ====================

def header(instance_id: str) -> str:
    """Styled instance header."""
    if _IS_TTY:
        line = dim("╌" * 56)
        return f"\n{line}\n  {bold(neon_cyan(instance_id))}\n{line}"
    else:
        line = "=" * 60
        return f"\n{line}\n  {instance_id}\n{line}"


def section(title: str) -> str:
    """Section separator."""
    if _IS_TTY:
        return f"  {neon_purple('┄┄┄')} {bold(neon_cyan(title))} {neon_purple('┄┄┄')}"
    else:
        return f"--- {title} ---"


def kv(key: str, value: str, color_fn=None) -> str:
    """Key-value pair."""
    v = color_fn(value) if color_fn else value
    if _IS_TTY:
        return f"  {dim(key)}  {v}"
    else:
        return f"  {key}  {value}"


def status(ok: bool, label: str) -> str:
    """Status indicator."""
    if ok:
        return f"  {neon_green(CHECK)} {label}"
    else:
        return f"  {dim(CROSS)} {dim(label)}"


def progress(current: int, total: int) -> str:
    """Progress indicator."""
    pct = (current / total * 100) if total > 0 else 0
    if _IS_TTY:
        bar_width = 20
        filled = int(bar_width * current / total) if total > 0 else 0
        bar_str = neon_cyan("━" * filled) + dim("╌" * (bar_width - filled))
        return f"  {bar_str} {dim(f'{current}/{total}')} {neon_purple(f'({pct:.0f}%)')}"
    else:
        return f"  [{current}/{total}] ({pct:.0f}%)"


def cost(amount: float, calls: int) -> str:
    """Format cost display."""
    if _IS_TTY:
        return f"{neon_orange(f'${amount:.4f}')} {dim(f'({calls} calls)')}"
    else:
        return f"${amount:.4f} ({calls} calls)"


def patch_info(patch: str) -> str:
    """Format patch info."""
    if patch:
        return f"{neon_green(f'{len(patch)} chars')}"
    else:
        return dim("(empty)")


def summary(total: int, submitted: int, preds_path: str) -> str:
    """Batch summary."""
    lines = [
        "",
        section("Summary"),
        kv("total    ", str(total)),
        kv("patched  ", neon_green(str(submitted)) if submitted else dim("0")),
        kv("results  ", neon_cyan(str(preds_path))),
        "",
    ]
    return "\n".join(lines)


def step(n: int, kind: str, text: str, width: int = 80) -> str:
    """Format a single-line agent step for live display.

    kind: tool name ("shell", "grep_files", "text_editor", "process")
          or "error"/"think"/"done".
    text: summary of the call (will be truncated).
    """
    prefix = f"  {dim(f'[{n}]')} "

    _TOOL_TAGS = {
        "shell":       (neon_cyan,   "$"),
        "bash":        (neon_cyan,   "$"),
        "ash":         (neon_cyan,   "$"),
        "grep_files":  (neon_purple, "grep"),
        "text_editor": (neon_orange, "edit"),
        "process":     (neon_green,  "proc"),
        "web_fetch":   (bright_cyan, "fetch"),
        "web_search":  (bright_cyan, "search"),
        "error":       (neon_pink,   "!"),
    }

    color_fn, label = _TOOL_TAGS.get(kind, (dim, kind[:6]))
    tag = color_fn(label)

    # Flatten to single line and truncate
    text = text.replace("\n", " ").replace("\r", "")
    max_text = width - len(f"  [{n}] {label} ") - 2
    if len(text) > max_text:
        text = text[:max_text] + dim("…")
    return f"{prefix}{tag} {text}"


def banner() -> str:
    """SWE-bench runner banner."""
    if _IS_TTY:
        w = 39
        top = neon_cyan(f"   ╔{'═' * w}╗")
        # ⚡ is 2-cell wide in terminal, so subtract 1 from padding
        pad = w - len("  ⚡ ASH swe-bench runner") - 1
        mid = neon_cyan("   ║") + bold(neon_pink("  ⚡ ASH ")) + dim("swe-bench runner") + " " * pad + neon_cyan("║")
        bot = neon_cyan(f"   ╚{'═' * w}╝")
        return f"\n{top}\n{mid}\n{bot}\n"
    else:
        return "\n  ASH swe-bench runner\n"
