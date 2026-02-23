"""Terminal styling for ash SWE-bench runner.

Lightweight ANSI color output. Auto-disables when not a TTY.
"""

import sys

# ==================== TTY Detection ====================

_IS_TTY = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


def _c(text: str, code: str) -> str:
    """Wrap text with ANSI code if TTY."""
    if _IS_TTY:
        return f"\033[{code}m{text}\033[0m"
    return text


# ==================== Colors ====================

def bold(t: str) -> str: return _c(t, "1")
def dim(t: str) -> str: return _c(t, "2")
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


# ==================== Symbols ====================

CHECK = "●" if _IS_TTY else "ok"
CROSS = "○" if _IS_TTY else "--"
ARROW = "▸" if _IS_TTY else ">"
DOT = "·"
BAR = "│" if _IS_TTY else "|"


# ==================== Formatters ====================

def header(instance_id: str) -> str:
    """Styled instance header."""
    if _IS_TTY:
        line = dim("─" * 56)
        return f"\n{line}\n  {bold(bright_cyan(instance_id))}\n{line}"
    else:
        line = "=" * 60
        return f"\n{line}\n  {instance_id}\n{line}"


def section(title: str) -> str:
    """Section separator."""
    if _IS_TTY:
        return f"{dim('───')} {bold(title)} {dim('───')}"
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
        return f"  {green(CHECK)} {label}"
    else:
        return f"  {dim(CROSS)} {dim(label)}"


def progress(current: int, total: int) -> str:
    """Progress indicator."""
    pct = (current / total * 100) if total > 0 else 0
    if _IS_TTY:
        bar_width = 20
        filled = int(bar_width * current / total) if total > 0 else 0
        bar_str = "█" * filled + dim("░" * (bar_width - filled))
        return f"  {bar_str} {dim(f'{current}/{total}')} {dim(f'({pct:.0f}%)')}"
    else:
        return f"  [{current}/{total}] ({pct:.0f}%)"


def cost(amount: float, calls: int) -> str:
    """Format cost display."""
    if _IS_TTY:
        return f"{yellow(f'${amount:.4f}')} {dim(f'({calls} calls)')}"
    else:
        return f"${amount:.4f} ({calls} calls)"


def patch_info(patch: str) -> str:
    """Format patch info."""
    if patch:
        return f"{green(f'{len(patch)} chars')}"
    else:
        return dim("(empty)")


def summary(total: int, submitted: int, preds_path: str) -> str:
    """Batch summary."""
    lines = [
        "",
        section("Summary"),
        kv("total    ", str(total)),
        kv("patched  ", bright_green(str(submitted)) if submitted else dim("0")),
        kv("results  ", cyan(str(preds_path))),
        "",
    ]
    return "\n".join(lines)


def step(n: int, kind: str, text: str, width: int = 80) -> str:
    """Format a single-line agent step for live display.

    kind: "bash", "think", "done", "error"
    text: command or assistant message (will be truncated).
    """
    prefix = f"  {dim(f'[{n}]')} "
    if kind in ("bash", "ash"):
        tag = cyan("$")
    elif kind == "error":
        tag = bright_red("!")
    else:
        tag = dim("…")

    # Truncate text to fit in one line
    max_text = width - len(f"  [{n}] X ") - 2
    if len(text) > max_text:
        text = text[:max_text] + dim("…")
    return f"{prefix}{tag} {text}"


def banner() -> str:
    """SWE-bench runner banner."""
    if _IS_TTY:
        return f"\n  {bold(bright_cyan('ash'))} {dim('swe-bench runner')}\n"
    else:
        return "\nash swe-bench runner\n"
