"""Shared Excalidraw generation: element builders + geometry validation.

Extracted from gen_architecture.py when a second diagram (the checkpoint flow)
needed the same machinery. The point of generating at all: the hand-written
diagrams went wrong twice -- typos that made the file unopenable, and text that
silently overflowed its box or boxes that overlapped, which looks fine to whoever
wrote it and is unreadable to whoever opens it. A :class:`Canvas` refuses to
write a file with any of those problems.

Font metrics approximate Excalidraw's default (Virgil/Excalifont), deliberately
pessimistic: a CJK glyph is a full em, latin ~0.58, so the checker over-estimates
width rather than passing something that will clip.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

BLUE = "#4dabf7"
GREEN = "#51cf66"
ORANGE = "#ffa94d"
YELLOW = "#ffd43b"
GREY = "#a0a0a0"
WHITE = "#e8e8e8"
VIOLET = "#b197fc"
RED = "#ff8787"

PAD = 14          # inner padding between a box edge and its text


def _char_w(ch: str, size: int) -> float:
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return size * 1.0
    return size * 0.58


def text_size(body: str, size: int) -> "tuple[float, float]":
    lines = body.split("\n")
    width = max((sum(_char_w(c, size) for c in line) for line in lines), default=0)
    return width, len(lines) * size * 1.25


class Canvas:
    """Collects elements; validates geometry; writes the document."""

    def __init__(self) -> None:
        self.els: list = []
        #: (box_id, [text_id, ...]) -- validate() checks each text fits its box
        self.contains: list = []
        self._seed = 0

    def _next_seed(self) -> int:
        self._seed += 1
        return self._seed

    # --- element builders ---------------------------------------------------
    def text(self, id_: str, x: float, y: float, body: str, *, size: int = 13,
             color: str = WHITE) -> dict:
        width, height = text_size(body, size)
        element = {
            "id": id_, "type": "text", "x": x, "y": y,
            "width": round(width, 1), "height": round(height, 1),
            "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 1, "strokeStyle": "solid",
            "roughness": 1, "opacity": 100, "groupIds": [],
            "roundness": None, "seed": self._next_seed(), "version": 1,
            "isDeleted": False, "text": body, "fontSize": size, "fontFamily": 1,
            "textAlign": "left", "verticalAlign": "top", "containerId": None,
            "originalText": body, "lineHeight": 1.25, "boundElements": [],
            "link": None, "locked": False, "autoResize": True,
        }
        self.els.append(element)
        return element

    def box(self, id_: str, x: float, y: float, w: float, h: float, *,
            color: str = BLUE, dashed: bool = False) -> dict:
        element = {
            "id": id_, "type": "rectangle", "x": x, "y": y, "width": w,
            "height": h, "angle": 0, "strokeColor": color,
            "backgroundColor": "transparent", "fillStyle": "solid",
            "strokeWidth": 1, "strokeStyle": "dashed" if dashed else "solid",
            "roughness": 1, "opacity": 100, "groupIds": [],
            "roundness": {"type": 3}, "seed": self._next_seed(), "version": 1,
            "isDeleted": False, "boundElements": [], "link": None,
            "locked": False,
        }
        self.els.append(element)
        return element

    def arrow(self, id_: str, x: float, y: float, points: list, *,
              color: str = WHITE, dashed: bool = False,
              head: "str | None" = "arrow") -> dict:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        element = {
            "id": id_, "type": "arrow", "x": x, "y": y,
            "width": max(xs) - min(xs), "height": max(1, max(ys) - min(ys)),
            "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 1,
            "strokeStyle": "dashed" if dashed else "solid",
            "roughness": 1, "opacity": 100, "groupIds": [],
            "roundness": {"type": 2}, "seed": self._next_seed(), "version": 1,
            "isDeleted": False, "points": points, "lastCommittedPoint": None,
            "startBinding": None, "endBinding": None, "startArrowhead": None,
            "endArrowhead": head, "boundElements": [], "link": None,
            "locked": False,
        }
        self.els.append(element)
        return element

    def panel(self, id_: str, x: float, y: float, w: float, title: str,
              body: str, *, color: str = BLUE, dashed: bool = False,
              title_size: int = 14, body_size: int = 12) -> dict:
        """A titled box sized to its contents. Returns its geometry.

        ``w`` is a *minimum*: the box widens to whatever its text needs.
        Trusting a hand-picked width is how the first diagram ended up with
        clipped lines -- the width looked plausible, and nothing checked.
        """
        tw, th = text_size(title, title_size)
        bw, bh = text_size(body, body_size) if body else (0, 0)
        w = max(w, tw + 2 * PAD + 6, bw + 2 * PAD + 6)
        h = PAD + th + (6 + bh if body else 0) + PAD
        self.box(id_, x, y, w, h, color=color, dashed=dashed)
        ids = [id_ + "-t"]
        self.text(id_ + "-t", x + PAD, y + PAD, title, size=title_size,
                  color=color)
        if body:
            self.text(id_ + "-b", x + PAD, y + PAD + th + 6, body,
                      size=body_size)
            ids.append(id_ + "-b")
        self.contains.append((id_, ids))
        return {"id": id_, "x": x, "y": y, "w": w, "h": h,
                "right": x + w, "bottom": y + h,
                "cx": x + w / 2}

    # --- validation -----------------------------------------------------------
    def validate(self, containers: "set[str] | None" = None) -> list:
        """Every geometric mistake the hand-written diagrams actually made."""
        containers = containers or set()
        problems = []
        by_id = {e["id"]: e for e in self.els}

        # 1. text must fit inside the box it is declared to live in
        for box_id, text_ids in self.contains:
            b = by_id[box_id]
            for tid in text_ids:
                t = by_id[tid]
                if t["x"] < b["x"] or t["y"] < b["y"]:
                    problems.append("%s starts before its box %s" % (tid, box_id))
                if t["x"] + t["width"] > b["x"] + b["width"] - 4:
                    problems.append(
                        "%s overflows %s horizontally by %.0fpx"
                        % (tid, box_id,
                           t["x"] + t["width"] - (b["x"] + b["width"])))
                if t["y"] + t["height"] > b["y"] + b["height"] - 2:
                    problems.append(
                        "%s overflows %s vertically by %.0fpx"
                        % (tid, box_id,
                           t["y"] + t["height"] - (b["y"] + b["height"])))

        # 2. panel boxes must not overlap (a declared container may hold panels)
        rects = [e for e in self.els if e["type"] == "rectangle"]
        panels = [b for b in rects if b["id"] not in containers]
        for i, a in enumerate(panels):
            for b in panels[i + 1:]:
                if (a["x"] < b["x"] + b["width"] and b["x"] < a["x"] + a["width"]
                        and a["y"] < b["y"] + b["height"]
                        and b["y"] < a["y"] + a["height"]):
                    problems.append("boxes %s and %s overlap" % (a["id"], b["id"]))

        # 3. free-standing text must not collide with other free-standing text
        contained = {t for _, ts in self.contains for t in ts}
        loose = [e for e in self.els
                 if e["type"] == "text" and e["id"] not in contained]
        for i, a in enumerate(loose):
            for b in loose[i + 1:]:
                if (a["x"] < b["x"] + b["width"] and b["x"] < a["x"] + a["width"]
                        and a["y"] < b["y"] + b["height"]
                        and b["y"] < a["y"] + a["height"]):
                    problems.append("labels %s and %s collide" % (a["id"], b["id"]))

        # 4. everything x-inside a container must fit it vertically too
        for cid in containers:
            c = by_id.get(cid)
            if c is None:
                continue
            for e in rects:
                if e["id"] in containers:
                    continue
                inside_x = (c["x"] <= e["x"]
                            and e["x"] + e["width"] <= c["x"] + c["width"])
                straddles_y = (c["y"] < e["y"] + e["height"]
                               and e["y"] < c["y"] + c["height"])
                if straddles_y and inside_x:
                    if e["y"] + e["height"] > c["y"] + c["height"]:
                        problems.append("%s escapes container %s vertically"
                                        % (e["id"], cid))
        return problems

    def write(self, path: Path, containers: "set[str] | None" = None) -> None:
        problems = self.validate(containers)
        if problems:
            raise SystemExit(
                "REFUSING to write -- %d geometry problem(s):\n  %s"
                % (len(problems), "\n  ".join(problems)))
        document = {
            "type": "excalidraw", "version": 2, "source": "claude",
            "appState": {"gridSize": None, "viewBackgroundColor": "#121212"},
            "files": {}, "elements": self.els,
        }
        Path(path).write_text(json.dumps(document, ensure_ascii=False, indent=1))
        print("wrote %s (%d elements, geometry checks passed)"
              % (path, len(self.els)))
