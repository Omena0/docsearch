"""Markdown parsing with line number preservation."""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Heading:
    """A Markdown heading with line info."""

    level: int
    text: str
    start_line: int
    end_line: int
    parent: Optional["Heading"] = None
    children: list["Heading"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    @property
    def path(self) -> tuple[str, ...]:
        """Get heading path from root."""
        path = []
        current = self
        while current is not None:
            if current.text:
                path.append(current.text)
            current = current.parent
        return tuple(reversed(path))


@dataclass
class ParsedDocument:
    """Parsed Markdown document."""

    file: str
    lines: list[str]
    headings: list[Heading]
    heading_map: dict[int, Heading]  # line -> heading


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
FENCE_RE = re.compile(r"^```|^~~~")


def parse_markdown(file: str, content: str) -> ParsedDocument:
    """Parse Markdown into headings with exact line numbers."""
    lines = content.splitlines(keepends=False)
    headings = []
    heading_map = {}
    stack: list[Heading] = []
    in_fence = False
    fence_char = ""

    for i, line in enumerate(lines, 1):
        # Check for fence start/end
        fence_match = FENCE_RE.match(line)
        if fence_match:
            if not in_fence:
                in_fence = True
                fence_char = line[:3]
            elif line.startswith(fence_char):
                in_fence = False
                fence_char = ""
            continue

        if in_fence:
            continue

        # Check for heading
        match = HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()

            # Pop stack to find parent
            while stack and stack[-1].level >= level:
                stack.pop()

            parent = stack[-1] if stack else None
            heading = Heading(
                level=level,
                text=text,
                start_line=i,
                end_line=len(lines),  # Will update later
                parent=parent,
            )
            if parent:
                parent.children.append(heading)
            stack.append(heading)
            headings.append(heading)
            heading_map[i] = heading

    # Update end_lines by finding next heading at same or higher level
    for idx, heading in enumerate(headings):
        if idx + 1 < len(headings):
            next_h = headings[idx + 1]
            heading.end_line = next_h.start_line - 1
        else:
            heading.end_line = len(lines)

    return ParsedDocument(
        file=file,
        lines=lines,
        headings=headings,
        heading_map=heading_map,
    )


def get_heading_at_line(parsed: ParsedDocument, line: int) -> Heading | None:
    """Get the deepest heading that contains the given line."""
    # Find the last heading that starts at or before this line
    best = None
    for h in parsed.headings:
        if h.start_line <= line <= h.end_line:
            best = h
    return best


def get_section_lines(parsed: ParsedDocument, heading: Heading) -> tuple[int, int]:
    """Get the line range for a heading section."""
    return (heading.start_line, heading.end_line)


def extract_section_text(parsed: ParsedDocument, start_line: int, end_line: int) -> str:
    """Extract text from line range (1-indexed, inclusive)."""
    # Convert to 0-indexed
    return "\n".join(parsed.lines[start_line - 1 : end_line])