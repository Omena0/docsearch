"""Tests for Markdown parsing."""

from docsearch.markdown import extract_section_text, get_heading_at_line, parse_markdown


def test_parse_simple_hierarchy():
    content = """# H1
Content 1
## H2
Content 2
### H3
Content 3
## H2 again
Content 4
"""
    parsed = parse_markdown("test.md", content)
    assert len(parsed.headings) == 4
    assert parsed.headings[0].level == 1
    assert parsed.headings[0].text == "H1"
    assert parsed.headings[1].level == 2
    assert parsed.headings[1].text == "H2"
    assert parsed.headings[2].level == 3
    assert parsed.headings[2].text == "H3"
    assert parsed.headings[3].level == 2
    assert parsed.headings[3].text == "H2 again"


def test_nested_headings_parent_child():
    content = """# Parent
## Child
### Grandchild
"""
    parsed = parse_markdown("test.md", content)
    assert parsed.headings[0].parent is None
    assert parsed.headings[1].parent == parsed.headings[0]
    assert parsed.headings[2].parent == parsed.headings[1]
    assert parsed.headings[0].children == [parsed.headings[1]]
    assert parsed.headings[1].children == [parsed.headings[2]]


def test_multiple_h1s():
    content = """# First
Content
# Second
Content
"""
    parsed = parse_markdown("test.md", content)
    assert len(parsed.headings) == 2
    assert parsed.headings[0].text == "First"
    assert parsed.headings[1].text == "Second"
    assert parsed.headings[0].parent is None
    assert parsed.headings[1].parent is None


def test_no_headings():
    content = """Just some text.
More text.
"""
    parsed = parse_markdown("test.md", content)
    assert len(parsed.headings) == 0


def test_headings_in_code_block():
    content = """# Real Heading
```text
# Not a heading
## Also not a heading
```
## Real H2
"""
    parsed = parse_markdown("test.md", content)
    assert len(parsed.headings) == 2
    assert parsed.headings[0].text == "Real Heading"
    assert parsed.headings[1].text == "Real H2"


def test_exact_line_numbers():
    content = """Line 1
# H1 at line 2
Line 4
## H2 at line 4
Line 7
"""
    parsed = parse_markdown("test.md", content)
    assert parsed.headings[0].start_line == 2
    assert parsed.headings[1].start_line == 4


def test_end_line_calculation():
    content = """# H1
content
## H2
content
# H1 again
"""
    parsed = parse_markdown("test.md", content)
    # H1 ends before H2 (H2 starts at line 3)
    assert parsed.headings[0].end_line == 2
    # H2 ends before next H1 (next H1 starts at line 5)
    assert parsed.headings[1].end_line == 4
    # Last H1 ends at end of file (5 lines)
    assert parsed.headings[2].end_line == 5


def test_heading_path():
    content = """# Kernel
## Processes
### fork
"""
    parsed = parse_markdown("test.md", content)
    h3 = parsed.headings[2]
    assert h3.path == ("Kernel", "Processes", "fork")


def test_get_heading_at_line():
    content = """# H1
content
## H2
content
"""
    parsed = parse_markdown("test.md", content)
    # Line 2 is in H1
    h = get_heading_at_line(parsed, 2)
    assert h is not None
    assert h.text == "H1"
    # Line 4 is in H2
    h = get_heading_at_line(parsed, 4)
    assert h is not None
    assert h.text == "H2"


def test_extract_section_text():
    content = """# H1
line 2
line 3
## H2
line 5
"""
    parsed = parse_markdown("test.md", content)
    text = extract_section_text(parsed, 1, 3)
    assert "H1" in text
    assert "line 2" in text
    assert "line 3" in text