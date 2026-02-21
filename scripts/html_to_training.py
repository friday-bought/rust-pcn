#!/usr/bin/env python3
"""Convert HTML doc sets into plain text training files for PCN.

Each subdirectory in the source becomes one .txt file in the output,
with all HTML files concatenated and converted to markdown.

Usage:
    python3 scripts/html_to_training.py data/books-code/ data/books-code-txt/
"""

import html2text
import os
import sys
from pathlib import Path


def make_converter():
    h = html2text.HTML2Text()
    h.body_width = 0          # no artificial line wrapping
    h.protect_links = False
    h.unicode_snob = True
    h.skip_internal_links = True
    h.ignore_images = True
    h.ignore_emphasis = False
    return h


def convert_file(path, converter):
    """Convert a single HTML file to markdown text."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        return converter.handle(content)
    except Exception as e:
        print(f"  Warning: failed to convert {path}: {e}", file=sys.stderr)
        return ""


def process_docset(docset_dir, output_dir, converter):
    """Process all HTML files in a doc set directory into one .txt file."""
    docset_name = docset_dir.name
    html_files = sorted(docset_dir.rglob("*.html"))

    if not html_files:
        print(f"  Skipping {docset_name}: no HTML files")
        return

    chunks = []
    for html_file in html_files:
        md = convert_file(html_file, converter)
        if md.strip():
            # Add a separator between documents
            page_name = html_file.stem.replace(".", " ").replace("-", " ").title()
            chunks.append(f"\n## {page_name}\n\n{md.strip()}\n")

    if not chunks:
        print(f"  Skipping {docset_name}: no content after conversion")
        return

    combined = f"# {docset_name} Documentation\n\n" + "\n".join(chunks)

    output_path = output_dir / f"{docset_name}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined)

    size_kb = output_path.stat().st_size / 1024
    print(f"  {docset_name}: {len(html_files)} files -> {size_kb:.0f} KB")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <source_dir> <output_dir>")
        sys.exit(1)

    source_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not source_dir.is_dir():
        print(f"Error: {source_dir} is not a directory")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    converter = make_converter()
    docsets = sorted([d for d in source_dir.iterdir() if d.is_dir()])

    print(f"Converting {len(docsets)} doc sets from {source_dir} -> {output_dir}")

    for docset in docsets:
        process_docset(docset, output_dir, converter)

    total_files = list(output_dir.glob("*.txt"))
    total_size = sum(f.stat().st_size for f in total_files) / (1024 * 1024)
    print(f"\nDone: {len(total_files)} training files, {total_size:.1f} MB total")


if __name__ == "__main__":
    main()
