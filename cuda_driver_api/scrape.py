#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests",
#     "beautifulsoup4",
#     "html2text",
#     "lxml",
# ]
# ///
"""Scrape CUDA Driver API documentation and convert to LLM-readable markdown.

Usage:
    uv run scrape.py
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urljoin

import html2text
import requests
from bs4 import BeautifulSoup, NavigableString

BASE_URL = "https://docs.nvidia.com/cuda/cuda-driver-api/"
OUTPUT_DIR = Path(__file__).parent / "texts" / "cuda_docs"

# All module/section pages to scrape
SECTIONS = {
    # Conceptual sections
    "driver-vs-runtime-api": "Driver vs Runtime API",
    "api-sync-behavior": "API Synchronization Behavior",
    "stream-sync-behavior": "Stream Synchronization Behavior",
    "graphs-thread-safety": "Graph Object Thread Safety",
    "version-mixing-rules": "Version Mixing Rules",

    # Module sections (group__ pages)
    "group__CUDA__TYPES": "Data Types",
    "group__CUDA__ERROR": "Error Handling",
    "group__CUDA__INITIALIZE": "Initialization",
    "group__CUDA__VERSION": "Version Management",
    "group__CUDA__DEVICE": "Device Management",
    "group__CUDA__DEVICE__DEPRECATED": "Device Management (Deprecated)",
    "group__CUDA__PRIMARY__CTX": "Primary Context Management",
    "group__CUDA__CTX": "Context Management",
    "group__CUDA__CTX__DEPRECATED": "Context Management (Deprecated)",
    "group__CUDA__MODULE": "Module Management",
    "group__CUDA__MODULE__DEPRECATED": "Module Management (Deprecated)",
    "group__CUDA__LIBRARY": "Library Management",
    "group__CUDA__MEM": "Memory Management",
    "group__CUDA__VA": "Virtual Memory Management",
    "group__CUDA__MALLOC__ASYNC": "Stream Ordered Memory Allocator",
    "group__CUDA__MULTICAST": "Multicast Object Management",
    "group__CUDA__UNIFIED": "Unified Addressing",
    "group__CUDA__STREAM": "Stream Management",
    "group__CUDA__EVENT": "Event Management",
    "group__CUDA__EXTRES__INTEROP": "External Resource Interoperability",
    "group__CUDA__STREAM__MEMORY": "Stream Memory Operations",
    "group__CUDA__EXEC": "Execution Control",
    "group__CUDA__EXEC__DEPRECATED": "Execution Control (Deprecated)",
    "group__CUDA__GRAPH": "Graph Management",
    "group__CUDA__OCCUPANCY": "Occupancy",
    "group__CUDA__TEXREF": "Texture Reference Management",
    "group__CUDA__TEXREF__DEPRECATED": "Texture Reference (Deprecated)",
    "group__CUDA__SURFREF": "Surface Reference Management",
    "group__CUDA__SURFREF__DEPRECATED": "Surface Reference (Deprecated)",
    "group__CUDA__TEXOBJECT": "Texture Object Management",
    "group__CUDA__SURFOBJECT": "Surface Object Management",
    "group__CUDA__TENSOR__MEMORY": "Tensor Map Object Management",
    "group__CUDA__PEER__ACCESS": "Peer Context Memory Access",
    "group__CUDA__GRAPHICS": "Graphics Interoperability",
    "group__CUDA__GL": "OpenGL Interoperability",
    "group__CUDA__GL__DEPRECATED": "OpenGL Interoperability (Deprecated)",
    "group__CUDA__D3D9": "Direct3D 9 Interoperability",
    "group__CUDA__D3D9__DEPRECATED": "Direct3D 9 (Deprecated)",
    "group__CUDA__D3D10": "Direct3D 10 Interoperability",
    "group__CUDA__D3D10__DEPRECATED": "Direct3D 10 (Deprecated)",
    "group__CUDA__D3D11": "Direct3D 11 Interoperability",
    "group__CUDA__D3D11__DEPRECATED": "Direct3D 11 (Deprecated)",
    "group__CUDA__VDPAU": "VDPAU Interoperability",
    "group__CUDA__EGL": "EGL Interoperability",
    "group__CUDA__PROFILER": "Profiler Control",
    "group__CUDA__GREEN__CONTEXTS": "Green Contexts",
    "group__CUDA__COREDUMP": "Coredump Attributes Control",
}

# Data structures page
DATA_STRUCTURES = "annotated"


def create_html2text_converter() -> html2text.HTML2Text:
    """Create a configured HTML2Text converter."""
    h = html2text.HTML2Text()
    h.body_width = 0  # No line wrapping
    h.ignore_links = False
    h.ignore_images = True
    h.ignore_emphasis = False
    h.skip_internal_links = False
    h.inline_links = True
    h.protect_links = True
    h.unicode_snob = True
    h.escape_snob = False
    return h


def fetch_page(url: str, max_retries: int = 3) -> str | None:
    """Fetch a page with retries."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CUDADocScraper/1.0)"
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def clean_text(text: str) -> str:
    """Clean up converted markdown text."""
    # Remove excessive blank lines
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    # Remove trailing whitespace
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    # Fix code blocks
    text = re.sub(r'```\n\n+', '```\n', text)
    text = re.sub(r'\n\n+```', '\n```', text)
    return text.strip()


def extract_main_content(soup: BeautifulSoup) -> BeautifulSoup:
    """Extract the main content area from the page."""
    # Try different content selectors
    main = soup.find('div', class_='contents')
    if not main:
        main = soup.find('div', id='content')
    if not main:
        main = soup.find('main')
    if not main:
        main = soup.find('article')
    if not main:
        main = soup.body
    return main


def process_function_doc(func_div) -> str:
    """Process a function documentation block into clean markdown."""
    lines = []

    # Get function name from the anchor
    anchor = func_div.find('a', class_='anchor')
    if anchor:
        func_id = anchor.get('id', '')
        if func_id:
            lines.append(f"### {func_id}\n")

    # Get the member title (function signature)
    memtitle = func_div.find('td', class_='memname') or func_div.find('div', class_='memtitle')
    if memtitle:
        # Get full signature from memitem table
        memproto = func_div.find('table', class_='memname')
        if memproto:
            sig_text = memproto.get_text(' ', strip=True)
            sig_text = re.sub(r'\s+', ' ', sig_text)
            lines.append(f"```c\n{sig_text}\n```\n")

    # Get description
    memdoc = func_div.find('div', class_='memdoc')
    if memdoc:
        converter = create_html2text_converter()
        doc_html = str(memdoc)
        doc_md = converter.handle(doc_html)
        lines.append(doc_md)

    return '\n'.join(lines)


def scrape_module_page(url: str, title: str) -> str:
    """Scrape a module documentation page and convert to markdown."""
    print(f"  Fetching: {url}")
    html = fetch_page(url)
    if not html:
        return f"# {title}\n\nFailed to fetch documentation.\n"

    soup = BeautifulSoup(html, 'lxml')
    main_content = extract_main_content(soup)

    # Remove navigation elements
    for elem in main_content.find_all(['nav', 'header', 'footer']):
        elem.decompose()
    for elem in main_content.find_all(class_=['breadcrumb', 'nav', 'sidebar']):
        elem.decompose()

    # Convert to markdown
    converter = create_html2text_converter()
    md_content = converter.handle(str(main_content))

    # Clean up
    md_content = clean_text(md_content)

    # Add title header
    final = f"# {title}\n\n{md_content}\n"
    return final


def scrape_conceptual_page(url: str, title: str) -> str:
    """Scrape a conceptual documentation page."""
    print(f"  Fetching: {url}")
    html = fetch_page(url)
    if not html:
        return f"# {title}\n\nFailed to fetch documentation.\n"

    soup = BeautifulSoup(html, 'lxml')
    main_content = extract_main_content(soup)

    # Remove navigation
    for elem in main_content.find_all(['nav', 'header', 'footer']):
        elem.decompose()

    converter = create_html2text_converter()
    md_content = converter.handle(str(main_content))
    md_content = clean_text(md_content)

    return f"# {title}\n\n{md_content}\n"


def slugify(text: str) -> str:
    """Convert title to filename-safe slug."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = re.sub(r'-+', '-', text)
    return text.strip('-')[:60]


def write_section_file(output_dir: Path, section_num: int, key: str, title: str, content: str) -> Path:
    """Write a section file."""
    slug = slugify(title)
    filename = f"{section_num:02d}-{slug}.md"
    out_path = output_dir / filename
    out_path.write_text(content, encoding='utf-8')
    print(f"  Wrote: {filename} ({len(content)} bytes)")
    return out_path


def build_index(sections: list[tuple[int, str, str, Path]], output_dir: Path) -> None:
    """Build an index file mapping sections to files."""
    index_path = output_dir / '_index.txt'
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write("# CUDA Driver API Documentation Index\n")
        f.write("# Section Number | Title | Filename\n\n")
        for num, key, title, path in sections:
            f.write(f"{num:02d}\t{title}\t{path.name}\n")
    print(f"  Wrote index: {index_path}")


def build_full_doc(sections: list[tuple[int, str, str, Path]], output_dir: Path) -> None:
    """Combine all sections into a single full document."""
    full_path = output_dir / 'cuda_driver_api_full.md'

    with open(full_path, 'w', encoding='utf-8') as f:
        f.write("# CUDA Driver API Reference\n\n")
        f.write("This document contains the complete CUDA Driver API reference documentation.\n\n")
        f.write("## Table of Contents\n\n")

        for num, key, title, path in sections:
            f.write(f"- [{title}](#{slugify(title)})\n")

        f.write("\n---\n\n")

        for num, key, title, path in sections:
            content = path.read_text(encoding='utf-8')
            f.write(content)
            f.write("\n\n---\n\n")

    print(f"  Wrote full doc: {full_path}")


def main():
    output_dir = OUTPUT_DIR / "sections"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("CUDA Driver API Documentation Scraper")
    print("=" * 50)

    sections_written: list[tuple[int, str, str, Path]] = []
    section_num = 1

    # Scrape all sections
    for key, title in SECTIONS.items():
        print(f"\n[{section_num}/{len(SECTIONS)}] {title}")

        if key.startswith("group__"):
            url = urljoin(BASE_URL, f"{key}.html")
            content = scrape_module_page(url, title)
        else:
            url = urljoin(BASE_URL, f"{key}.html")
            content = scrape_conceptual_page(url, title)

        out_path = write_section_file(output_dir, section_num, key, title, content)
        sections_written.append((section_num, key, title, out_path))
        section_num += 1

        # Be nice to the server
        time.sleep(0.5)

    # Scrape data structures page
    print(f"\n[{section_num}] Data Structures (annotated)")
    url = urljoin(BASE_URL, f"{DATA_STRUCTURES}.html")
    content = scrape_module_page(url, "Data Structures")
    out_path = write_section_file(output_dir, section_num, "data-structures", "Data Structures", content)
    sections_written.append((section_num, "data-structures", "Data Structures", out_path))

    print("\n" + "=" * 50)
    print("Building index and full document...")

    build_index(sections_written, OUTPUT_DIR)
    build_full_doc(sections_written, OUTPUT_DIR)

    print(f"\nDone! Created {len(sections_written)} section files.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
