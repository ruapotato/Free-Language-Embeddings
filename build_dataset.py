#!/usr/bin/env python3
"""
Build Dataset — Download, process, and prepare all DFSG-compliant training data.
================================================================================

V3: Common Pile (safe subset) + existing Linux specialization data.
All sources are human-written with explicit licenses permitting redistribution
and derivative works. No Common Crawl derivatives. No AI-generated content.

Sources (10B token target):
  General Text (~5B tokens, 50%):
    - Wikipedia (Common Pile wikimedia, CC-BY-SA)
    - Project Gutenberg (Common Pile gutenberg, Public Domain)

  Technical Q&A (~2B tokens, 20%):
    - Stack Exchange (Common Pile stackexchange, CC-BY-SA)

  Code (~1B tokens, 10%):
    - The Stack V2 FOSS subset (Common Pile the_stack_v2, per-file FOSS)

  Scientific (~1B tokens, 10%):
    - peS2o (Common Pile peS2o, CC licenses)
    - arXiv papers (Common Pile arxiv_papers, CC licenses)

  Linux Specialization (~365M tokens, 3.6%):
    - Existing V2 Linux data (man pages, kernel docs, RFCs, etc.)

  Government/Legal (~635M tokens, 6.4%):
    - USGPO (Common Pile usgpo, Public Domain)

  SKIPPED Common Pile sources (not DFSG-safe enough):
    - CCCC (web scrape with unreliable license detection)
    - Creative Commons YouTube (uploader self-declaration unreliable)

Output: data/pretrain/*.jsonl — one file per source, ready for train_pretrain.py

Usage:
    python build_dataset.py                    # build all sources
    python build_dataset.py --source wikipedia # build single source
    python build_dataset.py --list             # list sources and status
    python build_dataset.py --stats            # show token count estimates
"""

import argparse
import json
import os
import re
import subprocess
import sys
import html
import hashlib
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/pretrain")
RAW_DIR = Path("data/raw")
TOKENIZER_NAME = "HuggingFaceTB/cosmo2-tokenizer"

# Minimum text length (chars) to keep a document
MIN_DOC_LENGTH = 100

# Stack Exchange: minimum post score to include
SE_MIN_SCORE = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    print(f"[build_dataset] {msg}", flush=True)


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def count_lines(path):
    """Count lines in a JSONL file."""
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for _ in f:
            count += 1
    return count


def estimate_tokens(path):
    """Estimate token count from file size (rough: 1 byte ≈ 0.3 tokens for English)."""
    if not path.exists():
        return 0
    size_bytes = path.stat().st_size
    return int(size_bytes * 0.3)


def write_doc(f, text, source, license_tag):
    """Write a single document to JSONL output."""
    text = text.strip()
    if len(text) < MIN_DOC_LENGTH:
        return False
    doc = {"text": text, "source": source, "license": license_tag}
    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return True


def strip_html(text):
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Common Pile helper
# ---------------------------------------------------------------------------

def _build_common_pile_source(cp_name, out_path, source_label, license_tag):
    """Download and process a Common Pile source via HuggingFace datasets.

    Common Pile sources are available as common-pile/{cp_name} on HuggingFace.
    Each provides streaming access to pre-processed, deduplicated documents.
    """
    from datasets import load_dataset

    log(f"  Downloading common-pile/{cp_name} (streaming)...")

    try:
        ds = load_dataset(f"common-pile/{cp_name}", split="train", streaming=True)
    except Exception as e:
        log(f"  ERROR: Could not load common-pile/{cp_name}: {e}")
        log("  Make sure 'datasets' is installed: pip install datasets")
        return

    count = 0
    with open(out_path, "w") as f:
        for sample in ds:
            text = sample.get("text", "")
            if len(text) < MIN_DOC_LENGTH:
                continue

            if write_doc(f, text, source_label, license_tag):
                count += 1

            if count % 100_000 == 0 and count > 0:
                log(f"  Processed {count:,} documents...")

    log(f"  {source_label} complete: {count:,} documents → {out_path}")


# ---------------------------------------------------------------------------
# Source: Wikipedia (Common Pile wikimedia)
# ---------------------------------------------------------------------------

def build_wikipedia():
    """Download Wikipedia from Common Pile (better deduplication than raw dumps)."""
    out_path = DATA_DIR / "wikipedia.jsonl"
    if out_path.exists():
        n = count_lines(out_path)
        log(f"Wikipedia already exists ({n:,} docs), skipping. Delete to rebuild.")
        return

    log("Building Wikipedia from Common Pile...")
    _build_common_pile_source("wikimedia", out_path, "wikipedia", "CC-BY-SA-4.0")


# ---------------------------------------------------------------------------
# Source: Stack Exchange (Common Pile stackexchange)
# ---------------------------------------------------------------------------

def build_stackexchange():
    """Download Stack Exchange from Common Pile."""
    out_path = DATA_DIR / "stackexchange.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        n = count_lines(out_path)
        log(f"Stack Exchange already exists ({n:,} docs), skipping.")
        return

    log("Building Stack Exchange from Common Pile...")
    _build_common_pile_source("stackexchange", out_path, "stackexchange", "CC-BY-SA-4.0")


# ---------------------------------------------------------------------------
# Source: Project Gutenberg (Common Pile gutenberg)
# ---------------------------------------------------------------------------

def build_gutenberg():
    """Download Project Gutenberg from Common Pile."""
    out_path = DATA_DIR / "gutenberg.jsonl"
    if out_path.exists():
        n = count_lines(out_path)
        log(f"Project Gutenberg already exists ({n:,} docs), skipping.")
        return

    log("Building Project Gutenberg from Common Pile...")
    _build_common_pile_source("gutenberg", out_path, "gutenberg", "Public Domain")


# ---------------------------------------------------------------------------
# Source: Debian Man Pages
# ---------------------------------------------------------------------------

def build_manpages():
    """Extract man pages from the local system."""
    out_path = DATA_DIR / "manpages.jsonl"
    if out_path.exists():
        n = count_lines(out_path)
        log(f"Man pages already exists ({n:,} docs), skipping.")
        return

    log("Building man pages from local system...")

    man_dirs = [
        "/usr/share/man/man1", "/usr/share/man/man2",
        "/usr/share/man/man3", "/usr/share/man/man4",
        "/usr/share/man/man5", "/usr/share/man/man6",
        "/usr/share/man/man7", "/usr/share/man/man8",
    ]

    count = 0
    with open(out_path, "w") as f:
        for man_dir in man_dirs:
            if not os.path.isdir(man_dir):
                continue

            section = os.path.basename(man_dir)
            for filename in sorted(os.listdir(man_dir)):
                filepath = os.path.join(man_dir, filename)
                try:
                    # Use man to render the page as plain text
                    result = subprocess.run(
                        ["man", "--no-hyphenation", "--no-justification",
                         filepath],
                        capture_output=True, text=True, timeout=10,
                        env={**os.environ, "MANWIDTH": "80", "LANG": "C"}
                    )
                    if result.returncode != 0:
                        continue

                    text = result.stdout
                    # Clean up man page formatting artifacts
                    text = re.sub(r'.\x08', '', text)  # remove backspace overstrikes
                    text = re.sub(r'\x1b\[[0-9;]*m', '', text)  # remove ANSI codes

                    cmd_name = filename.split(".")[0]
                    text = f"# {cmd_name}({section})\n\n{text}"

                    if write_doc(f, text, "manpages", "GPL/BSD"):
                        count += 1

                except (subprocess.TimeoutExpired, Exception):
                    continue

    log(f"  Man pages complete: {count:,} pages → {out_path}")


# ---------------------------------------------------------------------------
# Source: Linux Kernel Documentation
# ---------------------------------------------------------------------------

def build_kernel_docs():
    """Download Linux kernel and extract Documentation/."""
    out_path = DATA_DIR / "kernel_docs.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        n = count_lines(out_path)
        log(f"Kernel docs already exists ({n:,} docs), skipping.")
        return

    log("Building Linux kernel documentation...")

    kernel_dir = RAW_DIR / "linux"
    doc_dir = kernel_dir / "Documentation"

    if not doc_dir.exists():
        _ensure_kernel_clone(kernel_dir)

    if not doc_dir.exists():
        log("  ERROR: Could not obtain kernel Documentation/")
        return

    count = 0
    with open(out_path, "w") as f:
        for root, dirs, files in os.walk(doc_dir):
            for fname in files:
                if not fname.endswith((".rst", ".txt", ".md")):
                    continue

                filepath = os.path.join(root, fname)
                try:
                    with open(filepath, "r", errors="replace") as doc:
                        text = doc.read()

                    rel_path = os.path.relpath(filepath, doc_dir)
                    text = f"# Linux Kernel: {rel_path}\n\n{text}"

                    if write_doc(f, text, "kernel_docs", "GPL-2.0"):
                        count += 1
                except Exception:
                    continue

    log(f"  Kernel docs complete: {count:,} files → {out_path}")


def _ensure_kernel_clone(kernel_dir):
    """Clone the Linux kernel with sparse checkout (only needed dirs)."""
    if kernel_dir.exists():
        # Check if it has content
        doc_dir = kernel_dir / "Documentation"
        if doc_dir.exists() and any(doc_dir.iterdir()):
            return
        # Empty clone, remove and retry
        import shutil
        shutil.rmtree(kernel_dir)

    log("  Cloning kernel repo (shallow, sparse checkout)...")
    try:
        # Mark directory as safe (git ownership check)
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", str(kernel_dir)],
            capture_output=True, timeout=10
        )
        # Step 1: Clone with no checkout
        subprocess.run(
            ["git", "clone", "--depth=1", "--no-checkout", "--filter=blob:none",
             "https://github.com/torvalds/linux.git", str(kernel_dir)],
            check=True, capture_output=True, timeout=300
        )
        # Step 2: Set up sparse checkout
        subprocess.run(
            ["git", "-C", str(kernel_dir), "sparse-checkout", "set",
             "Documentation", "kernel", "fs", "net", "drivers/net",
             "mm", "init", "security", "ipc", "lib", "scripts",
             "include/linux", "include/uapi"],
            check=True, capture_output=True, timeout=120
        )
        # Step 3: Checkout
        subprocess.run(
            ["git", "-C", str(kernel_dir), "checkout"],
            check=True, capture_output=True, timeout=600
        )
        log("  Kernel clone complete.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log(f"  Sparse checkout failed: {e}")
        log("  Trying curl download of tarball instead...")
        _download_kernel_tarball(kernel_dir)


def _download_kernel_tarball(kernel_dir):
    """Fallback: download kernel via curl and extract needed directories."""
    import tarfile

    tar_path = RAW_DIR / "linux-master.tar.gz"

    # Download with curl (handles redirects, large files better than urllib)
    if not tar_path.exists() or tar_path.stat().st_size < 100_000_000:
        if tar_path.exists():
            tar_path.unlink()
        log("    Downloading kernel tarball with curl (~2GB)...")
        result = subprocess.run(
            ["curl", "-L", "-o", str(tar_path), "--retry", "3",
             "https://github.com/torvalds/linux/archive/refs/heads/master.tar.gz"],
            capture_output=True, timeout=1800  # 30 min timeout
        )
        if result.returncode != 0:
            log(f"    curl download failed: {result.stderr.decode()[:200]}")
            return

    log("    Extracting needed directories from tarball...")
    kernel_dir.mkdir(parents=True, exist_ok=True)

    needed_prefixes = [
        "Documentation/", "kernel/", "fs/", "net/", "drivers/net/",
        "mm/", "init/", "security/", "ipc/", "lib/", "scripts/",
        "include/linux/", "include/uapi/",
    ]
    code_exts = {".c", ".h", ".S", ".rst", ".txt", ".md", ".py", ".sh"}
    special_names = {"Makefile", "Kconfig"}

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            # Strip top-level dir (linux-master/)
            parts = member.name.split("/", 1)
            if len(parts) < 2:
                continue
            rel_path = parts[1]

            # Check if in needed directory
            if not any(rel_path.startswith(p) for p in needed_prefixes):
                continue

            fname = os.path.basename(rel_path)
            ext = os.path.splitext(fname)[1].lower()
            if ext not in code_exts and fname not in special_names:
                continue

            # Extract to kernel_dir
            out_path = kernel_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                extracted = tar.extractfile(member)
                if extracted:
                    with open(out_path, "wb") as f:
                        f.write(extracted.read())
            except Exception:
                continue

    log("    Tarball extraction complete.")


# ---------------------------------------------------------------------------
# Source: RFC Documents
# ---------------------------------------------------------------------------

def build_rfcs():
    """Download RFC documents in plain text."""
    out_path = DATA_DIR / "rfcs.jsonl"
    if out_path.exists():
        n = count_lines(out_path)
        log(f"RFCs already exists ({n:,} docs), skipping.")
        return

    log("Building RFC documents...")

    import urllib.request
    import tarfile

    tar_url = "https://www.rfc-editor.org/in-notes/tar/RFC-all.tar.gz"
    tar_path = RAW_DIR / "rfc-all.tar.gz"

    if not tar_path.exists():
        log("  Downloading RFC archive (~300 MB)...")
        urllib.request.urlretrieve(tar_url, tar_path)

    count = 0
    with open(out_path, "w") as f:
        log("  Extracting and processing RFCs...")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if not member.name.endswith(".txt"):
                        continue
                    if not re.match(r'rfc\d+\.txt', os.path.basename(member.name)):
                        continue

                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue

                    text = extracted.read().decode("utf-8", errors="replace")

                    # Extract RFC number for title
                    rfc_num = re.search(r'rfc(\d+)', member.name)
                    if rfc_num:
                        text = f"# RFC {rfc_num.group(1)}\n\n{text}"

                    if write_doc(f, text, "rfcs", "IETF"):
                        count += 1
        except Exception as e:
            log(f"  Error processing RFC archive: {e}")

    log(f"  RFCs complete: {count:,} documents → {out_path}")


# ---------------------------------------------------------------------------
# Source: Arch Wiki
# ---------------------------------------------------------------------------

def build_archwiki():
    """Download and process Arch Wiki from HuggingFace (Dam-Buty/arch-wiki)."""
    out_path = DATA_DIR / "archwiki.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        n = count_lines(out_path)
        log(f"Arch Wiki already exists ({n:,} docs), skipping.")
        return

    log("Building Arch Wiki...")

    from datasets import load_dataset
    from collections import defaultdict

    log("  Loading Arch Wiki from HuggingFace (Dam-Buty/arch-wiki)...")
    ds = load_dataset("Dam-Buty/arch-wiki", split="train", streaming=True)

    # Group sections by title to rebuild full articles
    articles = defaultdict(list)
    for sample in ds:
        title = sample.get("title", "")
        section = sample.get("section", "")
        content = sample.get("content", "")

        # Skip talk pages, user pages, etc.
        if ":" in title and title.split(":")[0] in (
            "Talk", "User", "User talk", "Template", "Template talk",
            "Category", "File", "Help", "ArchWiki"
        ):
            continue

        articles[title].append((section, content))

    count = 0
    with open(out_path, "w") as f:
        for title, sections in articles.items():
            # Build full article from sections
            parts = [f"# {title}"]
            for section, content in sections:
                if section and section != title:
                    parts.append(f"\n## {section}\n")
                parts.append(content)

            text = "\n".join(parts)

            if write_doc(f, text, "archwiki", "GFDL"):
                count += 1

    log(f"  Arch Wiki complete: {count:,} articles → {out_path}")


# ---------------------------------------------------------------------------
# Source: TLDP (The Linux Documentation Project)
# ---------------------------------------------------------------------------

def build_tldp():
    """Download TLDP HOWTOs and guides."""
    out_path = DATA_DIR / "tldp.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        n = count_lines(out_path)
        log(f"TLDP already exists ({n:,} docs), skipping.")
        return

    log("Building TLDP guides...")

    import urllib.request
    import tarfile

    # TLDP HOWTOs and guides — try multiple sources
    guides_dir = RAW_DIR / "tldp"
    guides_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out_path, "w") as f:
        # Try archive.org mirror first (more reliable)
        tldp_urls = [
            "https://tldp.org/HOWTO/text/",
            "https://tldp.org/LDP/abs/html/",
        ]

        # Also try fetching the LDP HOWTO listing from web archive
        try:
            log("  Downloading TLDP HOWTO listing...")
            howto_url = "https://tldp.org/HOWTO/text/"
            index_data = urllib.request.urlopen(howto_url, timeout=30).read().decode()

            # Extract links (both .txt and directories)
            txt_links = re.findall(r'href="([A-Za-z][\w-]+(?:\.txt)?)"', index_data)
            # Filter to likely HOWTO files
            txt_links = [l for l in txt_links if len(l) > 3 and not l.startswith(".")]

            log(f"  Found {len(txt_links)} potential HOWTO entries")

            for link in txt_links:
                if not link.endswith(".txt"):
                    link = link.rstrip("/")
                    # Try as a directory with index
                    for suffix in [".txt", "/index.txt", ""]:
                        try:
                            full_url = howto_url + link + suffix
                            data = urllib.request.urlopen(full_url, timeout=10).read()
                            text = data.decode("utf-8", errors="replace")
                            if len(text) > 200:
                                text = f"# TLDP: {link}\n\n{text}"
                                if write_doc(f, text, "tldp", "GFDL"):
                                    count += 1
                                break
                        except Exception:
                            continue
                else:
                    try:
                        full_url = howto_url + link
                        data = urllib.request.urlopen(full_url, timeout=10).read()
                        text = data.decode("utf-8", errors="replace")
                        title = link.replace(".txt", "")
                        text = f"# TLDP: {title}\n\n{text}"
                        if write_doc(f, text, "tldp", "GFDL"):
                            count += 1
                    except Exception:
                        continue

        except Exception as e:
            log(f"  TLDP HOWTO download failed: {e}")

        # Also try to get the Advanced Bash Scripting Guide and others
        extra_guides = [
            ("https://tldp.org/LDP/abs/abs-guide.txt.gz", "Advanced-Bash-Scripting-Guide"),
            ("https://tldp.org/LDP/GNU-Linux-Tools-Summary/GNU-Linux-Tools-Summary.txt",
             "GNU-Linux-Tools-Summary"),
        ]
        for url, title in extra_guides:
            try:
                data = urllib.request.urlopen(url, timeout=15).read()
                if url.endswith(".gz"):
                    import gzip
                    data = gzip.decompress(data)
                text = data.decode("utf-8", errors="replace")
                text = f"# TLDP: {title}\n\n{text}"
                if write_doc(f, text, "tldp", "GFDL"):
                    count += 1
            except Exception:
                continue

    log(f"  TLDP complete: {count:,} guides → {out_path}")


# ---------------------------------------------------------------------------
# Source: GNU Manuals
# ---------------------------------------------------------------------------

def build_gnu_manuals():
    """Extract GNU info pages from the local system."""
    out_path = DATA_DIR / "gnu_manuals.jsonl"
    if out_path.exists():
        n = count_lines(out_path)
        log(f"GNU manuals already exists ({n:,} docs), skipping.")
        return

    log("Building GNU manuals from info pages...")

    info_dir = "/usr/share/info"
    count = 0

    with open(out_path, "w") as f:
        if not os.path.isdir(info_dir):
            log("  /usr/share/info not found, skipping.")
            return

        # Get list of unique info files (without .gz, section numbers)
        seen = set()
        for fname in sorted(os.listdir(info_dir)):
            base = fname.split(".info")[0] if ".info" in fname else fname.split(".")[0]
            if base in seen or not base:
                continue
            seen.add(base)

            try:
                result = subprocess.run(
                    ["info", "--subnodes", "-o", "-", base],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode != 0 or len(result.stdout) < 200:
                    continue

                text = f"# GNU {base}\n\n{result.stdout}"

                if write_doc(f, text, "gnu_manuals", "GFDL"):
                    count += 1

            except (subprocess.TimeoutExpired, Exception):
                continue

    log(f"  GNU manuals complete: {count:,} manuals → {out_path}")


# ---------------------------------------------------------------------------
# Source: FOSS Code (Common Pile the_stack_v2)
# ---------------------------------------------------------------------------

def build_thestack():
    """Download FOSS code from Common Pile's The Stack V2 subset."""
    out_path = DATA_DIR / "thestack.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        n = count_lines(out_path)
        log(f"FOSS code already exists ({n:,} docs), skipping.")
        return

    log("Building FOSS code from Common Pile (the_stack_v2)...")
    _build_common_pile_source("the_stack_v2", out_path, "thestack", "FOSS (per-file)")


# ---------------------------------------------------------------------------
# Source: peS2o (Common Pile — scientific papers)
# ---------------------------------------------------------------------------

def build_peS2o():
    """Download peS2o scientific papers from Common Pile."""
    out_path = DATA_DIR / "peS2o.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        n = count_lines(out_path)
        log(f"peS2o already exists ({n:,} docs), skipping.")
        return

    log("Building peS2o from Common Pile...")
    _build_common_pile_source("peS2o", out_path, "peS2o", "CC (per-paper)")


# ---------------------------------------------------------------------------
# Source: arXiv papers (Common Pile)
# ---------------------------------------------------------------------------

def build_arxiv():
    """Download arXiv papers from Common Pile."""
    out_path = DATA_DIR / "arxiv.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        n = count_lines(out_path)
        log(f"arXiv already exists ({n:,} docs), skipping.")
        return

    log("Building arXiv papers from Common Pile...")
    _build_common_pile_source("arxiv_papers", out_path, "arxiv", "CC (per-paper)")


# ---------------------------------------------------------------------------
# Source: USGPO (Common Pile — US Government Publishing Office)
# ---------------------------------------------------------------------------

def build_usgpo():
    """Download USGPO documents from Common Pile (Public Domain)."""
    out_path = DATA_DIR / "usgpo.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        n = count_lines(out_path)
        log(f"USGPO already exists ({n:,} docs), skipping.")
        return

    log("Building USGPO from Common Pile...")
    _build_common_pile_source("usgpo", out_path, "usgpo", "Public Domain")


# ---------------------------------------------------------------------------
# Source: Curated FOSS repos (existing V2 Linux data)
# ---------------------------------------------------------------------------

def build_foss_repos():
    """Use existing curated FOSS repos from V2 (already in data/pretrain/)."""
    # This is the old thestack data renamed — kept for backward compatibility
    # with V2 data that's already been built. If not present, skip silently.
    out_path = DATA_DIR / "foss_repos.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        n = count_lines(out_path)
        log(f"FOSS repos already exists ({n:,} docs), skipping.")
        return

    log("FOSS repos: not built (optional — use Common Pile thestack instead)")


# ---------------------------------------------------------------------------
# Source: Linux kernel source code
# ---------------------------------------------------------------------------

def build_kernel_source():
    """Extract key kernel source files (C, headers, Makefiles).

    Uses the same kernel clone as kernel_docs (sparse checkout or tarball).
    """
    out_path = DATA_DIR / "kernel_source.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        n = count_lines(out_path)
        log(f"Kernel source already exists ({n:,} docs), skipping.")
        return

    log("Building Linux kernel source code...")

    kernel_dir = RAW_DIR / "linux"
    _ensure_kernel_clone(kernel_dir)

    # Key directories for Linux systems knowledge
    key_dirs = [
        "kernel", "fs", "net", "drivers/net", "mm", "init",
        "security", "ipc", "lib", "scripts",
        "include/linux", "include/uapi",
    ]

    count = 0
    with open(out_path, "w") as f:
        for subdir in key_dirs:
            full_dir = kernel_dir / subdir
            if not full_dir.exists():
                continue

            for root, dirs, files in os.walk(full_dir):
                for fname in files:
                    if not fname.endswith((".c", ".h", ".S", "Makefile", "Kconfig")):
                        continue

                    filepath = os.path.join(root, fname)
                    try:
                        with open(filepath, "r", errors="replace") as src:
                            content = src.read()

                        if len(content) < 100 or len(content) > 500_000:
                            continue

                        rel_path = os.path.relpath(filepath, kernel_dir)
                        text = f"```c\n// Linux kernel: {rel_path}\n{content}\n```"

                        if write_doc(f, text, "kernel_source", "GPL-2.0"):
                            count += 1
                    except Exception:
                        continue

    log(f"  Kernel source complete: {count:,} files → {out_path}")


# ---------------------------------------------------------------------------
# Stats and status
# ---------------------------------------------------------------------------

SOURCES = {
    # General text (Common Pile)
    "wikipedia":     ("Wikipedia (Common Pile)", "CC-BY-SA-4.0",  build_wikipedia),
    "gutenberg":     ("Gutenberg (Common Pile)", "Public Domain", build_gutenberg),
    # Technical Q&A (Common Pile)
    "stackexchange": ("StackExchange (Common Pile)", "CC-BY-SA-4.0", build_stackexchange),
    # Code (Common Pile)
    "thestack":      ("The Stack V2 FOSS (Common Pile)", "FOSS (per-file)", build_thestack),
    # Scientific (Common Pile)
    "peS2o":         ("peS2o (Common Pile)",     "CC (per-paper)", build_peS2o),
    "arxiv":         ("arXiv (Common Pile)",     "CC (per-paper)", build_arxiv),
    # Government (Common Pile)
    "usgpo":         ("USGPO (Common Pile)",     "Public Domain", build_usgpo),
    # Linux specialization (existing V2 data)
    "manpages":      ("Debian Man Pages",        "GPL/BSD",       build_manpages),
    "kernel_docs":   ("Linux Kernel Docs",       "GPL-2.0",       build_kernel_docs),
    "rfcs":          ("RFC Documents",           "IETF",          build_rfcs),
    "archwiki":      ("Arch Wiki",               "GFDL",          build_archwiki),
    "tldp":          ("TLDP Guides",             "GFDL",          build_tldp),
    "gnu_manuals":   ("GNU Info Manuals",        "GFDL",          build_gnu_manuals),
    "kernel_source": ("Linux Kernel Source",     "GPL-2.0",       build_kernel_source),
    "foss_repos":    ("Curated FOSS Repos",      "MIT/GPL/Apache", build_foss_repos),
}


def show_status():
    """Show status of each data source."""
    log("Data source status:")
    log(f"  {'Source':<30s} {'License':<15s} {'Status':<10s} {'Docs':>10s} {'Est.Tokens':>12s}")
    log("  " + "-" * 85)

    total_docs = 0
    total_tokens = 0

    for key, (name, license_tag, _) in SOURCES.items():
        path = DATA_DIR / f"{key}.jsonl"
        if path.exists() and path.stat().st_size > 0:
            docs = count_lines(path)
            tokens = estimate_tokens(path)
            status = "READY"
            total_docs += docs
            total_tokens += tokens
        else:
            docs = 0
            tokens = 0
            status = "MISSING"

        log(f"  {name:<30s} {license_tag:<15s} {status:<10s} {docs:>10,d} {tokens/1e6:>10.1f}M")

    log("  " + "-" * 85)
    log(f"  {'TOTAL':<30s} {'':<15s} {'':<10s} {total_docs:>10,d} {total_tokens/1e6:>10.1f}M")
    log(f"\n  Chinchilla-optimal for 135M params: ~2.7B tokens (10B = ~74× Chinchilla)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build DFSG-compliant training data")
    parser.add_argument("--source", type=str, help="Build single source (e.g. wikipedia)")
    parser.add_argument("--list", action="store_true", help="List sources and status")
    parser.add_argument("--stats", action="store_true", help="Show token count estimates")
    args = parser.parse_args()

    ensure_dirs()

    if args.list or args.stats:
        show_status()
        return

    if args.source:
        if args.source not in SOURCES:
            log(f"Unknown source: {args.source}")
            log(f"Available: {', '.join(SOURCES.keys())}")
            sys.exit(1)
        name, license_tag, build_fn = SOURCES[args.source]
        log(f"Building {name} ({license_tag})...")
        build_fn()
        show_status()
        return

    # Build all sources
    log("Building all data sources...")
    log("=" * 60)

    for key, (name, license_tag, build_fn) in SOURCES.items():
        log(f"\n{'=' * 60}")
        log(f"Source: {name} ({license_tag})")
        log(f"{'=' * 60}")
        try:
            build_fn()
        except Exception as e:
            log(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    log(f"\n{'=' * 60}")
    log("Build complete!")
    log(f"{'=' * 60}\n")
    show_status()


if __name__ == "__main__":
    main()
