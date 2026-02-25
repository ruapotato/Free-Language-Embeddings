"""
Prepare Debian Documentation Corpus for Annealing
==================================================
Extracts documentation from the local Debian system and apt repository
into a training-ready JSONL file for the annealing stage.

Sources (all DFSG-free):
  1. Man pages         — /usr/share/man/man{1-8}/  (~7,200+ pages)
  2. Doc files         — /usr/share/doc/*/  (README, .txt, .md, .rst, HTML→text)
  3. Info pages        — /usr/share/info/  (~25 files)
  4. Package metadata  — Full repo Packages files (70k+ packages)
  5. Copyright files   — /usr/share/doc/*/copyright  (DEP-5 format)
  6. Changelogs        — /usr/share/doc/*/changelog*

Optionally installs curated documentation packages via apt first.

Output: data/debian_docs.jsonl

Usage:
    python prepare_debian_data.py
    python prepare_debian_data.py --skip-install    # skip apt install step
    python prepare_debian_data.py --no-changelogs   # exclude changelogs
"""

import argparse
import datetime
import gzip
import json
import os
import re
import subprocess
import sys
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "debian_docs.jsonl"

# Documentation packages to install (curated for OS assistant relevance)
DOC_PACKAGES = [
    "debian-handbook",
    "debian-policy",
    "developers-reference",
    "debian-faq",
    "doc-debian",
    "debian-reference-en",
    "debian-security-support",
    "debconf-doc",
    "apt-doc",
    "python-apt-doc",
    "dpkg-dev",
    "bash-doc",
    "python3-doc",
    "git-doc",
    "glibc-doc",
    "make-doc",
    "manpages",
    "manpages-dev",
    "info",
    "gcc-doc",
    "binutils-doc",
    "vim-doc",
    "gdb-doc",
    "gawk-doc",
    "diffutils-doc",
    "tar-doc",
    "perl-doc",
    "bind9-doc",
    "postfix-doc",
    "parted-doc",
    "nginx-doc",
    "apache2-doc",
    "postgresql-doc",
]

# Man page sections to extract (English only)
MAN_SECTIONS = ["man1", "man2", "man3", "man5", "man7", "man8"]

# Minimum text length to keep (chars)
MIN_TEXT_LENGTH = 200

# File extensions to skip in doc directories
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".bmp",
    ".pdf", ".ps", ".dvi", ".eps",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
    ".pyc", ".pyo", ".so", ".o", ".a",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# HTML to text
# ---------------------------------------------------------------------------

class HTMLTextExtractor(HTMLParser):
    """Simple HTML tag stripper that preserves text content."""

    def __init__(self):
        super().__init__()
        self._text = StringIO()
        self._skip = False
        self._skip_tags = {"script", "style", "head"}

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip = True
        elif tag in ("br", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                      "li", "tr", "dt", "dd", "blockquote"):
            self._text.write("\n")
        elif tag == "pre":
            self._text.write("\n")

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = False
        elif tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                      "li", "table", "pre", "blockquote"):
            self._text.write("\n")

    def handle_data(self, data):
        if not self._skip:
            self._text.write(data)

    def get_text(self):
        text = self._text.getvalue()
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def html_to_text(html_content):
    """Convert HTML to plain text."""
    extractor = HTMLTextExtractor()
    try:
        extractor.feed(html_content)
        return extractor.get_text()
    except Exception:
        # Fallback: crude tag stripping
        text = re.sub(r"<[^>]+>", " ", html_content)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------

def read_file(path, is_gz=False):
    """Read a file, optionally gzip-compressed. Returns text or None."""
    try:
        if is_gz or str(path).endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                return f.read()
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

def check_prerequisites():
    """Verify we're on a Debian-based system."""
    if not os.path.exists("/etc/os-release"):
        log("ERROR: /etc/os-release not found. Is this a Linux system?")
        sys.exit(1)

    os_id = ""
    with open("/etc/os-release") as f:
        for line in f:
            if line.startswith("ID="):
                os_id = line.strip().split("=", 1)[1].strip('"')
                break

    debian_based = {"debian", "ubuntu", "linuxmint", "pop", "kali", "raspbian"}
    if os_id not in debian_based:
        log(f"WARNING: OS is '{os_id}', not Debian-based. "
            "Some extraction steps may fail.")
    else:
        log(f"System: {os_id}")


# ---------------------------------------------------------------------------
# Step 1: Install doc packages
# ---------------------------------------------------------------------------

def install_doc_packages():
    """Install curated documentation packages via apt."""
    log("Installing documentation packages...")

    # Check which are already installed
    already = set()
    for pkg in DOC_PACKAGES:
        r = subprocess.run(
            ["dpkg-query", "-W", "-f", "${Status}", pkg],
            capture_output=True, text=True,
        )
        if "install ok installed" in r.stdout:
            already.add(pkg)

    to_install = [p for p in DOC_PACKAGES if p not in already]

    if not to_install:
        log(f"All {len(DOC_PACKAGES)} doc packages already installed.")
        return

    log(f"Already installed: {len(already)}/{len(DOC_PACKAGES)}")
    log(f"To install: {len(to_install)} packages")

    # Determine if we need sudo
    use_sudo = os.geteuid() != 0
    apt_prefix = ["sudo"] if use_sudo else []
    apt_env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}

    # Update package lists
    log("Running apt-get update...")
    subprocess.run(apt_prefix + ["apt-get", "update", "-qq"],
                   capture_output=True, env=apt_env)

    # Install (skip unavailable gracefully)
    cmd = apt_prefix + ["apt-get", "install", "-y", "--no-install-recommends"] + to_install
    log(f"Installing {len(to_install)} packages...")
    result = subprocess.run(cmd, capture_output=True, text=True, env=apt_env)

    if result.returncode != 0:
        # Some packages may not exist — install one by one
        log("Batch install had errors, trying packages individually...")
        installed = 0
        for pkg in to_install:
            r = subprocess.run(
                apt_prefix + ["apt-get", "install", "-y", "--no-install-recommends", pkg],
                capture_output=True, text=True, env=apt_env,
            )
            if r.returncode == 0:
                installed += 1
            else:
                log(f"  Could not install: {pkg}")
        log(f"Installed {installed}/{len(to_install)} packages")
    else:
        log(f"Installed {len(to_install)} packages successfully")


# ---------------------------------------------------------------------------
# Step 2: Extract man pages
# ---------------------------------------------------------------------------

def extract_man_pages():
    """Render all man pages to plain text."""
    log("Extracting man pages...")
    entries = []
    errors = 0
    total_chars = 0

    man_root = Path("/usr/share/man")
    env = {**os.environ, "COLUMNS": "80", "MAN_KEEP_FORMATTING": ""}

    for section in MAN_SECTIONS:
        section_dir = man_root / section
        if not section_dir.is_dir():
            continue

        pages = sorted(section_dir.glob("*.gz"))
        section_num = section[-1]

        for page_path in pages:
            name = page_path.name
            # Strip .gz and section extension: "ls.1.gz" → "ls"
            base = name
            if base.endswith(".gz"):
                base = base[:-3]
            # Remove section suffix: "ls.1" → "ls"
            if "." in base:
                base = base.rsplit(".", 1)[0]

            try:
                result = subprocess.run(
                    ["man", "--nh", "--nj", section_num, base],
                    capture_output=True, text=True, timeout=10, env=env,
                )
                text = result.stdout.strip()
            except (subprocess.TimeoutExpired, Exception):
                errors += 1
                continue

            if len(text) < MIN_TEXT_LENGTH:
                continue

            header = f"# Man page: {base}({section_num})"
            full_text = f"{header}\n\n{text}"
            entries.append({
                "text": full_text,
                "source": "debian_man",
            })
            total_chars += len(full_text)

        log(f"  {section}: {len(pages)} files processed")

    log(f"  Man pages: {len(entries)} extracted, {errors} errors, "
        f"{total_chars / 1e6:.1f}MB text")
    return entries


# ---------------------------------------------------------------------------
# Step 3: Extract doc files
# ---------------------------------------------------------------------------

def extract_doc_files():
    """Extract documentation files from /usr/share/doc/."""
    log("Extracting doc files from /usr/share/doc/...")
    entries = []
    total_chars = 0
    doc_root = Path("/usr/share/doc")

    if not doc_root.is_dir():
        log("  /usr/share/doc not found, skipping")
        return entries

    for pkg_dir in sorted(doc_root.iterdir()):
        if not pkg_dir.is_dir():
            continue

        pkg_name = pkg_dir.name

        for fpath in sorted(pkg_dir.rglob("*")):
            if not fpath.is_file():
                continue

            fname = fpath.name.lower()
            suffix = fpath.suffix.lower()

            # Skip binary files, changelogs (separate extractor), CSS/JS
            if suffix in SKIP_EXTENSIONS:
                continue
            if "changelog" in fname:
                continue

            # Determine how to read the file
            text = None
            is_gz = fname.endswith(".gz")
            base_name = fname[:-3] if is_gz else fname
            base_suffix = Path(base_name).suffix.lower()

            # HTML files (debian-handbook, apache2-doc, etc.)
            if base_suffix in (".html", ".htm"):
                raw = read_file(fpath, is_gz)
                if raw:
                    text = html_to_text(raw)
            # Plain text files
            elif (base_suffix in (".txt", ".md", ".rst", ".text", ".cfg")
                  or base_name in ("readme", "readme.debian", "news",
                                   "news.debian", "faq", "todo", "thanks",
                                   "bugs", "authors", "hacking", "install",
                                   "building")):
                text = read_file(fpath, is_gz)
            # Copyright files handled separately
            elif base_name == "copyright":
                continue

            if text and len(text.strip()) >= MIN_TEXT_LENGTH:
                # Relative path from doc root for the header
                rel = fpath.relative_to(doc_root)
                header = f"# Debian doc: {rel}"
                full_text = f"{header}\n\n{text.strip()}"
                entries.append({
                    "text": full_text,
                    "source": "debian_doc",
                })
                total_chars += len(full_text)

    log(f"  Doc files: {len(entries)} extracted, {total_chars / 1e6:.1f}MB text")
    return entries


# ---------------------------------------------------------------------------
# Step 4: Extract info pages
# ---------------------------------------------------------------------------

def extract_info_pages():
    """Extract and clean GNU info pages."""
    log("Extracting info pages...")
    entries = []
    total_chars = 0
    info_dir = Path("/usr/share/info")

    if not info_dir.is_dir():
        log("  /usr/share/info not found, skipping")
        return entries

    # Group multi-part info files (e.g., gnupg.info-1, gnupg.info-2)
    info_bases = {}
    for fpath in sorted(info_dir.iterdir()):
        if not fpath.is_file():
            continue
        name = fpath.name
        # Skip dir file (index)
        if name.startswith("dir"):
            continue
        # Get base name: "coreutils.info.gz" → "coreutils"
        base = name
        for ext in [".gz", ".bz2", ".xz"]:
            if base.endswith(ext):
                base = base[:-len(ext)]
        # Handle multi-part: "gnupg.info-1" → "gnupg"
        if re.match(r".*\.info-\d+$", base):
            base = re.sub(r"\.info-\d+$", "", base)
        elif base.endswith(".info"):
            base = base[:-5]

        if base not in info_bases:
            info_bases[base] = []
        info_bases[base].append(fpath)

    for base, parts in sorted(info_bases.items()):
        # Read and concatenate all parts
        all_text = []
        for fpath in sorted(parts):
            text = read_file(fpath)
            if text:
                all_text.append(text)

        if not all_text:
            continue

        combined = "\n".join(all_text)

        # Strip info formatting directives
        combined = re.sub(r"^\x1f\n.*?\n", "", combined, flags=re.MULTILINE)
        combined = re.sub(r"^Tag Table:.*", "", combined, flags=re.DOTALL)
        combined = re.sub(r"^\* Menu:.*?(?=\n[^ *]|\Z)", "", combined,
                          flags=re.MULTILINE | re.DOTALL)
        # Clean up node references
        combined = re.sub(r"^\s*\* [^:]+::\s*.*$", "", combined,
                          flags=re.MULTILINE)

        combined = combined.strip()
        if len(combined) < MIN_TEXT_LENGTH:
            continue

        header = f"# GNU Info: {base}"
        full_text = f"{header}\n\n{combined}"
        entries.append({
            "text": full_text,
            "source": "debian_info",
        })
        total_chars += len(full_text)

    log(f"  Info pages: {len(entries)} extracted, {total_chars / 1e6:.1f}MB text")
    return entries


# ---------------------------------------------------------------------------
# Step 5: Extract repo package metadata
# ---------------------------------------------------------------------------

def extract_repo_packages_metadata():
    """Parse apt Packages files for all packages in the repo."""
    log("Extracting repository package metadata...")
    entries = []
    total_chars = 0
    lists_dir = Path("/var/lib/apt/lists")

    if not lists_dir.is_dir():
        log("  /var/lib/apt/lists not found, skipping")
        return entries

    # Find all Packages files (uncompressed)
    pkg_files = sorted(lists_dir.glob("*_Packages"))

    if not pkg_files:
        log("  No Packages files found. Run 'apt-get update' first.")
        return entries

    log(f"  Found {len(pkg_files)} Packages files")

    for pkg_file in pkg_files:
        try:
            with open(pkg_file, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            log(f"  Error reading {pkg_file.name}: {e}")
            continue

        # Split into individual package records
        records = content.split("\n\n")

        for record in records:
            record = record.strip()
            if not record:
                continue

            # Parse key fields
            fields = {}
            current_key = None
            current_val = []

            for line in record.split("\n"):
                if line.startswith(" ") or line.startswith("\t"):
                    # Continuation of previous field
                    current_val.append(line.strip())
                elif ":" in line:
                    # Save previous field
                    if current_key:
                        fields[current_key] = "\n".join(current_val)
                    key, _, val = line.partition(":")
                    current_key = key.strip()
                    current_val = [val.strip()]

            if current_key:
                fields[current_key] = "\n".join(current_val)

            # Skip if no package name or description
            pkg_name = fields.get("Package", "")
            description = fields.get("Description", fields.get("Description-en", ""))
            if not pkg_name or not description:
                continue

            # Format as readable training text
            parts = [f"Package: {pkg_name}"]
            for key in ["Version", "Section", "Priority", "Architecture",
                        "Depends", "Recommends", "Suggests",
                        "Provides", "Conflicts", "Replaces",
                        "Homepage", "Maintainer"]:
                if key in fields and fields[key]:
                    parts.append(f"{key}: {fields[key]}")
            parts.append(f"Description: {description}")

            text = "\n".join(parts)
            if len(text) < 100:
                continue

            entries.append({
                "text": text,
                "source": "debian_repo_meta",
            })
            total_chars += len(text)

    log(f"  Repo metadata: {len(entries)} packages, {total_chars / 1e6:.1f}MB text")
    return entries


# ---------------------------------------------------------------------------
# Step 6: Extract copyright files
# ---------------------------------------------------------------------------

def extract_copyright_files():
    """Extract DEP-5 copyright files from installed packages."""
    log("Extracting copyright files...")
    entries = []
    total_chars = 0
    doc_root = Path("/usr/share/doc")

    if not doc_root.is_dir():
        return entries

    for pkg_dir in sorted(doc_root.iterdir()):
        if not pkg_dir.is_dir():
            continue

        copyright_file = pkg_dir / "copyright"
        if not copyright_file.is_file():
            continue

        text = read_file(copyright_file)
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            continue

        header = f"# Copyright: {pkg_dir.name}"
        full_text = f"{header}\n\n{text.strip()}"
        entries.append({
            "text": full_text,
            "source": "debian_copyright",
        })
        total_chars += len(full_text)

    log(f"  Copyright files: {len(entries)} extracted, {total_chars / 1e6:.1f}MB text")
    return entries


# ---------------------------------------------------------------------------
# Step 7: Extract changelogs
# ---------------------------------------------------------------------------

def extract_changelogs():
    """Extract package changelogs."""
    log("Extracting changelogs...")
    entries = []
    total_chars = 0
    doc_root = Path("/usr/share/doc")

    if not doc_root.is_dir():
        return entries

    for pkg_dir in sorted(doc_root.iterdir()):
        if not pkg_dir.is_dir():
            continue

        pkg_name = pkg_dir.name

        for fpath in sorted(pkg_dir.iterdir()):
            if not fpath.is_file():
                continue
            fname = fpath.name.lower()
            if "changelog" not in fname:
                continue

            text = read_file(fpath)
            if not text or len(text.strip()) < MIN_TEXT_LENGTH:
                continue

            header = f"# Changelog: {pkg_name}/{fpath.name}"
            full_text = f"{header}\n\n{text.strip()}"
            entries.append({
                "text": full_text,
                "source": "debian_changelog",
            })
            total_chars += len(full_text)

    log(f"  Changelogs: {len(entries)} extracted, {total_chars / 1e6:.1f}MB text")
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Debian documentation corpus for training"
    )
    parser.add_argument("--skip-install", action="store_true",
                        help="Skip installing doc packages via apt")
    parser.add_argument("--no-changelogs", action="store_true",
                        help="Exclude changelogs (saves ~240MB)")
    args = parser.parse_args()

    log("=" * 70)
    log("DEBIAN DOCUMENTATION CORPUS PREPARATION")
    log("=" * 70)

    check_prerequisites()

    # Step 1: Install doc packages
    if not args.skip_install:
        install_doc_packages()
    else:
        log("Skipping doc package installation (--skip-install)")

    # Step 2-7: Extract all documentation
    all_entries = []

    man_entries = extract_man_pages()
    all_entries.extend(man_entries)

    doc_entries = extract_doc_files()
    all_entries.extend(doc_entries)

    info_entries = extract_info_pages()
    all_entries.extend(info_entries)

    meta_entries = extract_repo_packages_metadata()
    all_entries.extend(meta_entries)

    copyright_entries = extract_copyright_files()
    all_entries.extend(copyright_entries)

    if not args.no_changelogs:
        changelog_entries = extract_changelogs()
        all_entries.extend(changelog_entries)
    else:
        log("Skipping changelogs (--no-changelogs)")
        changelog_entries = []

    # Write output
    log(f"\nWriting {len(all_entries)} entries to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            total_chars += len(entry["text"])

    file_size = OUTPUT_PATH.stat().st_size
    est_tokens = total_chars / 3.5

    # Summary
    log("\n" + "=" * 70)
    log("DEBIAN DOCUMENTATION CORPUS — COMPLETE")
    log("=" * 70)
    log(f"Output file:      {OUTPUT_PATH}")
    log(f"File size:        {file_size / 1e6:.1f}MB")
    log(f"Total entries:    {len(all_entries):,}")
    log(f"Total characters: {total_chars:,}")
    log(f"Est. tokens:      {est_tokens / 1e6:.1f}M (at ~3.5 chars/token)")
    log("")
    log("Breakdown by source:")
    from collections import Counter
    source_counts = Counter(e["source"] for e in all_entries)
    source_chars = {}
    for e in all_entries:
        source_chars[e["source"]] = source_chars.get(e["source"], 0) + len(e["text"])
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        chars = source_chars[src]
        log(f"  {src:25s}: {count:>7,} entries, {chars / 1e6:>7.1f}MB, "
            f"~{chars / 3.5 / 1e6:.1f}M tokens")
    log("=" * 70)


if __name__ == "__main__":
    main()
