#!/usr/bin/env python3
"""
generate_sft_v2.py — Human-Written SFT Data Pipeline for flm V2
================================================================

Processes 4 sources of DFSG-compliant, human-written conversation data:
  1. OASST2 (Apache-2.0)     — Multi-turn conversations, filter identity leaks
  2. Dolly 15K (CC-BY-SA-3.0) — Single-turn Q&A by Databricks employees
  3. Stack Exchange (CC-BY-SA-4.0) — Q&A pairs from 2.9M pretrain posts
  4. Ubuntu Dialogue (MIT)    — IRC support conversations from #ubuntu

All content is human-written. This script only filters and reformats.
AI is used for: system-prompt injection, quality filtering. NOT content generation.

Output: data/sft/flm_sft_v2.jsonl

Usage:
    python generate_sft_v2.py              # Run full pipeline
    python generate_sft_v2.py --skip-se    # Skip Stack Exchange (slow)
    python generate_sft_v2.py --skip-dolly # Skip Dolly download
    python generate_sft_v2.py --validate   # Validate existing output only
"""

import os
import sys
import json
import re
import random
import hashlib
import html
import time
import argparse
from pathlib import Path
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SFT_DIR = DATA_DIR / "sft"

# Input files (existing data)
OASST2_FILE = DATA_DIR / "oasst2_conversations.jsonl"
UBUNTU_FILE = DATA_DIR / "ubuntu_dialogue.jsonl"
SE_PRETRAIN = DATA_DIR / "pretrain" / "stackexchange.jsonl"

# Output files (per-source)
OASST2_OUT = SFT_DIR / "oasst2_filtered.jsonl"
DOLLY_OUT = SFT_DIR / "dolly_filtered.jsonl"
SE_OUT = SFT_DIR / "stackexchange_chat.jsonl"
UBUNTU_OUT = SFT_DIR / "ubuntu_irc.jsonl"
FINAL_OUT = SFT_DIR / "flm_sft_v2.jsonl"

# Stack Exchange settings
SE_MIN_ANSWER_CHARS = 100       # Skip very short answers
SE_MIN_KEYWORD_OVERLAP = 3     # Minimum keyword overlap for Q/A match
SE_MIN_TITLE_COVERAGE = 0.35   # Answer must match ≥35% of title keywords
SE_MAX_CONV_CHARS = 4000       # Keep conversations under ~1000 tokens
SE_SKIP_SITES = {"stackexchange/math"}  # LaTeX formulas don't work in text
SE_LINUX_SITES = {
    "stackexchange/unix", "stackexchange/askubuntu",
    "stackexchange/serverfault", "stackexchange/superuser",
}

# Ubuntu Dialogue settings
UBUNTU_MIN_TURNS = 3            # At least 3 conversational turns
UBUNTU_MIN_RESPONSE_WORDS = 35  # ~50 tokens across all assistant turns
UBUNTU_MAX_TURNS = 40           # Skip absurdly long IRC sessions

# Identity patterns to filter — catches assistant identity claims
# Targeted to avoid removing legitimate informational mentions
IDENTITY_FILTER_PATTERNS = [
    # Open Assistant / LAION identity
    re.compile(r"open\s*assistant", re.I),
    re.compile(r"\blaion\b", re.I),
    # Generic AI identity claims (assistant saying what it is)
    re.compile(r"as\s+an?\s+(?:ai|artificial intelligence)\s+(?:language\s+)?model", re.I),
    re.compile(r"i'?m\s+(?:just\s+)?an?\s+(?:ai|artificial intelligence)(?:\s+(?:language\s+)?model)?(?:\s|,|\.|$)", re.I),
    re.compile(r"i\s+am\s+a\s+large\s+language\s+model", re.I),
    re.compile(r"as\s+a\s+large\s+language\s+model", re.I),
    # Specific model identity claims
    re.compile(r"i\s+am\s+chatgpt", re.I),
    re.compile(r"i\s+am\s+gpt", re.I),
    re.compile(r"i(?:'m|\s+am)\s+(?:an\s+ai\s+)?(?:developed|created|made|trained)\s+by\s+openai", re.I),
    re.compile(r"(?:developed|created|trained)\s+by\s+openai", re.I),
    re.compile(r"i\s+am\s+claude", re.I),
    re.compile(r"(?:developed|created|trained)\s+by\s+anthropic", re.I),
]

# Stopwords for SE keyword extraction
STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "about", "what",
    "which", "who", "whom", "this", "that", "these", "those", "it", "its",
    "my", "your", "his", "her", "our", "their", "me", "him", "us",
    "them", "i", "you", "he", "she", "we", "they", "any", "many", "much",
    "also", "still", "already", "even", "well", "really", "quite", "get",
    "got", "like", "know", "want", "think", "make", "use", "using",
    "one", "two", "way", "work", "try", "new", "first", "last", "good",
    "best", "right", "don", "doesn", "didn", "won", "wouldn", "couldn",
    "shouldn", "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    """Print with timestamp."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def build_conversation(turns, source, system_prompt=None):
    """Build a conversation dict from (role, content) turns.

    Args:
        turns: list of ('user'|'assistant', content_string) tuples
        source: source identifier string
        system_prompt: optional system prompt string

    Returns:
        dict with 'text' and 'source' keys, or None if invalid
    """
    if not turns:
        return None

    parts = []
    if system_prompt:
        parts.append(f"<|system|>\n{system_prompt}")

    for role, content in turns:
        content = content.strip()
        if not content:
            continue
        if role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")

    text = "\n".join(parts)

    # Must have at least one user and one assistant turn
    if "<|user|>" not in text or "<|assistant|>" not in text:
        return None

    return {"text": text, "source": source}


def has_identity_leak(text):
    """Check if text contains AI identity patterns that should be filtered."""
    for pattern in IDENTITY_FILTER_PATTERNS:
        if pattern.search(text):
            return True
    return False


def clean_html(text):
    """Clean HTML artifacts from text (SE data may have residual HTML)."""
    text = html.unescape(text)
    # Convert block-level HTML to newlines
    text = re.sub(r"<(?:p|br|div|li|td|th)(?:\s[^>]*)?\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<(?:pre|code)(?:\s[^>]*)?>", "\n```\n", text, flags=re.I)
    text = re.sub(r"</(?:pre|code)>", "\n```\n", text, flags=re.I)
    text = re.sub(r"<(?:strong|b)(?:\s[^>]*)?>", "**", text, flags=re.I)
    text = re.sub(r"</(?:strong|b)>", "**", text, flags=re.I)
    text = re.sub(r"<(?:em|i)(?:\s[^>]*)?>", "*", text, flags=re.I)
    text = re.sub(r"</(?:em|i)>", "*", text, flags=re.I)
    # Convert links to markdown
    text = re.sub(r'<a\s+href="([^"]*)"[^>]*>([^<]*)</a>', r"[\2](\1)", text, flags=re.I)
    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Clean up excess whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_keywords(text, max_chars=300):
    """Extract significant keywords from text for SE matching."""
    text = text[:max_chars].lower()
    # Match words with 3+ chars, allowing dots/dashes/slashes for technical terms
    words = set(re.findall(r"\b[a-z][a-z0-9_./-]{2,}\b", text))
    return words - STOPWORDS


def write_jsonl(data, path):
    """Write list of dicts to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    log(f"  Wrote {len(data):,} conversations to {path}")


# ---------------------------------------------------------------------------
# Step 1: OASST2
# ---------------------------------------------------------------------------

def process_oasst2():
    """Filter OASST2 conversations and optionally add flm system prompts.

    - Removes conversations with AI identity leaks
    - Adds flm system prompt to ~50% of conversations (no system prompt → add one)
    - Preserves multi-turn structure
    """
    log("=" * 60)
    log("Step 1: Processing OASST2")
    log("=" * 60)

    if not OASST2_FILE.exists():
        log(f"  SKIP: {OASST2_FILE} not found")
        return []

    # Import system prompt generator
    sys.path.insert(0, str(PROJECT_DIR))
    from system_prompt import random_system_prompt

    rng = random.Random(42)
    results = []
    filtered_identity = 0
    total = 0
    added_system = 0

    with open(OASST2_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            entry = json.loads(line)
            text = entry.get("text", "")

            # Filter identity leaks
            if has_identity_leak(text):
                filtered_identity += 1
                continue

            # Add flm system prompt to ~50% of conversations without one
            if "<|system|>" not in text and rng.random() < 0.5:
                sys_prompt = random_system_prompt(
                    rng, include_none=False, short_chance=0.3
                )
                if sys_prompt:
                    text = f"<|system|>\n{sys_prompt}\n{text}"
                    added_system += 1

            results.append({
                "text": text,
                "source": entry.get("source", "oasst2"),
            })

    log(f"  Total: {total:,} | Kept: {len(results):,} | "
        f"Identity filtered: {filtered_identity:,} | Added system prompt: {added_system:,}")

    write_jsonl(results, OASST2_OUT)
    return results


# ---------------------------------------------------------------------------
# Step 2: Dolly 15K
# ---------------------------------------------------------------------------

def process_dolly():
    """Download and convert Databricks Dolly 15K to conversation format.

    Dolly contains 15K instruction-following examples written by Databricks
    employees. Categories: open_qa, closed_qa, information_extraction,
    summarization, brainstorming, classification, creative_writing.
    """
    log("=" * 60)
    log("Step 2: Processing Dolly 15K")
    log("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        log("  SKIP: 'datasets' library not installed (pip install datasets)")
        return []

    log("  Downloading databricks/databricks-dolly-15k...")
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    except Exception as e:
        log(f"  SKIP: Download failed: {e}")
        return []

    log(f"  Downloaded {len(ds):,} examples")

    sys.path.insert(0, str(PROJECT_DIR))
    from system_prompt import random_system_prompt

    rng = random.Random(43)
    results = []
    skipped_short = 0
    skipped_identity = 0
    categories = Counter()

    for example in ds:
        instruction = example.get("instruction", "").strip()
        context = example.get("context", "").strip()
        response = example.get("response", "").strip()
        category = example.get("category", "unknown")

        if not instruction or not response:
            skipped_short += 1
            continue

        # Skip very short responses (~20 tokens)
        if len(response) < 60:
            skipped_short += 1
            continue

        # Build user message
        if context:
            user_msg = f"{instruction}\n\nContext:\n{context}"
        else:
            user_msg = instruction

        # Check for identity leaks
        if has_identity_leak(response):
            skipped_identity += 1
            continue

        # Add system prompt to ~30% of conversations
        system_prompt = None
        if rng.random() < 0.3:
            system_prompt = random_system_prompt(
                rng, include_none=False, short_chance=0.3
            )

        conv = build_conversation(
            [("user", user_msg), ("assistant", response)],
            source=f"dolly_{category}",
            system_prompt=system_prompt,
        )
        if conv:
            results.append(conv)
            categories[category] += 1

    log(f"  Kept: {len(results):,} | Short: {skipped_short:,} | "
        f"Identity: {skipped_identity:,}")
    log(f"  By category:")
    for cat, count in categories.most_common():
        log(f"    {cat}: {count:,}")

    write_jsonl(results, DOLLY_OUT)
    return results


# ---------------------------------------------------------------------------
# Step 3: Stack Exchange
# ---------------------------------------------------------------------------

def process_stackexchange():
    """Pair Q/A from existing SE pretrain data using keyword matching.

    The pretrain file has 2.87M posts (1.09M questions, 1.78M answers) in
    PostId order per site. Questions start with '# Title'. Answers are plain
    text. We pair them using inverted-index keyword matching on titles.

    Algorithm:
      1. Stream through the pretrain file
      2. Questions: index by title keywords in a per-site inverted index
      3. Answers: find best-matching question via keyword overlap
      4. Quality filter on pair and output as conversation
    """
    log("=" * 60)
    log("Step 3: Processing Stack Exchange")
    log("=" * 60)

    if not SE_PRETRAIN.exists():
        log(f"  SKIP: {SE_PRETRAIN} not found")
        return []

    # Per-site question storage: site -> list of (title, body, keywords)
    site_qs = defaultdict(list)
    # Per-site inverted index: site -> keyword -> list of question indices
    site_idx = defaultdict(lambda: defaultdict(list))
    # Track which questions have been paired (one answer per question)
    used = set()

    results = []
    q_count = 0
    a_count = 0
    a_skipped_short = 0
    a_skipped_no_match = 0
    a_matched = 0

    t0 = time.time()

    with open(SE_PRETRAIN) as f:
        for line_no, raw in enumerate(f):
            if line_no % 200_000 == 0 and line_no > 0:
                elapsed = time.time() - t0
                rate = line_no / elapsed
                pct = line_no / 2_870_000 * 100
                log(f"  SE: {line_no:,} posts ({pct:.0f}%) | "
                    f"{a_matched:,} pairs | {rate:.0f} posts/s")

            obj = json.loads(raw)
            text = obj["text"]
            site = obj["source"]

            # Skip sites where Q&A format doesn't translate well
            if site in SE_SKIP_SITES:
                continue

            if text.startswith("#"):
                # --- Question ---
                q_count += 1
                nl = text.find("\n")
                if nl > 2:
                    title = text[2:nl].strip()
                    body = text[nl + 1:].strip()
                else:
                    title = text[2:].strip()
                    body = ""

                title = clean_html(title)
                body = clean_html(body)

                keywords = extract_keywords(title)
                if not keywords:
                    continue

                q_idx = len(site_qs[site])
                site_qs[site].append((title, body, keywords))

                # Update inverted index (cap posting list length)
                for kw in keywords:
                    posting = site_idx[site][kw]
                    if len(posting) < 1000:
                        posting.append(q_idx)

            else:
                # --- Answer ---
                a_count += 1

                if len(text) < SE_MIN_ANSWER_CHARS:
                    a_skipped_short += 1
                    continue

                qs = site_qs.get(site)
                if not qs:
                    continue

                answer_text = clean_html(text)
                if len(answer_text) < SE_MIN_ANSWER_CHARS:
                    a_skipped_short += 1
                    continue

                # Extract keywords from answer for matching
                a_keywords = extract_keywords(answer_text, max_chars=500)

                # Score questions using inverted index
                # Skip keywords with >500 entries (not discriminative)
                scores = Counter()
                idx = site_idx.get(site, {})
                for kw in a_keywords:
                    postings = idx.get(kw, ())
                    if len(postings) > 500:
                        continue
                    for qi in postings:
                        if (site, qi) not in used:
                            scores[qi] += 1

                if not scores:
                    a_skipped_no_match += 1
                    continue

                # Get best match
                best_qi, best_score = scores.most_common(1)[0]

                # Dynamic threshold: higher for non-Linux sites
                min_overlap = SE_MIN_KEYWORD_OVERLAP
                if site not in SE_LINUX_SITES:
                    min_overlap = 4

                if best_score < min_overlap:
                    a_skipped_no_match += 1
                    continue

                title, body, q_keywords = qs[best_qi]

                # Require answer covers enough of the title keywords
                # This prevents "load balancer" matching unrelated LB questions
                if len(q_keywords) > 0:
                    coverage = best_score / len(q_keywords)
                    if coverage < SE_MIN_TITLE_COVERAGE:
                        a_skipped_no_match += 1
                        continue

                used.add((site, best_qi))
                a_matched += 1

                # Build question text
                q_text = title
                if body:
                    q_text = title + "\n\n" + body

                # Length check — truncate if needed to fit SEQ_LEN
                total_chars = len(q_text) + len(answer_text)
                if total_chars > SE_MAX_CONV_CHARS:
                    if len(q_text) < 1500:
                        answer_text = answer_text[:SE_MAX_CONV_CHARS - len(q_text)]
                    else:
                        q_text = q_text[:1500]
                        answer_text = answer_text[:SE_MAX_CONV_CHARS - 1500]

                source_name = site.replace("stackexchange/", "se_")
                conv_text = f"<|user|>\n{q_text}\n<|assistant|>\n{answer_text}"
                results.append({"text": conv_text, "source": source_name})

    elapsed = time.time() - t0
    log(f"  Completed in {elapsed:.0f}s")
    log(f"  Questions indexed: {q_count:,} | Answers scanned: {a_count:,}")
    log(f"  Matched pairs: {a_matched:,}")
    log(f"  Skipped (short answer): {a_skipped_short:,} | "
        f"Skipped (no match): {a_skipped_no_match:,}")

    site_counts = Counter(item["source"] for item in results)
    log(f"  Per-site breakdown:")
    for site, count in site_counts.most_common(20):
        log(f"    {site}: {count:,}")

    write_jsonl(results, SE_OUT)
    return results


# ---------------------------------------------------------------------------
# Step 4: Ubuntu Dialogue
# ---------------------------------------------------------------------------

def process_ubuntu():
    """Reformat Ubuntu IRC dialogues from User/Helper to conversation format.

    The existing ubuntu_dialogue.jsonl has 251K conversations in
    'User: ...' / 'Helper: ...' format. We convert to <|user|>/<|assistant|>
    tags, merge consecutive same-speaker lines, and filter for quality.
    """
    log("=" * 60)
    log("Step 4: Processing Ubuntu Dialogue")
    log("=" * 60)

    if not UBUNTU_FILE.exists():
        log(f"  SKIP: {UBUNTU_FILE} not found")
        return []

    # Technical keywords — conversation must contain at least one
    TECH_KEYWORDS = re.compile(
        r"\b("
        r"sudo|apt|dpkg|apt-get|aptitude|snap|flatpak|"
        r"bash|terminal|command|shell|script|"
        r"install|package|repository|ppa|"
        r"error|fail|crash|bug|fix|broken|"
        r"kernel|driver|module|firmware|"
        r"grub|boot|partition|mount|fstab|"
        r"network|wifi|ethernet|dns|dhcp|ip|"
        r"permission|chmod|chown|root|"
        r"config|conf|settings|"
        r"file|directory|folder|path|"
        r"process|pid|kill|service|systemd|init|"
        r"update|upgrade|dist-upgrade|"
        r"xorg|display|nvidia|amd|gpu|"
        r"usb|sata|hard\s*drive|ssd|raid|"
        r"ssh|ftp|http|port|firewall|iptables|"
        r"log|dmesg|syslog|journalctl|"
        r"python|java|gcc|make|compile|"
        r"ubuntu|debian|linux|unix|fedora|arch"
        r")\b",
        re.I,
    )

    results = []
    total = 0
    skipped_short = 0
    skipped_quality = 0
    skipped_no_tech = 0

    with open(UBUNTU_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1

            entry = json.loads(line)
            text = entry.get("text", "")
            if not text:
                continue

            # Parse User:/Helper: format into turns
            raw_lines = text.split("\n")
            turns = []  # list of ('user'|'assistant', content)

            for ln in raw_lines:
                if ln.startswith("User: "):
                    content = ln[6:].strip()
                    if content:
                        if turns and turns[-1][0] == "user":
                            # Merge consecutive user lines
                            turns[-1] = ("user", turns[-1][1] + " " + content)
                        else:
                            turns.append(("user", content))
                elif ln.startswith("Helper: "):
                    content = ln[8:].strip()
                    if content:
                        if turns and turns[-1][0] == "assistant":
                            # Merge consecutive assistant lines
                            turns[-1] = ("assistant", turns[-1][1] + " " + content)
                        else:
                            turns.append(("assistant", content))

            # Quality filters
            if len(turns) < UBUNTU_MIN_TURNS:
                skipped_short += 1
                continue

            if len(turns) > UBUNTU_MAX_TURNS:
                skipped_quality += 1
                continue

            # Must have both roles
            has_user = any(r == "user" for r, _ in turns)
            has_asst = any(r == "assistant" for r, _ in turns)
            if not has_user or not has_asst:
                skipped_quality += 1
                continue

            # Check total assistant response length
            asst_text = " ".join(c for r, c in turns if r == "assistant")
            asst_words = len(asst_text.split())
            if asst_words < UBUNTU_MIN_RESPONSE_WORDS:
                skipped_short += 1
                continue

            # Technical content check
            full_text = " ".join(c for _, c in turns)
            if not TECH_KEYWORDS.search(full_text):
                skipped_no_tech += 1
                continue

            # Build conversation
            conv = build_conversation(turns, source="ubuntu_dialogue")
            if conv:
                results.append(conv)

    log(f"  Total: {total:,} | Kept: {len(results):,}")
    log(f"  Skipped (short/few turns): {skipped_short:,} | "
        f"Quality: {skipped_quality:,} | No tech content: {skipped_no_tech:,}")

    write_jsonl(results, UBUNTU_OUT)
    return results


# ---------------------------------------------------------------------------
# Step 5: Merge and Shuffle
# ---------------------------------------------------------------------------

def merge_all(sources):
    """Combine all sources, deduplicate by text hash, and shuffle."""
    log("=" * 60)
    log("Step 5: Merge, Deduplicate, and Shuffle")
    log("=" * 60)

    all_data = []
    for source_name, data in sources:
        log(f"  {source_name}: {len(data):,} conversations")
        all_data.extend(data)

    log(f"  Total before dedup: {len(all_data):,}")

    # Deduplicate by text hash
    seen = set()
    deduped = []
    for item in all_data:
        h = hashlib.md5(item["text"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(item)

    removed = len(all_data) - len(deduped)
    log(f"  Duplicates removed: {removed:,}")
    log(f"  Total after dedup: {len(deduped):,}")

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(deduped)

    write_jsonl(deduped, FINAL_OUT)

    # Print breakdown by source
    source_counts = Counter(item["source"] for item in deduped)
    log(f"\n  Final breakdown by source:")
    for source, count in source_counts.most_common(30):
        log(f"    {source}: {count:,}")

    return deduped


# ---------------------------------------------------------------------------
# Step 6: Validate
# ---------------------------------------------------------------------------

def validate(data=None):
    """Validate the final output — format, contamination, and stats."""
    log("=" * 60)
    log("Step 6: Validation")
    log("=" * 60)

    if data is None:
        if not FINAL_OUT.exists():
            log("  SKIP: No output file to validate")
            return
        data = []
        with open(FINAL_OUT) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

    total = len(data)
    log(f"  Total conversations: {total:,}")

    # Format check
    invalid_format = 0
    missing_user = 0
    missing_assistant = 0
    missing_source = 0

    for item in data:
        text = item.get("text", "")
        bad = False
        if "<|user|>" not in text:
            missing_user += 1
            bad = True
        if "<|assistant|>" not in text:
            missing_assistant += 1
            bad = True
        if "source" not in item:
            missing_source += 1
            bad = True
        if bad:
            invalid_format += 1

    log(f"  Format check: {invalid_format} invalid "
        f"({missing_user} no <|user|>, {missing_assistant} no <|assistant|>, "
        f"{missing_source} no source)")

    # Identity contamination audit
    identity_hits = 0
    identity_examples = []
    for item in data:
        text = item["text"]
        for pattern in IDENTITY_FILTER_PATTERNS:
            m = pattern.search(text)
            if m:
                identity_hits += 1
                if len(identity_examples) < 5:
                    identity_examples.append(
                        (item["source"], m.group(), text[:200])
                    )
                break

    log(f"  Identity contamination: {identity_hits:,} hits")
    for source, match, preview in identity_examples:
        log(f"    [{source}] matched '{match}': {preview[:100]}...")

    # Text length stats
    lengths = [len(item["text"]) for item in data]
    if lengths:
        avg_len = sum(lengths) / len(lengths)
        log(f"  Text length: avg {avg_len:.0f} chars | "
            f"min {min(lengths)} | max {max(lengths)}")

    # Multi-turn stats
    multi_turn = sum(
        1 for item in data
        if item["text"].count("<|user|>") > 1
        or item["text"].count("<|assistant|>") > 1
    )
    log(f"  Multi-turn: {multi_turn:,} ({multi_turn / max(total, 1) * 100:.1f}%)")

    # System prompt stats
    with_system = sum(1 for item in data if "<|system|>" in item["text"])
    log(f"  With system prompt: {with_system:,} "
        f"({with_system / max(total, 1) * 100:.1f}%)")

    # Source breakdown
    source_counts = Counter(item["source"] for item in data)
    log(f"  Source distribution:")
    for source, count in source_counts.most_common(15):
        pct = count / total * 100
        log(f"    {source}: {count:,} ({pct:.1f}%)")

    # Sample output
    log(f"\n  --- Sample conversations ---")
    rng = random.Random(42)
    samples = rng.sample(data, min(5, len(data)))
    for i, item in enumerate(samples):
        text = item["text"]
        log(f"\n  [{i + 1}] source={item['source']}")
        for ln in text.split("\n")[:8]:
            log(f"      {ln[:120]}")
        if text.count("\n") > 8:
            log(f"      ... ({text.count(chr(10))} lines total)")

    log("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="flm SFT V2 Data Pipeline — Human-Written, DFSG-Compliant"
    )
    parser.add_argument(
        "--skip-se", action="store_true",
        help="Skip Stack Exchange processing (slow, ~5-10 min)",
    )
    parser.add_argument(
        "--skip-dolly", action="store_true",
        help="Skip Dolly 15K download",
    )
    parser.add_argument(
        "--skip-ubuntu", action="store_true",
        help="Skip Ubuntu Dialogue processing",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Only validate existing output file",
    )
    args = parser.parse_args()

    log("=" * 60)
    log("flm SFT V2 Data Pipeline")
    log("Human-Written, DFSG-Compliant")
    log("=" * 60)

    if args.validate:
        validate()
        return

    t0 = time.time()
    sources = []

    # Step 1: OASST2
    oasst2_data = process_oasst2()
    if oasst2_data:
        sources.append(("OASST2", oasst2_data))

    # Step 2: Dolly 15K
    if not args.skip_dolly:
        dolly_data = process_dolly()
        if dolly_data:
            sources.append(("Dolly 15K", dolly_data))

    # Step 3: Stack Exchange
    if not args.skip_se:
        se_data = process_stackexchange()
        if se_data:
            sources.append(("Stack Exchange", se_data))

    # Step 4: Ubuntu Dialogue
    if not args.skip_ubuntu:
        ubuntu_data = process_ubuntu()
        if ubuntu_data:
            sources.append(("Ubuntu Dialogue", ubuntu_data))

    # Step 5: Merge
    if sources:
        final_data = merge_all(sources)

        # Step 6: Validate
        validate(final_data)
    else:
        log("WARNING: No data sources produced output!")

    elapsed = time.time() - t0
    log(f"\nTotal pipeline time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    log(f"Output: {FINAL_OUT}")


if __name__ == "__main__":
    main()
