#!/usr/bin/env python3
"""
Prepare Ubuntu Dialogue Corpus for training.

Downloads raw IRC dialogues from the Ubuntu Dialogue Corpus v1
(https://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/) and converts them
into cleaned JSONL training data.

Each conversation is a two-person exchange extracted from Ubuntu IRC
support channels. We clean IRC artifacts, filter out low-quality
conversations, and format them as natural User/Helper dialogues.

Output: data/ubuntu_dialogue.jsonl
"""

import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIALOGS_DIR = PROJECT_ROOT / "data" / "ubuntu_work" / "dialogs"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ubuntu_dialogue.jsonl"

MIN_TURNS = 3          # Minimum conversation turns to keep
MAX_TURNS = 200        # Skip absurdly long multi-day rambles
MIN_AVG_WORDS = 3      # Average words per turn must be at least this
MIN_TECHNICAL_RATIO = 0.0  # We keep all Ubuntu conversations (they're inherently technical)

# ---------------------------------------------------------------------------
# IRC cleaning patterns
# ---------------------------------------------------------------------------

# Bot / automated content patterns
BOT_NAMES = {
    "ubotu", "ubottu", "ubot5", "ubot3", "ubot4", "ubot2",
    "chanserv", "nickserv", "memoserv", "operserv",
    "floodbot1", "floodbot2", "floodbot3", "floodbot4",
    "lococount", "meetingology", "supybot",
    "logbot", "bot", "infobot", "dpkg", "apt", "judd",
}

# Patterns that indicate automated/bot messages
BOT_PATTERNS = re.compile(
    r"^("
    r"\*\*\*|"                          # IRC system messages (*** user joined)
    r"===|"                              # Mode changes
    r"\[.+\] has (joined|left|quit)|"    # Join/leave messages
    r"ChanServ|"
    r"Topic for|"
    r"Topic set by|"
    r"was kicked from|"
    r"sets mode|"
    r"changes topic to|"
    r"is now known as"
    r")",
    re.IGNORECASE,
)

# IRC artifact patterns to strip from message text
IRC_ARTIFACTS = [
    (re.compile(r"^\[?\d{2}:\d{2}(:\d{2})?\]?\s*"), ""),          # Timestamps like [14:30:00] or 14:30
    (re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s*"), ""),  # ISO timestamps
    (re.compile(r"^<[^>]+>\s*"), ""),                                # <nick> prefixes
    (re.compile(r"^\* \S+ "), ""),                                   # /me actions: * nick does something
    (re.compile(r"\x03\d{0,2}(?:,\d{1,2})?"), ""),                  # IRC color codes
    (re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]"), ""),             # Other IRC control characters
    (re.compile(r"\x02|\x0f|\x16|\x1d|\x1f"), ""),                  # Bold, reset, reverse, italic, underline
]

# URL shortening - keep URLs but clean up very long ones
LONG_URL_RE = re.compile(r"(https?://\S{120})\S+")

# Repeated characters
REPEATED_CHARS = re.compile(r"(.)\1{5,}")

# Repeated punctuation
REPEATED_PUNCT = re.compile(r"([!?.])\1{3,}")


def clean_message(text: str) -> str:
    """Clean a single IRC message."""
    if not text:
        return ""

    # Apply IRC artifact removals
    for pattern, replacement in IRC_ARTIFACTS:
        text = pattern.sub(replacement, text)

    # Truncate very long URLs (keep first 120 chars)
    text = LONG_URL_RE.sub(r"\1...", text)

    # Collapse repeated characters (e.g., "heeeeelp" -> "heeelp")
    text = REPEATED_CHARS.sub(r"\1\1\1", text)

    # Collapse repeated punctuation
    text = REPEATED_PUNCT.sub(r"\1\1\1", text)

    # Normalize whitespace
    text = " ".join(text.split())

    return text.strip()


def is_bot_message(speaker: str, text: str) -> bool:
    """Check if a message is from a bot or is automated content."""
    if speaker.lower().rstrip("_-0123456789") in BOT_NAMES:
        return True
    if BOT_PATTERNS.match(text):
        return True
    # Factoid responses from bots
    if text.startswith("!") and len(text.split()) <= 2:
        return True
    return False


def is_low_quality_message(text: str) -> bool:
    """Check if a message is too low-quality to keep."""
    if not text:
        return True
    if len(text) < 2:
        return True
    return False


def parse_dialog_file(filepath: str) -> list[dict]:
    """Parse a single TSV dialog file into a list of turns.

    We avoid csv.reader here because some messages contain unbalanced
    quotes which cause the CSV parser to merge multiple lines into one
    field.  Instead we split each line on tabs ourselves.
    """
    turns = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                raw_line = raw_line.rstrip("\n\r")
                if not raw_line:
                    continue
                parts = raw_line.split("\t")
                if len(parts) < 4:
                    continue
                timestamp = parts[0]
                speaker = parts[1]
                addressee = parts[2]
                # The message text may itself contain tabs, so rejoin the rest
                text = "\t".join(parts[3:])
                if not text.strip():
                    continue
                turns.append({
                    "timestamp": timestamp,
                    "speaker": speaker.strip(),
                    "addressee": addressee.strip(),
                    "text": text.strip(),
                })
    except Exception as e:
        # Skip files that can't be parsed
        return []
    return turns


def identify_roles(turns: list[dict]) -> tuple[str, str]:
    """
    Identify which speaker is the 'User' (asker) and which is the 'Helper'.

    Heuristic: The first speaker is usually the one asking for help.
    If the first message has no addressee, that person is the User.
    """
    if not turns:
        return "", ""

    speakers = set()
    for t in turns:
        speakers.add(t["speaker"])

    speakers = list(speakers)

    if len(speakers) != 2:
        # Multi-party or single-person - we only want two-person dialogues
        return "", ""

    # First speaker without an addressee is typically the person asking
    first_speaker = turns[0]["speaker"]
    other_speaker = speakers[1] if speakers[0] == first_speaker else speakers[0]

    # The asker (User) is the first speaker
    return first_speaker, other_speaker


def format_conversation(turns: list[dict], user: str, helper: str) -> str:
    """Format cleaned turns into a User/Helper dialogue string."""
    lines = []
    for turn in turns:
        cleaned = clean_message(turn["text"])
        if is_low_quality_message(cleaned):
            continue

        if turn["speaker"] == user:
            role = "User"
        elif turn["speaker"] == helper:
            role = "Helper"
        else:
            continue  # Skip messages from third parties

        lines.append(f"{role}: {cleaned}")

    return "\n".join(lines)


def conversation_quality_check(text: str, turns: list[dict]) -> bool:
    """Final quality check on the formatted conversation."""
    lines = text.strip().split("\n")

    # Must have minimum turns
    if len(lines) < MIN_TURNS:
        return False

    # Must have too many turns (multi-day rambling)
    if len(lines) > MAX_TURNS:
        return False

    # Must have both User and Helper
    has_user = any(line.startswith("User:") for line in lines)
    has_helper = any(line.startswith("Helper:") for line in lines)
    if not has_user or not has_helper:
        return False

    # Average word count per line must be reasonable
    total_words = sum(len(line.split()) - 1 for line in lines)  # -1 for role prefix
    avg_words = total_words / len(lines) if lines else 0
    if avg_words < MIN_AVG_WORDS:
        return False

    # Check that conversation isn't just greetings
    greeting_words = {"hi", "hello", "hey", "thanks", "thank", "bye", "np", "ok", "yes", "no", "yeah", "yep", "nope"}
    substantive_lines = 0
    for line in lines:
        words = set(line.lower().split())
        # Remove the role prefix word
        content_words = words - {"user:", "helper:"}
        if content_words - greeting_words:
            substantive_lines += 1

    if substantive_lines < 2:
        return False

    return True


def process_all_dialogs():
    """Process all dialog files and write to JSONL."""
    if not DIALOGS_DIR.exists():
        print(f"ERROR: Dialog directory not found: {DIALOGS_DIR}")
        print("Please download and extract the Ubuntu Dialogue Corpus first.")
        sys.exit(1)

    # Collect all TSV files
    tsv_files = sorted(DIALOGS_DIR.rglob("*.tsv"))
    total_files = len(tsv_files)
    print(f"Found {total_files:,} dialogue files")

    saved = 0
    skipped_bot = 0
    skipped_short = 0
    skipped_quality = 0
    skipped_multiparty = 0
    total_chars = 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for i, tsv_path in enumerate(tsv_files):
            if (i + 1) % 5000 == 0:
                print(f"  Processing {i+1:,}/{total_files:,} files... ({saved:,} saved so far)")

            turns = parse_dialog_file(str(tsv_path))
            if not turns:
                skipped_short += 1
                continue

            # Filter out bot messages
            clean_turns = []
            bot_count = 0
            for t in turns:
                if is_bot_message(t["speaker"], t["text"]):
                    bot_count += 1
                else:
                    clean_turns.append(t)

            # If most messages were bots, skip
            if bot_count > len(turns) * 0.5:
                skipped_bot += 1
                continue

            if len(clean_turns) < MIN_TURNS:
                skipped_short += 1
                continue

            # Identify roles
            user, helper = identify_roles(clean_turns)
            if not user or not helper:
                skipped_multiparty += 1
                continue

            # Format conversation
            text = format_conversation(clean_turns, user, helper)

            # Quality check
            if not conversation_quality_check(text, clean_turns):
                skipped_quality += 1
                continue

            # Write to JSONL
            record = {"text": text}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1
            total_chars += len(text)

    # Report statistics
    file_size = OUTPUT_PATH.stat().st_size
    estimated_tokens = total_chars // 3

    print("\n" + "=" * 60)
    print("UBUNTU DIALOGUE CORPUS - PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total dialogue files scanned:  {total_files:,}")
    print(f"Conversations saved:           {saved:,}")
    print(f"Skipped (too short/empty):     {skipped_short:,}")
    print(f"Skipped (bot content):         {skipped_bot:,}")
    print(f"Skipped (multi-party):         {skipped_multiparty:,}")
    print(f"Skipped (quality filter):      {skipped_quality:,}")
    print(f"Output file:                   {OUTPUT_PATH}")
    print(f"File size:                     {file_size / 1024 / 1024:.1f} MB")
    print(f"Total characters:              {total_chars:,}")
    print(f"Estimated tokens (~3 char/tok): {estimated_tokens:,}")
    print("=" * 60)

    # Print a few example conversations
    print("\n\nSAMPLE CONVERSATIONS:")
    print("-" * 60)
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            record = json.loads(line)
            text = record["text"]
            # Show first 800 chars of each
            preview = text[:800]
            if len(text) > 800:
                preview += "\n... [truncated]"
            print(f"\n--- Example {i+1} ---")
            print(preview)
            print()


if __name__ == "__main__":
    process_all_dialogs()
