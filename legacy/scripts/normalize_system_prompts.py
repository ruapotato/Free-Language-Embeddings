#!/usr/bin/env python3
"""
Normalize all SFT system prompts to the standardized format from system_prompt.py.

For each conversation that HAS a system prompt, this script:
1. Detects what OS the conversation is about (from prompt text + conversation content)
2. Replaces the system prompt with the full standardized format from the matching preset
3. Occasionally (20%) uses the short format for variety
4. Leaves conversations WITHOUT system prompts untouched

Also validates that OS assignment is consistent with conversation content
(e.g., pacman commands → Arch, not Debian).
"""

import json
import random
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from system_prompt import PRESETS, SYSTEM_PROMPT_TEMPLATE, SYSTEM_PROMPT_SHORT


random.seed(42)

INPUT = "data/sft/flm_combined.jsonl"
OUTPUT = "data/sft/flm_combined.jsonl"  # overwrite in place

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE, INPUT)
OUTPUT_PATH = os.path.join(BASE, OUTPUT)


def detect_os_from_prompt(sys_text):
    """Detect OS from existing system prompt text."""
    lower = sys_text.lower()
    if "arch linux" in lower or "arch" in lower and "pacman" in lower:
        return "arch"
    if "fedora" in lower:
        return "fedora_41"
    if "ubuntu 24.04" in lower or "noble" in lower:
        return "ubuntu_noble"
    if "ubuntu 22.04" in lower or "jammy" in lower:
        return "ubuntu_jammy"
    if "ubuntu" in lower:
        return "ubuntu_noble"  # default ubuntu
    if "bookworm" in lower or "debian 12" in lower or "debian gnu/linux 12" in lower:
        return "debian_bookworm"
    if "trixie" in lower or "debian 13" in lower or "debian gnu/linux 13" in lower:
        return "debian_trixie"
    if "server" in lower and "minimal" in lower:
        return "debian_server_minimal"
    if "debian" in lower:
        return "debian_trixie"  # default debian
    return None  # can't tell from prompt alone


def detect_os_from_content(conv_text):
    """Detect OS from conversation content (commands used, paths mentioned, etc.)."""
    lower = conv_text.lower()

    # Strong signals from package manager commands
    if "pacman -s" in lower or "pacman -q" in lower or "pacman -r" in lower or "makepkg" in lower or "aur" in lower:
        return "arch"
    if "dnf install" in lower or "dnf search" in lower or "rpm -q" in lower or "fedora" in lower:
        return "fedora_41"
    if "snap install" in lower or "ppa:" in lower or "add-apt-repository" in lower:
        return "ubuntu_noble"

    # Weaker signals
    if "apt install" in lower or "apt-get" in lower or "dpkg" in lower or "debian" in lower:
        # Could be debian or ubuntu
        if "ubuntu" in lower:
            return "ubuntu_noble"
        return "debian_trixie"

    return None


def pick_weighted_preset():
    """Pick a random OS preset with weighted distribution."""
    weights = [
        ("debian_trixie", 0.35),
        ("debian_bookworm", 0.15),
        ("ubuntu_noble", 0.15),
        ("ubuntu_jammy", 0.05),
        ("arch", 0.15),
        ("fedora_41", 0.10),
        ("debian_server_minimal", 0.05),
    ]
    names, probs = zip(*weights)
    return random.choices(names, weights=probs, k=1)[0]


def build_prompt(preset_name, short=False):
    """Build a standardized system prompt from a preset."""
    info = PRESETS[preset_name]
    template = SYSTEM_PROMPT_SHORT if short else SYSTEM_PROMPT_TEMPLATE
    return template.format(**info)


def is_already_standardized(sys_text):
    """Check if this system prompt already has the full standardized format."""
    return all(k in sys_text for k in ["Kernel:", "Shell:", "Packages:", "Pkg manager:"])


def normalize_conversation(obj):
    """Normalize a single conversation's system prompt. Returns modified obj."""
    text = obj["text"]

    # No system prompt → leave as-is
    if "<|system|>" not in text:
        return obj, "no_system"

    # Split out system prompt
    parts = text.split("<|system|>", 1)
    after_system = parts[1]
    sys_and_rest = after_system.split("<|user|>", 1)
    if len(sys_and_rest) < 2:
        return obj, "malformed"

    old_sys = sys_and_rest[0].strip()
    rest = sys_and_rest[1]

    # Already standardized → leave as-is
    if is_already_standardized(old_sys):
        return obj, "already_good"

    # Detect OS from prompt
    os_from_prompt = detect_os_from_prompt(old_sys)

    # Detect OS from conversation content
    os_from_content = detect_os_from_content(rest)

    # Decide which preset to use
    if os_from_content and os_from_prompt:
        # Content takes priority for consistency (if content uses pacman, must be Arch)
        # But check for conflicts
        if os_from_content == os_from_prompt:
            preset = os_from_content
        elif os_from_content in ("arch", "fedora_41"):
            # Content uses non-Debian tools → content wins
            preset = os_from_content
        else:
            # Prompt is more specific usually
            preset = os_from_prompt
    elif os_from_content:
        preset = os_from_content
    elif os_from_prompt:
        preset = os_from_prompt
    else:
        # Can't tell → weighted random
        preset = pick_weighted_preset()

    # Decide short vs full (20% short)
    short = random.random() < 0.20

    new_sys = build_prompt(preset, short=short)

    # Rebuild text
    new_text = f"<|system|>\n{new_sys}\n<|user|>\n{rest}"
    obj["text"] = new_text

    return obj, f"normalized:{preset}{'_short' if short else ''}"


def main():
    # Read all conversations
    with open(INPUT_PATH) as f:
        conversations = [json.loads(line) for line in f if line.strip()]

    print(f"Read {len(conversations)} conversations from {INPUT}")

    # Normalize
    stats = {}
    results = []
    for obj in conversations:
        obj, status = normalize_conversation(obj)
        results.append(obj)
        stats[status] = stats.get(status, 0) + 1

    # Write back
    with open(OUTPUT_PATH, "w") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(results)} conversations to {OUTPUT}")
    print("\nNormalization stats:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    # Verify
    print("\n--- Post-normalization audit ---")
    full = partial = none = 0
    os_dist = {}
    with open(OUTPUT_PATH) as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            if "<|system|>" not in text:
                none += 1
                continue
            sys_block = text.split("<|system|>")[1].split("<|user|>")[0].strip()
            if all(k in sys_block for k in ["Kernel:", "Shell:", "Packages:", "Pkg manager:"]):
                full += 1
                # Detect which OS
                for preset_name, info in PRESETS.items():
                    if info["pretty_name"] in sys_block:
                        os_dist[preset_name] = os_dist.get(preset_name, 0) + 1
                        break
            elif "System:" in sys_block and "Pkg manager:" in sys_block:
                # Short format
                full += 1
                for preset_name, info in PRESETS.items():
                    if info["pretty_name"] in sys_block:
                        os_dist[preset_name] = os_dist.get(preset_name, 0) + 1
                        break
            else:
                partial += 1

    print(f"  Full standard system prompts: {full}")
    print(f"  Still non-standard: {partial}")
    print(f"  No system prompt: {none}")
    print(f"\n  OS distribution in system prompts:")
    for k, v in sorted(os_dist.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
