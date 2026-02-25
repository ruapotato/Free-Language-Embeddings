"""
Standardized system prompt for flm — The Free Language Model.

This is the SINGLE SOURCE OF TRUTH for flm's system prompt format.
All SFT data, training scripts, and inference code should use this module.

The system prompt is built from /etc/os-release and system details detected
at runtime. During SFT data generation, we generate prompts for multiple
OS variants to teach flm to adapt its behavior.
"""

import os
import platform
import shutil
import subprocess


# ---------------------------------------------------------------------------
# Core template — every system prompt follows this exact structure
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = (
    "You are flm, the Free Language Model — free as in freedom. "
    "You were trained from scratch by David Hamner on a single RTX 3090. "
    "You are fully free software under GPL-3.0.\n"
    "System: {pretty_name}. Kernel: {kernel}. "
    "Shell: {shell}. Python: {python}. "
    "Packages: {pkg_count}. Pkg manager: {pkg_manager}."
)

# Short variant (for when we want less context)
SYSTEM_PROMPT_SHORT = (
    "You are flm, the Free Language Model. "
    "System: {pretty_name}. Pkg manager: {pkg_manager}."
)


# ---------------------------------------------------------------------------
# Detect real system details from /etc/os-release + environment
# ---------------------------------------------------------------------------
def detect_system():
    """Read /etc/os-release and system info. Returns a dict."""
    info = {
        "pretty_name": "Unknown Linux",
        "id": "linux",
        "version_id": "",
        "version_codename": "",
        "kernel": platform.release(),
        "shell": "",
        "python": platform.python_version(),
        "pkg_count": "unknown",
        "pkg_manager": "unknown",
    }

    # Parse /etc/os-release
    if os.path.exists("/etc/os-release"):
        with open("/etc/os-release") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    key, _, val = line.partition("=")
                    val = val.strip('"')
                    if key == "PRETTY_NAME":
                        info["pretty_name"] = val
                    elif key == "ID":
                        info["id"] = val
                    elif key == "VERSION_ID":
                        info["version_id"] = val
                    elif key == "VERSION_CODENAME":
                        info["version_codename"] = val

    # Shell
    shell = os.environ.get("SHELL", "")
    if shell:
        try:
            result = subprocess.run(
                [shell, "--version"], capture_output=True, text=True, timeout=5
            )
            first_line = result.stdout.strip().split("\n")[0]
            # Extract just "bash 5.2" or "zsh 5.9" style
            if "bash" in first_line.lower():
                ver = first_line.split("version ")[1].split("(")[0].strip() if "version " in first_line else ""
                info["shell"] = f"bash {ver}" if ver else "bash"
            elif "zsh" in first_line.lower():
                info["shell"] = first_line.split("(")[0].strip()
            else:
                info["shell"] = os.path.basename(shell)
        except Exception:
            info["shell"] = os.path.basename(shell)
    else:
        info["shell"] = "sh"

    # Package count and manager
    os_id = info["id"]
    if os_id in ("debian", "ubuntu", "linuxmint", "pop"):
        info["pkg_manager"] = "apt"
        try:
            result = subprocess.run(
                ["dpkg-query", "-f", ".\n", "-W"],
                capture_output=True, text=True, timeout=10
            )
            info["pkg_count"] = str(result.stdout.count("\n"))
        except Exception:
            pass
    elif os_id in ("fedora", "rhel", "centos", "rocky", "alma"):
        info["pkg_manager"] = "dnf"
        try:
            result = subprocess.run(
                ["rpm", "-qa"], capture_output=True, text=True, timeout=10
            )
            info["pkg_count"] = str(result.stdout.strip().count("\n") + 1)
        except Exception:
            pass
    elif os_id in ("arch", "manjaro", "endeavouros"):
        info["pkg_manager"] = "pacman"
        try:
            result = subprocess.run(
                ["pacman", "-Q"], capture_output=True, text=True, timeout=10
            )
            info["pkg_count"] = str(result.stdout.strip().count("\n") + 1)
        except Exception:
            pass
    elif os_id in ("opensuse", "opensuse-leap", "opensuse-tumbleweed"):
        info["pkg_manager"] = "zypper"

    return info


def build_system_prompt(info=None, short=False):
    """Build the system prompt from detected or provided system info."""
    if info is None:
        info = detect_system()
    template = SYSTEM_PROMPT_SHORT if short else SYSTEM_PROMPT_TEMPLATE
    return template.format(**info)


# ---------------------------------------------------------------------------
# Preset system details for SFT data generation
# These simulate different OS environments for training diversity
# ---------------------------------------------------------------------------
PRESETS = {
    "debian_trixie": {
        "pretty_name": "Debian GNU/Linux 13 (trixie)",
        "id": "debian",
        "version_id": "13",
        "version_codename": "trixie",
        "kernel": "6.12.6-amd64",
        "shell": "bash 5.2",
        "python": "3.12.8",
        "pkg_count": "1847",
        "pkg_manager": "apt",
    },
    "debian_bookworm": {
        "pretty_name": "Debian GNU/Linux 12 (bookworm)",
        "id": "debian",
        "version_id": "12",
        "version_codename": "bookworm",
        "kernel": "6.1.0-27-amd64",
        "shell": "bash 5.2",
        "python": "3.11.2",
        "pkg_count": "1523",
        "pkg_manager": "apt",
    },
    "ubuntu_noble": {
        "pretty_name": "Ubuntu 24.04 LTS (Noble Numbat)",
        "id": "ubuntu",
        "version_id": "24.04",
        "version_codename": "noble",
        "kernel": "6.8.0-45-generic",
        "shell": "bash 5.2",
        "python": "3.12.3",
        "pkg_count": "2104",
        "pkg_manager": "apt",
    },
    "ubuntu_jammy": {
        "pretty_name": "Ubuntu 22.04 LTS (Jammy Jellyfish)",
        "id": "ubuntu",
        "version_id": "22.04",
        "version_codename": "jammy",
        "kernel": "5.15.0-91-generic",
        "shell": "bash 5.1",
        "python": "3.10.12",
        "pkg_count": "1876",
        "pkg_manager": "apt",
    },
    "arch": {
        "pretty_name": "Arch Linux (rolling)",
        "id": "arch",
        "version_id": "",
        "version_codename": "",
        "kernel": "6.11.5-arch1-1",
        "shell": "zsh 5.9",
        "python": "3.12.7",
        "pkg_count": "943",
        "pkg_manager": "pacman",
    },
    "fedora_41": {
        "pretty_name": "Fedora Linux 41 (Workstation Edition)",
        "id": "fedora",
        "version_id": "41",
        "version_codename": "",
        "kernel": "6.11.4-301.fc41",
        "shell": "bash 5.2",
        "python": "3.13.0",
        "pkg_count": "2341",
        "pkg_manager": "dnf",
    },
    "debian_server_minimal": {
        "pretty_name": "Debian GNU/Linux 13 (trixie)",
        "id": "debian",
        "version_id": "13",
        "version_codename": "trixie",
        "kernel": "6.12.6-amd64",
        "shell": "bash 5.2",
        "python": "3.12.8",
        "pkg_count": "312",
        "pkg_manager": "apt",
    },
}


def get_preset_prompt(preset_name, short=False):
    """Get a system prompt for a specific OS preset."""
    return build_system_prompt(PRESETS[preset_name], short=short)


# ---------------------------------------------------------------------------
# Convenience: pick a random preset (weighted toward Debian)
# ---------------------------------------------------------------------------
def random_system_prompt(rng=None, include_none=True, short_chance=0.2):
    """Pick a random system prompt for SFT data generation.

    Weights:
        debian_trixie: 30%  (primary target)
        debian_bookworm: 10%
        ubuntu_noble: 10%
        ubuntu_jammy: 5%
        arch: 10%
        fedora_41: 5%
        debian_server_minimal: 5%
        None (no system prompt): 25%
    """
    import random
    rng = rng or random

    weights = [
        ("debian_trixie", 0.30),
        ("debian_bookworm", 0.10),
        ("ubuntu_noble", 0.10),
        ("ubuntu_jammy", 0.05),
        ("arch", 0.10),
        ("fedora_41", 0.05),
        ("debian_server_minimal", 0.05),
    ]
    if include_none:
        weights.append((None, 0.25))

    names, probs = zip(*weights)
    choice = rng.choices(names, weights=probs, k=1)[0]

    if choice is None:
        return None

    short = rng.random() < short_chance
    return get_preset_prompt(choice, short=short)


# ---------------------------------------------------------------------------
# CLI: print detected system prompt
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    info = detect_system()
    print("Detected system info:")
    print(json.dumps(info, indent=2))
    print()
    print("Full system prompt:")
    print(build_system_prompt(info))
    print()
    print("Short system prompt:")
    print(build_system_prompt(info, short=True))
    print()
    print("--- Presets ---")
    for name in PRESETS:
        print(f"\n[{name}]")
        print(get_preset_prompt(name))
