#!/bin/bash
# build_debian_dataset.sh — Build a Debian chroot for path2vec training data
#
# Creates a full Debian installation in a chroot, installs hundreds of
# packages for maximum filesystem coverage, then walks the tree to
# collect all paths as training data for V32 (path2vec).
#
# Requires: sudo, debootstrap
# Usage:
#   ./build_debian_dataset.sh          # Build chroot + collect paths
#   ./build_debian_dataset.sh --clean  # Remove chroot when done
#
# Output: checkpoints/path2vec_v32/paths.json
#         checkpoints/path2vec_v32/vocab.json

set -euo pipefail

CHROOT="/tmp/debian_chroot"
SUITE="trixie"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints/path2vec_v32"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
    log "ERROR: $*" >&2
    exit 1
}

check_deps() {
    # debootstrap lives in /usr/sbin which may not be in PATH
    if ! command -v debootstrap >/dev/null 2>&1 && ! [[ -x /usr/sbin/debootstrap ]]; then
        die "debootstrap not found. Install: sudo apt install debootstrap"
    fi
    command -v python3 >/dev/null 2>&1 || die "python3 not found"

    if [[ $EUID -eq 0 ]]; then
        SUDO=""
    else
        SUDO="sudo"
        $SUDO true 2>/dev/null || die "sudo access required"
    fi
}

# ---------------------------------------------------------------------------
# Mount / unmount virtual filesystems
# ---------------------------------------------------------------------------

mount_virtual_fs() {
    log "Mounting virtual filesystems..."
    for mp in proc sys dev; do
        mountpoint -q "${CHROOT}/${mp}" 2>/dev/null && continue
        case $mp in
            proc) $SUDO mount -t proc proc "${CHROOT}/proc" ;;
            sys)  $SUDO mount -t sysfs sysfs "${CHROOT}/sys" ;;
            dev)  $SUDO mount --bind /dev "${CHROOT}/dev" ;;
        esac
        log "  Mounted: ${CHROOT}/${mp}"
    done
}

unmount_virtual_fs() {
    log "Unmounting virtual filesystems..."
    for mp in dev sys proc; do
        if mountpoint -q "${CHROOT}/${mp}" 2>/dev/null; then
            $SUDO umount -l "${CHROOT}/${mp}" 2>/dev/null || true
            log "  Unmounted: ${CHROOT}/${mp}"
        fi
    done
}

# Cleanup on exit
cleanup() {
    unmount_virtual_fs
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Phase 1: Bootstrap
# ---------------------------------------------------------------------------

bootstrap() {
    if [[ -d "${CHROOT}/usr" ]]; then
        log "Chroot already exists at ${CHROOT}, skipping bootstrap"
        return
    fi

    log "Bootstrapping Debian ${SUITE} into ${CHROOT}..."
    $SUDO /usr/sbin/debootstrap --variant=minbase "${SUITE}" "${CHROOT}"
    log "Base system installed."
}

# ---------------------------------------------------------------------------
# Phase 2: Install packages
# ---------------------------------------------------------------------------

install_packages() {
    mount_virtual_fs

    # Update package lists
    log "Updating package lists..."
    $SUDO chroot "${CHROOT}" apt-get update -qq

    # Package list — organized by category for readability
    # Goal: maximize filesystem diversity (lots of /usr/lib, /usr/share,
    # /usr/include, /etc, /var paths from different domains)

    PACKAGES=(
        # --- Build tools & compilers ---
        build-essential cmake autoconf automake libtool pkg-config
        gcc g++ gfortran gdb valgrind strace ltrace
        bison flex nasm yasm

        # --- Version control ---
        git subversion mercurial

        # --- Languages & runtimes ---
        python3 python3-pip python3-venv python3-dev
        python3-numpy python3-scipy python3-matplotlib
        ruby perl lua5.4 gawk tcl tk
        default-jdk ant maven
        nodejs npm
        # golang-go rustc cargo  # large, may fail in minbase

        # --- Editors ---
        vim emacs-nox nano ed

        # --- Web servers ---
        apache2 nginx lighttpd

        # --- Databases ---
        postgresql mariadb-server sqlite3
        redis-server memcached

        # --- Mail ---
        postfix dovecot-imapd

        # --- Network servers ---
        openssh-server vsftpd proftpd-basic
        bind9 dnsmasq
        samba nfs-kernel-server

        # --- System tools ---
        systemd cron at logrotate
        rsyslog fail2ban ufw
        lvm2 mdadm cryptsetup
        parted gdisk dosfstools ntfs-3g
        iproute2 iptables nftables
        tcpdump nmap netcat-openbsd curl wget
        htop iotop sysstat
        tmux screen

        # --- Development libraries (adds /usr/lib, /usr/include) ---
        libssl-dev libcurl4-openssl-dev libxml2-dev libxslt1-dev
        libsqlite3-dev libpq-dev
        libboost-all-dev libglib2.0-dev
        libgtk-3-dev
        libpng-dev libjpeg-dev libtiff-dev libwebp-dev
        libfreetype-dev libfontconfig-dev
        libx11-dev libxext-dev libxrandr-dev libxi-dev
        libasound2-dev libpulse-dev
        libavcodec-dev libavformat-dev libswscale-dev
        libzmq3-dev libprotobuf-dev protobuf-compiler
        libsodium-dev liblz4-dev libzstd-dev

        # --- Desktop / GUI (lots of shared files) ---
        xfce4 lxde i3 openbox
        xterm x11-apps x11-utils

        # --- Documentation (fills /usr/share/doc, /usr/share/man) ---
        man-db manpages manpages-dev
        info texinfo doc-base

        # --- TeX (enormous file tree under /usr/share/texlive) ---
        texlive-base texlive-latex-base texlive-fonts-recommended

        # --- Multimedia ---
        imagemagick ffmpeg sox
        gimp inkscape

        # --- Compression ---
        zip unzip p7zip-full xz-utils zstd lz4 bzip2

        # --- Text processing ---
        jq xmlstarlet xsltproc
        groff ghostscript

        # --- Security ---
        gnupg ca-certificates openssl
        apparmor

        # --- Fonts (fills /usr/share/fonts) ---
        fonts-dejavu fonts-liberation fonts-noto-core
        fonts-freefont-ttf fonts-hack fonts-firacode

        # --- Locale / i18n ---
        locales tzdata

        # --- Misc ---
        file less tree
        bash-completion
        acl attr
        dbus policykit-1
    )

    # Install in batches — some packages may not exist or conflict
    BATCH_SIZE=20
    total=${#PACKAGES[@]}
    installed_ok=0
    failed_pkgs=()

    for ((i=0; i<total; i+=BATCH_SIZE)); do
        batch=("${PACKAGES[@]:i:BATCH_SIZE}")
        batch_num=$(( i/BATCH_SIZE + 1 ))
        log "Installing batch ${batch_num} (${#batch[@]} packages): ${batch[*]:0:5}..."

        if DEBIAN_FRONTEND=noninteractive $SUDO chroot "${CHROOT}" \
            apt-get install -y --no-install-recommends "${batch[@]}" \
            >/dev/null 2>&1; then
            installed_ok=$((installed_ok + ${#batch[@]}))
        else
            # Retry individually
            for pkg in "${batch[@]}"; do
                if DEBIAN_FRONTEND=noninteractive $SUDO chroot "${CHROOT}" \
                    apt-get install -y --no-install-recommends "$pkg" \
                    >/dev/null 2>&1; then
                    installed_ok=$((installed_ok + 1))
                else
                    failed_pkgs+=("$pkg")
                fi
            done
        fi
    done

    # Count what we got
    pkg_count=$($SUDO chroot "${CHROOT}" dpkg --list 2>/dev/null | grep -c '^ii ' || echo 0)
    log "Installation complete: ${pkg_count} packages installed"
    log "  Requested: ${total}, succeeded: ${installed_ok}, failed: ${#failed_pkgs[@]}"
    if [[ ${#failed_pkgs[@]} -gt 0 ]]; then
        log "  Failed: ${failed_pkgs[*]}"
    fi
}

# ---------------------------------------------------------------------------
# Phase 3: Collect paths
# ---------------------------------------------------------------------------

collect_paths() {
    mount_virtual_fs

    log "Collecting filesystem paths from ${CHROOT}..."
    mkdir -p "${CHECKPOINT_DIR}"

    # Walk the entire chroot, including virtual filesystems
    # Output: one path per line, relative to chroot root
    # Limit /proc to avoid infinite recursion on certain entries
    # Timeout find on /proc after collecting structure

    PATHS_RAW="${CHECKPOINT_DIR}/paths_raw.txt"

    # Main filesystem (excluding proc/sys which we handle specially)
    log "  Walking main filesystem..."
    $SUDO find "${CHROOT}" \
        -path "${CHROOT}/proc" -prune -o \
        -path "${CHROOT}/sys" -prune -o \
        -print 2>/dev/null | \
        sed "s|^${CHROOT}||" | \
        grep -v '^$' > "${PATHS_RAW}"
    main_count=$(wc -l < "${PATHS_RAW}")
    log "  Main filesystem: ${main_count} paths"

    # /proc — just the structure, not per-PID dynamic content
    # Grab /proc/[non-numeric]/* for the structural parts
    log "  Walking /proc (structural only)..."
    if mountpoint -q "${CHROOT}/proc" 2>/dev/null; then
        $SUDO find "${CHROOT}/proc" -maxdepth 1 -not -regex '.*/[0-9]+' -print 2>/dev/null | \
            sed "s|^${CHROOT}||" >> "${PATHS_RAW}" || true
        # Get one PID's structure as a template
        SAMPLE_PID=$($SUDO ls "${CHROOT}/proc" 2>/dev/null | grep -E '^[0-9]+$' | head -1)
        if [[ -n "${SAMPLE_PID}" ]]; then
            $SUDO find "${CHROOT}/proc/${SAMPLE_PID}" -maxdepth 2 -print 2>/dev/null | \
                sed "s|^${CHROOT}||;s|${SAMPLE_PID}|PID|g" >> "${PATHS_RAW}" || true
        fi
    fi

    # /sys — limit depth to avoid enormous output
    log "  Walking /sys (depth-limited)..."
    if mountpoint -q "${CHROOT}/sys" 2>/dev/null; then
        $SUDO find "${CHROOT}/sys" -maxdepth 4 -print 2>/dev/null | \
            sed "s|^${CHROOT}||" >> "${PATHS_RAW}" || true
    fi

    total_count=$(wc -l < "${PATHS_RAW}")
    log "  Total raw paths: ${total_count}"

    # Sort and deduplicate
    sort -u "${PATHS_RAW}" -o "${PATHS_RAW}"
    dedup_count=$(wc -l < "${PATHS_RAW}")
    log "  After dedup: ${dedup_count}"

    # Convert to tokenized JSON using Python
    log "Building vocabulary..."
    python3 - "${PATHS_RAW}" "${CHECKPOINT_DIR}" <<'PYEOF'
import sys
import json
from collections import Counter

paths_file = sys.argv[1]
out_dir = sys.argv[2]

# Read and tokenize paths
all_tokenized = []
with open(paths_file) as f:
    for line in f:
        path = line.strip()
        if not path:
            continue
        # Split on / and filter empty/long components
        parts = [p for p in path.split("/") if p and len(p) <= 100]
        if len(parts) >= 2:
            all_tokenized.append(parts)

# Statistics
token_counts = Counter()
lengths = []
for tokens in all_tokenized:
    token_counts.update(tokens)
    lengths.append(len(tokens))

import numpy as np
print(f"  Tokenized paths: {len(all_tokenized):,} (with >=2 components)")
print(f"  Unique tokens: {len(token_counts):,}")
print(f"  Avg path length: {np.mean(lengths):.1f} components")
print(f"  Max path length: {max(lengths)} components")
top30 = ", ".join(f"{w}({c})" for w, c in token_counts.most_common(30))
print(f"  Top 30: {top30}")

# Save tokenized paths
paths_out = f"{out_dir}/paths.json"
with open(paths_out, "w") as f:
    json.dump({"paths": all_tokenized}, f)
print(f"  Saved: {paths_out}")

# Build vocabulary (min_count=3)
MIN_COUNT = 3
MAX_VOCAB = 200_000
filtered = [(w, c) for w, c in token_counts.most_common() if c >= MIN_COUNT]
if len(filtered) > MAX_VOCAB:
    filtered = filtered[:MAX_VOCAB]

word2id = {w: i for i, (w, _) in enumerate(filtered)}
counts = [c for _, c in filtered]
total_count = sum(counts)

vocab_data = {
    "word2id": word2id,
    "counts": counts,
    "total_count": total_count,
}
vocab_out = f"{out_dir}/vocab.json"
with open(vocab_out, "w") as f:
    json.dump(vocab_data, f)

print(f"  Vocabulary: {len(word2id):,} tokens (min_count={MIN_COUNT})")
print(f"  Top 20: {', '.join(w for w, _ in filtered[:20])}")
print(f"  Saved: {vocab_out}")
PYEOF

    log "Data collection complete."
}

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean_chroot() {
    log "Removing chroot at ${CHROOT}..."
    unmount_virtual_fs
    $SUDO rm -rf "${CHROOT}"
    log "Chroot removed."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    check_deps

    case "${1:-}" in
        --clean)
            clean_chroot
            ;;
        --collect-only)
            collect_paths
            ;;
        --install-only)
            install_packages
            ;;
        *)
            bootstrap
            install_packages
            collect_paths
            log ""
            log "=========================================="
            log "Dataset ready! Now train with:"
            log "  python train_v32.py --fresh"
            log "=========================================="
            log ""
            log "To clean up the chroot later:"
            log "  ./build_debian_dataset.sh --clean"
            ;;
    esac
}

main "$@"
