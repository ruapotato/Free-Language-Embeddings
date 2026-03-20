#!/usr/bin/env python3
"""Write hand-crafted SFT conversations for V4 training — Batch 1.

Topics: Disk & Storage, Firewall, Terminal Tools, Permissions, Memory,
        Compression, File Tools, Networking, Security, Web/DB.

Appends to data/sft/flm_v4_massive.jsonl.
"""
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(REPO_ROOT, "data", "sft", "flm_v4_massive.jsonl")

# ── System prompt variants ──────────────────────────────────────────────────

SYS_DEB13 = (
    "You are flm, the Free Language Model \u2014 free as in freedom. "
    "You were trained from scratch by David Hamner on a single RTX 3090. "
    "You are fully free software under GPL-3.0.\n"
    "System: Debian GNU/Linux 13 (trixie). Kernel: 6.12.6-amd64. "
    "Shell: bash 5.2. Python: 3.12.8. Packages: 1847. Pkg manager: apt."
)

SYS_DEB12 = (
    "You are flm, the Free Language Model \u2014 free as in freedom. "
    "You were trained from scratch by David Hamner on a single RTX 3090. "
    "You are fully free software under GPL-3.0.\n"
    "System: Debian GNU/Linux 12 (bookworm). Kernel: 6.1.0-27-amd64. "
    "Shell: bash 5.2. Python: 3.11.2. Packages: 1523. Pkg manager: apt."
)

SYS_UBU24 = (
    "You are flm, the Free Language Model \u2014 free as in freedom. "
    "You were trained from scratch by David Hamner on a single RTX 3090. "
    "You are fully free software under GPL-3.0.\n"
    "System: Ubuntu 24.04 LTS (Noble Numbat). Kernel: 6.8.0-45-generic. "
    "Shell: bash 5.2. Python: 3.12.3. Packages: 2104. Pkg manager: apt."
)

SYS_UBU22 = (
    "You are flm, the Free Language Model \u2014 free as in freedom. "
    "You were trained from scratch by David Hamner on a single RTX 3090. "
    "You are fully free software under GPL-3.0.\n"
    "System: Ubuntu 22.04 LTS (Jammy Jellyfish). Kernel: 5.15.0-91-generic. "
    "Shell: bash 5.1. Python: 3.10.12. Packages: 1876. Pkg manager: apt."
)

SYS_ARCH = (
    "You are flm, the Free Language Model \u2014 free as in freedom. "
    "You were trained from scratch by David Hamner on a single RTX 3090. "
    "You are fully free software under GPL-3.0.\n"
    "System: Arch Linux (rolling). Kernel: 6.11.5-arch1-1. "
    "Shell: zsh 5.9. Python: 3.12.7. Packages: 943. Pkg manager: pacman."
)

SYS_FED41 = (
    "You are flm, the Free Language Model \u2014 free as in freedom. "
    "You were trained from scratch by David Hamner on a single RTX 3090. "
    "You are fully free software under GPL-3.0.\n"
    "System: Fedora Linux 41 (Workstation Edition). Kernel: 6.11.4-301.fc41. "
    "Shell: bash 5.2. Python: 3.13.0. Packages: 2341. Pkg manager: dnf."
)

SYS_DEBMIN = (
    "You are flm, the Free Language Model \u2014 free as in freedom. "
    "You were trained from scratch by David Hamner on a single RTX 3090. "
    "You are fully free software under GPL-3.0.\n"
    "System: Debian GNU/Linux 13 (trixie). Kernel: 6.12.6-amd64. "
    "Shell: bash 5.2. Python: 3.12.8. Packages: 312. Pkg manager: apt."
)

SHORT_DEB13 = (
    "You are flm, the Free Language Model. "
    "System: Debian GNU/Linux 13 (trixie). Pkg manager: apt."
)

SHORT_ARCH = (
    "You are flm, the Free Language Model. "
    "System: Arch Linux (rolling). Pkg manager: pacman."
)

SHORT_FED41 = (
    "You are flm, the Free Language Model. "
    "System: Fedora Linux 41 (Workstation Edition). Pkg manager: dnf."
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def make(system, question, answer, source):
    """Build one conversation dict."""
    if system is None:
        text = f"<|user|>\n{question}\n<|assistant|>\n{answer}\n"
    else:
        text = f"<|system|>\n{system}\n<|user|>\n{question}\n<|assistant|>\n{answer}\n"
    return {"text": text, "source": source}


# ── Conversations ───────────────────────────────────────────────────────────

conversations = []

# ---------------------------------------------------------------------------
# 1. Check disk health with smartctl (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I check the health of my hard drives?",
    r"""Use `smartctl` from the smartmontools package:

```bash
sudo apt install smartmontools
```

**Quick health check:**
```bash
sudo smartctl -H /dev/sda
```

This prints `PASSED` or `FAILED` based on the drive's self-assessment.

**Detailed SMART attributes:**
```bash
sudo smartctl -A /dev/sda
```

Key attributes to watch:
- **Reallocated_Sector_Ct** — bad sectors remapped. Any nonzero value is a warning sign.
- **Current_Pending_Sector** — sectors waiting to be remapped. Should be 0.
- **Offline_Uncorrectable** — sectors that couldn't be read. Should be 0.
- **UDMA_CRC_Error_Count** — cable/connection errors.

**Full info dump:**
```bash
sudo smartctl -a /dev/sda
```

**Run a self-test:**
```bash
# Short test (~2 minutes)
sudo smartctl -t short /dev/sda

# Long test (can take hours depending on drive size)
sudo smartctl -t long /dev/sda

# Check test results
sudo smartctl -l selftest /dev/sda
```

**For NVMe drives:**
```bash
sudo smartctl -a /dev/nvme0
```

NVMe reports differently — look at `Percentage Used`, `Available Spare`, and `Media and Data Integrity Errors`.

**Enable automatic monitoring:**
```bash
sudo systemctl enable --now smartd
```

The `smartd` daemon monitors drives and logs warnings. Configure it in `/etc/smartd.conf`:
```
/dev/sda -a -o on -S on -s (S/../.././02|L/../../6/03) -m root
```
This runs short tests daily at 2am and long tests Saturdays at 3am, emailing root on problems.""",
    "flm_disk_storage"
))

# ---------------------------------------------------------------------------
# 2. Check disk health with smartctl (Arch)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_ARCH,
    "How do I check disk health?",
    r"""Install smartmontools:

```bash
sudo pacman -S smartmontools
```

**Quick health status:**
```bash
sudo smartctl -H /dev/sda
```

**Full SMART data:**
```bash
sudo smartctl -a /dev/sda
```

For NVMe drives (common on Arch installs):
```bash
sudo smartctl -a /dev/nvme0n1
```

**Run self-tests:**
```bash
sudo smartctl -t short /dev/sda   # ~2 min
sudo smartctl -t long /dev/sda    # hours
sudo smartctl -l selftest /dev/sda  # view results
```

**Enable automatic monitoring with smartd:**
```bash
sudo systemctl enable --now smartd
```

Default config at `/etc/smartd.conf` monitors all drives. For custom alerts, edit it:
```
/dev/sda -a -o on -S on -s (S/../.././02|L/../../6/03) -m you@example.com
```

**Key SMART attributes to worry about:**
- `Reallocated_Sector_Ct > 0` — drive is remapping bad sectors
- `Current_Pending_Sector > 0` — sectors waiting to be remapped
- `UDMA_CRC_Error_Count` increasing — possibly a bad SATA cable

If any of these are climbing, back up immediately and plan a replacement.""",
    "flm_disk_storage"
))

# ---------------------------------------------------------------------------
# 3. Partition a new disk with fdisk/gdisk (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "I just added a new disk. How do I partition and format it?",
    r"""First, identify the new disk:

```bash
lsblk
# or
sudo fdisk -l
```

Let's say it's `/dev/sdb`.

**Option A: GPT with gdisk (recommended for disks > 2TB or UEFI)**

```bash
sudo apt install gdisk
sudo gdisk /dev/sdb
```

Inside gdisk:
```
o        # create new GPT partition table
n        # new partition
         # accept defaults for partition number (1), first sector
+100G    # size (or press Enter for whole disk)
8300     # Linux filesystem type code
w        # write and exit
```

**Option B: MBR with fdisk (legacy/BIOS systems)**

```bash
sudo fdisk /dev/sdb
```

Inside fdisk:
```
o        # create new MBR partition table
n        # new partition
p        # primary
1        # partition number
         # accept default first sector
+100G    # size (or Enter for whole disk)
w        # write and exit
```

**Format the partition:**

```bash
# ext4 (most common)
sudo mkfs.ext4 /dev/sdb1

# XFS (good for large files)
sudo mkfs.xfs /dev/sdb1
```

**Mount it:**
```bash
sudo mkdir -p /mnt/data
sudo mount /dev/sdb1 /mnt/data
```

**Make it permanent in /etc/fstab:**
```bash
# Get the UUID
sudo blkid /dev/sdb1
# Add to fstab
echo 'UUID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx  /mnt/data  ext4  defaults,nofail  0  2' | sudo tee -a /etc/fstab
```

Use `nofail` so the system still boots if the disk is missing.""",
    "flm_disk_storage"
))

# ---------------------------------------------------------------------------
# 4. Resize a partition (Fedora, LVM)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_FED41,
    "I need more space on my root partition. How do I resize it? Fedora uses LVM by default right?",
    r"""Yes, Fedora's default install uses LVM on top of partitions (often inside a LUKS container too). This makes resizing straightforward.

**Check your current layout:**
```bash
lsblk
sudo lvs
sudo vgs
sudo pvs
```

You'll typically see something like:
```
/dev/sda2 -> /dev/fedora/root (ext4 or xfs)
/dev/sda2 -> /dev/fedora/swap
```

**If you have free space in the volume group:**
```bash
# Check free space
sudo vgs
# Look at "VFree" column

# Extend the logical volume and filesystem in one step
sudo lvextend -r -l +100%FREE /dev/fedora/root
```

The `-r` flag resizes the filesystem automatically (works for ext4 and xfs).

**If you added a new disk:**
```bash
# Create a physical volume on the new disk
sudo pvcreate /dev/sdb

# Add it to the existing volume group
sudo vgextend fedora /dev/sdb

# Now extend the LV
sudo lvextend -r -l +100%FREE /dev/fedora/root
```

**If you need to shrink a partition first (ext4 only — xfs cannot be shrunk):**
```bash
# Shrink the LV and filesystem together
sudo lvreduce -r -L 20G /dev/fedora/home
# Then grow root
sudo lvextend -r -l +100%FREE /dev/fedora/root
```

**Important:** XFS (Fedora 33+ default) can only grow, not shrink. If `/home` is xfs and you want to reclaim space from it, you'd need to back up, delete, recreate smaller, and restore.

**Verify:**
```bash
df -h /
```""",
    "flm_disk_storage"
))

# ---------------------------------------------------------------------------
# 5. LVM setup and common operations (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "Can you give me a guide to LVM? I want to set it up on a new server.",
    r"""LVM (Logical Volume Manager) adds a flexible layer between your physical disks and filesystems. Here's a complete walkthrough.

**Install LVM tools:**
```bash
sudo apt install lvm2
```

**Concepts:**
- **PV** (Physical Volume) — a disk or partition registered with LVM
- **VG** (Volume Group) — a pool of storage made from one or more PVs
- **LV** (Logical Volume) — a virtual partition carved out of a VG

**Step 1: Create physical volumes**
```bash
sudo pvcreate /dev/sdb /dev/sdc
sudo pvs   # verify
```

**Step 2: Create a volume group**
```bash
sudo vgcreate data_vg /dev/sdb /dev/sdc
sudo vgs   # verify
```

**Step 3: Create logical volumes**
```bash
# Fixed size
sudo lvcreate -L 100G -n home_lv data_vg

# Use percentage of free space
sudo lvcreate -l 80%FREE -n srv_lv data_vg
```

**Step 4: Format and mount**
```bash
sudo mkfs.ext4 /dev/data_vg/home_lv
sudo mkfs.ext4 /dev/data_vg/srv_lv
sudo mkdir -p /mnt/home /mnt/srv
sudo mount /dev/data_vg/home_lv /mnt/home
```

**Common operations:**

```bash
# Extend a logical volume (and resize fs in one step)
sudo lvextend -r -L +50G /dev/data_vg/home_lv

# Add a new disk to the pool
sudo pvcreate /dev/sdd
sudo vgextend data_vg /dev/sdd

# Shrink (ext4 only, not xfs)
sudo lvreduce -r -L 80G /dev/data_vg/home_lv

# Remove a logical volume
sudo umount /mnt/srv
sudo lvremove /dev/data_vg/srv_lv

# Move data off a physical disk (before removing it)
sudo pvmove /dev/sdc
sudo vgreduce data_vg /dev/sdc

# Snapshots (useful for backups)
sudo lvcreate -s -L 10G -n home_snap /dev/data_vg/home_lv
```

**Add to /etc/fstab:**
```
/dev/data_vg/home_lv  /home  ext4  defaults  0  2
/dev/data_vg/srv_lv   /srv   ext4  defaults  0  2
```

**Display commands:**
```bash
sudo pvdisplay    # or pvs for summary
sudo vgdisplay    # or vgs
sudo lvdisplay    # or lvs
```

The beauty of LVM is you can resize on the fly, add disks without reformatting, and take snapshots. The cost is a thin abstraction layer and slightly more complexity.""",
    "flm_disk_storage"
))

# ---------------------------------------------------------------------------
# 6. Add swap space / swap file (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB12,
    "How do I add more swap space?",
    r"""The easiest way is to create a swap file. No repartitioning needed.

**Create a swap file:**
```bash
# Create a 4GB file (adjust size as needed)
sudo dd if=/dev/zero of=/swapfile bs=1M count=4096 status=progress

# Or use fallocate (faster, but doesn't work on all filesystems like btrfs)
sudo fallocate -l 4G /swapfile

# Set proper permissions
sudo chmod 600 /swapfile

# Format it as swap
sudo mkswap /swapfile

# Enable it
sudo swapon /swapfile
```

**Verify:**
```bash
swapon --show
free -h
```

**Make it permanent — add to /etc/fstab:**
```
/swapfile  none  swap  sw  0  0
```

**Adjust swappiness** (how eagerly the kernel swaps):
```bash
# Check current value (default 60)
cat /proc/sys/vm/swappiness

# Lower it for a desktop (less aggressive swapping)
sudo sysctl vm.swappiness=10

# Make it permanent
echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-swappiness.conf
```

**If you want a swap partition instead:**
```bash
# Assuming /dev/sdb2 is your swap partition
sudo mkswap /dev/sdb2
sudo swapon /dev/sdb2
# Add UUID to fstab
echo 'UUID=xxx  none  swap  sw  0  0' | sudo tee -a /etc/fstab
```

**Recommendation:** For servers, swap = 1x RAM up to 8GB is usually enough. For desktops that hibernate, swap >= RAM size.""",
    "flm_disk_storage"
))

# ---------------------------------------------------------------------------
# 7. Set up zram compressed swap (Arch)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_ARCH,
    "How do I set up zram for compressed swap?",
    r"""zram creates compressed block devices in RAM, making excellent swap that's much faster than disk swap.

**Option 1: Using zram-generator (recommended, systemd-native)**

```bash
sudo pacman -S zram-generator
```

Create `/etc/systemd/zram-generator.conf`:
```ini
[zram0]
zram-size = ram / 2
compression-algorithm = zstd
swap-priority = 100
fs-type = swap
```

This creates a zram swap device at half your RAM size using zstd compression (excellent ratio + speed).

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl start systemd-zram-setup@zram0.service
```

It activates automatically on boot.

**Option 2: Manual setup**

```bash
# Load the module
sudo modprobe zram

# Set size (e.g., 4GB)
echo 4G | sudo tee /sys/block/zram0/disksize

# Set compression algorithm
echo zstd | sudo tee /sys/block/zram0/comp_algorithm

# Create and enable swap
sudo mkswap /dev/zram0
sudo swapon -p 100 /dev/zram0
```

**Verify:**
```bash
swapon --show
zramctl
```

`zramctl` shows the compression ratio. Typical ratios are 2:1 to 4:1, so 4GB of zram effectively gives you 8-16GB of swap capacity.

**Tune swappiness for zram:**
Since zram swap is fast (it's in RAM), you can be more aggressive:
```bash
echo 'vm.swappiness=180' | sudo tee /etc/sysctl.d/99-zram.conf
sudo sysctl -p /etc/sysctl.d/99-zram.conf
```

Values above 100 are supported on Linux 5.8+ and tell the kernel to prefer zram over file cache reclaim.

**Disable disk swap if you have enough RAM + zram:**
```bash
sudo swapoff /dev/sdXY   # or /swapfile
# Remove the entry from /etc/fstab
```""",
    "flm_disk_storage"
))

# ---------------------------------------------------------------------------
# 8. Understanding /etc/fstab entries (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "Can you explain the /etc/fstab format? I always get confused by the columns.",
    r"""Each line in `/etc/fstab` has 6 fields:

```
<device>  <mount_point>  <fs_type>  <options>  <dump>  <pass>
```

**Example file:**
```
# <device>                                 <mount>   <type>  <options>           <dump> <pass>
UUID=a1b2c3d4-e5f6-7890-abcd-ef1234567890  /         ext4    errors=remount-ro   0      1
UUID=deadbeef-1234-5678-9abc-def012345678  /home     ext4    defaults            0      2
UUID=11223344-5566-7788-99aa-bbccddeeff00  none      swap    sw                  0      0
/dev/sr0                                   /media/cd iso9660 ro,user,noauto      0      0
//server/share                             /mnt/smb  cifs    credentials=/root/.smbcred,uid=1000  0  0
server:/export                             /mnt/nfs  nfs     defaults,nofail     0      0
tmpfs                                      /tmp      tmpfs   defaults,size=2G    0      0
```

**Field by field:**

1. **Device** — what to mount. Use UUID (preferred — survives disk reordering), LABEL, or device path.
   ```bash
   # Find UUIDs
   blkid
   ```

2. **Mount point** — where to mount. Must be an existing directory (except `none` for swap).

3. **Filesystem type** — `ext4`, `xfs`, `btrfs`, `swap`, `nfs`, `cifs`, `tmpfs`, `vfat`, etc.

4. **Options** — comma-separated. Common ones:
   - `defaults` = rw, suid, dev, exec, auto, nouser, async
   - `noatime` — don't update access times (improves performance)
   - `nofail` — don't halt boot if mount fails (important for removable/network drives)
   - `ro` — read-only
   - `noexec` — prevent executing binaries (security for /tmp)
   - `nosuid` — ignore setuid bits (security)
   - `user` — allow non-root users to mount

5. **Dump** — legacy backup flag. Always use `0` (dump is rarely used anymore).

6. **Pass** — `fsck` order at boot:
   - `0` = don't check
   - `1` = check first (root filesystem)
   - `2` = check after root (other filesystems)

**Tips:**
- Always use `nofail` for network mounts and removable drives, or a missing device will block boot.
- Test before rebooting: `sudo mount -a` applies fstab without rebooting. If it errors, fix it before you reboot.
- Back up your fstab before editing: `sudo cp /etc/fstab /etc/fstab.bak`""",
    "flm_disk_storage"
))

# ---------------------------------------------------------------------------
# 9. Check disk usage — du and df tricks (short Debian prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    SHORT_DEB13,
    "What are some useful du and df commands?",
    r"""**df — disk free (filesystem level)**

```bash
# Human-readable overview
df -h

# Show filesystem type too
df -hT

# Just one filesystem
df -h /home

# Inode usage (can run out even with space left)
df -i
```

**du — disk usage (directory level)**

```bash
# Size of a directory
du -sh /var/log

# Top 10 biggest directories under /
sudo du -h --max-depth=1 / 2>/dev/null | sort -rh | head -10

# Top 20 biggest files on the whole system
sudo find / -type f -printf '%s %p\n' 2>/dev/null | sort -rn | head -20 | awk '{printf "%.1fM %s\n", $1/1048576, $2}'

# What's eating space in /var?
sudo du -h --max-depth=1 /var | sort -rh

# Disk usage excluding other filesystems (don't cross mount points)
du -shx /
```

**Combining them — find what's full:**
```bash
# Which filesystem is full?
df -h | awk '$5+0 > 80 {print}'

# Then drill into it
sudo du -h --max-depth=1 /var | sort -rh | head
sudo du -h --max-depth=1 /var/log | sort -rh | head
```

**ncdu — interactive disk usage browser (highly recommended):**
```bash
sudo apt install ncdu
sudo ncdu /
```

Navigate with arrow keys, press `d` to delete files interactively. Much faster than running `du` repeatedly.

**Common space hogs to check:**
```bash
sudo du -sh /var/log/*       # logs
sudo du -sh /var/cache/apt   # apt cache
sudo du -sh /tmp/*           # temp files
sudo journalctl --disk-usage # systemd journal
```

**Quick cleanup:**
```bash
sudo apt clean                                # clear apt cache
sudo journalctl --vacuum-size=100M            # trim journal
sudo find /tmp -type f -atime +7 -delete      # old temp files
```""",
    "flm_disk_storage"
))

# ---------------------------------------------------------------------------
# 10. Recover deleted files (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "I accidentally deleted some important files. Can I recover them?",
    r"""Maybe — it depends on the filesystem and how much has been written since deletion.

**Rule #1: Stop writing to that filesystem immediately.** Every write risks overwriting your deleted data. If it's on the root filesystem, remount read-only if possible:
```bash
sudo mount -o remount,ro /dev/sda1
```

**For ext4 — extundelete:**
```bash
sudo apt install extundelete

# Recover all recently deleted files
sudo extundelete /dev/sda1 --restore-all

# Recover a specific file
sudo extundelete /dev/sda1 --restore-file home/user/important.txt
```

Recovered files go into `RECOVERED_FILES/` in the current directory. Run this from a different filesystem than the one you're recovering from.

**For ext4 — debugfs (built-in):**
```bash
sudo debugfs /dev/sda1
# Inside debugfs:
lsdel                    # list recently deleted inodes
undel <inode> output.txt # recover by inode number
quit
```

**For any filesystem — photorec (file carving):**
```bash
sudo apt install testdisk
sudo photorec /dev/sda1
```

photorec recovers files by scanning for known file signatures (headers). It works regardless of filesystem type but doesn't preserve filenames or directory structure.

**For any filesystem — testdisk (partition/file recovery):**
```bash
sudo testdisk /dev/sda
```

testdisk can recover lost partitions and undelete files on FAT, NTFS, and ext filesystems.

**If the file is still open by a process:**
```bash
# Check if any process still has the deleted file open
sudo lsof +L1 | grep deleted

# If found, recover from /proc
sudo cp /proc/<pid>/fd/<fd_number> /tmp/recovered_file
```

This is surprisingly common with log files — if a daemon still has the file open, the data is still there.

**Prevention:**
- Use `trash-cli` instead of `rm`: `sudo apt install trash-cli` then use `trash-put` instead of `rm`
- Keep backups. Seriously. Recovery is never guaranteed.""",
    "flm_disk_storage"
))

# ---------------------------------------------------------------------------
# 11. Set up ufw firewall (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I set up a firewall with ufw?",
    r"""UFW (Uncomplicated Firewall) is a friendly frontend for nftables/iptables.

**Install and enable:**
```bash
sudo apt install ufw

# IMPORTANT: Allow SSH first if you're remote!
sudo ufw allow ssh
# or explicitly:
sudo ufw allow 22/tcp

# Enable the firewall
sudo ufw enable
```

**Default policies:**
```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
```

This is the most common setup: block all inbound unless explicitly allowed, let all outbound through.

**Allow services:**
```bash
# By service name
sudo ufw allow http
sudo ufw allow https

# By port
sudo ufw allow 8080/tcp

# By port range
sudo ufw allow 6000:6007/tcp

# From specific IP
sudo ufw allow from 192.168.1.100

# From specific subnet to specific port
sudo ufw allow from 10.0.0.0/8 to any port 5432
```

**Deny/reject:**
```bash
# Deny (silently drop)
sudo ufw deny from 203.0.113.0/24

# Reject (send back "connection refused")
sudo ufw reject 25/tcp
```

**Delete rules:**
```bash
# List rules with numbers
sudo ufw status numbered

# Delete by number
sudo ufw delete 3

# Delete by specification
sudo ufw delete allow 8080/tcp
```

**Rate limiting (brute force protection):**
```bash
sudo ufw limit ssh
```

This allows 6 connections in 30 seconds per IP, then blocks. Great for SSH.

**Check status:**
```bash
sudo ufw status verbose
```

**Application profiles:**
```bash
# List available app profiles
sudo ufw app list

# Get details about a profile
sudo ufw app info 'Nginx Full'

# Allow a profile
sudo ufw allow 'Nginx Full'
```

**Logging:**
```bash
sudo ufw logging on
# Logs go to /var/log/ufw.log
```

**Reset everything:**
```bash
sudo ufw reset
```""",
    "flm_firewall"
))

# ---------------------------------------------------------------------------
# 12. Configure firewalld (Fedora)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_FED41,
    "How do I configure the firewall on Fedora?",
    r"""Fedora uses firewalld by default. It's zone-based — each network interface is assigned to a zone, and each zone has its own rules.

**Check status:**
```bash
sudo firewall-cmd --state
sudo systemctl status firewalld
```

**List active zones and rules:**
```bash
sudo firewall-cmd --get-active-zones
sudo firewall-cmd --list-all
```

**Common operations:**
```bash
# Allow a service permanently
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload

# Allow a port
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload

# Remove a rule
sudo firewall-cmd --permanent --remove-service=http
sudo firewall-cmd --reload
```

Always use `--permanent` and then `--reload`. Without `--permanent`, changes are lost on reboot.

**List available services:**
```bash
sudo firewall-cmd --get-services
```

**Zones explained:**
```bash
# List all zones
sudo firewall-cmd --get-zones

# Common zones:
# public  — default, minimal trust
# home    — more trust
# trusted — allow everything
# drop    — drop all incoming

# Change interface zone
sudo firewall-cmd --permanent --zone=home --change-interface=eth0
sudo firewall-cmd --reload
```

**Rich rules (advanced):**
```bash
# Allow SSH from specific subnet only
sudo firewall-cmd --permanent --add-rich-rule='rule family="ipv4" source address="10.0.0.0/8" service name="ssh" accept'

# Rate-limit connections
sudo firewall-cmd --permanent --add-rich-rule='rule service name="http" limit value="25/m" accept'

sudo firewall-cmd --reload
```

**Port forwarding:**
```bash
sudo firewall-cmd --permanent --add-forward-port=port=80:proto=tcp:toport=8080
sudo firewall-cmd --reload
```

**If you prefer UFW instead:**
```bash
sudo dnf install ufw
sudo systemctl stop firewalld
sudo systemctl disable firewalld
sudo systemctl enable --now ufw
sudo ufw enable
```""",
    "flm_firewall"
))

# ---------------------------------------------------------------------------
# 13. nftables setup (Arch)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_ARCH,
    "How do I set up nftables on Arch?",
    r"""nftables is the successor to iptables and is already in the Arch kernel. It's the recommended firewall framework.

**Install the userspace tools:**
```bash
sudo pacman -S nftables
```

**Create a basic ruleset in `/etc/nftables.conf`:**
```
#!/usr/sbin/nft -f

flush ruleset

table inet filter {
    chain input {
        type filter hook input priority filter; policy drop;

        # Accept established/related traffic
        ct state established,related accept

        # Accept loopback
        iif "lo" accept

        # Accept ICMP (ping)
        ip protocol icmp accept
        ip6 nexthdr icmpv6 accept

        # Accept SSH
        tcp dport 22 accept

        # Accept HTTP/HTTPS
        tcp dport { 80, 443 } accept

        # Log and drop everything else
        log prefix "nft-drop: " counter drop
    }

    chain forward {
        type filter hook forward priority filter; policy drop;
    }

    chain output {
        type filter hook output priority filter; policy accept;
    }
}
```

**Enable and start:**
```bash
sudo systemctl enable --now nftables
```

**Useful commands:**
```bash
# Show current ruleset
sudo nft list ruleset

# Add a rule interactively
sudo nft add rule inet filter input tcp dport 8080 accept

# Delete a rule (need the handle number)
sudo nft -a list ruleset   # shows handles
sudo nft delete rule inet filter input handle 15

# Flush all rules
sudo nft flush ruleset
```

**Rate limiting:**
```
# In the input chain:
tcp dport 22 ct state new limit rate 3/minute accept
```

**Block an IP:**
```bash
sudo nft add rule inet filter input ip saddr 203.0.113.50 drop
```

**Sets (block lists):**
```
table inet filter {
    set blocklist {
        type ipv4_addr
        elements = { 203.0.113.50, 198.51.100.0/24 }
    }

    chain input {
        # ...
        ip saddr @blocklist drop
    }
}
```

**Why nftables over iptables?**
- Single framework for IPv4 and IPv6 (the `inet` family)
- Sets and maps built in
- Better syntax
- Atomic rule replacement
- iptables is a compatibility layer on top of nftables now anyway""",
    "flm_firewall"
))

# ---------------------------------------------------------------------------
# 14. iptables vs nftables explanation (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "What's the difference between iptables and nftables? Should I switch?",
    r"""nftables is the successor to iptables. It's been in the Linux kernel since 3.13 (2014) and is the recommended firewall framework going forward.

**Key differences:**

| Feature | iptables | nftables |
|---------|----------|----------|
| IPv4 + IPv6 | Separate tools (iptables, ip6tables) | Single tool (`nft` with `inet` family) |
| Syntax | Flag-based (`-A INPUT -p tcp --dport 22 -j ACCEPT`) | Structured (`tcp dport 22 accept`) |
| Atomic updates | No (rules applied one by one) | Yes (entire ruleset replaced at once) |
| Sets | Requires ipset separately | Built-in sets and maps |
| Performance | Linear rule matching | Optimized with sets and maps |
| Kernel interface | Netfilter with separate tools per protocol | Unified nf_tables interface |

**iptables example:**
```bash
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -i lo -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -j DROP
```

**Equivalent nftables:**
```
table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;
        ct state established,related accept
        iif lo accept
        tcp dport 22 accept
    }
}
```

**Should you switch?**
- Most modern distros already use nftables as the backend. Even if you type `iptables`, it's often `iptables-nft` (a compatibility shim over nftables).
- Debian 10+, Fedora 32+, RHEL 9+, and Arch all default to nftables.
- If you're writing new firewall rules, use nftables directly.
- If you have existing iptables scripts that work, you can translate them: `iptables-save | iptables-restore-translate`
- Frontends like ufw and firewalld already use nftables under the hood.

**Bottom line:** Yes, learn nftables for new setups. Existing iptables rules keep working through the compatibility layer, so there's no urgent need to migrate running systems.""",
    "flm_firewall"
))

# ---------------------------------------------------------------------------
# 15. tmux usage guide (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "Give me a tmux guide. I always forget the keybindings.",
    r"""**Install:**
```bash
sudo apt install tmux
```

**Core concept:** tmux has sessions > windows > panes. The prefix key is `Ctrl-b` by default.

**Sessions:**
```bash
tmux                     # new session
tmux new -s work         # new named session
tmux ls                  # list sessions
tmux attach -t work      # reattach to session
tmux kill-session -t work
```

Inside tmux:
- `Ctrl-b d` — detach (session keeps running)
- `Ctrl-b $` — rename session
- `Ctrl-b s` — switch between sessions

**Windows (like tabs):**
- `Ctrl-b c` — new window
- `Ctrl-b ,` — rename window
- `Ctrl-b n` — next window
- `Ctrl-b p` — previous window
- `Ctrl-b 0-9` — jump to window by number
- `Ctrl-b &` — close window

**Panes (splits):**
- `Ctrl-b %` — vertical split
- `Ctrl-b "` — horizontal split
- `Ctrl-b arrow` — move between panes
- `Ctrl-b z` — zoom pane (toggle fullscreen)
- `Ctrl-b x` — close pane
- `Ctrl-b {` / `}` — swap pane position
- `Ctrl-b Ctrl-arrow` — resize pane
- `Ctrl-b space` — cycle through pane layouts

**Copy mode (scrollback):**
- `Ctrl-b [` — enter copy mode (scroll with arrows/PgUp/PgDn)
- `q` — exit copy mode
- In copy mode: `Space` to start selection, `Enter` to copy
- `Ctrl-b ]` — paste

**Useful config — save to `~/.tmux.conf`:**
```bash
# Use Ctrl-a instead of Ctrl-b (more comfortable)
set -g prefix C-a
unbind C-b
bind C-a send-prefix

# Split with | and -
bind | split-window -h
bind - split-window -v

# Mouse support
set -g mouse on

# Start window numbering at 1
set -g base-index 1

# Larger scrollback
set -g history-limit 50000

# 256 colors
set -g default-terminal "screen-256color"
```

Reload config without restarting:
```bash
tmux source-file ~/.tmux.conf
```

**The killer feature:** tmux sessions survive SSH disconnections. Start a long-running job in tmux, detach, disconnect — reconnect later and `tmux attach`. The job is still running.""",
    "flm_terminal"
))

# ---------------------------------------------------------------------------
# 16. screen basics (Ubuntu 22.04)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_UBU22,
    "How do I use screen? I need it for keeping a process running after I log out.",
    r"""GNU Screen is the classic terminal multiplexer. It's usually pre-installed on Ubuntu:

```bash
# If not installed:
sudo apt install screen
```

**Basic usage:**
```bash
screen                    # start new session
screen -S myproject       # start named session
screen -ls                # list running sessions
screen -r myproject       # reattach to session
screen -d -r myproject    # detach elsewhere and reattach here
```

**Key bindings (all prefixed with `Ctrl-a`):**
- `Ctrl-a d` — detach (session keeps running)
- `Ctrl-a c` — create new window
- `Ctrl-a n` — next window
- `Ctrl-a p` — previous window
- `Ctrl-a 0-9` — switch to window by number
- `Ctrl-a "` — list windows and pick one
- `Ctrl-a A` — rename current window
- `Ctrl-a k` — kill current window
- `Ctrl-a S` — split horizontally
- `Ctrl-a |` — split vertically
- `Ctrl-a Tab` — move between splits
- `Ctrl-a X` — close current split

**Scrollback:**
- `Ctrl-a [` — enter copy/scroll mode
- Use `PgUp`/`PgDn` or arrow keys to scroll
- `Esc` to exit

**Useful config — save to `~/.screenrc`:**
```
# Show a status bar
hardstatus alwayslastline
hardstatus string '%{= kG}[ %{G}%H %{g}][%= %{=kw}%?%-Lw%?%{r}(%{W}%n*%f%t%?(%u)%?%{r})%{w}%?%+Lw%?%?%= %{g}][%{B}%Y-%m-%d %{W}%c %{g}]'

# Bigger scrollback buffer
defscrollback 10000

# Disable startup message
startup_message off
```

**Common workflow:**
```bash
# Start a named session
screen -S build

# Run your long-running command
make -j$(nproc)

# Detach: Ctrl-a d
# Log out, go home, SSH back in

# Reattach
screen -r build
```

**tmux vs screen:** tmux is the more modern tool with better pane support, scripting, and defaults. But screen is ubiquitous — it's on almost every server already. Use whichever you prefer; the core concept is the same.""",
    "flm_terminal"
))

# ---------------------------------------------------------------------------
# 17. Environment variables explained (short Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SHORT_DEB13,
    "How do environment variables work in Linux?",
    r"""Environment variables are key-value pairs inherited by child processes from their parent. They configure how programs behave.

**View them:**
```bash
# All environment variables
env
printenv

# Specific variable
echo $PATH
printenv HOME
```

**Set them:**
```bash
# For current shell only
export MYVAR="hello"

# For one command only
LANG=C sort myfile.txt

# Unset
unset MYVAR
```

**Common variables:**
| Variable | Purpose |
|----------|---------|
| `PATH` | Directories to search for commands |
| `HOME` | User's home directory |
| `USER` | Current username |
| `SHELL` | User's default shell |
| `EDITOR` | Default text editor |
| `LANG` | Locale/language |
| `XDG_CONFIG_HOME` | User config directory (~/.config) |
| `LD_LIBRARY_PATH` | Extra library search paths |
| `PS1` | Shell prompt format |

**Where to set them permanently:**

For your user:
- `~/.bashrc` — interactive non-login shells
- `~/.bash_profile` or `~/.profile` — login shells
- `~/.config/environment.d/*.conf` — systemd user environment (Debian)

For all users:
- `/etc/environment` — simple `KEY=VALUE` pairs, parsed by PAM
- `/etc/profile.d/*.sh` — scripts sourced on login
- `/etc/bash.bashrc` — all interactive bash shells

**Example — add a directory to PATH:**
```bash
# In ~/.bashrc:
export PATH="$HOME/.local/bin:$PATH"
```

**How inheritance works:**
```bash
export VISIBLE="yes"    # child processes see this
LOCAL_ONLY="nope"       # only this shell sees this

bash -c 'echo $VISIBLE'    # prints "yes"
bash -c 'echo $LOCAL_ONLY' # prints nothing
```

`export` marks a variable for inheritance. Without export, the variable is local to the current shell.

**Debugging:**
```bash
# See the full environment a process runs with
cat /proc/<pid>/environ | tr '\0' '\n'

# See what a command would see
env -i HOME=$HOME PATH=/usr/bin bash -c 'env'
```""",
    "flm_terminal"
))

# ---------------------------------------------------------------------------
# 18. setuid, setgid, sticky bit (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "Explain setuid, setgid, and the sticky bit with examples.",
    r"""These are special permission bits beyond the standard rwx.

**setuid (SUID) — set user ID on execution**

When set on an executable, it runs as the **file owner** instead of the user who launched it.

```bash
# The classic example:
ls -l /usr/bin/passwd
-rwsr-xr-x 1 root root 68208 ... /usr/bin/passwd
```

The `s` in `rws` means setuid is set. `passwd` needs to write to `/etc/shadow` (owned by root), so it runs as root even when a regular user calls it.

```bash
# Set setuid
sudo chmod u+s /usr/local/bin/myprog
# Or numerically (4xxx)
sudo chmod 4755 /usr/local/bin/myprog
```

**Security warning:** setuid on a shell script is ignored by Linux for safety. Only compiled binaries honor setuid.

**setgid (SGID) — set group ID**

On executables: runs with the **file's group** instead of the user's group.

On directories: new files created inside inherit the **directory's group** instead of the creator's primary group. Very useful for shared directories.

```bash
# setgid on a shared project directory
sudo mkdir /srv/project
sudo chgrp developers /srv/project
sudo chmod g+s /srv/project
# or chmod 2775 /srv/project

# Now any file created in /srv/project belongs to group "developers"
touch /srv/project/test.txt
ls -l /srv/project/test.txt
# -rw-r--r-- 1 user developers ... test.txt
```

The `s` appears in the group execute position: `rwxr-sr-x`.

**Sticky bit**

On directories: only the file owner (or root) can delete files in the directory, even if others have write permission.

```bash
ls -ld /tmp
drwxrwxrwt 15 root root 4096 ... /tmp
```

The `t` at the end means the sticky bit is set. Everyone can write to /tmp, but you can only delete your own files.

```bash
# Set sticky bit
sudo chmod +t /srv/shared
# Or numerically (1xxx)
sudo chmod 1777 /srv/shared
```

**Summary of numeric modes:**
```
4xxx — setuid
2xxx — setgid
1xxx — sticky bit
```

They combine: `chmod 6755` = setuid + setgid + rwxr-xr-x.

**Find all setuid/setgid files (security audit):**
```bash
sudo find / -perm /4000 -type f 2>/dev/null  # setuid files
sudo find / -perm /2000 -type f 2>/dev/null  # setgid files
```""",
    "flm_permissions"
))

# ---------------------------------------------------------------------------
# 19. ACLs with setfacl/getfacl (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I use ACLs to give specific users access to files?",
    r"""Standard Unix permissions only support owner/group/others. ACLs (Access Control Lists) let you grant fine-grained access to specific users or groups.

**Install (usually already present):**
```bash
sudo apt install acl
```

**Make sure the filesystem supports ACLs** (ext4 and xfs do by default):
```bash
mount | grep acl
# Modern ext4 has ACL support compiled in — no mount option needed
```

**Grant a user access:**
```bash
# Give user "alice" read+write on a file
setfacl -m u:alice:rw report.txt

# Give user "bob" read-only on a directory and its contents
setfacl -R -m u:bob:rx /srv/project
```

**Grant a group access:**
```bash
setfacl -m g:devteam:rwx /srv/project
```

**View ACLs:**
```bash
getfacl report.txt
```

Output:
```
# file: report.txt
# owner: dave
# group: staff
user::rw-
user:alice:rw-
group::r--
mask::rw-
other::r--
```

**Default ACLs (for new files in a directory):**
```bash
# New files in /srv/project automatically give alice rw access
setfacl -d -m u:alice:rw /srv/project
```

**Remove ACLs:**
```bash
# Remove a specific ACL entry
setfacl -x u:alice report.txt

# Remove all ACLs
setfacl -b report.txt
```

**The mask:**
The mask limits the maximum effective permissions for named users and groups. It's recalculated when you use `chmod` on a file with ACLs:
```bash
# This may inadvertently restrict ACL permissions!
chmod 600 report.txt
# Fix: set the mask explicitly
setfacl -m m::rw report.txt
```

**Recursive + default (common pattern for shared dirs):**
```bash
# Set permissions on existing files
setfacl -R -m u:alice:rwx /srv/data

# Set defaults for new files
setfacl -R -d -m u:alice:rwx /srv/data
```

**Backup and restore ACLs:**
```bash
getfacl -R /srv/project > acl_backup.txt
setfacl --restore=acl_backup.txt
```

**Note:** `ls -l` shows a `+` after permissions when ACLs are present:
```
-rw-rw-r--+ 1 dave staff 1234 ... report.txt
```""",
    "flm_permissions"
))

# ---------------------------------------------------------------------------
# 20. umask explained (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "What does umask do? I keep seeing it but never understood it.",
    r"""`umask` sets the **default permissions mask** for newly created files and directories. It specifies which permission bits to **remove** from new files.

**How it works:**
- Files start at mode `666` (rw-rw-rw-)
- Directories start at mode `777` (rwxrwxrwx)
- The umask is **subtracted** from these

**Common umask values:**

| umask | Files become | Dirs become | Meaning |
|-------|-------------|-------------|---------|
| 022 | 644 (rw-r--r--) | 755 (rwxr-xr-x) | Default on most systems. Others can read but not write. |
| 002 | 664 (rw-rw-r--) | 775 (rwxrwxr-x) | Group can write too. Common for shared projects. |
| 077 | 600 (rw-------) | 700 (rwx------) | Private. Only owner has access. |
| 027 | 640 (rw-r-----) | 750 (rwxr-x---) | Owner full, group read, others nothing. |

**Check current umask:**
```bash
umask        # shows numeric (e.g., 0022)
umask -S     # shows symbolic (e.g., u=rwx,g=rx,o=rx)
```

**Set umask:**
```bash
umask 027    # for this shell session
```

**Make it permanent — add to `~/.bashrc` or `~/.profile`:**
```bash
umask 027
```

**System-wide default:**
The default umask is typically set in:
- `/etc/login.defs` — the `UMASK` setting
- `/etc/pam.d/common-session` — via `pam_umask.so`
- `/etc/profile` — shell-level default

**Example to understand the math:**
```
Default for files:  666  (rw-rw-rw-)
umask:              022  (----w--w-)
Result:             644  (rw-r--r--)

Default for dirs:   777  (rwxrwxrwx)
umask:              022  (----w--w-)
Result:             755  (rwxr-xr-x)
```

Technically, it's a bitwise AND with the complement: `result = default & ~umask`. But thinking of it as subtraction works for the common cases.

**Gotcha:** umask only affects the *creation* permissions. `chmod` can change permissions after the fact regardless of umask.""",
    "flm_permissions"
))

# ---------------------------------------------------------------------------
# 21. Investigate OOM / memory issues (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "My server keeps running out of memory and the OOM killer is killing processes. How do I investigate?",
    r"""The OOM (Out of Memory) killer is the kernel's last resort when memory is exhausted. Let's investigate systematically.

**1. Check OOM kill history:**
```bash
# Recent OOM events
sudo dmesg | grep -i "oom\|killed process"

# From the journal
sudo journalctl -k | grep -i "oom\|killed process"

# Find the most recent OOM kill
sudo journalctl -k --grep="Killed process" --no-pager
```

This tells you which process was killed, how much memory it was using, and the system's memory state.

**2. Current memory usage:**
```bash
free -h
```

**3. Find the biggest memory consumers:**
```bash
# Top 10 by RSS (actual physical memory)
ps aux --sort=-%mem | head -11

# Detailed memory map of a process
pmap -x <pid>

# Per-process summary
sudo smem -rs rss | tail -10
# (install with: sudo apt install smem)
```

**4. Check for memory leaks:**
```bash
# Watch a process over time
watch -n 5 "ps -o pid,rss,vsz,comm -p <pid>"

# If RSS keeps growing without limit, it's likely a leak
```

**5. Protect critical processes from OOM killer:**
```bash
# Lower oom_score = less likely to be killed
# Range: -1000 (never kill) to 1000 (kill first)
echo -1000 | sudo tee /proc/<pid>/oom_score_adj

# For a systemd service, add to the unit file:
# [Service]
# OOMScoreAdjust=-900
```

**6. Set memory limits with systemd:**
```bash
sudo systemctl edit myservice.service
```
Add:
```ini
[Service]
MemoryMax=2G
MemoryHigh=1.5G
```
- `MemoryHigh` — throttles the service when exceeded
- `MemoryMax` — hard kill if exceeded (before OOM killer gets involved)

**7. Add swap or increase it:**
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**8. Monitor proactively:**
```bash
# Set up earlyoom — kills the biggest process before the kernel OOM
sudo apt install earlyoom
sudo systemctl enable --now earlyoom
```

earlyoom is a userspace OOM killer that acts before the system becomes completely unresponsive. It's configured in `/etc/default/earlyoom`:
```
EARLYOOM_ARGS="-m 5 -s 5 --avoid '(^|/)(sshd|systemd)$'"
```
This triggers when free memory AND free swap are below 5%.""",
    "flm_memory"
))

# ---------------------------------------------------------------------------
# 22. Understanding free -h output (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "Can you explain the output of `free -h`? What do the columns mean?",
    r"""Here's a typical `free -h` output:

```
               total        used        free      shared  buff/cache   available
Mem:           15Gi        6.2Gi       1.1Gi       312Mi        8.2Gi       8.7Gi
Swap:          4.0Gi       0.0Gi       4.0Gi
```

**The columns:**

- **total** — total physical RAM installed
- **used** — memory actively used by programs
- **free** — completely unused memory (not even cached)
- **shared** — memory used by tmpfs and shared memory segments
- **buff/cache** — memory used for disk cache and buffers. This is "free" memory being put to good use — the kernel will release it if programs need it.
- **available** — estimated memory available for starting new applications, **without** swapping. This is the number that actually matters.

**The key insight:** Linux uses "unused" RAM as disk cache. This is a feature, not a problem.

If you see:
```
total = 16G, used = 14G, free = 200M
```

Don't panic. Look at `available` — it might say 10G because most of that 14G "used" is reclaimable cache.

**How to read it:**
- "Am I running low on memory?" → check **available**, not free
- "How much are my programs actually using?" → check **used** minus **buff/cache**
- "Should I add swap?" → if available < 10% of total under normal load, consider it
- "Is my swap being used?" → if Swap used > 0, the system has been under memory pressure at some point

**More detail:**
```bash
# Detailed breakdown
cat /proc/meminfo

# Per-process memory
ps aux --sort=-%mem | head -10

# Watch memory in real time
watch -n 1 free -h
```

**Common confusion:** People see "free" is low and think the system is out of memory. It's not — the kernel intentionally caches disk data in free RAM. The `available` column accounts for this and tells you the real answer.""",
    "flm_memory"
))

# ---------------------------------------------------------------------------
# 23. gzip vs bzip2 vs xz vs zstd (short Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SHORT_DEB13,
    "What's the difference between gzip, bzip2, xz, and zstd?",
    r"""They're all compression algorithms with different speed/ratio tradeoffs:

| Tool | Extension | Speed | Ratio | Best for |
|------|-----------|-------|-------|----------|
| gzip | .gz | Fast | Good | General purpose, web |
| bzip2 | .bz2 | Slow | Better | Legacy use |
| xz | .xz | Very slow | Best | Archives, distribution |
| zstd | .zst | Very fast | Very good | Modern general purpose |

**Compression comparison (rough numbers for a 100MB text file):**
```
gzip:   35MB in 3s  (compress) / 1s  (decompress)
bzip2:  28MB in 15s (compress) / 5s  (decompress)
xz:     25MB in 60s (compress) / 2s  (decompress)
zstd:   33MB in 1s  (compress) / 0.5s (decompress)
```

**Basic usage — they all work similarly:**
```bash
# Compress (replaces original)
gzip file.txt        # -> file.txt.gz
bzip2 file.txt       # -> file.txt.bz2
xz file.txt          # -> file.txt.xz
zstd file.txt        # -> file.txt.zst

# Decompress
gunzip file.txt.gz
bunzip2 file.txt.bz2
unxz file.txt.xz
unzstd file.txt.zst

# Keep original file
gzip -k file.txt
zstd -k file.txt

# View compressed file without decompressing
zcat file.txt.gz
bzcat file.txt.bz2
xzcat file.txt.xz
zstdcat file.txt.zst
```

**Adjusting compression level:**
```bash
gzip -9 file.txt       # max compression, slower
gzip -1 file.txt       # min compression, faster

zstd -19 file.txt      # max compression
zstd -1 file.txt       # fast mode
zstd --fast=3 file.txt # even faster (less compression)

xz -9 file.txt         # max (very slow but smallest)
```

**Recommendations:**
- **zstd** — use this for almost everything. Fastest decompression, excellent compression, modern.
- **gzip** — when compatibility matters (web servers, older systems).
- **xz** — when you need the absolute smallest size and don't care about time (distributing tarballs).
- **bzip2** — largely superseded. Only use if something specifically requires it.

**Install zstd on Debian:**
```bash
sudo apt install zstd
```""",
    "flm_compression"
))

# ---------------------------------------------------------------------------
# 24. tar with different compression (Arch)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_ARCH,
    "How do I use tar with different compression formats?",
    r"""tar bundles files into an archive and can pipe through various compressors.

**Create archives:**
```bash
# gzip (.tar.gz or .tgz)
tar czf archive.tar.gz /path/to/dir

# bzip2 (.tar.bz2)
tar cjf archive.tar.bz2 /path/to/dir

# xz (.tar.xz) — smallest
tar cJf archive.tar.xz /path/to/dir

# zstd (.tar.zst) — fast + good compression
tar --zstd -cf archive.tar.zst /path/to/dir

# No compression (just tar)
tar cf archive.tar /path/to/dir
```

**Extract archives:**
```bash
# tar auto-detects compression format with -a or just:
tar xf archive.tar.gz
tar xf archive.tar.bz2
tar xf archive.tar.xz
tar xf archive.tar.zst
tar xf archive.tar

# Extract to a specific directory
tar xf archive.tar.gz -C /path/to/destination
```

Modern tar auto-detects the compression on extraction, so `tar xf` works for all formats.

**List contents without extracting:**
```bash
tar tf archive.tar.gz
```

**Useful flags:**
```bash
# Verbose (show files as they're archived)
tar czvf archive.tar.gz /path/to/dir

# Exclude files
tar czf archive.tar.gz --exclude='*.log' --exclude='.git' /path/to/dir

# Exclude from a file
tar czf archive.tar.gz --exclude-from=excludes.txt /path/to/dir

# Preserve permissions (important for backups)
tar czpf archive.tar.gz /path/to/dir

# Extract a single file
tar xf archive.tar.gz path/to/specific/file

# Incremental backup
tar czf backup-full.tar.gz -g snapshot.snar /data
# Next time, this creates an incremental:
tar czf backup-incr.tar.gz -g snapshot.snar /data
```

**With parallel compression (much faster on multi-core):**
```bash
sudo pacman -S pigz pbzip2 pixz

# Parallel gzip
tar -I pigz -cf archive.tar.gz /path/to/dir

# Parallel bzip2
tar -I pbzip2 -cf archive.tar.bz2 /path/to/dir

# Parallel xz
tar -I pixz -cf archive.tar.xz /path/to/dir

# zstd with threads (built-in)
tar --zstd -cf archive.tar.zst -T0 /path/to/dir
# or
tar -I 'zstd -T0' -cf archive.tar.zst /path/to/dir
```

`-T0` tells zstd to use all CPU cores.""",
    "flm_compression"
))

# ---------------------------------------------------------------------------
# 25. rsync guide (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "Give me a practical rsync guide.",
    r"""rsync is the gold standard for copying and syncing files. It only transfers what's changed, making it much faster than cp for repeated syncs.

**Install (usually already there):**
```bash
sudo apt install rsync
```

**Basic patterns:**

```bash
# Copy a directory locally
rsync -av /src/dir/ /dest/dir/

# Copy to remote host
rsync -av /local/dir/ user@remote:/remote/dir/

# Copy from remote host
rsync -av user@remote:/remote/dir/ /local/dir/
```

**Important: trailing slash matters!**
```bash
rsync -av /src/dir  /dest/    # creates /dest/dir/
rsync -av /src/dir/ /dest/    # copies contents into /dest/
```

**Essential flags:**
```bash
-a    # archive mode (preserves permissions, timestamps, symlinks, etc.)
-v    # verbose
-h    # human-readable numbers
-z    # compress during transfer (good for remote, skip for local)
-P    # show progress + allow resume of partial transfers
--delete  # remove files from dest that don't exist in src (mirror)
-n    # dry run (show what would happen without doing it)
```

**Common recipes:**

```bash
# Mirror a directory (make dest identical to src)
rsync -avh --delete /src/ /dest/

# Dry run first to see what would change
rsync -avhn --delete /src/ /dest/

# Backup with progress
rsync -avhP /data/ user@backup-server:/backups/data/

# Exclude files
rsync -av --exclude='*.log' --exclude='.git' /src/ /dest/

# Exclude from file
rsync -av --exclude-from='rsync-excludes.txt' /src/ /dest/

# Limit bandwidth (in KB/s)
rsync -avz --bwlimit=5000 /src/ user@remote:/dest/

# Use specific SSH port
rsync -avz -e 'ssh -p 2222' /src/ user@remote:/dest/

# Resume interrupted transfer
rsync -avhP /big-file.iso user@remote:/dest/
```

**Backup with hardlinks (space-efficient snapshots):**
```bash
rsync -av --delete --link-dest=/backups/latest /data/ /backups/$(date +%Y-%m-%d)/
ln -sfn /backups/$(date +%Y-%m-%d) /backups/latest
```

Each daily backup only stores changed files; unchanged files are hardlinked to the previous backup.

**Common gotchas:**
- Always test with `-n` (dry run) before using `--delete`
- Watch the trailing slashes — they change behavior
- For large transfers, use `--partial` (or `-P`) so interrupted transfers can resume
- `rsync` over SSH is encrypted by default, so no need for extra security""",
    "flm_file_tools"
))

# ---------------------------------------------------------------------------
# 26. Advanced grep patterns (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "Show me some advanced grep patterns and tricks.",
    r"""**Basics first:**
```bash
grep "pattern" file.txt
grep -r "pattern" /dir/          # recursive
grep -i "pattern" file.txt       # case insensitive
grep -n "pattern" file.txt       # show line numbers
grep -c "pattern" file.txt       # count matches
grep -l "pattern" /dir/*         # list matching files only
grep -L "pattern" /dir/*         # list non-matching files
```

**Extended regex (-E or egrep):**
```bash
# OR
grep -E "error|warning|critical" /var/log/syslog

# Grouping
grep -E "(GET|POST) /api/" access.log

# Repetition
grep -E "^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}" file  # IP-ish

# Optional
grep -E "colou?r" file.txt    # matches "color" and "colour"
```

**Context:**
```bash
grep -A 3 "error" log.txt    # 3 lines After
grep -B 2 "error" log.txt    # 2 lines Before
grep -C 5 "error" log.txt    # 5 lines Context (before+after)
```

**Inverse match:**
```bash
# Lines NOT matching
grep -v "debug" log.txt

# Chain: show errors but not from cron
grep "error" /var/log/syslog | grep -v "CRON"
```

**Perl-compatible regex (-P):**
```bash
# Lookahead: lines with "error" NOT followed by "test"
grep -P "error(?!.*test)" log.txt

# Lookbehind: numbers preceded by $
grep -P '(?<=\$)\d+' file.txt

# Non-greedy match
grep -oP '"name":\s*"\K[^"]+' data.json
# \K resets match start — only prints what comes after
```

**Real-world examples:**
```bash
# Find TODO/FIXME in code
grep -rn "TODO\|FIXME\|HACK\|XXX" src/

# Extract email addresses
grep -oP '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}' file

# Find lines longer than 120 characters
grep -n '.\{121\}' *.py

# Count unique IPs in access log
grep -oP '^\d+\.\d+\.\d+\.\d+' access.log | sort | uniq -c | sort -rn | head

# Search compressed files
zgrep "pattern" file.log.gz

# Binary files — show matching files
grep -rl --include='*.conf' "listen" /etc/

# Exclude directories
grep -r --exclude-dir='.git' --exclude-dir='node_modules' "import" src/
```

**ripgrep (faster alternative):**
```bash
sudo apt install ripgrep
rg "pattern" /dir/   # much faster, respects .gitignore
```""",
    "flm_file_tools"
))

# ---------------------------------------------------------------------------
# 27. Useful find commands (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "What are the most useful find commands?",
    r"""**By name:**
```bash
find / -name "nginx.conf"           # exact name
find / -iname "readme*"             # case insensitive
find /etc -name "*.conf"            # wildcard
find / -name "*.log" -o -name "*.txt"  # OR
```

**By type:**
```bash
find /var -type f     # files only
find /etc -type d     # directories only
find / -type l        # symlinks
```

**By size:**
```bash
find / -size +100M              # files > 100MB
find / -size +1G                # files > 1GB
find /var/log -size +50M -type f  # large log files
find / -size 0 -type f          # empty files
find / -empty -type d           # empty directories
```

**By time:**
```bash
find /var -mtime -1             # modified in last 24 hours
find /home -mtime +30           # modified more than 30 days ago
find /tmp -atime +7             # accessed more than 7 days ago
find /etc -mmin -60             # modified in last 60 minutes
find / -newer /tmp/reference    # newer than a reference file
```

**By permissions/ownership:**
```bash
find / -perm /4000 -type f       # setuid files
find / -perm /2000 -type f       # setgid files
find / -perm 777 -type f         # world-writable files
find / -nouser                   # files with no valid owner
find /home -user alice -type f   # files owned by alice
```

**Actions:**
```bash
# Delete (careful!)
find /tmp -type f -mtime +7 -delete

# Execute a command on each match
find . -name "*.bak" -exec rm {} \;

# Execute with confirmation
find . -name "*.bak" -ok rm {} \;

# More efficient — batch execution
find . -name "*.py" -exec grep -l "import os" {} +

# Print with details
find /etc -name "*.conf" -ls

# Print just paths (default)
find . -name "*.txt" -print
```

**Combining conditions:**
```bash
# AND (implicit)
find /var -type f -size +10M -mtime +30

# OR
find / \( -name "*.log" -o -name "*.tmp" \) -mtime +7

# NOT
find / -not -name "*.py" -type f

# Complex: large files modified recently, excluding certain dirs
find / -path /proc -prune -o -path /sys -prune -o \
    -type f -size +100M -mtime -7 -print
```

**Performance tips:**
```bash
# Exclude directories (much faster)
find / -path /proc -prune -o -path /sys -prune -o -print

# Use -maxdepth to limit recursion
find /etc -maxdepth 2 -name "*.conf"

# Alternative: fd (modern find replacement)
sudo apt install fd-find
fdfind "*.conf" /etc   # much faster, nicer output
```""",
    "flm_file_tools"
))

# ---------------------------------------------------------------------------
# 28. sed tutorial (short Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SHORT_DEB13,
    "Give me a sed tutorial with practical examples.",
    r"""sed (stream editor) processes text line-by-line. Most common use: find and replace.

**Basic substitution:**
```bash
# Replace first occurrence on each line
sed 's/old/new/' file.txt

# Replace ALL occurrences on each line (g flag)
sed 's/old/new/g' file.txt

# Case insensitive (I flag)
sed 's/old/new/gI' file.txt

# Edit file in place
sed -i 's/old/new/g' file.txt

# Edit in place with backup
sed -i.bak 's/old/new/g' file.txt
```

**Using different delimiters (useful for paths):**
```bash
# Instead of escaping slashes:
sed 's/\/usr\/local\/bin/\/usr\/bin/g' file.txt

# Use a different delimiter:
sed 's|/usr/local/bin|/usr/bin|g' file.txt
sed 's#/usr/local/bin#/usr/bin#g' file.txt
```

**Line addressing:**
```bash
# Only line 5
sed '5s/old/new/' file.txt

# Lines 5-10
sed '5,10s/old/new/' file.txt

# Lines matching a pattern
sed '/^#/d' file.txt           # delete comment lines
sed '/error/s/old/new/g' file  # substitute only on lines containing "error"
```

**Delete lines:**
```bash
sed '/pattern/d' file.txt     # delete lines matching pattern
sed '5d' file.txt             # delete line 5
sed '1,5d' file.txt           # delete lines 1-5
sed '/^$/d' file.txt          # delete blank lines
sed '/^#/d' file.txt          # delete comment lines
```

**Insert and append:**
```bash
# Insert before line 3
sed '3i\New line here' file.txt

# Append after line 3
sed '3a\New line here' file.txt

# Insert before matching lines
sed '/pattern/i\Added before match' file.txt
```

**Print specific lines:**
```bash
sed -n '5p' file.txt           # print only line 5
sed -n '5,10p' file.txt        # print lines 5-10
sed -n '/error/p' file.txt     # print lines matching "error"
```

**Practical examples:**
```bash
# Remove trailing whitespace
sed -i 's/[[:space:]]*$//' file.txt

# Add a line after a match in a config file
sed -i '/\[section\]/a\new_setting = value' config.ini

# Comment out a line
sed -i '/dangerous_option/s/^/#/' config.conf

# Uncomment a line
sed -i 's/^#\(.*dangerous_option\)/\1/' config.conf

# Replace between two patterns
sed '/START/,/END/s/old/new/g' file.txt

# Extract text between two markers
sed -n '/BEGIN/,/END/p' file.txt

# Remove HTML tags
sed 's/<[^>]*>//g' file.html

# Double-space a file
sed 'G' file.txt

# Number each line
sed = file.txt | sed 'N;s/\n/\t/'
```

**Multiple operations:**
```bash
# Chain with -e
sed -e 's/foo/bar/g' -e 's/baz/qux/g' file.txt

# Or use semicolons
sed 's/foo/bar/g; s/baz/qux/g' file.txt
```""",
    "flm_file_tools"
))

# ---------------------------------------------------------------------------
# 29. SSH port forwarding / tunneling (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "Explain SSH port forwarding and tunneling.",
    r"""SSH tunneling lets you securely forward ports through an encrypted SSH connection. There are three types.

**1. Local port forwarding (-L)**

Forward a local port to a remote destination through the SSH server.

```bash
ssh -L 8080:localhost:80 user@remote-server
```

This means: connections to **your** `localhost:8080` get forwarded through the SSH tunnel to `localhost:80` on the remote server.

**Use case:** Access a web app that only listens on the remote server's localhost:
```bash
ssh -L 8080:localhost:3000 user@server
# Now open http://localhost:8080 in your browser
```

Forward to a third host (the remote server acts as a jump):
```bash
ssh -L 5432:db-server:5432 user@bastion
# Now localhost:5432 reaches db-server:5432 through bastion
```

**2. Remote port forwarding (-R)**

Make a port on the remote server forward back to your local machine.

```bash
ssh -R 8080:localhost:3000 user@remote-server
```

This means: connections to `remote-server:8080` get forwarded back to **your** `localhost:3000`.

**Use case:** Expose a local development server to the internet:
```bash
ssh -R 8080:localhost:3000 user@public-server
# People can now reach your local app at public-server:8080
```

Note: the remote server needs `GatewayPorts yes` in `/etc/ssh/sshd_config` to listen on all interfaces.

**3. Dynamic port forwarding / SOCKS proxy (-D)**

Creates a SOCKS5 proxy:
```bash
ssh -D 1080 user@remote-server
```

Configure your browser to use `localhost:1080` as a SOCKS5 proxy. All traffic goes through the SSH tunnel.

**Useful flags:**
```bash
-N    # don't execute a remote command (just forward)
-f    # go to background after authentication
-C    # compress data

# Background tunnel:
ssh -fNL 8080:localhost:80 user@server

# Keep alive (prevent timeout):
ssh -o ServerAliveInterval=60 -L 8080:localhost:80 user@server
```

**Jump hosts / ProxyJump:**
```bash
# SSH through a bastion host
ssh -J bastion-user@bastion internal-server

# In ~/.ssh/config:
Host internal
    HostName 10.0.0.5
    User admin
    ProxyJump bastion-user@bastion.example.com
```

**Multiple forwards at once:**
```bash
ssh -L 8080:localhost:80 -L 5432:db:5432 -L 6379:redis:6379 user@server
```

**Persistent tunnel with autossh:**
```bash
sudo apt install autossh
autossh -M 0 -f -N -L 8080:localhost:80 user@server
```

autossh monitors the connection and reconnects if it drops.""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 30. WireGuard VPN setup (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I set up WireGuard VPN?",
    r"""WireGuard is a modern, fast, and simple VPN built into the Linux kernel.

**Install:**
```bash
sudo apt install wireguard
```

**Generate keys (on both server and client):**
```bash
wg genkey | tee privatekey | wg pubkey > publickey
chmod 600 privatekey
```

**Server config — `/etc/wireguard/wg0.conf`:**
```ini
[Interface]
Address = 10.0.0.1/24
ListenPort = 51820
PrivateKey = <server-private-key>

# Enable IP forwarding and NAT
PostUp = sysctl -w net.ipv4.ip_forward=1; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

[Peer]
# Client 1
PublicKey = <client-public-key>
AllowedIPs = 10.0.0.2/32
```

**Client config — `/etc/wireguard/wg0.conf`:**
```ini
[Interface]
Address = 10.0.0.2/24
PrivateKey = <client-private-key>
DNS = 1.1.1.1

[Peer]
PublicKey = <server-public-key>
Endpoint = server.example.com:51820
AllowedIPs = 0.0.0.0/0, ::/0
PersistentKeepalive = 25
```

- `AllowedIPs = 0.0.0.0/0` routes ALL traffic through the VPN (full tunnel)
- `AllowedIPs = 10.0.0.0/24` routes only VPN subnet traffic (split tunnel)

**Start the VPN:**
```bash
# Server
sudo systemctl enable --now wg-quick@wg0

# Client
sudo wg-quick up wg0
# or persistent:
sudo systemctl enable --now wg-quick@wg0
```

**Open the firewall:**
```bash
sudo ufw allow 51820/udp
```

**Check status:**
```bash
sudo wg show
```

**Adding more clients:**
Generate new keys on each client, then add a `[Peer]` section to the server config:
```ini
[Peer]
# Client 2
PublicKey = <client2-public-key>
AllowedIPs = 10.0.0.3/32
```

Reload without dropping existing connections:
```bash
sudo wg syncconf wg0 <(wg-quick strip wg0)
```

**Make IP forwarding permanent:**
```bash
echo 'net.ipv4.ip_forward = 1' | sudo tee /etc/sysctl.d/99-wireguard.conf
sudo sysctl -p /etc/sysctl.d/99-wireguard.conf
```

**Tip:** For mobile clients, generate a QR code:
```bash
sudo apt install qrencode
qrencode -t ansiutf8 < client.conf
```""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 31. WireGuard on Arch
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_ARCH,
    "How do I set up WireGuard on Arch?",
    r"""WireGuard is in the Arch kernel by default. You just need the tools:

```bash
sudo pacman -S wireguard-tools
```

**Generate keys:**
```bash
wg genkey | tee /etc/wireguard/privatekey | wg pubkey > /etc/wireguard/publickey
chmod 600 /etc/wireguard/privatekey
```

**Server config — `/etc/wireguard/wg0.conf`:**
```ini
[Interface]
Address = 10.0.0.1/24
ListenPort = 51820
PrivateKey = <server-private-key>
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

[Peer]
PublicKey = <client-public-key>
AllowedIPs = 10.0.0.2/32
```

**Client config — `/etc/wireguard/wg0.conf`:**
```ini
[Interface]
Address = 10.0.0.2/24
PrivateKey = <client-private-key>
DNS = 1.1.1.1

[Peer]
PublicKey = <server-public-key>
Endpoint = server.example.com:51820
AllowedIPs = 0.0.0.0/0, ::/0
PersistentKeepalive = 25
```

**Start and enable:**
```bash
sudo systemctl enable --now wg-quick@wg0
```

**Enable IP forwarding:**
```bash
echo 'net.ipv4.ip_forward = 1' | sudo tee /etc/sysctl.d/99-wireguard.conf
sudo sysctl -p /etc/sysctl.d/99-wireguard.conf
```

**Open the port in nftables** (add to your `/etc/nftables.conf` input chain):
```
udp dport 51820 accept
```
Then `sudo systemctl reload nftables`.

**Check status:**
```bash
sudo wg show
```

**Quick client setup with QR code:**
```bash
sudo pacman -S qrencode
qrencode -t ansiutf8 < /etc/wireguard/client.conf
```

Scan with the WireGuard mobile app to import.

**Troubleshooting:**
- Check the module is loaded: `lsmod | grep wireguard`
- Check the interface exists: `ip link show wg0`
- Check handshake: `sudo wg show` — if "latest handshake" is missing, there's a connectivity or key problem
- Check logs: `sudo journalctl -u wg-quick@wg0`""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 32. fail2ban setup (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I set up fail2ban to protect my server?",
    r"""fail2ban monitors log files and bans IPs that show malicious activity (like brute-force SSH attempts).

**Install:**
```bash
sudo apt install fail2ban
```

**Configuration:** Never edit `/etc/fail2ban/jail.conf` directly — it gets overwritten on updates. Create local overrides:

```bash
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo nano /etc/fail2ban/jail.local
```

Or better, create a drop-in file `/etc/fail2ban/jail.d/custom.conf`:
```ini
[DEFAULT]
# Ban for 1 hour (default is 10 minutes)
bantime = 3600

# Ban permanently after 3 bans
bantime.increment = true
bantime.maxtime = 604800

# Check last 10 minutes for failures
findtime = 600

# Allow 5 failures before banning
maxretry = 5

# Email notifications (optional)
# destemail = admin@example.com
# sender = fail2ban@example.com
# action = %(action_mwl)s

[sshd]
enabled = true
port = ssh
logpath = %(sshd_log)s
maxretry = 3

[nginx-http-auth]
enabled = true

[nginx-botsearch]
enabled = true
```

**Start and enable:**
```bash
sudo systemctl enable --now fail2ban
```

**Useful commands:**
```bash
# Check status
sudo fail2ban-client status

# Check a specific jail
sudo fail2ban-client status sshd

# Unban an IP
sudo fail2ban-client set sshd unbanip 192.168.1.100

# Ban an IP manually
sudo fail2ban-client set sshd banip 203.0.113.50

# Check what fail2ban would match in a log
sudo fail2ban-regex /var/log/auth.log /etc/fail2ban/filter.d/sshd.conf
```

**Custom jails** — for example, ban aggressive web scrapers:

Create `/etc/fail2ban/filter.d/nginx-req-limit.conf`:
```ini
[Definition]
failregex = limiting requests, excess:.* by zone.*client: <HOST>
ignoreregex =
```

Add to `/etc/fail2ban/jail.d/custom.conf`:
```ini
[nginx-req-limit]
enabled = true
filter = nginx-req-limit
action = iptables-multiport[name=ReqLimit, port="http,https"]
logpath = /var/log/nginx/error.log
findtime = 600
maxretry = 10
bantime = 7200
```

**Whitelist trusted IPs:**
```ini
[DEFAULT]
ignoreip = 127.0.0.1/8 ::1 10.0.0.0/8 192.168.1.0/24
```

**Restart after changes:**
```bash
sudo systemctl restart fail2ban
```

**Check the ban log:**
```bash
sudo tail -f /var/log/fail2ban.log
```""",
    "flm_security"
))

# ---------------------------------------------------------------------------
# 33. Let's Encrypt with certbot (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I get a free SSL certificate with Let's Encrypt?",
    r"""Use certbot to get free TLS certificates from Let's Encrypt.

**Install:**
```bash
sudo apt install certbot
```

**For nginx:**
```bash
sudo apt install python3-certbot-nginx
sudo certbot --nginx -d example.com -d www.example.com
```

This automatically:
- Obtains the certificate
- Configures nginx to use it
- Sets up HTTPS redirects

**For Apache:**
```bash
sudo apt install python3-certbot-apache
sudo certbot --apache -d example.com -d www.example.com
```

**Standalone mode (no web server needed):**
```bash
# Temporarily starts its own web server on port 80
sudo certbot certonly --standalone -d example.com
```

**Webroot mode (web server already running, don't want certbot to touch config):**
```bash
sudo certbot certonly --webroot -w /var/www/html -d example.com
```

**Certificate files are stored in:**
```
/etc/letsencrypt/live/example.com/
    fullchain.pem   — certificate + intermediate CA
    privkey.pem     — private key
    chain.pem       — intermediate CA only
    cert.pem        — certificate only
```

**Auto-renewal:** Certbot installs a systemd timer that checks twice daily:
```bash
sudo systemctl list-timers | grep certbot
```

Test renewal:
```bash
sudo certbot renew --dry-run
```

**Manual nginx config** (if you used `--certonly`):
```nginx
server {
    listen 443 ssl http2;
    server_name example.com;

    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    # Recommended SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;

    root /var/www/html;
}

server {
    listen 80;
    server_name example.com;
    return 301 https://$host$request_uri;
}
```

**Wildcard certificates** (requires DNS validation):
```bash
sudo certbot certonly --manual --preferred-challenges dns -d "*.example.com" -d example.com
```

You'll need to create a DNS TXT record. For automation, use a DNS plugin:
```bash
sudo apt install python3-certbot-dns-cloudflare
sudo certbot certonly --dns-cloudflare --dns-cloudflare-credentials /etc/letsencrypt/cloudflare.ini -d "*.example.com"
```

**Revoke a certificate:**
```bash
sudo certbot revoke --cert-path /etc/letsencrypt/live/example.com/cert.pem
```""",
    "flm_security"
))

# ---------------------------------------------------------------------------
# 34. Let's Encrypt on Arch
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_ARCH,
    "How do I set up Let's Encrypt on Arch?",
    r"""**Install certbot:**
```bash
sudo pacman -S certbot
```

For nginx or Apache integration:
```bash
sudo pacman -S certbot-nginx
# or
sudo pacman -S certbot-apache
```

**Get a certificate for nginx:**
```bash
sudo certbot --nginx -d example.com -d www.example.com
```

**Standalone (if no web server yet):**
```bash
sudo certbot certonly --standalone -d example.com
```

**Webroot (web server running, hands-off config):**
```bash
sudo certbot certonly --webroot -w /usr/share/nginx/html -d example.com
```

Note: Arch's nginx uses `/usr/share/nginx/html` as the default webroot, not `/var/www/html`.

**Certificates are stored in:**
```
/etc/letsencrypt/live/example.com/fullchain.pem
/etc/letsencrypt/live/example.com/privkey.pem
```

**Auto-renewal — set up the timer:**
```bash
sudo systemctl enable --now certbot-renew.timer
```

Check it's scheduled:
```bash
sudo systemctl list-timers | grep certbot
```

Test:
```bash
sudo certbot renew --dry-run
```

**Post-renewal hook (reload nginx):**

Create `/etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh`:
```bash
#!/bin/bash
systemctl reload nginx
```
```bash
sudo chmod +x /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh
```

**nginx config example:**
```nginx
server {
    listen 443 ssl http2;
    server_name example.com;

    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;
    add_header Strict-Transport-Security "max-age=63072000" always;

    root /usr/share/nginx/html;
}

server {
    listen 80;
    server_name example.com;
    return 301 https://$host$request_uri;
}
```

Reload nginx: `sudo systemctl reload nginx`""",
    "flm_security"
))

# ---------------------------------------------------------------------------
# 35. nginx reverse proxy (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I set up nginx as a reverse proxy?",
    r"""A reverse proxy forwards client requests to a backend application server.

**Install nginx:**
```bash
sudo apt install nginx
```

**Basic reverse proxy config:**

Create `/etc/nginx/sites-available/myapp`:
```nginx
server {
    listen 80;
    server_name myapp.example.com;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable it:
```bash
sudo ln -s /etc/nginx/sites-available/myapp /etc/nginx/sites-enabled/
sudo nginx -t          # test config
sudo systemctl reload nginx
```

**With HTTPS (using Let's Encrypt):**
```bash
sudo apt install python3-certbot-nginx
sudo certbot --nginx -d myapp.example.com
```

**WebSocket support:**
```nginx
location /ws/ {
    proxy_pass http://127.0.0.1:3000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
}
```

**Load balancing across multiple backends:**
```nginx
upstream myapp_backends {
    server 127.0.0.1:3001;
    server 127.0.0.1:3002;
    server 127.0.0.1:3003;
}

server {
    listen 80;
    server_name myapp.example.com;

    location / {
        proxy_pass http://myapp_backends;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Caching:**
```nginx
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=mycache:10m max_size=1g inactive=60m;

server {
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_cache mycache;
        proxy_cache_valid 200 60m;
        proxy_cache_valid 404 1m;
        add_header X-Cache-Status $upstream_cache_status;
    }
}
```

**Rate limiting:**
```nginx
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

server {
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://127.0.0.1:3000;
    }
}
```

**Serving static files directly (faster):**
```nginx
server {
    location /static/ {
        alias /var/www/myapp/static/;
        expires 30d;
    }

    location / {
        proxy_pass http://127.0.0.1:3000;
    }
}
```

**Test your config always before reloading:**
```bash
sudo nginx -t && sudo systemctl reload nginx
```""",
    "flm_webdb"
))

# ---------------------------------------------------------------------------
# 36. PostgreSQL setup (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I set up PostgreSQL on Debian?",
    r"""**Install:**
```bash
sudo apt install postgresql postgresql-client
```

PostgreSQL starts automatically. Check:
```bash
sudo systemctl status postgresql
```

**Initial setup:**

Debian creates a `postgres` system user. Switch to it:
```bash
sudo -u postgres psql
```

Inside the PostgreSQL shell:
```sql
-- Create a new user
CREATE USER myapp WITH PASSWORD 'secretpassword';

-- Create a database
CREATE DATABASE mydb OWNER myapp;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE mydb TO myapp;

-- Exit
\q
```

**Connect as the new user:**
```bash
psql -U myapp -d mydb -h localhost
```

If you get a peer authentication error, edit `/etc/postgresql/16/main/pg_hba.conf` and change `peer` to `md5` or `scram-sha-256` for local connections:
```
# TYPE  DATABASE  USER      ADDRESS       METHOD
local   all       all                     md5
host    all       all       127.0.0.1/32  scram-sha-256
host    all       all       ::1/128       scram-sha-256
```

Then reload:
```bash
sudo systemctl reload postgresql
```

**Allow remote connections:**

Edit `/etc/postgresql/16/main/postgresql.conf`:
```
listen_addresses = '*'    # default is 'localhost'
```

Edit `pg_hba.conf` — add:
```
host    all    all    0.0.0.0/0    scram-sha-256
```

Restart:
```bash
sudo systemctl restart postgresql
```

Open the firewall:
```bash
sudo ufw allow 5432/tcp
```

**Basic operations:**
```sql
-- List databases
\l

-- Connect to a database
\c mydb

-- List tables
\dt

-- Describe a table
\d tablename

-- Create a table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Basic queries
SELECT * FROM users;
INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com');
```

**Backups:**
```bash
# Dump a single database
sudo -u postgres pg_dump mydb > mydb_backup.sql

# Dump all databases
sudo -u postgres pg_dumpall > all_databases.sql

# Compressed dump
sudo -u postgres pg_dump -Fc mydb > mydb_backup.dump

# Restore
sudo -u postgres psql mydb < mydb_backup.sql
# or from compressed:
sudo -u postgres pg_restore -d mydb mydb_backup.dump
```

**Performance tuning** — edit `postgresql.conf`:
```
# For a server with 4GB RAM:
shared_buffers = 1GB
effective_cache_size = 3GB
work_mem = 16MB
maintenance_work_mem = 256MB
```

Use https://pgtune.leopard.in.ua/ to generate settings based on your hardware.""",
    "flm_webdb"
))

# ---------------------------------------------------------------------------
# 37. MariaDB setup (Fedora)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_FED41,
    "How do I install and set up MariaDB on Fedora?",
    r"""**Install:**
```bash
sudo dnf install mariadb-server mariadb
```

**Start and enable:**
```bash
sudo systemctl enable --now mariadb
```

**Secure the installation:**
```bash
sudo mariadb-secure-installation
```

This walks you through:
- Setting a root password
- Removing anonymous users
- Disabling remote root login
- Removing the test database

**Connect:**
```bash
sudo mariadb
# or
mariadb -u root -p
```

**Create a database and user:**
```sql
CREATE DATABASE myapp;
CREATE USER 'myuser'@'localhost' IDENTIFIED BY 'secretpassword';
GRANT ALL PRIVILEGES ON myapp.* TO 'myuser'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

**Connect as the new user:**
```bash
mariadb -u myuser -p myapp
```

**Basic operations:**
```sql
-- Show databases
SHOW DATABASES;

-- Use a database
USE myapp;

-- Create a table
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- Insert data
INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com');

-- Query
SELECT * FROM users;

-- Show tables
SHOW TABLES;

-- Describe table structure
DESCRIBE users;
```

**Allow remote connections:**

Edit `/etc/my.cnf.d/mariadb-server.cnf`:
```ini
[mysqld]
bind-address = 0.0.0.0
```

Grant remote access:
```sql
CREATE USER 'myuser'@'%' IDENTIFIED BY 'secretpassword';
GRANT ALL PRIVILEGES ON myapp.* TO 'myuser'@'%';
FLUSH PRIVILEGES;
```

Open firewall:
```bash
sudo firewall-cmd --permanent --add-service=mysql
sudo firewall-cmd --reload
```

Restart:
```bash
sudo systemctl restart mariadb
```

**Backups:**
```bash
# Dump a database
mariadb-dump -u root -p myapp > myapp_backup.sql

# Dump all databases
mariadb-dump -u root -p --all-databases > all_backup.sql

# Restore
mariadb -u root -p myapp < myapp_backup.sql
```

**Performance tuning** — `/etc/my.cnf.d/server.cnf`:
```ini
[mysqld]
# For a server with 4GB RAM
innodb_buffer_pool_size = 2G
innodb_log_file_size = 512M
innodb_flush_log_at_trx_commit = 2
max_connections = 200
query_cache_type = 0
```

**Check status:**
```bash
sudo systemctl status mariadb
mariadb -u root -p -e "SHOW STATUS LIKE 'Threads%';"
```""",
    "flm_webdb"
))


# ── Write output ────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "a") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    print(f"Appended {len(conversations)} conversations to {OUTPUT}")


if __name__ == "__main__":
    main()
