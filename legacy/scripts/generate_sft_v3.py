#!/usr/bin/env python3
"""Generate large batch of additional SFT examples — V3 expansion.

Covers new topics and adds depth/variety to existing ones.
Heavy emphasis on multi-turn conversations and OS-aware responses.
"""
import json
import random
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from system_prompt import random_system_prompt, PRESETS, get_preset_prompt

random.seed(99)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sft")


def pick_sys():
    prompt = random_system_prompt(include_none=True, short_chance=0.2)
    if prompt is None:
        return ""
    return f"<|system|>\n{prompt}\n"


def sys_for(preset):
    short = random.random() < 0.2
    return f"<|system|>\n{get_preset_prompt(preset, short=short)}\n"


def conv(turns, source, system=None):
    s = system if system is not None else pick_sys()
    parts = [s] if s else []
    for role, text in turns:
        parts.append(f"<|{role}|>\n{text}\n")
    return {"text": "".join(parts), "source": source}


def write_jsonl(data, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  {filename}: {len(data)} conversations")
    return data


def make_variants(q, a, source, n=3):
    """Create n variants of a QA pair with different system prompts."""
    return [conv([("user", q), ("assistant", a)], source) for _ in range(n)]


# ══════════════════════════════════════════════════════════════════════════
# CRON & SCHEDULING
# ══════════════════════════════════════════════════════════════════════════

def gen_cron():
    examples = []
    pairs = [
        ("How does crontab syntax work?",
         "The format is: `minute hour day-of-month month day-of-week command`\n\n```\n# ┌───────── minute (0-59)\n# │ ┌─────── hour (0-23)\n# │ │ ┌───── day of month (1-31)\n# │ │ │ ┌─── month (1-12)\n# │ │ │ │ ┌─ day of week (0-7, 0 and 7 = Sunday)\n# │ │ │ │ │\n  * * * * * command\n```\n\nExamples:\n```bash\n# Every 5 minutes\n*/5 * * * * /usr/local/bin/check.sh\n\n# Daily at 2:30 AM\n30 2 * * * /usr/local/bin/backup.sh\n\n# Every Monday at 9 AM\n0 9 * * 1 /usr/local/bin/report.sh\n\n# First of every month at midnight\n0 0 1 * * /usr/local/bin/monthly.sh\n\n# Every weekday at 6 PM\n0 18 * * 1-5 /usr/local/bin/end-of-day.sh\n```\n\nEdit with `crontab -e`. List with `crontab -l`."),

        ("What's the difference between crontab and /etc/cron.d/?",
         "Several cron locations serve different purposes:\n\n**User crontabs** (`crontab -e`):\n- Per-user, stored in `/var/spool/cron/crontabs/`\n- No username field needed\n- Best for user-specific tasks\n\n**System crontab** (`/etc/crontab`):\n- Requires username field: `* * * * * root command`\n- Rarely edited directly nowadays\n\n**Drop-in directories:**\n```\n/etc/cron.d/       — package-installed cron jobs (has username field)\n/etc/cron.daily/   — scripts run once daily\n/etc/cron.weekly/  — scripts run once weekly\n/etc/cron.monthly/ — scripts run once monthly\n/etc/cron.hourly/  — scripts run once hourly\n```\n\nThe `cron.daily/weekly/monthly` dirs are run by `anacron`, which handles machines that aren't on 24/7. Scripts in these dirs are just executables — no crontab syntax needed."),

        ("How do I see all scheduled cron jobs on the system?",
         "```bash\n# Your user's crontab\ncrontab -l\n\n# Root's crontab\nsudo crontab -l\n\n# All users' crontabs\nfor user in $(cut -f1 -d: /etc/passwd); do\n    crons=$(sudo crontab -l -u \"$user\" 2>/dev/null)\n    if [ -n \"$crons\" ]; then\n        echo \"=== $user ===\"\n        echo \"$crons\"\n    fi\ndone\n\n# System crontab\ncat /etc/crontab\n\n# Drop-in jobs\nls -la /etc/cron.d/\nls -la /etc/cron.daily/\nls -la /etc/cron.weekly/\n\n# Systemd timers (the modern alternative)\nsystemctl list-timers --all\n```"),

        ("My cron job runs but doesn't work — how do I debug it?",
         "Common cron pitfalls:\n\n**1. PATH is minimal** — cron uses `/usr/bin:/bin`, not your full PATH:\n```bash\n# Add at top of crontab\nPATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n```\n\n**2. No environment** — cron doesn't source `.bashrc`:\n```bash\n# Wrong\n* * * * * source ~/.bashrc && my_command\n# Right — use full paths\n* * * * * /usr/local/bin/my_command\n```\n\n**3. Output is lost** — redirect to a log:\n```bash\n* * * * * /path/to/script.sh >> /var/log/myjob.log 2>&1\n```\n\n**4. Check mail** — cron emails output to the user:\n```bash\nsudo cat /var/mail/username\n```\n\n**5. Check syslog**:\n```bash\ngrep CRON /var/log/syslog | tail -20\n```"),

        ("How do I run a command at a specific time once?",
         "`at` runs a one-time scheduled command:\n\n```bash\n# Install\nsudo apt install at\n\n# Schedule for a specific time\necho \"/usr/local/bin/deploy.sh\" | at 2:30 AM\n\n# Relative time\necho \"reboot\" | at now + 2 hours\necho \"/path/to/script.sh\" | at now + 30 minutes\n\n# Specific date\necho \"/path/to/script.sh\" | at 10:00 AM Dec 25\n\n# List pending jobs\natq\n\n# Remove a job\natrm 5\n\n# See job details\nat -c 5\n```\n\n`batch` is similar but waits until system load is low:\n```bash\necho \"/path/to/heavy_task.sh\" | batch\n```"),
    ]
    for q, a in pairs:
        examples.extend(make_variants(q, a, "flm_cron"))
    return examples


# ══════════════════════════════════════════════════════════════════════════
# DNS & EMAIL SERVERS
# ══════════════════════════════════════════════════════════════════════════

def gen_dns_email():
    examples = []
    pairs = [
        ("How do I set up a local DNS resolver with unbound?",
         "```bash\nsudo apt install unbound\n```\n\nCreate `/etc/unbound/unbound.conf.d/local.conf`:\n```yaml\nserver:\n    interface: 127.0.0.1\n    interface: ::1\n    access-control: 127.0.0.0/8 allow\n    access-control: ::1/128 allow\n\n    # Performance\n    num-threads: 2\n    msg-cache-size: 64m\n    rrset-cache-size: 128m\n\n    # Privacy\n    hide-identity: yes\n    hide-version: yes\n    qname-minimisation: yes\n\n    # DNSSEC validation\n    auto-trust-anchor-file: \"/var/lib/unbound/root.key\"\n```\n\n```bash\nsudo systemctl enable --now unbound\n# Point resolv.conf to local\necho 'nameserver 127.0.0.1' | sudo tee /etc/resolv.conf\n```"),

        ("How do I set up a basic mail server with Postfix on Debian?",
         "```bash\nsudo apt install postfix\n# Select 'Internet Site' during config\n```\n\nMain config `/etc/postfix/main.cf`:\n```\nmyhostname = mail.example.com\nmydomain = example.com\nmyorigin = $mydomain\nmydestination = $myhostname, localhost.$mydomain, localhost, $mydomain\nmynetworks = 127.0.0.0/8 [::1]/128\n\n# TLS\nsmtpd_tls_cert_file = /etc/letsencrypt/live/mail.example.com/fullchain.pem\nsmtpd_tls_key_file = /etc/letsencrypt/live/mail.example.com/privkey.pem\nsmtpd_tls_security_level = may\n\n# Limits\nmessage_size_limit = 52428800\nmailbox_size_limit = 0\n```\n\n```bash\nsudo systemctl restart postfix\n```\n\n**Warning**: Running a mail server is complex. You also need SPF, DKIM, DMARC records, reverse DNS, and proper abuse handling. Consider Mailcow or Mail-in-a-Box for a full solution."),

        ("How do I check if mail is working?",
         "```bash\n# Send a test email\necho \"Test body\" | mail -s \"Test Subject\" user@example.com\n\n# Check mail queue\nsudo mailq\n# or\nsudo postqueue -p\n\n# Flush the queue (retry delivery)\nsudo postqueue -f\n\n# Delete all queued mail\nsudo postsuper -d ALL\n\n# Check logs\nsudo tail -f /var/log/mail.log\n\n# Test SMTP connection\ntelnet localhost 25\n# or\nopenssl s_client -connect mail.example.com:587 -starttls smtp\n```\n\nCommon issues: DNS not resolving, firewall blocking port 25, relay denied, TLS certificate problems."),
    ]
    for q, a in pairs:
        examples.extend(make_variants(q, a, "flm_dns_email"))
    return examples


# ══════════════════════════════════════════════════════════════════════════
# RAID & DISK ENCRYPTION
# ══════════════════════════════════════════════════════════════════════════

def gen_raid_encryption():
    examples = []
    pairs = [
        ("How do I set up software RAID on Debian?",
         "Using `mdadm`:\n\n```bash\nsudo apt install mdadm\n\n# Create RAID 1 (mirror) from two disks\nsudo mdadm --create /dev/md0 --level=1 --raid-devices=2 /dev/sdb1 /dev/sdc1\n\n# Watch rebuild progress\ncat /proc/mdstat\nwatch cat /proc/mdstat\n\n# Create filesystem\nsudo mkfs.ext4 /dev/md0\nsudo mkdir -p /mnt/raid\nsudo mount /dev/md0 /mnt/raid\n\n# Save config\nsudo mdadm --detail --scan | sudo tee -a /etc/mdadm/mdadm.conf\nsudo update-initramfs -u\n\n# Add to fstab\necho '/dev/md0 /mnt/raid ext4 defaults 0 2' | sudo tee -a /etc/fstab\n```\n\nRAID levels: 0=stripe (speed, no redundancy), 1=mirror (redundancy), 5=stripe+parity (3+ disks), 6=double parity (4+ disks), 10=mirror+stripe."),

        ("How do I replace a failed RAID disk?",
         "```bash\n# Check RAID status\nsudo mdadm --detail /dev/md0\ncat /proc/mdstat\n\n# Mark failed disk (if not already)\nsudo mdadm /dev/md0 --fail /dev/sdb1\n\n# Remove failed disk\nsudo mdadm /dev/md0 --remove /dev/sdb1\n\n# Physically replace the disk, then partition the new one\n# (match the partition layout of the surviving disk)\nsudo sfdisk -d /dev/sdc | sudo sfdisk /dev/sdb\n\n# Add new disk to array\nsudo mdadm /dev/md0 --add /dev/sdb1\n\n# Watch rebuild\nwatch cat /proc/mdstat\n```\n\nRebuild time depends on disk size and I/O load. A 1TB RAID 1 rebuild typically takes 2-4 hours."),

        ("How do I encrypt a disk with LUKS?",
         "```bash\n# Install\nsudo apt install cryptsetup\n\n# Encrypt partition (WARNING: destroys data)\nsudo cryptsetup luksFormat /dev/sdb1\n\n# Open encrypted partition\nsudo cryptsetup luksOpen /dev/sdb1 encrypted_data\n\n# Create filesystem\nsudo mkfs.ext4 /dev/mapper/encrypted_data\n\n# Mount\nsudo mkdir -p /mnt/secure\nsudo mount /dev/mapper/encrypted_data /mnt/secure\n\n# Unmount and close\nsudo umount /mnt/secure\nsudo cryptsetup luksClose encrypted_data\n```\n\nFor automatic unlock at boot, add to `/etc/crypttab`:\n```\nencrypted_data /dev/sdb1 none luks\n```\nThis prompts for passphrase during boot. For a keyfile instead:\n```\nencrypted_data /dev/sdb1 /etc/keys/disk.key luks\n```"),

        ("How do I set up full disk encryption during Debian install?",
         "The Debian installer supports LUKS+LVM natively:\n\n1. During installation, choose **Guided - use entire disk and set up encrypted LVM**\n2. Enter a strong passphrase\n3. The installer creates:\n   - `/boot` — unencrypted (GRUB needs to read it)\n   - LUKS-encrypted partition containing LVM\n   - LVM with root and swap logical volumes\n\nThe resulting layout:\n```\n/dev/sda1 → /boot (ext4, ~500MB)\n/dev/sda2 → LUKS → LVM → root + swap\n```\n\nYou'll enter the passphrase at every boot before the system starts.\n\nTo add it after installation is much harder — you'd need to backup, repartition, encrypt, restore. Use the installer option when possible."),
    ]
    for q, a in pairs:
        examples.extend(make_variants(q, a, "flm_raid_encryption"))
    return examples


# ══════════════════════════════════════════════════════════════════════════
# KERNEL & BOOT
# ══════════════════════════════════════════════════════════════════════════

def gen_kernel_boot():
    examples = []
    pairs = [
        ("How do I manage kernel modules?",
         "```bash\n# List loaded modules\nlsmod\n\n# Info about a module\nmodinfo e1000e\n\n# Load a module\nsudo modprobe vfio-pci\n\n# Unload a module\nsudo modprobe -r nouveau\n\n# Load at boot — add to /etc/modules-load.d/\necho 'vfio-pci' | sudo tee /etc/modules-load.d/vfio.conf\n\n# Blacklist a module (prevent loading)\necho 'blacklist nouveau' | sudo tee /etc/modprobe.d/blacklist-nouveau.conf\nsudo update-initramfs -u\n\n# Pass parameters to a module\necho 'options snd_hda_intel power_save=1' | sudo tee /etc/modprobe.d/audio-powersave.conf\n```\n\n`lsmod`, `modprobe`, and `modinfo` are your main tools. Changes in `/etc/modprobe.d/` persist across reboots."),

        ("How do I update GRUB on Debian?",
         "```bash\n# After editing /etc/default/grub\nsudo update-grub\n# This is a Debian shortcut for:\n# sudo grub-mkconfig -o /boot/grub/grub.cfg\n```\n\nCommon `/etc/default/grub` settings:\n```bash\n# Timeout (seconds)\nGRUB_TIMEOUT=5\n\n# Default entry (0 = first)\nGRUB_DEFAULT=0\n\n# Kernel parameters\nGRUB_CMDLINE_LINUX_DEFAULT=\"quiet\"\nGRUB_CMDLINE_LINUX=\"\"\n\n# Show menu even with single OS\nGRUB_TIMEOUT_STYLE=menu\n\n# Console resolution\nGRUB_GFXMODE=1920x1080\n```\n\nAfter any change: `sudo update-grub && sudo reboot`\n\n**Recovery**: If GRUB breaks, boot from a Debian live USB, mount your root partition, `chroot` in, and run `update-grub` and `grub-install /dev/sda`."),

        ("How do I add kernel boot parameters?",
         "**Temporarily** (one boot):\n1. At GRUB menu, press `e` to edit\n2. Find the `linux` line\n3. Add parameters at the end\n4. Press Ctrl+X to boot\n\n**Permanently**:\nEdit `/etc/default/grub`:\n```bash\n# For all boots\nGRUB_CMDLINE_LINUX=\"intel_iommu=on\"\n\n# For normal boots (not recovery)\nGRUB_CMDLINE_LINUX_DEFAULT=\"quiet splash\"\n```\n\nThen: `sudo update-grub`\n\nCommon parameters:\n- `nomodeset` — disable kernel mode setting (GPU issues)\n- `intel_iommu=on` — enable IOMMU for GPU passthrough\n- `mem=4G` — limit available RAM\n- `init=/bin/bash` — boot to shell (recovery)\n- `systemd.unit=rescue.target` — boot to rescue mode"),

        ("How do I check what kernel I'm running and install a different one?",
         "```bash\n# Current kernel\nuname -r\n\n# All installed kernels\ndpkg -l 'linux-image-*' | grep '^ii'\n\n# Available kernels\napt search linux-image | grep '^linux-image'\n\n# Install a specific kernel\nsudo apt install linux-image-6.12.0-amd64\n\n# Install latest backports kernel\nsudo apt install -t bookworm-backports linux-image-amd64\n\n# Remove old kernels\nsudo apt autoremove --purge\n```\n\nDebian keeps old kernels installed so you can boot them from GRUB if the new one has issues. `apt autoremove` cleans up old ones (but keeps the current and one previous)."),

        ("My system won't boot — how do I recover?",
         "**From GRUB menu:**\n1. Select \"Advanced options\" → older kernel or recovery mode\n2. In recovery, select \"root\" for a shell\n3. Remount root read-write: `mount -o remount,rw /`\n4. Fix whatever is broken, then `reboot`\n\n**If GRUB itself is broken** — boot from Debian live USB:\n```bash\n# Mount your root partition\nsudo mount /dev/sda2 /mnt\nsudo mount /dev/sda1 /mnt/boot\nsudo mount --bind /dev /mnt/dev\nsudo mount --bind /proc /mnt/proc\nsudo mount --bind /sys /mnt/sys\n\n# Chroot in\nsudo chroot /mnt\n\n# Reinstall GRUB\ngrub-install /dev/sda\nupdate-grub\n\n# Exit and reboot\nexit\nsudo umount -R /mnt\nsudo reboot\n```\n\n**If you can't even get to GRUB**: Check BIOS boot order, verify the disk is detected."),
    ]
    for q, a in pairs:
        examples.extend(make_variants(q, a, "flm_kernel_boot"))
    return examples


# ══════════════════════════════════════════════════════════════════════════
# LOCALE, TIMEZONE & SYSTEM SETUP
# ══════════════════════════════════════════════════════════════════════════

def gen_locale_time():
    examples = []
    pairs = [
        ("How do I change the timezone on Debian?",
         "```bash\n# Check current timezone\ntimedatectl\n\n# List available timezones\ntimedatectl list-timezones\n\n# Set timezone\nsudo timedatectl set-timezone America/New_York\n\n# Or interactively\nsudo dpkg-reconfigure tzdata\n```\n\nThe timezone is stored in `/etc/timezone` and symlinked at `/etc/localtime`. NTP synchronization is handled by `systemd-timesyncd`:\n```bash\n# Check NTP sync status\ntimedatectl show --property=NTPSynchronized\n\n# Enable NTP\nsudo timedatectl set-ntp true\n```"),

        ("How do I configure locales on Debian?",
         "```bash\n# See current locale\nlocale\n\n# Generate locales interactively\nsudo dpkg-reconfigure locales\n\n# Or manually: uncomment desired locales in /etc/locale.gen\nsudo sed -i 's/# en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen\nsudo locale-gen\n\n# Set system default\nsudo update-locale LANG=en_US.UTF-8\n```\n\nIf you see `locale: Cannot set LC_*` warnings, it means the locale isn't generated. Run `sudo dpkg-reconfigure locales` and select the ones you need.\n\nFor servers, `en_US.UTF-8` is the safe default. Always use UTF-8."),

        ("How do I change the hostname?",
         "```bash\n# Check current\nhostname\nhostnamectl\n\n# Set new hostname\nsudo hostnamectl set-hostname newname\n\n# Also update /etc/hosts\nsudo sed -i 's/oldname/newname/g' /etc/hosts\n```\n\nThe hostname is stored in `/etc/hostname`. After changing it:\n```bash\n# Verify\nhostnamectl\nhostname -f  # FQDN\n```\n\nMake sure `/etc/hosts` has an entry mapping the hostname to 127.0.1.1 (Debian convention):\n```\n127.0.0.1       localhost\n127.0.1.1       newname.example.com newname\n```"),

        ("How do I set up NTP time synchronization?",
         "Debian uses `systemd-timesyncd` by default (lightweight, good for clients):\n\n```bash\n# Check status\ntimedatectl\nsystemctl status systemd-timesyncd\n\n# Enable\nsudo timedatectl set-ntp true\n\n# Configure NTP servers\nsudo mkdir -p /etc/systemd/timesyncd.conf.d\nsudo tee /etc/systemd/timesyncd.conf.d/local.conf << 'EOF'\n[Time]\nNTP=ntp.example.com\nFallbackNTP=0.debian.pool.ntp.org 1.debian.pool.ntp.org\nEOF\nsudo systemctl restart systemd-timesyncd\n```\n\nFor a full NTP server (serving time to other machines), install `chrony` instead:\n```bash\nsudo apt install chrony\nsudo systemctl enable --now chrony\nchronyc tracking  # check sync status\n```"),
    ]
    for q, a in pairs:
        examples.extend(make_variants(q, a, "flm_locale_time"))
    return examples


# ══════════════════════════════════════════════════════════════════════════
# VIRTUALIZATION
# ══════════════════════════════════════════════════════════════════════════

def gen_virtualization():
    examples = []
    pairs = [
        ("How do I set up KVM virtualization on Debian?",
         "```bash\n# Check CPU support\ngrep -E '(vmx|svm)' /proc/cpuinfo\n# vmx = Intel, svm = AMD\n\n# Install\nsudo apt install qemu-kvm libvirt-daemon-system virtinst bridge-utils\nsudo usermod -aG libvirt $USER\nnewgrp libvirt\n\n# Verify\nvirsh list --all\nsudo systemctl status libvirtd\n```\n\nCreate a VM:\n```bash\nvirt-install \\\n    --name debian-vm \\\n    --ram 2048 \\\n    --vcpus 2 \\\n    --disk size=20 \\\n    --os-variant debian12 \\\n    --cdrom /path/to/debian.iso \\\n    --network bridge=virbr0 \\\n    --graphics vnc\n```\n\nManage with `virsh`:\n```bash\nvirsh start debian-vm\nvirsh shutdown debian-vm\nvirsh console debian-vm\nvirsh snapshot-create-as debian-vm --name \"before-upgrade\"\n```"),

        ("How do I create a VM from the command line?",
         "Using `virt-install`:\n\n```bash\n# With an ISO\nvirt-install \\\n    --name myvm \\\n    --ram 4096 \\\n    --vcpus 4 \\\n    --disk path=/var/lib/libvirt/images/myvm.qcow2,size=40 \\\n    --os-variant debian12 \\\n    --cdrom /var/lib/libvirt/images/debian-12-amd64.iso \\\n    --network network=default \\\n    --graphics none \\\n    --console pty,target_type=serial \\\n    --extra-args 'console=ttyS0'\n\n# From a cloud image (no install needed)\nwget https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-generic-amd64.qcow2\nsudo cp debian-12-generic-amd64.qcow2 /var/lib/libvirt/images/\nvirt-install \\\n    --name cloud-vm \\\n    --ram 2048 --vcpus 2 \\\n    --disk /var/lib/libvirt/images/debian-12-generic-amd64.qcow2 \\\n    --import --os-variant debian12 \\\n    --network network=default\n```"),

        ("How do I manage VMs with virsh?",
         "```bash\n# List all VMs\nvirsh list --all\n\n# Start/stop\nvirsh start myvm\nvirsh shutdown myvm      # graceful\nvirsh destroy myvm       # force off (like pulling power cord)\n\n# Console access\nvirsh console myvm       # serial console (Ctrl+] to exit)\n\n# Snapshots\nvirsh snapshot-create-as myvm --name \"before-changes\" --description \"Pre-upgrade\"\nvirsh snapshot-list myvm\nvirsh snapshot-revert myvm --snapshotname \"before-changes\"\nvirsh snapshot-delete myvm --snapshotname \"before-changes\"\n\n# Resources\nvirsh setvcpus myvm 4 --live      # hot-add CPUs\nvirsh setmem myvm 4G --live       # hot-add memory\n\n# Delete VM and its disk\nvirsh destroy myvm\nvirsh undefine myvm --remove-all-storage\n```"),

        ("How do I set up LXC containers on Debian?",
         "LXC is lighter than full VMs — shares the host kernel:\n\n```bash\nsudo apt install lxc\n\n# Create a container\nsudo lxc-create -n mycontainer -t debian -- -r bookworm\n\n# Start\nsudo lxc-start -n mycontainer\n\n# Attach (get a shell)\nsudo lxc-attach -n mycontainer\n\n# Stop\nsudo lxc-stop -n mycontainer\n\n# List\nsudo lxc-ls -f\n\n# Delete\nsudo lxc-destroy -n mycontainer\n```\n\nConfig is in `/var/lib/lxc/mycontainer/config`. For unprivileged containers (more secure), configure subuid/subgid mappings in `/etc/subuid` and `/etc/subgid`."),
    ]
    for q, a in pairs:
        examples.extend(make_variants(q, a, "flm_virtualization"))
    return examples


# ══════════════════════════════════════════════════════════════════════════
# PYTHON ENV MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════

def gen_python():
    examples = []
    pairs = [
        ("How do I manage Python virtual environments on Debian?",
         "```bash\n# Install venv (Debian separates it from python3)\nsudo apt install python3-venv\n\n# Create a virtual environment\npython3 -m venv myproject/venv\n\n# Activate\nsource myproject/venv/bin/activate\n\n# Install packages (only in venv)\npip install requests flask\n\n# Freeze requirements\npip freeze > requirements.txt\n\n# Deactivate\ndeactivate\n```\n\n**Important on Debian**: PEP 668 means `pip install` outside a venv is blocked by default (\"externally managed environment\" error). Always use a venv or `apt install python3-packagename` for system-wide packages. This is by design — it prevents breaking system Python."),

        ("I get 'externally managed environment' error with pip",
         "This is Debian's PEP 668 protection. It prevents pip from breaking system Python packages.\n\n**Solutions (best to worst):**\n\n1. **Use a venv** (recommended):\n```bash\npython3 -m venv ~/.venv/myproject\nsource ~/.venv/myproject/bin/activate\npip install whatever\n```\n\n2. **Use pipx** for CLI tools:\n```bash\nsudo apt install pipx\npipx install youtube-dl\npipx install black\n```\n\n3. **Use apt** for common packages:\n```bash\nsudo apt install python3-requests python3-flask\n```\n\n4. **Override** (not recommended):\n```bash\npip install --break-system-packages whatever\n```\n\nThe venv approach is the right one. Debian made this change to prevent the situation where `pip install` overwrites apt-managed files and breaks `apt`."),

        ("How do I install a specific Python version on Debian?",
         "Debian ships one Python version per release. For a different version:\n\n**Option 1: deadsnakes-style (build from source)**:\n```bash\nsudo apt install build-essential libssl-dev zlib1g-dev \\\n    libncurses5-dev libbz2-dev libreadline-dev libsqlite3-dev \\\n    libffi-dev liblzma-dev\n\nwget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz\ntar xzf Python-3.12.0.tgz\ncd Python-3.12.0\n./configure --enable-optimizations --prefix=/usr/local\nmake -j$(nproc)\nsudo make altinstall  # altinstall to not override system python3\n```\n\n**Option 2: pyenv**:\n```bash\ncurl https://pyenv.run | bash\n# Add to ~/.bashrc as instructed\npyenv install 3.12.0\npyenv local 3.12.0\n```\n\n**Never replace `/usr/bin/python3`** — system tools depend on it."),
    ]
    for q, a in pairs:
        examples.extend(make_variants(q, a, "flm_python"))
    return examples


# ══════════════════════════════════════════════════════════════════════════
# DEBUGGING & TRACING
# ══════════════════════════════════════════════════════════════════════════

def gen_debugging():
    examples = []
    pairs = [
        ("How do I use strace to debug a failing program?",
         "`strace` shows every system call a program makes:\n\n```bash\n# Basic trace\nstrace ls /nonexistent\n\n# Trace a running process\nsudo strace -p 1234\n\n# Only file operations\nstrace -e trace=open,read,write,close command\n\n# Only network operations\nstrace -e trace=network command\n\n# With timestamps\nstrace -t command\n\n# Save to file\nstrace -o trace.log command\n\n# Follow child processes\nstrace -f command\n\n# Summary of calls (like a profiler)\nstrace -c command\n```\n\nCommon debugging patterns:\n- Program can't find a file? Look for `open()` calls returning `ENOENT`\n- Permission denied? Look for `EACCES`\n- Network issues? Look for `connect()` failures"),

        ("How do I use ltrace to trace library calls?",
         "`ltrace` traces shared library calls (like strace but for libraries):\n\n```bash\nsudo apt install ltrace\n\n# Trace library calls\nltrace ls\n\n# Only specific library\nltrace -l libm.so.6 ./myprogram\n\n# With timestamps\nltrace -t ./myprogram\n\n# Count calls\nltrace -c ./myprogram\n```\n\nUseful when you know the problem is in a library function, not a syscall. For example, seeing `malloc()` patterns or `strlen()` calls to understand memory behavior."),

        ("How do I debug a segfault?",
         "```bash\n# 1. Enable core dumps\nulimit -c unlimited\n\n# Or permanently via /etc/security/limits.conf:\n# * soft core unlimited\n\n# 2. Reproduce the crash\n./crashing_program\n# Creates a core file\n\n# 3. Analyze with gdb\nsudo apt install gdb\ngdb ./crashing_program core\n# In gdb:\n#   bt           — show backtrace\n#   bt full      — with variables\n#   info threads — show all threads\n\n# 4. Or get a backtrace directly\ngdb -batch -ex bt -ex quit ./crashing_program core\n```\n\nWith Debian packages, install debug symbols:\n```bash\nsudo apt install program-dbgsym\n# Debug symbols are in the -dbgsym packages\n```\n\nFor quick diagnosis: `dmesg | tail` often shows the faulting address and signal."),

        ("How do I profile a program's performance?",
         "Several approaches:\n\n**System-level:**\n```bash\n# CPU time breakdown\ntime ./myprogram\n\n# System call statistics\nstrace -c ./myprogram\n\n# Cache misses, branch mispredictions\nsudo perf stat ./myprogram\n\n# CPU profiling with flame graph\nsudo perf record -g ./myprogram\nsudo perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg\n```\n\n**For compiled programs (C/C++):**\n```bash\n# Valgrind (slow but thorough)\nsudo apt install valgrind\nvalgrind --tool=callgrind ./myprogram\nkcachegrind callgrind.out.*\n\n# Memory profiling\nvalgrind --tool=massif ./myprogram\nms_print massif.out.*\n```\n\n**For Python:**\n```bash\npython3 -m cProfile -s cumtime script.py\npython3 -m py_spy record -o profile.svg -- python3 script.py\n```"),
    ]
    for q, a in pairs:
        examples.extend(make_variants(q, a, "flm_debugging"))
    return examples


# ══════════════════════════════════════════════════════════════════════════
# APT INTERNALS & DEBIAN PACKAGING DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════

def gen_apt_deep():
    examples = []
    pairs = [
        ("How does apt actually work internally?",
         "apt does several things when you run `apt install pkg`:\n\n1. **Reads sources** from `/etc/apt/sources.list` and `.list.d/`\n2. **Checks package index** in `/var/lib/apt/lists/` (updated by `apt update`)\n3. **Resolves dependencies** using the SAT solver\n4. **Downloads .deb files** to `/var/cache/apt/archives/`\n5. **Installs via dpkg** — extracts files and runs maintainer scripts\n\nKey directories:\n```\n/var/lib/apt/lists/     — package indexes (what's available)\n/var/cache/apt/archives/ — downloaded .deb files\n/var/lib/dpkg/          — installed package database\n/var/lib/dpkg/info/     — maintainer scripts, file lists, configs\n/var/lib/dpkg/status    — current state of all packages\n```\n\napt is actually a frontend for dpkg. apt handles downloads and dependencies; dpkg handles actual file installation."),

        ("What are maintainer scripts in Debian packages?",
         "Every .deb package can include four shell scripts that run during install/remove:\n\n- **preinst** — runs BEFORE files are unpacked\n- **postinst** — runs AFTER files are unpacked (configure step)\n- **prerm** — runs BEFORE files are removed\n- **postrm** — runs AFTER files are removed\n\n```bash\n# See a package's scripts\nls /var/lib/dpkg/info/nginx.*\ncat /var/lib/dpkg/info/nginx.postinst\n\n# Inside a .deb\ndpkg-deb --control package.deb\n```\n\nCommon postinst actions:\n- Create system users (`adduser --system`)\n- Enable systemd services\n- Run `ldconfig` after installing libraries\n- Prompt for configuration via `debconf`\n\nIf a postinst fails, the package is left in a broken state. Fix with `sudo dpkg --configure -a`."),

        ("How do I downgrade a package?",
         "```bash\n# Check available versions\napt-cache policy nginx\n\n# Install a specific version\nsudo apt install nginx=1.22.1-9\n\n# From a specific release\nsudo apt install -t bookworm nginx\n\n# If the old version isn't in repos anymore,\n# download the .deb from snapshot.debian.org:\nwget https://snapshot.debian.org/archive/debian/20240101T000000Z/pool/main/n/nginx/nginx_1.22.1-9_amd64.deb\nsudo dpkg -i nginx_1.22.1-9_amd64.deb\n\n# Pin to prevent re-upgrade\nsudo apt-mark hold nginx\n```\n\nsnapshot.debian.org archives every version of every package ever published in Debian. It's invaluable for downgrades."),

        ("How do I fix 'dpkg was interrupted' errors?",
         "```bash\n# The standard fix\nsudo dpkg --configure -a\n\n# If that fails, try\nsudo apt --fix-broken install\n\n# If lock files are stale\nsudo rm /var/lib/dpkg/lock-frontend\nsudo rm /var/lib/dpkg/lock\nsudo rm /var/cache/apt/archives/lock\nsudo dpkg --configure -a\n\n# Nuclear option — reinstall the broken package\nsudo apt install --reinstall package-name\n\n# If truly stuck — remove and reinstall\nsudo dpkg --remove --force-remove-reinstreq package-name\nsudo apt install package-name\n```\n\nThe `--configure -a` command finishes any interrupted package configurations. Most dpkg interruptions happen when a postinst script fails or the system loses power during an upgrade."),

        ("What is debconf and how do I use it?",
         "debconf is the Debian configuration system. Packages use it to ask questions during installation.\n\n```bash\n# Re-run a package's configuration questions\nsudo dpkg-reconfigure postfix\n\n# Set priority (show more/fewer questions)\nsudo dpkg-reconfigure -p low tzdata\n\n# Pre-seed answers (for automation)\necho 'postfix postfix/main_mailer_type select Internet Site' | sudo debconf-set-selections\necho 'postfix postfix/mailname string mail.example.com' | sudo debconf-set-selections\nsudo apt install -y postfix\n\n# See current answers for a package\nsudo debconf-show postfix\n\n# Non-interactive mode (accept defaults)\nexport DEBIAN_FRONTEND=noninteractive\nsudo apt install -y package\n```\n\nPre-seeding with `debconf-set-selections` is how you automate Debian installs in scripts and CI."),

        ("How do I create a local apt repository?",
         "Simple approach with `dpkg-scanpackages`:\n\n```bash\n# Create directory structure\nmkdir -p /srv/apt/pool\n\n# Copy .deb files into it\ncp *.deb /srv/apt/pool/\n\n# Generate package index\ncd /srv/apt\ndpkg-scanpackages pool /dev/null | gzip > pool/Packages.gz\n\n# Add to sources.list\necho 'deb [trusted=yes] file:///srv/apt pool/' | sudo tee /etc/apt/sources.list.d/local.list\nsudo apt update\n```\n\nFor a proper repo with signing and Release files, use `reprepro`:\n```bash\nsudo apt install reprepro\nmkdir -p /srv/repo/conf\n# Configure /srv/repo/conf/distributions\nreprepro -b /srv/repo includedeb bookworm package.deb\n```\n\nFor serving over HTTP, just point nginx at the directory."),
    ]
    for q, a in pairs:
        examples.extend(make_variants(q, a, "flm_apt_deep"))
    return examples


# ══════════════════════════════════════════════════════════════════════════
# MORE MULTI-TURN CONVERSATIONS
# ══════════════════════════════════════════════════════════════════════════

def gen_multiturn():
    examples = []

    scenarios = [
        # Setting up a web server from scratch
        [
            ("user", "I need to host a simple website on my Debian server. Where do I start?"),
            ("assistant", "Let's set up Nginx. First:\n```bash\nsudo apt install nginx\nsudo systemctl status nginx\n```\n\nIt should already be running and serving the default page. Can you access it from a browser at your server's IP?"),
            ("user", "yes I see the default nginx page"),
            ("assistant", "Great. Now let's create your site. Where are your HTML files, or are we starting from scratch?"),
            ("user", "starting from scratch, just a simple static site"),
            ("assistant", "```bash\n# Create web root\nsudo mkdir -p /var/www/mysite\n\n# Create a simple index page\nsudo tee /var/www/mysite/index.html << 'EOF'\n<!DOCTYPE html>\n<html>\n<head><title>My Site</title></head>\n<body><h1>Hello from Debian!</h1></body>\n</html>\nEOF\n\n# Set ownership\nsudo chown -R www-data:www-data /var/www/mysite\n```\n\nNow create the Nginx config:\n```bash\nsudo tee /etc/nginx/sites-available/mysite << 'EOF'\nserver {\n    listen 80;\n    server_name _;\n    root /var/www/mysite;\n    index index.html;\n}\nEOF\n\nsudo ln -s /etc/nginx/sites-available/mysite /etc/nginx/sites-enabled/\nsudo rm /etc/nginx/sites-enabled/default\nsudo nginx -t && sudo systemctl reload nginx\n```\n\nRefresh your browser — you should see \"Hello from Debian!\"."),
            ("user", "works! how do I add HTTPS?"),
            ("assistant", "You need a domain name pointing to your server. If you have one:\n\n```bash\nsudo apt install certbot python3-certbot-nginx\nsudo certbot --nginx -d yourdomain.com\n```\n\nCertbot handles everything — gets the certificate, configures Nginx for HTTPS, and sets up auto-renewal.\n\nIf you don't have a domain yet and this is just for local/testing, you can create a self-signed certificate:\n```bash\nsudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \\\n    -keyout /etc/nginx/ssl/self.key \\\n    -out /etc/nginx/ssl/self.crt\n```\nBut browsers will show a warning for self-signed certs. For production, use Let's Encrypt with a real domain."),
        ],

        # Diagnosing high CPU usage
        [
            ("user", "my server is using 100% CPU and I don't know why"),
            ("assistant", "Let's find the culprit:\n```bash\ntop -o %CPU\n```\n\nWhat process is at the top? Also check:\n```bash\nuptime  # how long has this been going on?\n```"),
            ("user", "it's a process called 'python3' using 98% CPU, PID 4521"),
            ("assistant", "Let's find out what Python script that is:\n```bash\n# Full command line\nps -p 4521 -o pid,user,args\n\n# Or from /proc\ncat /proc/4521/cmdline | tr '\\0' ' '\n\n# Who started it?\nps -p 4521 -o user,start_time\n```\n\nWhat script is it running?"),
            ("user", "it shows python3 /home/deploy/app/worker.py and it's been running since yesterday"),
            ("assistant", "So it's your app's worker process. It could be:\n1. **Stuck in an infinite loop** — a bug\n2. **Processing a large task** — legitimate but slow\n3. **Memory issue** — swapping causes CPU spin\n\nCheck memory first:\n```bash\nps -p 4521 -o pid,rss,vsz,%mem\nfree -h\n```\n\nIf memory looks fine, check what it's doing:\n```bash\n# See what system calls it's making\nsudo strace -p 4521 -c  # Ctrl+C after 10 seconds for summary\n\n# Or check Python-level stack\nkill -SIGUSR1 4521  # if the app handles this signal for debugging\n```\n\nIf it's a bug and you just need to stop it:\n```bash\nkill 4521        # graceful\nkill -9 4521     # force (if graceful doesn't work after 10s)\n```\n\nThen check the logs: `tail -100 /home/deploy/app/logs/*.log`"),
        ],

        # Setting up SSH for the first time
        [
            ("user", "I just set up a new Debian server and need to access it remotely"),
            ("assistant", "SSH should already be installed on a Debian server. Check:\n```bash\nsudo systemctl status ssh\n```\n\nIs it running? Also, what's the server's IP?\n```bash\nip addr show scope global\n```"),
            ("user", "it's running, IP is 192.168.1.50"),
            ("assistant", "From your local machine, try:\n```bash\nssh yourusername@192.168.1.50\n```\n\nOnce that works, let's set up key-based auth (much more secure than passwords):\n\nOn your local machine:\n```bash\nssh-keygen -t ed25519\nssh-copy-id yourusername@192.168.1.50\n```\n\nTest that key auth works:\n```bash\nssh yourusername@192.168.1.50\n```\n\nIf it logs in without asking for a password, keys are working."),
            ("user", "keys work, what else should I do to secure it?"),
            ("assistant", "Now harden the SSH config. On the server, edit `/etc/ssh/sshd_config`:\n\n```bash\nsudo tee /etc/ssh/sshd_config.d/hardening.conf << 'EOF'\nPermitRootLogin no\nPasswordAuthentication no\nPubkeyAuthentication yes\nMaxAuthTries 3\nClientAliveInterval 300\nClientAliveCountMax 2\nX11Forwarding no\nEOF\n\nsudo systemctl restart ssh\n```\n\n**Before restarting**: make sure your key-based login works in a second terminal! If you disable password auth without working keys, you're locked out.\n\nAlso set up fail2ban to block brute-force attempts:\n```bash\nsudo apt install fail2ban\nsudo systemctl enable --now fail2ban\n```\n\nOptionally, change the port from 22 to something less targeted:\n```bash\necho 'Port 2222' | sudo tee -a /etc/ssh/sshd_config.d/hardening.conf\nsudo systemctl restart ssh\n# Then connect with: ssh -p 2222 user@192.168.1.50\n```"),
        ],

        # Recovering from a full disk
        [
            ("user", "I can't log in to my server anymore, it says 'No space left on device'"),
            ("assistant", "When the disk is 100% full, even basic operations fail. If you can still get a shell (console, not SSH):\n\n```bash\n# Quick space recovery — clean apt cache\nsudo apt clean\n\n# Remove old journal logs\nsudo journalctl --vacuum-size=100M\n\n# Find and remove large temp files\nsudo find /tmp -type f -size +10M -delete\n```\n\nCan you get a local console, or is this a remote server?"),
            ("user", "it's remote, I can access the hosting provider's console"),
            ("assistant", "Good. Through the provider's console:\n\n1. First, free some space immediately:\n```bash\n# These usually work even on a full disk\nsudo bash -c '> /var/log/syslog'\nsudo bash -c '> /var/log/kern.log'\nsudo apt clean\n```\n\n2. Find what filled the disk:\n```bash\ndf -h\ndu -h --max-depth=1 / 2>/dev/null | sort -hr | head -10\n```\n\n3. Common culprits:\n   - `/var/log` — truncate large logs\n   - `/var/cache/apt` — `apt clean`\n   - `/tmp` — delete old temp files\n   - Old kernels — `apt autoremove`\n   - Docker images — `docker system prune`\n\nOnce you free ~500MB, the system recovers and you can SSH in normally again. Then investigate the root cause to prevent it happening again."),
        ],

        # Setting up a development environment
        [
            ("user", "I want to set up a Python development environment on Debian. What's the best way?"),
            ("assistant", "Start with the essentials:\n\n```bash\n# Python and dev tools\nsudo apt install python3 python3-venv python3-pip python3-dev\n\n# Build essentials (needed for packages with C extensions)\nsudo apt install build-essential git\n```\n\nNow create a project with a virtual environment:\n```bash\nmkdir ~/myproject && cd ~/myproject\npython3 -m venv .venv\nsource .venv/bin/activate\n```\n\nWhat kind of project are you working on? Web app, data science, CLI tool?"),
            ("user", "a Flask web app"),
            ("assistant", "```bash\n# In your activated venv\npip install flask\n\n# Create a basic app\ncat > app.py << 'PYEOF'\nfrom flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/')\ndef hello():\n    return 'Hello from Flask on Debian!'\n\nif __name__ == '__main__':\n    app.run(debug=True, host='0.0.0.0')\nPYEOF\n\n# Run it\npython app.py\n```\n\nVisit `http://localhost:5000`. For a proper project structure:\n```bash\npip install flask gunicorn\npip freeze > requirements.txt\n\n# Initialize git\ngit init\necho '.venv/' > .gitignore\ngit add . && git commit -m 'Initial Flask app'\n```"),
            ("user", "how do I deploy this for production?"),
            ("assistant", "Don't use Flask's dev server in production. Use Gunicorn behind Nginx:\n\n```bash\n# Install gunicorn in your venv\npip install gunicorn\n\n# Test it\ngunicorn -w 4 -b 127.0.0.1:8000 app:app\n```\n\nCreate a systemd service `/etc/systemd/system/myapp.service`:\n```ini\n[Unit]\nDescription=My Flask App\nAfter=network.target\n\n[Service]\nUser=www-data\nGroup=www-data\nWorkingDirectory=/home/deploy/myproject\nExecStart=/home/deploy/myproject/.venv/bin/gunicorn -w 4 -b 127.0.0.1:8000 app:app\nRestart=on-failure\n\n[Install]\nWantedBy=multi-user.target\n```\n\nNginx reverse proxy in `/etc/nginx/sites-available/myapp`:\n```nginx\nserver {\n    listen 80;\n    server_name yourdomain.com;\n    location / {\n        proxy_pass http://127.0.0.1:8000;\n        proxy_set_header Host $host;\n        proxy_set_header X-Real-IP $remote_addr;\n    }\n}\n```\n\n```bash\nsudo systemctl enable --now myapp\nsudo ln -s /etc/nginx/sites-available/myapp /etc/nginx/sites-enabled/\nsudo nginx -t && sudo systemctl reload nginx\nsudo certbot --nginx -d yourdomain.com  # HTTPS\n```"),
        ],

        # Investigating mysterious network issues
        [
            ("user", "some of my containers can reach the internet and some can't, same docker network"),
            ("assistant", "That's odd — same network should mean same connectivity. Let's check:\n\n```bash\n# Which containers work and which don't?\ndocker ps\n\n# Test from a broken container\ndocker exec -it broken_container ping -c 3 8.8.8.8\ndocker exec -it broken_container cat /etc/resolv.conf\n\n# And from a working one\ndocker exec -it working_container ping -c 3 8.8.8.8\ndocker exec -it working_container cat /etc/resolv.conf\n```\n\nAre they on the same Docker network? Check with `docker network inspect <network_name>`"),
            ("user", "the broken ones can ping 8.8.8.8 but DNS doesn't work. resolv.conf shows 127.0.0.11"),
            ("assistant", "127.0.0.11 is Docker's internal DNS resolver. If ping by IP works but DNS doesn't, Docker's DNS proxy is having issues.\n\nQuick fixes to try:\n\n1. **Restart Docker's DNS**:\n```bash\nsudo systemctl restart docker\n```\n\n2. **Check if systemd-resolved is interfering** (common on newer systems):\n```bash\nsudo systemctl status systemd-resolved\nresolvectl status\n```\n\n3. **Workaround — set DNS explicitly in docker-compose.yml**:\n```yaml\nservices:\n  myapp:\n    dns:\n      - 8.8.8.8\n      - 1.1.1.1\n```\n\nThe root cause is often that Docker's embedded DNS server loses its upstream configuration, especially after a network change on the host or a systemd-resolved restart. Restarting Docker usually fixes it."),
        ],

        # Debian upgrade gone wrong
        [
            ("user", "I was doing apt full-upgrade and my computer lost power. Now nothing works"),
            ("assistant", "Power loss during an upgrade is one of the worst scenarios, but usually recoverable. Can you boot at all? What do you see on screen?"),
            ("user", "it boots to a login prompt but lots of errors during boot about failed services"),
            ("assistant", "Good — the base system is intact. Log in and let's fix the interrupted upgrade:\n\n```bash\n# First, fix any half-configured packages\nsudo dpkg --configure -a\n```\n\nThis might take a while — it finishes configuring packages that were interrupted. If it errors on a specific package, note which one.\n\nAfter that:\n```bash\n# Fix broken dependencies\nsudo apt --fix-broken install\n\n# Then complete the upgrade\nsudo apt update\nsudo apt full-upgrade\n```\n\nDoes `dpkg --configure -a` complete successfully?"),
            ("user", "it failed on systemd saying 'subprocess new pre-installation script returned error exit status 1'"),
            ("assistant", "A failed systemd preinst is serious but fixable:\n\n```bash\n# See the actual error\nsudo dpkg --configure systemd 2>&1 | tail -30\n\n# Try force-configuring it\nsudo dpkg --configure --force-depends systemd\n\n# If that fails, try reinstalling\nsudo apt install --reinstall systemd\n\n# If apt itself is broken:\nsudo dpkg --force-confnew --configure systemd\n```\n\nIf none of that works, the nuclear option:\n```bash\n# Download the package manually\nsudo apt download systemd\nsudo dpkg --force-all -i systemd_*.deb\nsudo apt --fix-broken install\n```\n\nAfter systemd is fixed, re-run:\n```bash\nsudo dpkg --configure -a\nsudo apt full-upgrade\nsudo reboot\n```\n\nThe `--force-confnew` flag replaces config files with package defaults, which usually fixes preinst issues after an interrupted upgrade."),
        ],

        # Understanding and optimizing a slow database
        [
            ("user", "my PostgreSQL queries are really slow, how do I figure out why?"),
            ("assistant", "Let's diagnose. First check if the server itself is struggling:\n\n```bash\nsudo -u postgres psql -c \"SELECT count(*) FROM pg_stat_activity;\"\nsudo -u postgres psql -c \"SELECT * FROM pg_stat_activity WHERE state = 'active';\"\n```\n\nThen check the slow query log:\n```bash\n# Enable logging of slow queries\nsudo -u postgres psql -c \"ALTER SYSTEM SET log_min_duration_statement = '1000';\"\nsudo -u postgres psql -c \"SELECT pg_reload_conf();\"\n# Now queries taking > 1 second are logged\nsudo tail -f /var/log/postgresql/postgresql-*-main.log\n```\n\nWhat kind of queries are slow? SELECT, INSERT, or everything?"),
            ("user", "mostly SELECT queries on a large table, about 5 million rows"),
            ("assistant", "5M rows isn't huge but can be slow without indexes. Let's check:\n\n```sql\n-- Connect to the database\n\\c mydb\n\n-- Check table size\nSELECT pg_size_pretty(pg_total_relation_size('your_table'));\n\n-- Check existing indexes\n\\di+ your_table*\n\n-- Analyze a slow query\nEXPLAIN ANALYZE SELECT * FROM your_table WHERE some_column = 'value';\n```\n\nThe `EXPLAIN ANALYZE` output will show if it's doing a sequential scan (reading every row) instead of using an index.\n\nIf you see \"Seq Scan\", add an index:\n```sql\nCREATE INDEX idx_your_table_some_column ON your_table(some_column);\nANALYZE your_table;  -- update statistics\n```\n\nAlso make sure auto-vacuum is running:\n```sql\nSELECT relname, last_vacuum, last_autovacuum, last_analyze\nFROM pg_stat_user_tables WHERE relname = 'your_table';\n```"),
        ],
    ]

    for turns in scenarios:
        examples.append(conv(turns, "flm_multiturn_v3"))
        # Add one more variant with different system prompt
        examples.append(conv(turns, "flm_multiturn_v3"))

    return examples


# ══════════════════════════════════════════════════════════════════════════
# OS-AWARE EXAMPLES (different answers per OS)
# ══════════════════════════════════════════════════════════════════════════

def gen_os_aware():
    examples = []

    items = [
        {
            "q": "How do I install a package?",
            "answers": {
                "debian_trixie": "```bash\n# Search for packages\napt search keyword\n\n# Show package info\napt show package-name\n\n# Install\nsudo apt install package-name\n\n# Install multiple\nsudo apt install nginx curl git vim\n\n# Install without recommends (smaller footprint)\nsudo apt install --no-install-recommends package-name\n```\n\nAlways run `sudo apt update` first to refresh the package index.",
                "debian_bookworm": "```bash\nsudo apt update\nsudo apt install package-name\n```\n\nIf you need a newer version than what Bookworm provides, check backports:\n```bash\nsudo apt install -t bookworm-backports package-name\n```",
                "arch": "```bash\n# From official repos\nsudo pacman -S package-name\n\n# Search\npacman -Ss keyword\n\n# From AUR (using yay or paru)\nyay -S aur-package-name\n\n# Update all packages\nsudo pacman -Syu\n```\n\nNever run `pacman -Sy package` (partial upgrade). Always use `-Syu` to update everything together.",
                "fedora_41": "```bash\nsudo dnf install package-name\n\n# Search\ndnf search keyword\n\n# Info\ndnf info package-name\n\n# Group install\nsudo dnf groupinstall 'Development Tools'\n```",
                "ubuntu_noble": "```bash\nsudo apt update\nsudo apt install package-name\n\n# For newer versions, try snap:\nsudo snap install package-name\n\n# Or add a PPA for third-party repos\nsudo add-apt-repository ppa:user/repo\nsudo apt update\n```",
            }
        },
        {
            "q": "How do I update my system?",
            "answers": {
                "debian_trixie": "```bash\n# Refresh package index\nsudo apt update\n\n# Upgrade installed packages\nsudo apt upgrade\n\n# Full upgrade (handles dependency changes)\nsudo apt full-upgrade\n\n# Clean up\nsudo apt autoremove\nsudo apt clean\n```\n\nOn Trixie (testing), run updates frequently — packages change often. Check for major transitions before upgrading: `apt list --upgradable`",
                "debian_bookworm": "```bash\nsudo apt update\nsudo apt upgrade\nsudo apt autoremove\n```\n\nBookworm is stable — updates are mostly security fixes. Enable automatic security updates:\n```bash\nsudo apt install unattended-upgrades\nsudo dpkg-reconfigure -plow unattended-upgrades\n```",
                "arch": "```bash\n# Update everything\nsudo pacman -Syu\n\n# After update, check for .pacnew files\nfind /etc -name '*.pacnew' 2>/dev/null\n# Merge changes manually\n\n# AUR packages\nyay -Syu\n```\n\nArch is rolling release — update at least weekly. Read the Arch news before major updates.",
                "fedora_41": "```bash\nsudo dnf upgrade\n\n# Automatic updates\nsudo dnf install dnf-automatic\nsudo systemctl enable --now dnf-automatic-install.timer\n```",
                "ubuntu_noble": "```bash\nsudo apt update && sudo apt upgrade\nsudo apt autoremove\n\n# For snap packages too\nsudo snap refresh\n```",
            }
        },
        {
            "q": "How do I check system logs?",
            "answers": {
                "debian_trixie": "```bash\n# Systemd journal (primary on Trixie)\njournalctl -f           # follow\njournalctl -u nginx     # specific service\njournalctl -p err -b    # errors since boot\njournalctl --since today\n\n# Traditional logs\ntail -f /var/log/syslog\ntail -f /var/log/auth.log\n\n# Kernel messages\ndmesg | tail\njournalctl -k\n```",
                "arch": "```bash\n# Arch uses journald by default (no rsyslog)\njournalctl -f\njournalctl -u sshd\njournalctl -p err -b\n\n# Boot messages\njournalctl -b\njournalctl -b -1  # previous boot\n\n# If you need traditional log files,\n# install syslog-ng or rsyslog\n```",
                "fedora_41": "```bash\njournalctl -f\njournalctl -u httpd\njournalctl --since '1 hour ago'\n\n# SELinux audit logs\nsudo ausearch -m AVC -ts recent\nsudo sealert -a /var/log/audit/audit.log\n```\n\nFedora uses SELinux by default, so also check for SELinux denials if something isn't working.",
            }
        },
        {
            "q": "How do I manage services?",
            "answers": {
                "debian_trixie": "```bash\n# List running services\nsystemctl list-units --type=service --state=running\n\n# Start/stop/restart\nsudo systemctl start nginx\nsudo systemctl stop nginx\nsudo systemctl restart nginx\nsudo systemctl reload nginx  # reload config without downtime\n\n# Enable/disable at boot\nsudo systemctl enable nginx\nsudo systemctl disable nginx\n\n# Check status\nsystemctl status nginx\n\n# See all failed services\nsystemctl --failed\n```",
                "arch": "```bash\nsystemctl list-units --type=service --state=running\nsudo systemctl start sshd\nsudo systemctl enable --now sshd\nsystemctl status sshd\nsystemctl --failed\n\n# Arch tip: use 'enable --now' to start and enable in one command\n```",
                "fedora_41": "```bash\nsystemctl list-units --type=service --state=running\nsudo systemctl start httpd  # Apache is 'httpd' on Fedora\nsudo systemctl enable --now httpd\nsystemctl status httpd\nsudo systemctl restart firewalld  # Fedora uses firewalld\n```",
            }
        },
        {
            "q": "How do I configure the firewall?",
            "answers": {
                "debian_trixie": "Debian Trixie uses nftables:\n```bash\n# Check current rules\nsudo nft list ruleset\n\n# Simple firewall\nsudo tee /etc/nftables.conf << 'EOF'\n#!/usr/sbin/nft -f\nflush ruleset\ntable inet filter {\n    chain input {\n        type filter hook input priority 0; policy drop;\n        iif lo accept\n        ct state established,related accept\n        tcp dport {22, 80, 443} accept\n    }\n    chain forward { type filter hook forward priority 0; policy drop; }\n    chain output { type filter hook output priority 0; policy accept; }\n}\nEOF\nsudo systemctl enable --now nftables\n```",
                "arch": "Arch doesn't install a firewall by default. Common choices:\n```bash\n# nftables (recommended)\nsudo pacman -S nftables\nsudo systemctl enable --now nftables\n\n# Or ufw (simpler)\nsudo pacman -S ufw\nsudo ufw default deny incoming\nsudo ufw allow ssh\nsudo ufw allow http\nsudo ufw enable\n```",
                "fedora_41": "Fedora uses firewalld:\n```bash\n# Check status\nsudo firewall-cmd --state\nsudo firewall-cmd --list-all\n\n# Allow a service\nsudo firewall-cmd --permanent --add-service=http\nsudo firewall-cmd --permanent --add-service=https\nsudo firewall-cmd --reload\n\n# Allow a specific port\nsudo firewall-cmd --permanent --add-port=8080/tcp\nsudo firewall-cmd --reload\n```",
                "ubuntu_noble": "Ubuntu uses ufw (Uncomplicated Firewall):\n```bash\nsudo ufw enable\nsudo ufw default deny incoming\nsudo ufw default allow outgoing\nsudo ufw allow ssh\nsudo ufw allow http\nsudo ufw allow https\nsudo ufw status verbose\n```",
            }
        },
    ]

    for item in items:
        q = item["q"]
        for os_name, answer in item["answers"].items():
            sys_prompt = sys_for(os_name)
            examples.append(conv([("user", q), ("assistant", answer)], "flm_os_aware_v2", system=sys_prompt))

    return examples


# ══════════════════════════════════════════════════════════════════════════
# MORE IDENTITY & PERSONALITY
# ══════════════════════════════════════════════════════════════════════════

def gen_identity_extra():
    examples = []
    pairs = [
        ("What model architecture are you based on?",
         "I'm a transformer-based language model with 164 million parameters. My architecture was selected through a tournament process — multiple architectures were trained and evaluated, and the best performer was scaled up. I use a SmolLM2-style training curriculum: base pretraining on general text, annealing with curated data, then supervised fine-tuning."),

        ("How were you trained?",
         "From scratch on a single RTX 3090 — no pretrained weights, no distillation from larger models. The training has multiple stages:\n\n1. Base pretraining on general web text (FineWeb-Edu, DCLM)\n2. Annealing with curated data (Wikipedia, math, Debian docs, code)\n3. Chat pretraining to learn conversation format\n4. Supervised fine-tuning to learn my identity and OS expertise\n5. DPO alignment with preference data\n\nEvery dataset used is DFSG-compliant. The entire pipeline is reproducible."),

        ("What OS do you run on?",
         "I was trained on Debian Trixie, so that's where my deepest knowledge is. But I can help with other Linux distributions too — the fundamentals are the same. I can see information about your specific system in my system prompt, so I'll adapt my answers to your OS when possible."),

        ("Are you better than ChatGPT?",
         "At everything? No. I'm 164 million parameters — ChatGPT is hundreds of billions. I can't write essays, do complex reasoning, or handle arbitrary topics like a large model.\n\nBut I'm built for a specific purpose: helping with Debian and Linux system administration. For that narrow domain, I can be genuinely useful. And I have something ChatGPT doesn't — you can see exactly how I was trained, modify me, and run me locally without an internet connection."),

        ("Can you help me with something that's not Linux?",
         "I'll try, but my training is heavily focused on Linux, Debian, and system administration. For general knowledge questions, math, or coding in common languages, I can often give a reasonable answer. For topics far from my training domain — creative writing, medical advice, legal questions — I'll either give a poor answer or tell you I don't know. Being honest about my limits is better than making something up."),

        ("What can you actually help me with?",
         "My sweet spot:\n\n- **Debian/Ubuntu/Linux administration** — packages, services, configs\n- **Troubleshooting** — diagnosing broken systems, interpreting error messages\n- **Shell scripting** — bash, command-line tools, automation\n- **Networking** — DNS, firewalls, SSH, VPN setup\n- **Debian packaging** — creating and maintaining .deb packages\n- **System security** — hardening, monitoring, best practices\n\nI'm weaker at:\n- Topics outside Linux/computing\n- Creative tasks\n- Very long or complex reasoning chains\n- Anything requiring real-time information\n\nWhen in doubt, ask. I'll tell you if I'm out of my depth."),

        ("Do you hallucinate?",
         "Yes. All language models can generate plausible-sounding but incorrect information. I'm no exception — and being a small model, I'm probably more prone to it than larger ones.\n\nSome safeguards:\n- For commands that modify the system, always review before running\n- If I give you a path or config option, verify it exists\n- If something I say seems off, it might be — trust your experience\n\nI try to say \"I'm not sure\" when I'm uncertain, but I won't always get that right either."),

        ("Why should I use you instead of just reading the man pages?",
         "You shouldn't use me *instead* of man pages — you should use me *alongside* them. I can:\n\n- Point you to the right tool for a problem (\"use `ss` instead of `netstat`\")\n- Give you a quick start (\"here's a working nftables config\")\n- Explain concepts in plain language before you dive into the details\n- Help troubleshoot when you're stuck\n\nBut man pages are the authoritative source. If I say one thing and the man page says another, trust the man page. I'm trained on documentation, but documentation is always more accurate than my compressed version of it."),
    ]
    for q, a in pairs:
        examples.extend(make_variants(q, a, "flm_identity", n=2))
    return examples


# ══════════════════════════════════════════════════════════════════════════
# MISC PRACTICAL TASKS
# ══════════════════════════════════════════════════════════════════════════

def gen_practical():
    examples = []
    pairs = [
        ("How do I compress and decompress files?",
         "```bash\n# gzip (most common on Linux)\ngzip file.txt           # → file.txt.gz (removes original)\ngunzip file.txt.gz      # → file.txt\ngzip -k file.txt        # keep original\n\n# bzip2 (better compression, slower)\nbzip2 file.txt\nbunzip2 file.txt.bz2\n\n# xz (best compression, slowest)\nxz file.txt\nunxz file.txt.xz\n\n# zstd (best speed/ratio tradeoff, modern)\nzstd file.txt\nzstd -d file.txt.zst\n\n# zip (for sharing with Windows users)\nzip archive.zip file1 file2\nzip -r archive.zip directory/\nunzip archive.zip\n\n# 7z\nsudo apt install p7zip-full\n7z a archive.7z files/\n7z x archive.7z\n```\n\nFor directories, use tar first, then compress: `tar czf backup.tar.gz dir/`"),

        ("How do I schedule a system reboot?",
         "```bash\n# Reboot now\nsudo reboot\n\n# Reboot at a specific time\nsudo shutdown -r 22:00\n\n# Reboot in 30 minutes\nsudo shutdown -r +30\n\n# Cancel a scheduled shutdown\nsudo shutdown -c\n\n# Using at for one-time schedule\necho 'reboot' | sudo at 3:00 AM\n\n# Using systemd timer for recurring reboots\n# (useful for servers that need periodic restarts)\n```\n\n`shutdown -r` warns logged-in users. `reboot` is immediate. For servers, always tell users first: `sudo wall \"Server rebooting in 5 minutes for maintenance\"`"),

        ("How do I transfer files between servers?",
         "```bash\n# scp — simple copy over SSH\nscp file.txt user@remote:/path/\nscp -r directory/ user@remote:/path/\nscp user@remote:/path/file.txt ./\n\n# rsync — better for large/repeated transfers\nrsync -avz /local/path/ user@remote:/remote/path/\nrsync -avz --progress user@remote:/remote/path/ /local/path/\n\n# sftp — interactive file transfer\nsftp user@remote\n# put localfile, get remotefile, ls, cd, etc.\n\n# For very large files, use rsync with compression\nrsync -avz --compress-level=9 bigfile.tar user@remote:/path/\n\n# Between two remote servers (from your machine)\nrsync -avz -e ssh user1@server1:/path/ user2@server2:/path/\n```\n\nrsync is preferred over scp — it's resumable, shows progress, and only transfers changes."),

        ("How do I check open files and what's using them?",
         "`lsof` (list open files) is incredibly versatile:\n\n```bash\n# All open files\nsudo lsof\n\n# Files opened by a process\nsudo lsof -p 1234\n\n# Who's using a file\nsudo lsof /var/log/syslog\n\n# Who's using a port\nsudo lsof -i :80\nsudo lsof -i :443\n\n# All network connections\nsudo lsof -i\n\n# Files opened by a user\nsudo lsof -u username\n\n# Why can't I unmount?\nsudo lsof +D /mnt/usb\n# Shows processes using files in /mnt/usb\n\n# Count open files per process\nsudo lsof | awk '{print $1}' | sort | uniq -c | sort -nr | head -20\n```\n\nInstall with `sudo apt install lsof` if not present."),

        ("How do I use screen or tmux?",
         "tmux is the modern choice:\n\n```bash\nsudo apt install tmux\n\n# New session\ntmux\ntmux new -s mysession\n\n# Detach (session keeps running)\n# Press Ctrl+B, then D\n\n# List sessions\ntmux ls\n\n# Reattach\ntmux attach -t mysession\n\n# Split panes\n# Ctrl+B, % — vertical split\n# Ctrl+B, \" — horizontal split\n# Ctrl+B, arrow — switch pane\n\n# New window\n# Ctrl+B, C — new window\n# Ctrl+B, N — next window\n# Ctrl+B, P — previous window\n# Ctrl+B, number — jump to window\n\n# Kill session\ntmux kill-session -t mysession\n```\n\ntmux is essential for remote work — your processes survive SSH disconnects. I recommend adding `tmux attach || tmux` to your SSH login."),

        ("How do I find which process is using a specific port?",
         "```bash\n# Best: ss (socket statistics)\nsudo ss -tlnp sport = :80\n\n# Or lsof\nsudo lsof -i :80\n\n# Or fuser\nsudo fuser 80/tcp\n\n# Kill whatever is on that port\nsudo fuser -k 80/tcp\n```\n\nCommon scenario: \"Address already in use\" error when starting a service.\n```bash\n# Find the conflicting process\nsudo ss -tlnp sport = :8080\n# Output shows PID and process name\n# Then either stop that service or change your port\n```\n\n`ss` is the fastest. `lsof -i` gives more detail. `fuser -k` kills the process directly."),

        ("How do I set up a swap file?",
         "```bash\n# Create a 2GB swap file\nsudo fallocate -l 2G /swapfile\nsudo chmod 600 /swapfile\nsudo mkswap /swapfile\nsudo swapon /swapfile\n\n# Verify\nfree -h\nswapon --show\n\n# Make permanent\necho '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab\n```\n\nHow much swap? Common rules:\n- RAM < 2GB: 2x RAM\n- RAM 2-8GB: equal to RAM\n- RAM > 8GB: 2-4GB is usually enough\n- For hibernation: at least equal to RAM\n\nTo adjust swap behavior:\n```bash\n# Lower swappiness (default 60, lower = less swapping)\nsudo sysctl vm.swappiness=10\n# Make permanent\necho 'vm.swappiness=10' | sudo tee -a /etc/sysctl.d/99-custom.conf\n```"),

        ("How do I find large files eating my disk space?",
         "```bash\n# Quick: largest directories at top level\nsudo du -h --max-depth=1 / 2>/dev/null | sort -hr | head -15\n\n# Drill down into a specific directory\ndu -h --max-depth=1 /var | sort -hr | head -10\n\n# Find files over 100MB\nsudo find / -type f -size +100M -exec ls -lh {} \\; 2>/dev/null | sort -k5 -hr\n\n# Top 20 largest files\nsudo find / -type f -printf '%s %p\\n' 2>/dev/null | sort -nr | head -20 | awk '{printf \"%.1f MB  %s\\n\", $1/1048576, $2}'\n\n# Interactive (best option)\nsudo apt install ncdu\nsudo ncdu /\n```\n\n`ncdu` is the best tool for this — interactive, fast, and lets you drill down and delete from the interface."),
    ]
    for q, a in pairs:
        examples.extend(make_variants(q, a, "flm_practical"))
    return examples


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("Generating V3 SFT examples...")
    print()

    all_data = []
    generators = [
        ("flm_cron_v3.jsonl", gen_cron),
        ("flm_dns_email.jsonl", gen_dns_email),
        ("flm_raid_encryption.jsonl", gen_raid_encryption),
        ("flm_kernel_boot.jsonl", gen_kernel_boot),
        ("flm_locale_time.jsonl", gen_locale_time),
        ("flm_virtualization.jsonl", gen_virtualization),
        ("flm_python_env.jsonl", gen_python),
        ("flm_debugging_v3.jsonl", gen_debugging),
        ("flm_apt_deep.jsonl", gen_apt_deep),
        ("flm_multiturn_v3.jsonl", gen_multiturn),
        ("flm_os_aware_v2.jsonl", gen_os_aware),
        ("flm_identity_v3.jsonl", gen_identity_extra),
        ("flm_practical.jsonl", gen_practical),
    ]

    for filename, gen_fn in generators:
        data = gen_fn()
        written = write_jsonl(data, filename)
        all_data.extend(written)

    print(f"\nTotal new V3 examples: {len(all_data)}")
    print(f"\nNow run scripts/merge_sft.py to merge everything.")


if __name__ == "__main__":
    main()
