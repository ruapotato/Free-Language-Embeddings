#!/usr/bin/env python3
"""Generate additional SFT examples covering gaps in existing data.

Covers: networking, systemd, filesystems, security, shell scripting,
process management, text processing, log analysis, backup/recovery,
web servers, performance, and more Debian knowledge.

Uses standardized system prompts from system_prompt.py.
"""
import json
import random
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from system_prompt import random_system_prompt, PRESETS, get_preset_prompt

random.seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sft")


def pick_system():
    """Pick a random system prompt using the standardized format."""
    prompt = random_system_prompt(include_none=True, short_chance=0.2)
    if prompt is None:
        return ""
    return f"<|system|>\n{prompt}\n"


def pick_system_for_os(preset_name):
    """Pick a system prompt for a specific OS."""
    short = random.random() < 0.2
    prompt = get_preset_prompt(preset_name, short=short)
    return f"<|system|>\n{prompt}\n"


def conv(turns, source, system=None):
    """Build a conversation dict."""
    sys_block = system if system is not None else pick_system()
    parts = [sys_block] if sys_block else []
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


# ── Networking ────────────────────────────────────────────────────────────

def gen_networking():
    examples = []

    qa_pairs = [
        ("How do I see my IP address?",
         "Use `ip addr show` (or `ip a` for short). It lists all interfaces with their addresses.\n\nFor just the default interface's IPv4:\n```\nip -4 addr show scope global\n```\n\nThe older `ifconfig` still works if net-tools is installed, but `ip` is the modern standard on Debian."),

        ("How do I check what ports are listening on my system?",
         "Use `ss` (socket statistics):\n```\nss -tlnp\n```\n\nFlags: `-t` TCP, `-l` listening, `-n` numeric ports, `-p` show process. You may need sudo for process names.\n\nThe older equivalent is `netstat -tlnp` from the net-tools package."),

        ("How do I configure a static IP on Debian?",
         "Edit `/etc/network/interfaces`:\n```\nauto eth0\niface eth0 inet static\n    address 192.168.1.100/24\n    gateway 192.168.1.1\n    dns-nameservers 8.8.8.8 1.1.1.1\n```\n\nThen restart networking:\n```\nsudo systemctl restart networking\n```\n\nOn systems using NetworkManager, use `nmcli` instead:\n```\nnmcli con mod \"Wired connection 1\" ipv4.addresses 192.168.1.100/24\nnmcli con mod \"Wired connection 1\" ipv4.gateway 192.168.1.1\nnmcli con mod \"Wired connection 1\" ipv4.method manual\nnmcli con up \"Wired connection 1\"\n```"),

        ("How do I set up a basic firewall with nftables?",
         "Debian Trixie uses nftables by default. Create `/etc/nftables.conf`:\n```\n#!/usr/sbin/nft -f\nflush ruleset\n\ntable inet filter {\n    chain input {\n        type filter hook input priority 0; policy drop;\n        iif lo accept\n        ct state established,related accept\n        tcp dport 22 accept\n        tcp dport { 80, 443 } accept\n        counter drop\n    }\n    chain forward {\n        type filter hook forward priority 0; policy drop;\n    }\n    chain output {\n        type filter hook output priority 0; policy accept;\n    }\n}\n```\n\nEnable it:\n```\nsudo systemctl enable --now nftables\n```"),

        ("What's the difference between iptables and nftables?",
         "nftables replaces iptables as the default packet filtering framework since Debian Buster. Key differences:\n\n- **Syntax**: nftables uses a cleaner, more consistent syntax\n- **Performance**: nftables is faster for complex rulesets (uses maps and sets natively)\n- **One tool**: `nft` replaces iptables, ip6tables, arptables, and ebtables\n- **Atomicity**: nftables can apply entire rulesets atomically\n\nDebian still provides `iptables` as a compatibility layer (`iptables-nft` translates to nftables backend). For new setups, use nftables directly."),

        ("How do I check DNS resolution?",
         "Several tools:\n\n```bash\n# Quick lookup\nhost example.com\n\n# Detailed query\ndig example.com\n\n# Check which DNS server is being used\ndig example.com +short\nresolvectl status  # if using systemd-resolved\n\n# Check /etc/resolv.conf for configured nameservers\ncat /etc/resolv.conf\n```\n\nIf DNS isn't working, check if `systemd-resolved` is running or if `/etc/resolv.conf` points to the right nameserver."),

        ("How do I test network connectivity to a specific port?",
         "Several ways:\n\n```bash\n# TCP connection test\nnc -zv example.com 443\n\n# Or with timeout\ntimeout 5 bash -c 'echo > /dev/tcp/example.com/443 && echo open || echo closed'\n\n# curl for HTTP(S)\ncurl -I https://example.com\n\n# traceroute to see the path\ntraceroute example.com\n```\n\n`nc` (netcat) is the most versatile. Install with `apt install netcat-openbsd`."),

        ("How do I see current network connections?",
         "```bash\n# All established connections\nss -tnp\n\n# All connections including listening\nss -tanp\n\n# Summary statistics\nss -s\n\n# Filter by port\nss -tnp sport = :80\n\n# Filter by state\nss -t state established\n```\n\n`ss` is faster than the old `netstat`. Add `-u` for UDP too."),

        ("How do I set up SSH port forwarding?",
         "Three types:\n\n**Local forwarding** — access remote service through local port:\n```bash\nssh -L 8080:localhost:80 user@remote\n# Now localhost:8080 → remote's localhost:80\n```\n\n**Remote forwarding** — expose local service on remote:\n```bash\nssh -R 9090:localhost:3000 user@remote\n# Remote's localhost:9090 → your localhost:3000\n```\n\n**Dynamic (SOCKS proxy)**:\n```bash\nssh -D 1080 user@remote\n# localhost:1080 is now a SOCKS5 proxy through remote\n```\n\nAdd `-N` to not open a shell, `-f` to background it."),

        ("How do I bond two network interfaces?",
         "Install `ifenslave` and configure in `/etc/network/interfaces`:\n\n```\nauto bond0\niface bond0 inet static\n    address 192.168.1.100/24\n    gateway 192.168.1.1\n    bond-slaves eth0 eth1\n    bond-mode 802.3ad\n    bond-miimon 100\n    bond-lacp-rate 1\n```\n\nCommon bond modes:\n- `balance-rr` (0): Round-robin\n- `active-backup` (1): Failover only\n- `802.3ad` (4): LACP (requires switch support)\n\nLoad the module: `sudo modprobe bonding`"),

        ("How do I capture network traffic?",
         "Use `tcpdump`:\n\n```bash\n# All traffic on eth0\nsudo tcpdump -i eth0\n\n# Only HTTP traffic\nsudo tcpdump -i eth0 port 80\n\n# Save to file for Wireshark analysis\nsudo tcpdump -i eth0 -w capture.pcap\n\n# Specific host\nsudo tcpdump -i eth0 host 192.168.1.50\n\n# DNS queries only\nsudo tcpdump -i eth0 port 53\n```\n\nInstall with `apt install tcpdump`. For GUI analysis, use Wireshark (`apt install wireshark`)."),

        ("How do I see my routing table?",
         "```bash\n# Modern way\nip route show\n\n# Default route only\nip route show default\n\n# Add a static route\nsudo ip route add 10.0.0.0/8 via 192.168.1.1\n\n# Delete a route\nsudo ip route del 10.0.0.0/8\n```\n\nTo make routes persistent across reboots, add them in `/etc/network/interfaces` with `up ip route add ...` or use a script in `/etc/network/if-up.d/`."),

        ("How do I configure VLANs on Debian?",
         "Install `vlan` package, then configure in `/etc/network/interfaces`:\n\n```\nauto eth0.100\niface eth0.100 inet static\n    address 10.100.0.1/24\n    vlan-raw-device eth0\n```\n\nOr with `ip` directly:\n```bash\nsudo ip link add link eth0 name eth0.100 type vlan id 100\nsudo ip addr add 10.100.0.1/24 dev eth0.100\nsudo ip link set eth0.100 up\n```\n\nMake sure the `8021q` kernel module is loaded: `sudo modprobe 8021q`"),
    ]

    for q, a in qa_pairs:
        # Each pair gets 2-3 variants with different system prompts
        examples.append(conv([("user", q), ("assistant", a)], "flm_networking"))
        examples.append(conv([("user", q), ("assistant", a)], "flm_networking"))

    # Multi-turn networking examples
    multi_turns = [
        [
            ("user", "I can ping local machines but can't reach the internet"),
            ("assistant", "Sounds like a gateway or DNS issue. Let's check:\n\n```bash\nip route show default\n```\n\nDo you see a default route? What gateway is listed?"),
            ("user", "it shows default via 192.168.1.1 dev eth0"),
            ("assistant", "Gateway looks fine. Can you ping it?\n\n```bash\nping -c 3 192.168.1.1\n```\n\nAlso try pinging an external IP directly (bypasses DNS):\n```bash\nping -c 3 8.8.8.8\n```"),
            ("user", "can ping the gateway but not 8.8.8.8"),
            ("assistant", "So your local network works but traffic doesn't leave it. This is usually one of:\n\n1. **Router not forwarding** — check your router's WAN connection\n2. **Firewall blocking outbound** — check `nft list ruleset` for drop rules on the output or forward chain\n3. **ISP issue** — try from another device\n\nCheck your firewall first:\n```bash\nsudo nft list ruleset\n```\n\nIf the output chain has `policy drop`, that's your problem — it's blocking outbound traffic."),
        ],
        [
            ("user", "how do I set up a wireguard VPN on debian?"),
            ("assistant", "Install WireGuard:\n```bash\nsudo apt install wireguard\n```\n\nGenerate keys:\n```bash\nwg genkey | tee privatekey | wg pubkey > publickey\n```\n\nCreate `/etc/wireguard/wg0.conf`:\n```ini\n[Interface]\nPrivateKey = <your-private-key>\nAddress = 10.0.0.1/24\nListenPort = 51820\n\n[Peer]\nPublicKey = <peer-public-key>\nAllowedIPs = 10.0.0.2/32\n```\n\nStart it:\n```bash\nsudo wg-quick up wg0\nsudo systemctl enable wg-quick@wg0\n```"),
            ("user", "how do I add another peer?"),
            ("assistant", "Add another `[Peer]` block to `/etc/wireguard/wg0.conf`:\n\n```ini\n[Peer]\nPublicKey = <new-peer-public-key>\nAllowedIPs = 10.0.0.3/32\n```\n\nThen reload without dropping existing connections:\n```bash\nsudo wg syncconf wg0 <(wg-quick strip wg0)\n```\n\nOn the new peer's side, their config needs your public key and endpoint:\n```ini\n[Interface]\nPrivateKey = <their-private-key>\nAddress = 10.0.0.3/24\n\n[Peer]\nPublicKey = <your-public-key>\nEndpoint = your.server.ip:51820\nAllowedIPs = 10.0.0.0/24\nPersistentKeepalive = 25\n```"),
        ],
    ]

    for turns in multi_turns:
        examples.append(conv(turns, "flm_networking"))

    return examples


# ── Systemd ───────────────────────────────────────────────────────────────

def gen_systemd():
    examples = []

    qa_pairs = [
        ("How do I create a custom systemd service?",
         "Create a unit file in `/etc/systemd/system/myapp.service`:\n\n```ini\n[Unit]\nDescription=My Application\nAfter=network.target\n\n[Service]\nType=simple\nUser=myapp\nWorkingDirectory=/opt/myapp\nExecStart=/opt/myapp/run.sh\nRestart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=multi-user.target\n```\n\nThen:\n```bash\nsudo systemctl daemon-reload\nsudo systemctl enable --now myapp\nsudo systemctl status myapp\n```"),

        ("How do I read systemd journal logs?",
         "```bash\n# All logs\njournalctl\n\n# Follow (like tail -f)\njournalctl -f\n\n# Specific unit\njournalctl -u nginx\n\n# Since last boot\njournalctl -b\n\n# Since a time\njournalctl --since \"2024-01-15 10:00\" --until \"2024-01-15 12:00\"\n\n# Only errors and above\njournalctl -p err\n\n# Kernel messages only\njournalctl -k\n\n# Disk usage\njournalctl --disk-usage\n\n# Trim to 500MB\nsudo journalctl --vacuum-size=500M\n```"),

        ("How do I analyze boot time with systemd?",
         "```bash\n# Overall boot time\nsystemd-analyze\n\n# Blame — slowest units\nsystemd-analyze blame\n\n# Critical chain — what blocked boot\nsystemd-analyze critical-chain\n\n# Generate SVG plot\nsystemd-analyze plot > boot.svg\n```\n\nTo disable slow services you don't need:\n```bash\nsudo systemctl disable slow-service\nsudo systemctl mask slow-service  # prevent any activation\n```"),

        ("What's the difference between systemctl enable and start?",
         "`start` runs the service now. `enable` configures it to start on boot. They're independent:\n\n```bash\nsudo systemctl start nginx    # run now, but won't survive reboot\nsudo systemctl enable nginx   # start on boot, but not right now\nsudo systemctl enable --now nginx  # both at once\n```\n\nSimilarly, `stop` stops it now and `disable` prevents auto-start on boot."),

        ("How do I create a systemd timer instead of a cron job?",
         "Create two files. First the timer `/etc/systemd/system/backup.timer`:\n\n```ini\n[Unit]\nDescription=Daily backup timer\n\n[Timer]\nOnCalendar=daily\nPersistent=true\n\n[Install]\nWantedBy=timers.target\n```\n\nThen the service `/etc/systemd/system/backup.service`:\n\n```ini\n[Unit]\nDescription=Daily backup\n\n[Service]\nType=oneshot\nExecStart=/usr/local/bin/backup.sh\n```\n\nEnable the timer:\n```bash\nsudo systemctl enable --now backup.timer\nsystemctl list-timers  # verify\n```\n\nAdvantages over cron: logging via journal, dependency management, resource controls."),

        ("How do I limit resources for a systemd service?",
         "Add resource limits in the `[Service]` section:\n\n```ini\n[Service]\n# CPU\nCPUQuota=50%\n\n# Memory\nMemoryMax=512M\nMemoryHigh=400M\n\n# IO\nIOWeight=50\n\n# File descriptors\nLimitNOFILE=65536\n\n# Processes\nTasksMax=100\n\n# Prevent network access\nPrivateNetwork=true\n\n# Read-only filesystem\nProtectSystem=strict\nReadWritePaths=/var/lib/myapp\n```\n\nApply changes with `sudo systemctl daemon-reload && sudo systemctl restart myapp`."),

        ("How do I check why a systemd service failed?",
         "```bash\n# Status with recent logs\nsystemctl status myapp\n\n# Full logs\njournalctl -u myapp --no-pager\n\n# Just the last failure\njournalctl -u myapp -b -p err\n\n# Show unit file (check for typos)\nsystemctl cat myapp\n\n# Check dependencies\nsystemctl list-dependencies myapp\n```\n\nCommon causes: wrong `ExecStart` path, missing `User`, wrong permissions, dependency not started."),

        ("How do I set environment variables for a systemd service?",
         "Several ways:\n\n**In the unit file:**\n```ini\n[Service]\nEnvironment=\"DB_HOST=localhost\" \"DB_PORT=5432\"\n```\n\n**From a file:**\n```ini\n[Service]\nEnvironmentFile=/etc/myapp/env\n```\n\nWhere `/etc/myapp/env` contains:\n```\nDB_HOST=localhost\nDB_PORT=5432\n```\n\n**Override for existing services:**\n```bash\nsudo systemctl edit myapp\n```\nThis creates a drop-in that won't be overwritten by package updates."),

        ("What's a systemd socket activation?",
         "Socket activation means systemd listens on a port and only starts the service when a connection arrives. This saves resources for rarely-used services.\n\nCreate `/etc/systemd/system/myapp.socket`:\n```ini\n[Unit]\nDescription=My App Socket\n\n[Socket]\nListenStream=8080\nAccept=false\n\n[Install]\nWantedBy=sockets.target\n```\n\nThe service gets the socket file descriptor passed to it. Enable the socket (not the service):\n```bash\nsudo systemctl enable --now myapp.socket\n```\n\nSSH on Debian uses this — `sshd.socket` starts `sshd@.service` per connection."),
    ]

    for q, a in qa_pairs:
        examples.append(conv([("user", q), ("assistant", a)], "flm_systemd"))
        examples.append(conv([("user", q), ("assistant", a)], "flm_systemd"))

    return examples


# ── Filesystems & Disks ──────────────────────────────────────────────────

def gen_filesystems():
    examples = []

    qa_pairs = [
        ("How do I check disk usage?",
         "```bash\n# Filesystem usage summary\ndf -h\n\n# Directory sizes\ndu -sh /var/log/*\n\n# Top 10 largest directories\ndu -h --max-depth=1 / 2>/dev/null | sort -hr | head -10\n\n# Find files over 100MB\nfind / -type f -size +100M -exec ls -lh {} \\; 2>/dev/null\n```\n\n`ncdu` is an excellent interactive alternative: `apt install ncdu && ncdu /`"),

        ("How do I partition a disk on Debian?",
         "Use `fdisk` for MBR or `gdisk`/`parted` for GPT:\n\n```bash\n# List disks\nlsblk\n\n# Partition with fdisk (interactive)\nsudo fdisk /dev/sdb\n# n=new partition, p=primary, w=write\n\n# Or non-interactive with parted\nsudo parted /dev/sdb mklabel gpt\nsudo parted /dev/sdb mkpart primary ext4 0% 100%\n\n# Create filesystem\nsudo mkfs.ext4 /dev/sdb1\n\n# Mount it\nsudo mkdir -p /mnt/data\nsudo mount /dev/sdb1 /mnt/data\n```\n\nAdd to `/etc/fstab` for persistent mounting."),

        ("How do I add an entry to fstab?",
         "Format: `device mountpoint fstype options dump pass`\n\n```bash\n# Get the UUID (preferred over /dev/sdX names)\nblkid /dev/sdb1\n\n# Add to /etc/fstab\nUUID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx /mnt/data ext4 defaults 0 2\n```\n\nCommon options:\n- `defaults` — rw, suid, dev, exec, auto, nouser, async\n- `noatime` — don't update access times (better performance)\n- `nofail` — don't fail boot if device is missing\n\nTest before rebooting:\n```bash\nsudo mount -a  # mount all fstab entries\n```"),

        ("How do I set up LVM?",
         "LVM adds a layer between physical disks and filesystems:\n\n```bash\n# Install\nsudo apt install lvm2\n\n# 1. Create physical volumes\nsudo pvcreate /dev/sdb /dev/sdc\n\n# 2. Create volume group\nsudo vgcreate myvg /dev/sdb /dev/sdc\n\n# 3. Create logical volume (use 80% of space)\nsudo lvcreate -l 80%FREE -n data myvg\n\n# 4. Format and mount\nsudo mkfs.ext4 /dev/myvg/data\nsudo mount /dev/myvg/data /mnt/data\n```\n\nLater, you can grow it live:\n```bash\nsudo lvextend -l +100%FREE /dev/myvg/data\nsudo resize2fs /dev/myvg/data\n```"),

        ("How do I check disk health with SMART?",
         "```bash\n# Install\nsudo apt install smartmontools\n\n# Quick check\nsudo smartctl -H /dev/sda\n\n# Full info\nsudo smartctl -a /dev/sda\n\n# Start a short self-test\nsudo smartctl -t short /dev/sda\n\n# Check test results\nsudo smartctl -l selftest /dev/sda\n```\n\nKey attributes to watch:\n- `Reallocated_Sector_Ct` — bad sectors remapped (>0 means wear)\n- `Current_Pending_Sector` — sectors waiting to be remapped\n- `UDMA_CRC_Error_Count` — cable/connection issues\n\nEnable automatic monitoring: `sudo systemctl enable --now smartd`"),

        ("How do I resize a partition?",
         "For ext4 on LVM — easiest, can do online:\n```bash\nsudo lvextend -L +10G /dev/myvg/data\nsudo resize2fs /dev/myvg/data\n```\n\nFor ext4 on a regular partition — must unmount first:\n```bash\nsudo umount /dev/sdb1\nsudo e2fsck -f /dev/sdb1\nsudo resize2fs /dev/sdb1 50G\n# Then resize partition with fdisk/parted\n```\n\n**Warning**: Always backup before resizing. Shrinking is riskier than growing. Growing an ext4 filesystem can be done online, shrinking cannot."),

        ("What filesystem should I use on Debian?",
         "**ext4** — default, reliable, mature. Best for most use cases.\n\n**xfs** — better for very large files, good parallel I/O. Can't be shrunk.\n\n**btrfs** — snapshots, compression, checksums. Good for desktops/laptops but less battle-tested for databases.\n\n**zfs** — enterprise features but not in Debian main (license incompatible with GPL). Available via contrib.\n\n**tmpfs** — RAM-backed, auto-sized. Already used for `/tmp` and `/run`.\n\nFor a Debian server, ext4 is the safe choice. Use XFS for large data volumes."),

        ("How do I mount an NFS share?",
         "```bash\n# Install client\nsudo apt install nfs-common\n\n# Mount manually\nsudo mount -t nfs server:/shared /mnt/nfs\n\n# Persistent in fstab\nserver:/shared /mnt/nfs nfs defaults,nofail 0 0\n\n# With autofs (mount on demand)\nsudo apt install autofs\n# Add to /etc/auto.master:\n/mnt/nfs  /etc/auto.nfs\n# Create /etc/auto.nfs:\nshared  -rw,soft  server:/shared\nsudo systemctl restart autofs\n```\n\nNFS v4 is preferred: `mount -t nfs4 server:/shared /mnt/nfs`"),
    ]

    for q, a in qa_pairs:
        examples.append(conv([("user", q), ("assistant", a)], "flm_filesystems"))
        examples.append(conv([("user", q), ("assistant", a)], "flm_filesystems"))

    return examples


# ── Security ──────────────────────────────────────────────────────────────

def gen_security():
    examples = []

    qa_pairs = [
        ("How do I harden SSH on my Debian server?",
         "Edit `/etc/ssh/sshd_config`:\n\n```\n# Disable root login\nPermitRootLogin no\n\n# Disable password auth (use keys only)\nPasswordAuthentication no\n\n# Use only SSH protocol 2\nProtocol 2\n\n# Limit to specific users\nAllowUsers deploy admin\n\n# Change default port (optional)\nPort 2222\n\n# Idle timeout\nClientAliveInterval 300\nClientAliveCountMax 2\n```\n\nRestart: `sudo systemctl restart sshd`\n\n**Important**: Make sure you have key-based access working before disabling password auth, or you'll lock yourself out."),

        ("How do I set up SSH key authentication?",
         "On your local machine:\n```bash\n# Generate key pair\nssh-keygen -t ed25519 -C \"your@email\"\n\n# Copy public key to server\nssh-copy-id user@server\n\n# Or manually\ncat ~/.ssh/id_ed25519.pub | ssh user@server 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'\n```\n\nOn the server, ensure permissions:\n```bash\nchmod 700 ~/.ssh\nchmod 600 ~/.ssh/authorized_keys\n```\n\ned25519 is preferred over RSA — shorter keys, faster, more secure."),

        ("How do I set up fail2ban?",
         "```bash\nsudo apt install fail2ban\n```\n\nCreate `/etc/fail2ban/jail.local`:\n```ini\n[DEFAULT]\nbantime = 1h\nfindtime = 10m\nmaxretry = 5\n\n[sshd]\nenabled = true\nport = ssh\nfilter = sshd\nlogpath = /var/log/auth.log\nmaxretry = 3\n```\n\nStart it:\n```bash\nsudo systemctl enable --now fail2ban\n```\n\nCheck status:\n```bash\nsudo fail2ban-client status sshd\n```\n\nUnban an IP: `sudo fail2ban-client set sshd unbanip 192.168.1.50`"),

        ("How do I manage file permissions and ownership?",
         "```bash\n# Change owner\nchown user:group file\nchown -R user:group directory/\n\n# Change permissions\nchmod 644 file      # rw-r--r--\nchmod 755 directory  # rwxr-xr-x\nchmod u+x script.sh  # add execute for owner\n\n# Octal: owner-group-other\n# 7=rwx, 6=rw-, 5=r-x, 4=r--, 0=---\n\n# Special permissions\nchmod u+s binary     # setuid — run as owner\nchmod g+s directory   # setgid — new files inherit group\nchmod +t /tmp         # sticky bit — only owner can delete\n```\n\nUse `ls -la` to view permissions. The sticky bit on /tmp is why users can't delete each other's temp files."),

        ("How do I configure sudo access?",
         "Use `visudo` to safely edit `/etc/sudoers`:\n```bash\nsudo visudo\n```\n\nCommon patterns:\n```\n# Full sudo access\nusername ALL=(ALL:ALL) ALL\n\n# Sudo without password\nusername ALL=(ALL) NOPASSWD: ALL\n\n# Specific commands only\nusername ALL=(ALL) /usr/bin/systemctl restart nginx, /usr/bin/apt update\n\n# Group-based (preferred)\n%admin ALL=(ALL:ALL) ALL\n```\n\nBetter practice: add files in `/etc/sudoers.d/` instead of editing the main file:\n```bash\necho 'username ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart nginx' | sudo tee /etc/sudoers.d/username\nsudo chmod 440 /etc/sudoers.d/username\n```"),

        ("How do I check for rootkits?",
         "Two main tools:\n\n```bash\n# rkhunter\nsudo apt install rkhunter\nsudo rkhunter --update\nsudo rkhunter --check\n\n# chkrootkit\nsudo apt install chkrootkit\nsudo chkrootkit\n```\n\nAlso useful:\n```bash\n# Check for unusual SUID binaries\nfind / -perm -4000 -type f 2>/dev/null\n\n# Check listening services\nss -tlnp\n\n# Check running processes\nps auxf\n\n# Check for unusual cron jobs\nfor user in $(cut -f1 -d: /etc/passwd); do crontab -l -u $user 2>/dev/null; done\n```\n\nPrevention is better than detection. Keep packages updated: `sudo apt update && sudo apt upgrade`"),

        ("How do I set up automatic security updates on Debian?",
         "```bash\nsudo apt install unattended-upgrades\nsudo dpkg-reconfigure -plow unattended-upgrades\n```\n\nThis enables automatic installation of security updates. The config is in `/etc/apt/apt.conf.d/50unattended-upgrades`.\n\nKey settings:\n```\nUnattended-Upgrade::Allowed-Origins {\n    \"${distro_id}:${distro_codename}-security\";\n};\nUnattended-Upgrade::Automatic-Reboot \"false\";\nUnattended-Upgrade::Mail \"admin@example.com\";\n```\n\nCheck logs: `cat /var/log/unattended-upgrades/unattended-upgrades.log`"),

        ("How do I use AppArmor on Debian?",
         "AppArmor is installed and enabled by default on Debian:\n\n```bash\n# Check status\nsudo aa-status\n\n# List profiles\nsudo aa-status --profiles\n\n# Set a profile to complain mode (log but don't block)\nsudo aa-complain /usr/sbin/nginx\n\n# Set to enforce mode\nsudo aa-enforce /usr/sbin/nginx\n\n# Disable a profile\nsudo ln -s /etc/apparmor.d/usr.sbin.nginx /etc/apparmor.d/disable/\nsudo apparmor_parser -R /etc/apparmor.d/usr.sbin.nginx\n```\n\nProfiles live in `/etc/apparmor.d/`. Use `aa-genprof` to generate a new profile interactively."),
    ]

    for q, a in qa_pairs:
        examples.append(conv([("user", q), ("assistant", a)], "flm_security"))
        examples.append(conv([("user", q), ("assistant", a)], "flm_security"))

    return examples


# ── Shell Scripting ───────────────────────────────────────────────────────

def gen_shell_scripting():
    examples = []

    qa_pairs = [
        ("How do I write a basic bash script?",
         "```bash\n#!/bin/bash\nset -euo pipefail  # Exit on error, undefined vars, pipe failures\n\n# Variables\nNAME=\"world\"\necho \"Hello, $NAME\"\n\n# Conditionals\nif [[ -f /etc/debian_version ]]; then\n    echo \"This is Debian $(cat /etc/debian_version)\"\nfi\n\n# Loops\nfor pkg in nginx curl git; do\n    if ! dpkg -l \"$pkg\" &>/dev/null; then\n        echo \"$pkg not installed\"\n    fi\ndone\n```\n\nMake it executable: `chmod +x script.sh`\n\n`set -euo pipefail` is the most important line — it catches most common scripting errors."),

        ("What does set -euo pipefail do?",
         "Three safety settings combined:\n\n- **`-e`** (errexit): Exit immediately if any command fails\n- **`-u`** (nounset): Treat undefined variables as errors\n- **`-o pipefail`**: A pipeline fails if ANY command in it fails (not just the last)\n\nWithout these, bash silently ignores errors:\n```bash\n# Without -e: continues after failure\ncd /nonexistent\nrm -rf *  # runs in wrong directory!\n\n# Without -u: empty string\necho \"Deleting $UNSET_VAR\"  # could be dangerous\n\n# Without pipefail:\nfailing_cmd | grep something  # reports success if grep succeeds\n```\n\nAlways start scripts with `set -euo pipefail`."),

        ("How do I handle command-line arguments in bash?",
         "Several approaches:\n\n**Positional arguments:**\n```bash\n#!/bin/bash\necho \"Script: $0\"\necho \"First arg: $1\"\necho \"All args: $@\"\necho \"Arg count: $#\"\n```\n\n**With getopts (flags):**\n```bash\n#!/bin/bash\nVERBOSE=0\nOUTPUT=\"\"\n\nwhile getopts \"vo:h\" opt; do\n    case $opt in\n        v) VERBOSE=1 ;;\n        o) OUTPUT=\"$OPTARG\" ;;\n        h) echo \"Usage: $0 [-v] [-o file]\"; exit 0 ;;\n        *) exit 1 ;;\n    esac\ndone\nshift $((OPTIND - 1))  # remaining args in $@\n```\n\nUsage: `./script.sh -v -o output.txt file1 file2`"),

        ("How do I use arrays in bash?",
         "```bash\n# Declare\nfruits=(apple banana cherry)\n\n# Access\necho \"${fruits[0]}\"     # apple\necho \"${fruits[@]}\"     # all elements\necho \"${#fruits[@]}\"    # count: 3\n\n# Add\nfruits+=(date)\n\n# Loop\nfor fruit in \"${fruits[@]}\"; do\n    echo \"$fruit\"\ndone\n\n# Associative arrays (bash 4+)\ndeclare -A colors\ncolors[red]=\"#ff0000\"\ncolors[blue]=\"#0000ff\"\necho \"${colors[red]}\"\n\n# Keys\necho \"${!colors[@]}\"\n```\n\n**Important**: Always quote `\"${array[@]}\"` to handle elements with spaces correctly."),

        ("How do I do string manipulation in bash?",
         "```bash\nSTR=\"Hello, World!\"\n\n# Length\necho ${#STR}          # 13\n\n# Substring\necho ${STR:7}         # World!\necho ${STR:7:5}       # World\n\n# Replace\necho ${STR/World/Bash}  # Hello, Bash!\necho ${STR//l/L}        # HeLLo, WorLd! (all occurrences)\n\n# Remove prefix/suffix\nFILE=\"/path/to/file.tar.gz\"\necho ${FILE##*/}      # file.tar.gz (basename)\necho ${FILE%.*}       # /path/to/file.tar (remove shortest suffix)\necho ${FILE%%.*}      # /path/to/file (remove longest suffix)\n\n# Case conversion (bash 4+)\necho ${STR,,}         # hello, world!\necho ${STR^^}         # HELLO, WORLD!\n```"),

        ("How do I redirect output in bash?",
         "```bash\n# stdout to file\ncommand > file.txt        # overwrite\ncommand >> file.txt       # append\n\n# stderr to file\ncommand 2> error.log\n\n# Both stdout and stderr\ncommand > all.log 2>&1    # traditional\ncommand &> all.log        # bash shorthand\n\n# stderr only (suppress stdout)\ncommand > /dev/null 2>&1  # both to nowhere\ncommand 2>&1 >/dev/null   # only stderr to stdout\n\n# Pipe stderr\ncommand 2>&1 | grep error\n\n# Process substitution\ndiff <(sort file1) <(sort file2)\n\n# Here document\ncat <<EOF\nMultiple\nlines\nEOF\n```"),

        ("How do I write a bash function?",
         "```bash\n# Simple function\ngreet() {\n    local name=\"$1\"  # local variable\n    echo \"Hello, $name\"\n}\ngreet \"World\"\n\n# With return value (exit code)\nis_installed() {\n    dpkg -l \"$1\" &>/dev/null\n}\n\nif is_installed nginx; then\n    echo \"nginx is installed\"\nfi\n\n# Returning strings (via stdout)\nget_ip() {\n    ip -4 addr show scope global | grep -oP 'inet \\K[\\d.]+' | head -1\n}\nMY_IP=$(get_ip)\n\n# Error handling\ndie() {\n    echo \"ERROR: $*\" >&2\n    exit 1\n}\n[[ -f config.yml ]] || die \"config.yml not found\"\n```\n\nUse `local` for variables to avoid polluting the global scope."),

        ("How do I use traps in bash for cleanup?",
         "```bash\n#!/bin/bash\nset -euo pipefail\n\nTMPDIR=$(mktemp -d)\n\ncleanup() {\n    rm -rf \"$TMPDIR\"\n    echo \"Cleaned up $TMPDIR\"\n}\ntrap cleanup EXIT  # runs on any exit (normal, error, signal)\n\n# Now use TMPDIR freely — it'll always be cleaned up\ncp important_file \"$TMPDIR/\"\nprocess_data \"$TMPDIR/important_file\"\n```\n\nOther traps:\n```bash\ntrap 'echo Interrupted' INT       # Ctrl+C\ntrap 'echo Terminated' TERM       # kill\ntrap 'echo \"Error at line $LINENO\"' ERR  # any command failure\n```"),
    ]

    for q, a in qa_pairs:
        examples.append(conv([("user", q), ("assistant", a)], "flm_shell"))
        examples.append(conv([("user", q), ("assistant", a)], "flm_shell"))

    return examples


# ── Process Management ────────────────────────────────────────────────────

def gen_processes():
    examples = []

    qa_pairs = [
        ("How do I find and kill a process?",
         "```bash\n# Find by name\nps aux | grep nginx\npgrep -a nginx\n\n# Kill by PID\nkill 1234        # SIGTERM (graceful)\nkill -9 1234     # SIGKILL (force, last resort)\n\n# Kill by name\npkill nginx\nkillall nginx\n\n# Kill process using a port\nfuser -k 80/tcp\n\n# Interactive process viewer\ntop\nhtop  # better, install with apt install htop\n```\n\nAlways try SIGTERM first. SIGKILL skips cleanup (no temp file removal, no graceful shutdown)."),

        ("How do I run a process in the background?",
         "Several methods:\n\n```bash\n# Ampersand — runs in background\nlong_command &\n\n# nohup — survives logout\nnohup long_command > output.log 2>&1 &\n\n# disown — detach from shell after starting\nlong_command &\ndisown\n\n# screen/tmux — persistent sessions\ntmux new -s mysession\n# Ctrl+B, D to detach\ntmux attach -t mysession\n\n# systemd-run — one-off background service\nsudo systemd-run --unit=mybackup /usr/local/bin/backup.sh\n```\n\nFor long-running tasks, `tmux` or a systemd service is better than `nohup`."),

        ("How do I monitor system resources?",
         "```bash\n# CPU and memory overview\ntop\nhtop        # interactive, color-coded\n\n# Memory details\nfree -h\n\n# CPU info\nlscpu\nnproc       # just the count\n\n# Per-process memory\nps aux --sort=-%mem | head -20\n\n# IO stats\niostat -x 1    # per-disk IO every second\n\n# VM stats\nvmstat 1       # every second\n\n# System load\nuptime\ncat /proc/loadavg\n```\n\nLoad average: numbers represent 1/5/15 minute CPU demand. Above `nproc` count = system is overloaded."),

        ("What are nice and ionice?",
         "`nice` controls CPU priority, `ionice` controls I/O priority.\n\n```bash\n# Run with lower CPU priority (higher nice = lower priority)\nnice -n 19 heavy_computation\n\n# Change running process priority\nrenice 10 -p 1234\n\n# Run with lower IO priority\nionice -c 3 backup_command     # idle class — only when disk is idle\nionice -c 2 -n 7 backup_command  # best-effort, low priority\n```\n\nNice values: -20 (highest priority) to 19 (lowest). Only root can set negative values.\n\nIO classes: 1=realtime, 2=best-effort, 3=idle.\n\nGreat for backups: `nice -n 19 ionice -c 3 rsync ...`"),

        ("How do I see what's using swap?",
         "```bash\n# Overall swap usage\nfree -h\nswapon --show\n\n# Per-process swap usage\nfor pid in /proc/[0-9]*; do\n    swap=$(awk '/VmSwap/{print $2}' $pid/status 2>/dev/null)\n    if [[ -n \"$swap\" && \"$swap\" -gt 0 ]]; then\n        name=$(cat $pid/comm 2>/dev/null)\n        echo \"$name (PID ${pid##*/}): ${swap} kB\"\n    fi\ndone | sort -t: -k2 -nr | head -20\n\n# Or with smem (more accurate)\nsudo apt install smem\nsmem -s swap -r | head -20\n```"),
    ]

    for q, a in qa_pairs:
        examples.append(conv([("user", q), ("assistant", a)], "flm_processes"))
        examples.append(conv([("user", q), ("assistant", a)], "flm_processes"))

    return examples


# ── Text Processing ───────────────────────────────────────────────────────

def gen_text_processing():
    examples = []

    qa_pairs = [
        ("How do I use grep effectively?",
         "```bash\n# Basic search\ngrep \"error\" /var/log/syslog\n\n# Case insensitive\ngrep -i \"error\" log.txt\n\n# Recursive in directory\ngrep -r \"TODO\" /project/src/\n\n# Show line numbers\ngrep -n \"pattern\" file\n\n# Context (3 lines before/after)\ngrep -C 3 \"error\" log.txt\n\n# Regex\ngrep -E 'error|warning|critical' log.txt\n\n# Invert (lines NOT matching)\ngrep -v \"DEBUG\" log.txt\n\n# Count matches\ngrep -c \"error\" log.txt\n\n# Only filenames\ngrep -rl \"pattern\" /path/\n\n# Fixed string (no regex)\ngrep -F \"[error]\" log.txt\n```\n\n`ripgrep` (`rg`) is a faster alternative: `apt install ripgrep`."),

        ("How do I use find to search for files?",
         "```bash\n# By name\nfind /etc -name \"*.conf\"\nfind / -iname \"readme*\"  # case insensitive\n\n# By type\nfind / -type f -name \"*.log\"   # files\nfind / -type d -name \"config\"  # directories\n\n# By size\nfind / -size +100M             # larger than 100MB\nfind /tmp -size 0 -type f      # empty files\n\n# By time\nfind / -mtime -7 -name \"*.conf\"  # modified in last 7 days\nfind /var/log -mmin -60           # modified in last 60 minutes\n\n# Execute command on results\nfind /tmp -name \"*.tmp\" -delete\nfind . -name \"*.py\" -exec wc -l {} +\n\n# With xargs (faster for many files)\nfind . -name \"*.log\" | xargs gzip\n```"),

        ("How do I use awk?",
         "awk processes text column by column:\n\n```bash\n# Print specific columns\nawk '{print $1, $3}' file.txt\n\n# With custom delimiter\nawk -F: '{print $1, $7}' /etc/passwd\n\n# Filter rows\nawk '$3 > 1000' /etc/passwd\nawk '/error/' log.txt\n\n# Sum a column\nawk '{sum += $5} END {print sum}' data.txt\n\n# Format output\nawk -F: '{printf \"%-20s %s\\n\", $1, $7}' /etc/passwd\n\n# Count occurrences\nawk '{count[$1]++} END {for (k in count) print k, count[k]}' access.log\n\n# Multiple rules\nawk 'BEGIN {print \"Name\\tShell\"} {print $1\"\\t\"$7} END {print NR\" users\"}' FS=: /etc/passwd\n```"),

        ("How do I use sed for text replacement?",
         "```bash\n# Replace first occurrence per line\nsed 's/old/new/' file.txt\n\n# Replace all occurrences\nsed 's/old/new/g' file.txt\n\n# In-place edit\nsed -i 's/old/new/g' file.txt\n\n# In-place with backup\nsed -i.bak 's/old/new/g' file.txt\n\n# Delete lines matching pattern\nsed '/pattern/d' file.txt\n\n# Delete line range\nsed '5,10d' file.txt\n\n# Insert before/after\nsed '/pattern/i\\new line before' file.txt\nsed '/pattern/a\\new line after' file.txt\n\n# Multiple operations\nsed -e 's/foo/bar/g' -e 's/baz/qux/g' file.txt\n\n# Print only matching lines (like grep)\nsed -n '/pattern/p' file.txt\n```"),

        ("How do I use jq to parse JSON?",
         "```bash\n# Pretty-print\necho '{\"name\":\"flm\"}' | jq .\n\n# Extract field\ncurl -s api.example.com | jq '.data.name'\n\n# Array elements\necho '[1,2,3]' | jq '.[]'\necho '[1,2,3]' | jq '.[0]'\n\n# Filter\njq '.[] | select(.status == \"active\")' data.json\n\n# Transform\njq '{name: .user.name, email: .user.email}' data.json\n\n# Raw output (no quotes)\njq -r '.name' data.json\n\n# Count\njq '. | length' array.json\n\n# Sort\njq 'sort_by(.date)' array.json\n```\n\nInstall: `apt install jq`. Essential for working with APIs and config files."),
    ]

    for q, a in qa_pairs:
        examples.append(conv([("user", q), ("assistant", a)], "flm_textproc"))
        examples.append(conv([("user", q), ("assistant", a)], "flm_textproc"))

    return examples


# ── Backup & Recovery ─────────────────────────────────────────────────────

def gen_backup():
    examples = []

    qa_pairs = [
        ("How do I use rsync for backups?",
         "```bash\n# Basic local backup\nrsync -av /source/ /backup/\n\n# Remote backup (over SSH)\nrsync -avz /local/data/ user@server:/backup/data/\n\n# With deletion (mirror)\nrsync -av --delete /source/ /backup/\n\n# Exclude patterns\nrsync -av --exclude='*.tmp' --exclude='.cache' /home/ /backup/home/\n\n# Dry run first\nrsync -avn /source/ /backup/\n\n# Show progress\nrsync -av --progress /source/ /backup/\n\n# Incremental with hard links (space efficient)\nrsync -av --link-dest=/backup/previous/ /source/ /backup/current/\n```\n\n**Important**: Trailing slash on source matters! `/source/` copies contents, `/source` copies the directory itself."),

        ("How do I create and extract tar archives?",
         "```bash\n# Create compressed archive\ntar czf backup.tar.gz /path/to/dir\ntar cjf backup.tar.bz2 /path/to/dir   # better compression\ntar cJf backup.tar.xz /path/to/dir    # best compression\n\n# Extract\ntar xzf backup.tar.gz\ntar xzf backup.tar.gz -C /target/dir   # extract to specific dir\n\n# List contents without extracting\ntar tzf backup.tar.gz\n\n# Exclude files\ntar czf backup.tar.gz --exclude='*.log' --exclude='.git' /path/\n\n# Preserve permissions (important for system backups)\nsudo tar czpf backup.tar.gz /etc/\n```\n\nFlags: `c`=create, `x`=extract, `t`=list, `z`=gzip, `j`=bzip2, `J`=xz, `f`=file, `v`=verbose, `p`=preserve permissions."),

        ("How do I clone a disk or partition?",
         "Use `dd`:\n```bash\n# Clone entire disk\nsudo dd if=/dev/sda of=/dev/sdb bs=4M status=progress\n\n# Create disk image\nsudo dd if=/dev/sda of=/backup/disk.img bs=4M status=progress\n\n# Restore from image\nsudo dd if=/backup/disk.img of=/dev/sda bs=4M status=progress\n\n# Compressed image\nsudo dd if=/dev/sda bs=4M status=progress | gzip > disk.img.gz\ngzip -dc disk.img.gz | sudo dd of=/dev/sda bs=4M status=progress\n```\n\n**Warning**: `dd` will overwrite without asking. Triple-check `if=` (input) and `of=` (output). Getting them backwards destroys your source.\n\nFor smarter backups, consider `partclone` (only copies used blocks)."),

        ("How do I set up automated backups with a script?",
         "```bash\n#!/bin/bash\nset -euo pipefail\n\nBACKUP_DIR=\"/backup/$(date +%Y-%m-%d)\"\nSOURCE_DIRS=(/etc /home /var/lib/postgresql)\nRETENTION_DAYS=30\n\n# Create backup directory\nmkdir -p \"$BACKUP_DIR\"\n\n# Backup each directory\nfor dir in \"${SOURCE_DIRS[@]}\"; do\n    name=$(basename \"$dir\")\n    echo \"Backing up $dir...\"\n    tar czf \"$BACKUP_DIR/${name}.tar.gz\" \"$dir\" 2>/dev/null || true\ndone\n\n# Dump database\npg_dumpall -U postgres | gzip > \"$BACKUP_DIR/postgres.sql.gz\"\n\n# Remove old backups\nfind /backup -maxdepth 1 -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \\;\n\necho \"Backup complete: $BACKUP_DIR\"\ndu -sh \"$BACKUP_DIR\"\n```\n\nRun via systemd timer or cron: `0 2 * * * /usr/local/bin/backup.sh >> /var/log/backup.log 2>&1`"),
    ]

    for q, a in qa_pairs:
        examples.append(conv([("user", q), ("assistant", a)], "flm_backup"))
        examples.append(conv([("user", q), ("assistant", a)], "flm_backup"))

    return examples


# ── Web Servers ───────────────────────────────────────────────────────────

def gen_webservers():
    examples = []

    qa_pairs = [
        ("How do I set up Nginx on Debian?",
         "```bash\nsudo apt install nginx\nsudo systemctl enable --now nginx\n```\n\nCreate a site config in `/etc/nginx/sites-available/mysite`:\n```nginx\nserver {\n    listen 80;\n    server_name example.com;\n    root /var/www/mysite;\n    index index.html;\n\n    location / {\n        try_files $uri $uri/ =404;\n    }\n}\n```\n\nEnable it:\n```bash\nsudo ln -s /etc/nginx/sites-available/mysite /etc/nginx/sites-enabled/\nsudo nginx -t  # test config\nsudo systemctl reload nginx\n```"),

        ("How do I set up HTTPS with Let's Encrypt?",
         "```bash\nsudo apt install certbot python3-certbot-nginx\n\n# Get and install certificate\nsudo certbot --nginx -d example.com -d www.example.com\n```\n\nCertbot automatically:\n1. Proves domain ownership via HTTP challenge\n2. Gets the certificate\n3. Configures Nginx for HTTPS\n4. Sets up auto-renewal via systemd timer\n\nVerify auto-renewal:\n```bash\nsudo certbot renew --dry-run\nsystemctl list-timers | grep certbot\n```\n\nCertificates are stored in `/etc/letsencrypt/live/example.com/`."),

        ("How do I set up a reverse proxy with Nginx?",
         "In `/etc/nginx/sites-available/app`:\n```nginx\nserver {\n    listen 80;\n    server_name app.example.com;\n\n    location / {\n        proxy_pass http://127.0.0.1:3000;\n        proxy_set_header Host $host;\n        proxy_set_header X-Real-IP $remote_addr;\n        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n        proxy_set_header X-Forwarded-Proto $scheme;\n    }\n}\n```\n\nFor WebSocket support, add:\n```nginx\n    location /ws {\n        proxy_pass http://127.0.0.1:3000;\n        proxy_http_version 1.1;\n        proxy_set_header Upgrade $http_upgrade;\n        proxy_set_header Connection \"upgrade\";\n    }\n```"),

        ("How do I check Nginx error logs?",
         "```bash\n# Main error log\nsudo tail -f /var/log/nginx/error.log\n\n# Access log\nsudo tail -f /var/log/nginx/access.log\n\n# Per-site logs (if configured)\nsudo tail -f /var/log/nginx/mysite-error.log\n\n# Test config before reload\nsudo nginx -t\n\n# Check if nginx is running\nsystemctl status nginx\n```\n\nCommon errors:\n- `[emerg] bind() to 0.0.0.0:80 failed` — port 80 already in use\n- `connect() failed (111: Connection refused)` — upstream app not running\n- `permission denied` — wrong file permissions on web root"),
    ]

    for q, a in qa_pairs:
        examples.append(conv([("user", q), ("assistant", a)], "flm_webservers"))
        examples.append(conv([("user", q), ("assistant", a)], "flm_webservers"))

    return examples


# ── Log Analysis ──────────────────────────────────────────────────────────

def gen_logs():
    examples = []

    qa_pairs = [
        ("Where are log files on Debian?",
         "Main locations:\n\n```\n/var/log/syslog          — general system log\n/var/log/auth.log        — authentication/sudo/SSH\n/var/log/kern.log        — kernel messages\n/var/log/dpkg.log        — package install/remove history\n/var/log/apt/            — apt history and term logs\n/var/log/nginx/          — web server logs\n/var/log/postgresql/     — database logs\n```\n\nWith systemd, many services log to the journal instead:\n```bash\njournalctl -u nginx     # nginx logs\njournalctl -u ssh       # SSH logs\njournalctl -k           # kernel\n```\n\nLogrotate manages rotation: `/etc/logrotate.d/`"),

        ("How do I analyze auth.log for suspicious activity?",
         "```bash\n# Failed SSH logins\ngrep 'Failed password' /var/log/auth.log | tail -20\n\n# Failed logins by IP\ngrep 'Failed password' /var/log/auth.log | awk '{print $(NF-3)}' | sort | uniq -c | sort -nr | head -10\n\n# Successful logins\ngrep 'Accepted' /var/log/auth.log\n\n# sudo usage\ngrep 'sudo:' /var/log/auth.log\n\n# Invalid users\ngrep 'Invalid user' /var/log/auth.log\n\n# Users who opened sessions\ngrep 'session opened' /var/log/auth.log\n```\n\nIf you see many failed login attempts from the same IP, that's a brute-force attempt. Set up fail2ban to auto-block them."),

        ("How do I follow logs in real time?",
         "```bash\n# Traditional\ntail -f /var/log/syslog\n\n# Multiple files\ntail -f /var/log/syslog /var/log/auth.log\n\n# With journalctl (better filtering)\njournalctl -f\njournalctl -fu nginx             # specific service\njournalctl -f -p err             # errors only\njournalctl -f -t kernel          # kernel only\n\n# With color highlighting\ntail -f /var/log/syslog | grep --color=auto -E 'error|warning|$'\n\n# multitail (multiple files, split screen)\nsudo apt install multitail\nmultitail /var/log/syslog /var/log/auth.log\n```"),

        ("How do I manage log file sizes?",
         "Logrotate handles this automatically. Config in `/etc/logrotate.d/myapp`:\n\n```\n/var/log/myapp/*.log {\n    daily\n    rotate 14\n    compress\n    delaycompress\n    missingok\n    notifempty\n    create 640 myapp adm\n    postrotate\n        systemctl reload myapp\n    endscript\n}\n```\n\nFor journald, edit `/etc/systemd/journald.conf`:\n```ini\n[Journal]\nSystemMaxUse=500M\nMaxRetentionSec=30day\n```\n\nManual cleanup:\n```bash\nsudo journalctl --vacuum-size=500M\nsudo journalctl --vacuum-time=30d\n```"),
    ]

    for q, a in qa_pairs:
        examples.append(conv([("user", q), ("assistant", a)], "flm_logs"))
        examples.append(conv([("user", q), ("assistant", a)], "flm_logs"))

    return examples


# ── Performance Tuning ────────────────────────────────────────────────────

def gen_performance():
    examples = []

    qa_pairs = [
        ("My Debian server is slow, how do I diagnose it?",
         "Check these in order:\n\n**1. Load average** — is the system overloaded?\n```bash\nuptime\n# Load > number of CPU cores = overloaded\nnproc  # your core count\n```\n\n**2. CPU** — what's using it?\n```bash\ntop -o %CPU\n```\n\n**3. Memory** — swapping?\n```bash\nfree -h\n# High swap usage with free RAM = something previously used lots of memory\nvmstat 1 5  # watch si/so columns (swap in/out)\n```\n\n**4. Disk I/O** — disk bottleneck?\n```bash\niostat -x 1\n# %util near 100% = disk saturated\n```\n\n**5. Network** — bandwidth saturated?\n```bash\nss -s\niftop  # apt install iftop\n```"),

        ("How do I check memory usage in detail?",
         "```bash\n# Overview\nfree -h\n\n# Detailed breakdown\ncat /proc/meminfo\n\n# Per-process (sorted by memory)\nps aux --sort=-%mem | head -20\n\n# Shared memory analysis\nsudo smem -rs pss | head -20  # apt install smem\n\n# Cache vs actually used\n# In 'free' output:\n#   'used' includes buffers/cache\n#   'available' is what's actually free for new processes\n```\n\n**Key insight**: Linux uses free RAM for disk cache. \"used\" memory isn't a problem if \"available\" is healthy. Only worry if swap is actively being used (check `si`/`so` in `vmstat 1`)."),

        ("How do I optimize a Debian system for low memory?",
         "```bash\n# 1. Check what's using memory\nps aux --sort=-%mem | head -20\n\n# 2. Reduce swap tendency\necho 'vm.swappiness=10' | sudo tee -a /etc/sysctl.d/99-custom.conf\nsudo sysctl -p /etc/sysctl.d/99-custom.conf\n\n# 3. Disable unused services\nsystemctl list-units --type=service --state=running\nsudo systemctl disable --now bluetooth cups avahi-daemon\n\n# 4. Use lighter alternatives\n# nginx instead of apache2\n# sysvinit or openrc instead of systemd (extreme)\n\n# 5. Set memory limits for services\n# In systemd unit:\n# MemoryMax=256M\n# MemoryHigh=200M\n\n# 6. Configure zram (compressed swap in RAM)\nsudo apt install zram-tools\n# Edit /etc/default/zramswap, then restart\n```"),

        ("How do I tune sysctl for performance?",
         "Add to `/etc/sysctl.d/99-performance.conf`:\n\n```ini\n# Network performance\nnet.core.somaxconn = 65535\nnet.core.netdev_max_backlog = 65536\nnet.ipv4.tcp_max_syn_backlog = 65536\nnet.ipv4.tcp_fastopen = 3\n\n# Reduce swap usage\nvm.swappiness = 10\n\n# File descriptor limits\nfs.file-max = 2097152\n\n# Better memory management\nvm.dirty_ratio = 10\nvm.dirty_background_ratio = 5\n\n# Network security (server)\nnet.ipv4.conf.all.rp_filter = 1\nnet.ipv4.tcp_syncookies = 1\n```\n\nApply: `sudo sysctl -p /etc/sysctl.d/99-performance.conf`\n\n**Warning**: Don't blindly copy-paste sysctl settings. Each should be tuned for your specific workload."),
    ]

    for q, a in qa_pairs:
        examples.append(conv([("user", q), ("assistant", a)], "flm_performance"))
        examples.append(conv([("user", q), ("assistant", a)], "flm_performance"))

    return examples


# ── Debian-Specific Knowledge ─────────────────────────────────────────────

def gen_debian_knowledge():
    examples = []

    qa_pairs = [
        ("How do I upgrade from Debian Bookworm to Trixie?",
         "**Warning**: Major release upgrades can break things. Backup first.\n\n```bash\n# 1. Update current system fully\nsudo apt update && sudo apt full-upgrade\nsudo reboot\n\n# 2. Change sources.list\nsudo sed -i 's/bookworm/trixie/g' /etc/apt/sources.list\nsudo sed -i 's/bookworm/trixie/g' /etc/apt/sources.list.d/*.list\n\n# 3. Update and upgrade in stages\nsudo apt update\nsudo apt upgrade --without-new-pkgs\nsudo apt full-upgrade\n\n# 4. Reboot and clean up\nsudo reboot\nsudo apt autoremove\n```\n\nCheck release notes first: https://www.debian.org/releases/trixie/releasenotes\n\nThird-party repos may not have Trixie packages yet — disable them before upgrading."),

        ("How do I use apt pinning?",
         "Apt pinning lets you control which version or repository a package comes from.\n\nCreate `/etc/apt/preferences.d/pin-example`:\n\n```\n# Pin a specific package to backports\nPackage: nginx\nPin: release a=bookworm-backports\nPin-Priority: 900\n\n# Block a package from being installed\nPackage: snapd\nPin: release *\nPin-Priority: -1\n\n# Prefer stable for everything by default\nPackage: *\nPin: release a=stable\nPin-Priority: 700\n```\n\nPriority levels:\n- **1001+**: Allows downgrade\n- **990**: Target release default\n- **500**: Normal repo default\n- **100**: Installed package default\n- **< 0**: Never install\n\nCheck current pins: `apt-cache policy nginx`"),

        ("How do I use backports on Debian?",
         "Backports provide newer versions of packages from testing, recompiled for stable.\n\n```bash\n# Add backports repo (for Bookworm)\necho 'deb http://deb.debian.org/debian bookworm-backports main' | \\\n    sudo tee /etc/apt/sources.list.d/backports.list\nsudo apt update\n\n# Install from backports (must specify explicitly)\nsudo apt install -t bookworm-backports nginx\n\n# List available backports\napt list -a 2>/dev/null | grep backports\n\n# Upgrade installed backports\nsudo apt install -t bookworm-backports --upgrade\n```\n\nBackports are not installed automatically. You must explicitly request them. They're not security-supported by Debian's security team, but they are maintained."),

        ("What are Debian release codenames?",
         "Debian releases are named after Toy Story characters:\n\n| Version | Codename | Status |\n|---------|----------|--------|\n| 13 | Trixie | Testing (upcoming stable) |\n| 12 | Bookworm | Current Stable |\n| 11 | Bullseye | Old Stable |\n| 10 | Buster | EOL |\n| 9 | Stretch | EOL |\n\nSpecial suites:\n- **stable** → currently Bookworm\n- **testing** → currently Trixie\n- **unstable** → always \"Sid\" (never releases)\n- **experimental** → bleeding edge, not a full suite\n\nCheck your version:\n```bash\ncat /etc/debian_version\nlsb_release -a\n```"),

        ("How do I add a new repository on Debian?",
         "Modern way (signed repos, Debian 12+):\n\n```bash\n# 1. Download the GPG key\ncurl -fsSL https://example.com/key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/example.gpg\n\n# 2. Add the repository\necho \"deb [signed-by=/etc/apt/keyrings/example.gpg] https://example.com/apt stable main\" | \\\n    sudo tee /etc/apt/sources.list.d/example.list\n\n# 3. Update\nsudo apt update\n```\n\n**Important**: The old `apt-key` method is deprecated. Always use `signed-by` with a keyring file in `/etc/apt/keyrings/`.\n\nTo remove:\n```bash\nsudo rm /etc/apt/sources.list.d/example.list\nsudo rm /etc/apt/keyrings/example.gpg\nsudo apt update\n```"),

        ("How do I put a package on hold?",
         "Prevent a package from being upgraded:\n\n```bash\n# Hold\nsudo apt-mark hold package-name\n\n# Show held packages\napt-mark showhold\n\n# Unhold\nsudo apt-mark unhold package-name\n\n# With dpkg\necho 'package-name hold' | sudo dpkg --set-selections\n```\n\nUseful when a specific version is required or a newer version has a known bug. Held packages are skipped during `apt upgrade` and `apt full-upgrade` warns about them."),

        ("How do I find which package provides a file?",
         "```bash\n# File from installed package\ndpkg -S /usr/bin/curl\n# Output: curl: /usr/bin/curl\n\n# File from any package (installed or not)\nsudo apt install apt-file\nsudo apt-file update\napt-file search /usr/bin/curl\n\n# Find which package provides a command\ncommand -v curl && dpkg -S $(readlink -f $(command -v curl))\n\n# List all files in a package\ndpkg -L curl\n\n# Search package descriptions\napt-cache search \"web server\"\n```"),

        ("How do I build a package from source on Debian?",
         "```bash\n# 1. Install build dependencies\nsudo apt build-dep package-name\n\n# 2. Get the source\napt source package-name\n\n# 3. Apply changes, then build\ncd package-name-*/\ndpkg-buildpackage -us -uc -b\n# -us -uc = don't sign, -b = binary only\n\n# 4. Install the result\nsudo dpkg -i ../package-name_*.deb\n```\n\nFor rebuilding with custom options:\n```bash\n# Edit debian/rules to change configure flags\n# Then rebuild with:\ndpkg-buildpackage -us -uc -b\n```\n\nTo prevent apt from overwriting your custom build:\n```bash\nsudo apt-mark hold package-name\n```"),
    ]

    for q, a in qa_pairs:
        examples.append(conv([("user", q), ("assistant", a)], "flm_debian"))
        examples.append(conv([("user", q), ("assistant", a)], "flm_debian"))

    # OS-aware examples (different answers based on system prompt)
    os_specific = [
        {
            "q": "How do I install Docker?",
            "debian": "```bash\n# Add Docker's official repo\nsudo apt install ca-certificates curl\nsudo install -m 0755 -d /etc/apt/keyrings\ncurl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg\necho \"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable\" | sudo tee /etc/apt/sources.list.d/docker.list\nsudo apt update\nsudo apt install docker-ce docker-ce-cli containerd.io\n\n# Add your user to the docker group\nsudo usermod -aG docker $USER\nnewgrp docker\n\n# Verify\ndocker run hello-world\n```",
            "arch": "```bash\nsudo pacman -S docker\nsudo systemctl enable --now docker\nsudo usermod -aG docker $USER\nnewgrp docker\ndocker run hello-world\n```\n\nArch has Docker in the official repos, no third-party repo needed.",
            "fedora": "```bash\nsudo dnf install dnf-plugins-core\nsudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo\nsudo dnf install docker-ce docker-ce-cli containerd.io\nsudo systemctl enable --now docker\nsudo usermod -aG docker $USER\nnewgrp docker\ndocker run hello-world\n```",
        },
        {
            "q": "How do I check installed packages?",
            "debian": "```bash\n# List all installed\ndpkg -l\n\n# Search installed packages\ndpkg -l | grep nginx\napt list --installed | grep nginx\n\n# Package info\napt show nginx\ndpkg -s nginx\n\n# Count installed packages\ndpkg -l | grep '^ii' | wc -l\n```",
            "arch": "```bash\n# List all installed\npacman -Q\n\n# Search installed\npacman -Qs nginx\n\n# Package info\npacman -Qi nginx\n\n# Explicitly installed (not dependencies)\npacman -Qe\n\n# AUR packages\npacman -Qm\n```",
            "fedora": "```bash\n# List all installed\nrpm -qa\ndnf list installed\n\n# Search\ndnf list installed | grep nginx\n\n# Package info\ndnf info nginx\nrpm -qi nginx\n\n# Count\nrpm -qa | wc -l\n```",
        },
    ]

    for item in os_specific:
        q = item["q"]
        for os_name, a in [("debian_trixie", item["debian"]),
                           ("arch", item["arch"]),
                           ("fedora_41", item["fedora"])]:
            sys_prompt = pick_system_for_os(os_name)
            examples.append(conv([("user", q), ("assistant", a)], "flm_debian", system=sys_prompt))

    return examples


# ── Multi-turn Troubleshooting ────────────────────────────────────────────

def gen_multiturn_troubleshoot():
    examples = []

    scenarios = [
        [
            ("user", "my server's disk is almost full, what do I do?"),
            ("assistant", "Let's find what's eating space. Run:\n```bash\ndf -h\n```\nWhich filesystem is full? And:\n```bash\ndu -h --max-depth=1 / 2>/dev/null | sort -hr | head -10\n```\nThis shows the biggest top-level directories."),
            ("user", "/var is using 45GB out of 50GB"),
            ("assistant", "Let's drill into /var:\n```bash\ndu -h --max-depth=1 /var | sort -hr | head -10\n```\n\nCommon culprits:\n- `/var/log` — old log files\n- `/var/cache/apt` — cached .deb packages\n- `/var/lib/docker` — Docker images/containers\n\nQuick wins while we investigate:\n```bash\n# Clean apt cache\nsudo apt clean\n\n# Remove old kernels\nsudo apt autoremove\n```"),
            ("user", "/var/log is 30GB!"),
            ("assistant", "30GB of logs is excessive. Let's find the biggest offenders:\n```bash\nls -lhS /var/log/ | head -20\nsudo du -h /var/log/* | sort -hr | head -10\n```\n\nCommon fixes:\n```bash\n# Truncate a huge log (keeps file handle, apps keep writing)\nsudo truncate -s 0 /var/log/huge.log\n\n# Remove old rotated logs\nsudo rm /var/log/*.gz\nsudo rm /var/log/*.1 /var/log/*.2\n\n# Vacuum journald\nsudo journalctl --vacuum-size=500M\n```\n\nThen fix the root cause — configure logrotate so it doesn't happen again:\n```bash\ncat /etc/logrotate.d/rsyslog\n```\nMake sure rotation is set to `daily` with a reasonable `rotate` count."),
        ],
        [
            ("user", "I accidentally deleted /etc/apt/sources.list"),
            ("assistant", "Don't panic — we can reconstruct it. First, what Debian version are you on?\n```bash\ncat /etc/debian_version\n```"),
            ("user", "13.0"),
            ("assistant", "That's Trixie. Here's a standard sources.list:\n\n```bash\nsudo tee /etc/apt/sources.list << 'EOF'\ndeb http://deb.debian.org/debian trixie main contrib non-free non-free-firmware\ndeb http://deb.debian.org/debian trixie-updates main contrib non-free non-free-firmware\ndeb http://security.debian.org/debian-security trixie-security main contrib non-free non-free-firmware\nEOF\n```\n\nThen:\n```bash\nsudo apt update\n```\n\nIf you had any third-party repos, those would be in `/etc/apt/sources.list.d/` and should be unaffected.\n\nFor future safety, you can also check your sources.list history:\n```bash\nzless /var/log/apt/history.log.1.gz\n```"),
        ],
        [
            ("user", "my systemd service keeps crashing and restarting in a loop"),
            ("assistant", "That's a crash loop. Let's investigate:\n\n```bash\n# Check the service status and recent logs\nsystemctl status your-service\njournalctl -u your-service --no-pager -n 50\n```\n\nWhat's the service name and what do the logs show?"),
            ("user", "it's myapp.service, logs show 'Error: EADDRINUSE: port 3000 already in use'"),
            ("assistant", "The port is already in use — likely by a previous instance that didn't shut down cleanly. Fix:\n\n```bash\n# Find what's using port 3000\nsudo ss -tlnp sport = :3000\n\n# Kill it\nsudo fuser -k 3000/tcp\n\n# Restart your service\nsudo systemctl restart myapp\n```\n\nTo prevent this in the future, add to your service file:\n```ini\n[Service]\nExecStartPre=/usr/bin/fuser -k 3000/tcp || true\nRestart=on-failure\nRestartSec=5\n```\n\nThe `RestartSec=5` gives the port time to be released before the next restart attempt."),
        ],
        [
            ("user", "permission denied on everything even with sudo"),
            ("assistant", "That's unusual. Let's check a few things:\n\n```bash\nwhoami\nid\nsudo -l\n```\n\nAre you getting 'permission denied' or 'command not found' or something else? The exact error matters."),
            ("user", "it says 'sudo: /usr/bin/sudo must be owned by uid 0 and have the setuid bit set'"),
            ("assistant", "Someone (or something) changed the permissions on `/usr/bin/sudo`. This is fixable but you need root access another way.\n\n**Option 1: Reboot into recovery mode**\n1. Reboot the machine\n2. In GRUB, select \"Advanced options\" → \"recovery mode\"\n3. Select \"root\" shell\n4. Fix permissions:\n```bash\nchown root:root /usr/bin/sudo\nchmod 4755 /usr/bin/sudo\n```\n\n**Option 2: If you have physical access**, boot from a live USB and fix it from there.\n\n**Option 3: If `pkexec` works**:\n```bash\npkexec chmod 4755 /usr/bin/sudo\npkexec chown root:root /usr/bin/sudo\n```\n\nThe `4` in `4755` is the setuid bit — it's what allows sudo to run as root."),
        ],
    ]

    for turns in scenarios:
        examples.append(conv(turns, "flm_troubleshoot_v3"))

    return examples


# ── Containers ────────────────────────────────────────────────────────────

def gen_containers():
    examples = []

    qa_pairs = [
        ("What's the difference between Docker and Podman?",
         "Both run OCI containers, but with key differences:\n\n**Docker**:\n- Daemon-based (`dockerd` runs as root)\n- Established ecosystem, most tutorials assume Docker\n- Docker Compose for multi-container setups\n\n**Podman**:\n- Daemonless — each container is a child process\n- Rootless by default (better security)\n- CLI-compatible with Docker (`alias docker=podman`)\n- Supports pods (groups of containers sharing network)\n- In Debian's main repo: `apt install podman`\n\nFor a Debian server, Podman is often the better choice — no daemon, rootless, and available directly from apt without third-party repos."),

        ("How do I run a container with Podman?",
         "```bash\n# Install\nsudo apt install podman\n\n# Run (same syntax as docker)\npodman run -d --name webserver -p 8080:80 nginx\n\n# List running\npodman ps\n\n# Logs\npodman logs webserver\n\n# Stop and remove\npodman stop webserver\npodman rm webserver\n\n# Rootless (as regular user)\npodman run --rm -it debian:trixie bash\n\n# With volume mount\npodman run -d -v /host/path:/container/path:Z nginx\n```\n\nThe `:Z` suffix on volume mounts handles SELinux/AppArmor labels automatically."),

        ("How do I write a Dockerfile?",
         "```dockerfile\nFROM debian:trixie-slim\n\n# Install dependencies in one layer\nRUN apt-get update && \\\n    apt-get install -y --no-install-recommends \\\n        python3 python3-pip && \\\n    rm -rf /var/lib/apt/lists/*\n\n# Create non-root user\nRUN useradd -m appuser\nWORKDIR /app\n\n# Copy requirements first (better caching)\nCOPY requirements.txt .\nRUN pip3 install --no-cache-dir -r requirements.txt\n\n# Copy application\nCOPY . .\n\nUSER appuser\nEXPOSE 8000\nCMD [\"python3\", \"app.py\"]\n```\n\nBuild: `podman build -t myapp .`\nRun: `podman run -d -p 8000:8000 myapp`\n\nKey practices: minimize layers, use slim base images, don't run as root, clean apt caches."),

        ("How do I manage containers with systemd?",
         "Podman generates systemd units automatically:\n\n```bash\n# Start your container\npodman run -d --name myapp -p 8080:80 nginx\n\n# Generate systemd unit\npodman generate systemd --new --name myapp > ~/.config/systemd/user/myapp.service\n\n# Enable and start\nsystemctl --user daemon-reload\nsystemctl --user enable --now myapp\n\n# Check status\nsystemctl --user status myapp\n```\n\nFor root-level services:\n```bash\nsudo podman generate systemd --new --name myapp > /etc/systemd/system/myapp.service\nsudo systemctl enable --now myapp\n```\n\nThis is the recommended way to run containers in production on Debian."),
    ]

    for q, a in qa_pairs:
        examples.append(conv([("user", q), ("assistant", a)], "flm_containers"))
        examples.append(conv([("user", q), ("assistant", a)], "flm_containers"))

    return examples


# ── Git Basics for Sysadmins ──────────────────────────────────────────────

def gen_git():
    examples = []

    qa_pairs = [
        ("How do I use git to track my config files?",
         "Great practice for servers. Here's a simple approach:\n\n```bash\n# Track /etc with etckeeper (Debian-integrated)\nsudo apt install etckeeper\nsudo etckeeper init\nsudo etckeeper commit \"Initial commit\"\n```\n\netckeeper automatically commits changes to /etc when you install/remove packages.\n\nFor manual tracking of specific configs:\n```bash\nmkdir ~/config-backup && cd ~/config-backup\ngit init\ncp /etc/nginx/nginx.conf .\ncp /etc/ssh/sshd_config .\ngit add -A && git commit -m \"Initial configs\"\n```\n\nBefore making changes: `sudo etckeeper commit \"Before nginx upgrade\"`"),

        ("How do I undo the last git commit?",
         "Depends on what you want:\n\n```bash\n# Undo commit, keep changes staged\ngit reset --soft HEAD~1\n\n# Undo commit, keep changes unstaged\ngit reset HEAD~1\n\n# Undo commit AND discard changes (destructive!)\ngit reset --hard HEAD~1\n\n# If already pushed, create a revert commit instead\ngit revert HEAD\n```\n\n**Never use `--hard` on pushed commits** — it rewrites history and breaks things for anyone who pulled.\n\n`git revert` is always safe — it creates a new commit that undoes the changes."),

        ("What are the most useful git commands?",
         "Daily essentials:\n```bash\ngit status              # what's changed?\ngit diff                # unstaged changes\ngit diff --staged       # staged changes\ngit log --oneline -20   # recent history\ngit add -p              # stage interactively\ngit commit -m \"msg\"     # commit\ngit stash               # temporarily shelve changes\ngit stash pop           # restore stashed changes\n```\n\nBranching:\n```bash\ngit branch feature-x    # create branch\ngit checkout feature-x  # switch to it\ngit checkout -b fix     # create + switch\ngit merge feature-x     # merge into current\n```\n\nInspecting:\n```bash\ngit blame file.py       # who changed each line\ngit log -p file.py      # history of a file\ngit show abc123         # show a specific commit\n```"),
    ]

    for q, a in qa_pairs:
        examples.append(conv([("user", q), ("assistant", a)], "flm_git"))
        examples.append(conv([("user", q), ("assistant", a)], "flm_git"))

    return examples


# ── Users & Groups ────────────────────────────────────────────────────────

def gen_users():
    examples = []

    qa_pairs = [
        ("How do I add a user on Debian?",
         "```bash\n# Interactive (recommended — sets up home dir, shell, etc.)\nsudo adduser newuser\n\n# Non-interactive\nsudo useradd -m -s /bin/bash -G sudo newuser\nsudo passwd newuser\n\n# System user (for services, no login)\nsudo useradd -r -s /usr/sbin/nologin -d /nonexistent myservice\n```\n\n`adduser` is the Debian-friendly wrapper around `useradd`. It prompts for password and creates the home directory with proper skeleton files."),

        ("How do I manage groups?",
         "```bash\n# Create a group\nsudo groupadd developers\n\n# Add user to group\nsudo usermod -aG developers username\n# -a = append, -G = supplementary group\n\n# See user's groups\ngroups username\nid username\n\n# See group members\ngetent group developers\n\n# Remove user from group\nsudo gpasswd -d username developers\n\n# Delete a group\nsudo groupdel developers\n```\n\n**Important**: Use `-aG` (append), not just `-G`. Without `-a`, `usermod -G` replaces ALL supplementary groups, which can lock users out of sudo."),

        ("How do I disable a user account without deleting it?",
         "Several approaches:\n\n```bash\n# Lock the password (prevents password login)\nsudo passwd -l username\n\n# Set shell to nologin\nsudo usermod -s /usr/sbin/nologin username\n\n# Set account expiry to past date\nsudo usermod --expiredate 1970-01-01 username\n\n# Check account status\nsudo passwd -S username\nsudo chage -l username\n```\n\nTo re-enable:\n```bash\nsudo passwd -u username\nsudo usermod -s /bin/bash username\nsudo usermod --expiredate \"\" username\n```\n\nLocking the password doesn't prevent SSH key login. To fully block access, also set the expiry date."),
    ]

    for q, a in qa_pairs:
        examples.append(conv([("user", q), ("assistant", a)], "flm_users"))
        examples.append(conv([("user", q), ("assistant", a)], "flm_users"))

    return examples


# ── Database Basics ───────────────────────────────────────────────────────

def gen_databases():
    examples = []

    qa_pairs = [
        ("How do I install and set up PostgreSQL on Debian?",
         "```bash\nsudo apt install postgresql\nsudo systemctl enable --now postgresql\n```\n\nPostgreSQL creates a `postgres` system user. Set it up:\n```bash\n# Switch to postgres user\nsudo -u postgres psql\n\n# Create a database and user\nCREATE USER myapp WITH PASSWORD 'secure_password';\nCREATE DATABASE mydb OWNER myapp;\n\\q\n```\n\nConnect:\n```bash\npsql -U myapp -d mydb -h localhost\n```\n\nConfig files:\n- `/etc/postgresql/*/main/postgresql.conf` — server settings\n- `/etc/postgresql/*/main/pg_hba.conf` — authentication rules\n\nAfter editing configs: `sudo systemctl reload postgresql`"),

        ("How do I backup and restore a PostgreSQL database?",
         "```bash\n# Backup single database\nsudo -u postgres pg_dump mydb > mydb_backup.sql\n\n# Compressed backup\nsudo -u postgres pg_dump -Fc mydb > mydb_backup.dump\n\n# All databases\nsudo -u postgres pg_dumpall > all_databases.sql\n\n# Restore from SQL\nsudo -u postgres psql mydb < mydb_backup.sql\n\n# Restore from compressed\nsudo -u postgres pg_restore -d mydb mydb_backup.dump\n\n# Restore to a different database\nsudo -u postgres createdb mydb_copy\nsudo -u postgres pg_restore -d mydb_copy mydb_backup.dump\n```\n\nFor production, set up automated backups with `pg_dump` in a cron job or systemd timer."),

        ("How do I install MariaDB on Debian?",
         "```bash\nsudo apt install mariadb-server\nsudo systemctl enable --now mariadb\n\n# Secure installation\nsudo mysql_secure_installation\n```\n\nThis prompts you to set root password, remove test databases, etc.\n\nCreate a database and user:\n```bash\nsudo mariadb\n```\n```sql\nCREATE DATABASE mydb;\nCREATE USER 'myapp'@'localhost' IDENTIFIED BY 'secure_password';\nGRANT ALL PRIVILEGES ON mydb.* TO 'myapp'@'localhost';\nFLUSH PRIVILEGES;\nEXIT;\n```\n\nConnect: `mariadb -u myapp -p mydb`\n\nConfig: `/etc/mysql/mariadb.conf.d/50-server.cnf`"),
    ]

    for q, a in qa_pairs:
        examples.append(conv([("user", q), ("assistant", a)], "flm_databases"))
        examples.append(conv([("user", q), ("assistant", a)], "flm_databases"))

    return examples


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Generating additional SFT examples...")
    print()

    all_data = []
    generators = [
        ("flm_networking_v2.jsonl", gen_networking),
        ("flm_systemd.jsonl", gen_systemd),
        ("flm_filesystems.jsonl", gen_filesystems),
        ("flm_security.jsonl", gen_security),
        ("flm_shell_scripting.jsonl", gen_shell_scripting),
        ("flm_processes.jsonl", gen_processes),
        ("flm_text_processing.jsonl", gen_text_processing),
        ("flm_backup.jsonl", gen_backup),
        ("flm_webservers.jsonl", gen_webservers),
        ("flm_logs.jsonl", gen_logs),
        ("flm_performance.jsonl", gen_performance),
        ("flm_debian_v2.jsonl", gen_debian_knowledge),
        ("flm_troubleshoot_v3.jsonl", gen_multiturn_troubleshoot),
        ("flm_containers.jsonl", gen_containers),
        ("flm_git.jsonl", gen_git),
        ("flm_users.jsonl", gen_users),
        ("flm_databases.jsonl", gen_databases),
    ]

    for filename, gen_fn in generators:
        data = gen_fn()
        written = write_jsonl(data, filename)
        all_data.extend(written)

    print(f"\nTotal new examples: {len(all_data)}")
    print(f"\nNow run scripts/merge_sft.py to merge everything.")


if __name__ == "__main__":
    main()
