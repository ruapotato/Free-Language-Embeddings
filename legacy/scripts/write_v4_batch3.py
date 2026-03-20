#!/usr/bin/env python3
"""Write hand-crafted SFT conversations for V4 training — Batch 3.

Topics: Networking & DNS, WiFi & Wireless, Containers & Virtualization,
        Git, Python, Multi-turn troubleshooting.

Appends to data/sft/flm_v4_massive.jsonl.
"""
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(REPO_ROOT, "data", "sft", "flm_v4_massive.jsonl")

# -- System prompt variants ---------------------------------------------------

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

SHORT_UBU24 = (
    "You are flm, the Free Language Model. "
    "System: Ubuntu 24.04 LTS (Noble Numbat). Pkg manager: apt."
)


# -- Helpers -------------------------------------------------------------------

def make(system, question, answer, source):
    """Build one single-turn conversation dict."""
    if system is None:
        text = f"<|user|>\n{question}\n<|assistant|>\n{answer}\n"
    else:
        text = (
            f"<|system|>\n{system}\n<|user|>\n{question}\n"
            f"<|assistant|>\n{answer}\n"
        )
    return {"text": text, "source": source}


def make_multi(system, turns, source):
    """Build a multi-turn conversation.

    turns is a list of (role, content) tuples where role is 'user' or
    'assistant'.
    """
    if system is not None:
        text = f"<|system|>\n{system}\n"
    else:
        text = ""
    for role, content in turns:
        text += f"<|{role}|>\n{content}\n"
    return {"text": text, "source": source}


# -- Conversations -------------------------------------------------------------

conversations = []

# ---------------------------------------------------------------------------
# 1. Configure a static IP with /etc/network/interfaces (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I configure a static IP address on Debian using /etc/network/interfaces?",
    r"""First, identify your interface name:

```bash
ip link show
```

Common names are `eth0`, `ens3`, `enp0s3`, etc. Suppose it's `eth0`.

Edit the interfaces file:

```bash
sudo nano /etc/network/interfaces
```

Replace the DHCP line (or add a new stanza):

```
# The primary network interface
auto eth0
iface eth0 inet static
    address 192.168.1.100/24
    gateway 192.168.1.1
    dns-nameservers 1.1.1.1 9.9.9.10
```

The `dns-nameservers` line requires the `resolvconf` package to take effect:

```bash
sudo apt install resolvconf
```

Without `resolvconf`, set DNS manually in `/etc/resolv.conf`:

```
nameserver 1.1.1.1
nameserver 9.9.9.10
```

Apply the changes:

```bash
sudo systemctl restart networking
```

Or bring the interface down and back up:

```bash
sudo ifdown eth0 && sudo ifup eth0
```

Verify:

```bash
ip addr show eth0
ip route show
ping -c 3 1.1.1.1
```

**Important notes:**
- The `/24` in the address line is CIDR notation for a 255.255.255.0 netmask. You can also write `netmask 255.255.255.0` on a separate line.
- If you're on a headless server, be careful — a typo means you lose connectivity. Test with `ifup --no-act eth0` first if available.
- On Debian 13, NetworkManager may be installed on desktop systems and can conflict with `/etc/network/interfaces`. Check with `systemctl status NetworkManager`. If it's active, either disable it or use `nmcli` instead.""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 2. Configure a static IP with NetworkManager/nmcli (Fedora)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_FED41,
    "How do I set a static IP with nmcli on Fedora?",
    r"""List your connections:

```bash
nmcli connection show
```

This shows connection names, UUIDs, types, and devices. Suppose the connection is named `Wired connection 1` on device `enp1s0`.

Set a static IPv4 address:

```bash
nmcli connection modify "Wired connection 1" \
    ipv4.addresses 192.168.1.50/24 \
    ipv4.gateway 192.168.1.1 \
    ipv4.dns "1.1.1.1 9.9.9.10" \
    ipv4.method manual
```

Apply the changes:

```bash
nmcli connection up "Wired connection 1"
```

Verify:

```bash
ip addr show enp1s0
nmcli connection show "Wired connection 1" | grep ipv4
```

**Add a second IP address (alias):**

```bash
nmcli connection modify "Wired connection 1" \
    +ipv4.addresses 192.168.1.51/24
nmcli connection up "Wired connection 1"
```

**Switch back to DHCP:**

```bash
nmcli connection modify "Wired connection 1" \
    ipv4.method auto \
    ipv4.addresses "" \
    ipv4.gateway "" \
    ipv4.dns ""
nmcli connection up "Wired connection 1"
```

**Create a brand new connection profile:**

```bash
nmcli connection add con-name "static-lan" \
    type ethernet ifname enp1s0 \
    ipv4.addresses 10.0.0.5/24 \
    ipv4.gateway 10.0.0.1 \
    ipv4.dns "1.1.1.1" \
    ipv4.method manual
```

The configuration is stored in `/etc/NetworkManager/system-connections/` as keyfile format on Fedora 41.""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 3. Configure a static IP with systemd-networkd (Arch)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_ARCH,
    "How do I configure a static IP with systemd-networkd?",
    r"""First, make sure systemd-networkd is enabled:

```bash
sudo systemctl enable --now systemd-networkd
sudo systemctl enable --now systemd-resolved
```

Find your interface name:

```bash
ip link show
```

Suppose it's `enp0s3`. Create a network file:

```bash
sudo nano /etc/systemd/network/20-static.network
```

```ini
[Match]
Name=enp0s3

[Network]
Address=192.168.1.100/24
Gateway=192.168.1.1
DNS=1.1.1.1
DNS=9.9.9.10
```

If you also want IPv6:

```ini
[Network]
Address=192.168.1.100/24
Gateway=192.168.1.1
Address=fd00::100/64
Gateway=fd00::1
DNS=1.1.1.1
DNS=2606:4700:4700::1111
```

Restart networkd to apply:

```bash
sudo systemctl restart systemd-networkd
```

Verify:

```bash
networkctl status enp0s3
ip addr show enp0s3
ip route show
```

**For DNS resolution**, link resolv.conf to systemd-resolved:

```bash
sudo ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf
```

Check DNS works:

```bash
resolvectl status
resolvectl query archlinux.org
```

**Important:** If you had NetworkManager or dhcpcd running, disable them first to avoid conflicts:

```bash
sudo systemctl disable --now NetworkManager
sudo systemctl disable --now dhcpcd
```

Files in `/etc/systemd/network/` are processed in lexicographic order. The `20-` prefix lets you slot in more specific configs with lower numbers (e.g., `10-vlan.network`) that take priority.""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 4. DNS troubleshooting - dig, nslookup, host (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I troubleshoot DNS issues? What tools should I use?",
    r"""Install the DNS tools if you don't have them:

```bash
sudo apt install dnsutils   # provides dig, nslookup, host
```

**`dig` — the most detailed tool:**

```bash
# Basic query
dig example.com

# Query a specific record type
dig example.com MX
dig example.com AAAA
dig example.com TXT

# Query a specific DNS server
dig @8.8.8.8 example.com

# Short answer only
dig +short example.com

# Trace the full resolution path
dig +trace example.com

# Reverse lookup
dig -x 93.184.216.34
```

**Reading dig output:**
- `status: NOERROR` — query succeeded
- `status: NXDOMAIN` — domain doesn't exist
- `status: SERVFAIL` — DNS server failed to answer
- `ANSWER SECTION` — the actual records returned
- `Query time` — how long the lookup took (high values = slow DNS)

**`host` — simpler output:**

```bash
host example.com
host -t MX example.com
host 93.184.216.34     # reverse lookup
```

**`nslookup` — interactive or one-shot:**

```bash
nslookup example.com
nslookup -type=mx example.com
nslookup example.com 8.8.8.8
```

**Troubleshooting checklist:**

1. Check what DNS server you're using:
```bash
cat /etc/resolv.conf
resolvectl status   # if using systemd-resolved
```

2. Test if DNS works at all:
```bash
dig @1.1.1.1 example.com +short
```

3. If external DNS works but your resolver doesn't, the problem is local:
```bash
dig @127.0.0.53 example.com   # systemd-resolved
dig @192.168.1.1 example.com  # local router
```

4. Check for DNS caching issues:
```bash
sudo resolvectl flush-caches    # systemd-resolved
# or restart the resolver
sudo systemctl restart systemd-resolved
```

5. Test from a clean slate:
```bash
dig +trace example.com @1.1.1.1
```
This bypasses all caches and walks the DNS tree from root.""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 5. Configure /etc/hosts and /etc/resolv.conf (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "How do /etc/hosts and /etc/resolv.conf work?",
    r"""`/etc/hosts` is a local, static mapping of hostnames to IP addresses. It's checked before DNS.

**Format:**
```
127.0.0.1       localhost
127.0.1.1       myhostname
192.168.1.10    fileserver fileserver.lan
10.0.0.5        db.internal
```

Use cases:
- Override DNS for testing: point a domain to a local IP
- Define names for LAN machines that don't have DNS entries
- Block domains by pointing them to `0.0.0.0`

---

`/etc/resolv.conf` tells the system which DNS servers to use.

**Format:**
```
nameserver 1.1.1.1
nameserver 9.9.9.10
search lan example.com
options timeout:2 attempts:3
```

- `nameserver` — up to 3 DNS server IPs, queried in order
- `search` — domain suffixes to try for short hostnames (so `ping db` tries `db.lan`, then `db.example.com`)
- `options timeout:2` — wait 2 seconds per query before trying next server

**The lookup order** is controlled by `/etc/nsswitch.conf`:
```
hosts: files dns
```
This means: check `/etc/hosts` first (`files`), then DNS (`dns`).

**Common pitfall:** On systems using `systemd-resolved` or `NetworkManager`, `/etc/resolv.conf` is often a symlink that gets overwritten. Check:

```bash
ls -la /etc/resolv.conf
```

If it points to `/run/systemd/resolve/stub-resolv.conf`, systemd-resolved manages it. To set DNS persistently, use `resolvectl` or NetworkManager rather than editing the file directly.

If you need manual control, you can break the symlink:

```bash
sudo rm /etc/resolv.conf
sudo tee /etc/resolv.conf << 'EOF'
nameserver 1.1.1.1
nameserver 9.9.9.10
EOF
```

But be aware that DHCP clients and NetworkManager may overwrite it again. On Debian with ifupdown, install `resolvconf` to manage it properly, or use `chattr +i /etc/resolv.conf` as a last resort to prevent changes.""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 6. Set up a local DNS resolver with dnsmasq (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I set up dnsmasq as a local DNS resolver?",
    r"""Install dnsmasq:

```bash
sudo apt install dnsmasq
```

If `systemd-resolved` is running, it binds to port 53 and conflicts. Disable it:

```bash
sudo systemctl disable --now systemd-resolved
sudo rm /etc/resolv.conf
echo "nameserver 127.0.0.1" | sudo tee /etc/resolv.conf
```

Edit the dnsmasq config:

```bash
sudo nano /etc/dnsmasq.conf
```

Key settings:

```ini
# Listen only on localhost and LAN
listen-address=127.0.0.1,192.168.1.1
bind-interfaces

# Upstream DNS servers
server=1.1.1.1
server=9.9.9.10

# Cache size (default 150, increase for busy networks)
cache-size=1000

# Log queries for debugging (disable in production)
log-queries

# Don't read /etc/resolv.conf for upstream servers
no-resolv

# Local domain
local=/lan/
domain=lan

# Static host entries (in addition to /etc/hosts)
address=/myapp.lan/192.168.1.50
address=/db.lan/192.168.1.51
```

Start and enable:

```bash
sudo systemctl enable --now dnsmasq
```

Test it:

```bash
dig @127.0.0.1 example.com
dig @127.0.0.1 myapp.lan
```

**For LAN-wide DNS**, point your DHCP server (router) to give out your dnsmasq machine's IP as the DNS server. Or run DHCP on dnsmasq itself by adding:

```ini
dhcp-range=192.168.1.100,192.168.1.200,12h
dhcp-option=option:router,192.168.1.1
```

**Check the cache:**

```bash
sudo kill -USR1 $(pidof dnsmasq)
# Then check syslog for cache stats
journalctl -u dnsmasq --since "1 minute ago"
```""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 7. tcpdump basics (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I capture and analyze network traffic with tcpdump?",
    r"""Install tcpdump if needed:

```bash
sudo apt install tcpdump
```

**Basic capture on an interface:**

```bash
sudo tcpdump -i eth0
```

This floods the terminal. Use filters:

**Capture only specific traffic:**

```bash
# Only HTTP traffic
sudo tcpdump -i eth0 port 80

# Only DNS queries
sudo tcpdump -i eth0 port 53

# Only traffic to/from a specific host
sudo tcpdump -i eth0 host 192.168.1.50

# Only TCP SYN packets (connection attempts)
sudo tcpdump -i eth0 'tcp[tcpflags] & tcp-syn != 0'

# Combine filters
sudo tcpdump -i eth0 'host 10.0.0.5 and port 443'
```

**Save to a file for later analysis:**

```bash
sudo tcpdump -i eth0 -w capture.pcap
```

Open it later with tcpdump or Wireshark:

```bash
tcpdump -r capture.pcap
tcpdump -r capture.pcap -A   # show ASCII payload
tcpdump -r capture.pcap -X   # show hex + ASCII
```

**Useful flags:**

```bash
# Don't resolve hostnames (faster)
sudo tcpdump -i eth0 -n

# Don't resolve hostnames or port names
sudo tcpdump -i eth0 -nn

# Limit capture to N packets
sudo tcpdump -i eth0 -c 100

# Show verbose output
sudo tcpdump -i eth0 -v     # verbose
sudo tcpdump -i eth0 -vv    # more verbose

# Capture only first N bytes of each packet
sudo tcpdump -i eth0 -s 96  # just headers
```

**Real-world examples:**

```bash
# Debug why SSH connections are slow (check DNS reverse lookups)
sudo tcpdump -i eth0 -nn port 53

# See who's talking to your web server
sudo tcpdump -i eth0 -nn dst port 80 or dst port 443

# Capture everything for Wireshark analysis
sudo tcpdump -i eth0 -w /tmp/debug.pcap -c 10000
```

**Rotate capture files (long-running captures):**

```bash
sudo tcpdump -i eth0 -w /tmp/cap_%Y%m%d_%H%M.pcap -G 3600 -W 24
```

This creates a new file every hour (`-G 3600`) and keeps at most 24 files (`-W 24`).""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 8. ss command — replacement for netstat (short Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SHORT_DEB13,
    "How do I use the ss command? I used to use netstat.",
    r"""`ss` is the modern replacement for `netstat`. It's faster and more detailed.

**Common equivalents:**

| netstat | ss |
|---|---|
| `netstat -tlnp` | `ss -tlnp` |
| `netstat -ulnp` | `ss -ulnp` |
| `netstat -an` | `ss -an` |
| `netstat -s` | `ss -s` |

**Show listening TCP ports:**

```bash
ss -tlnp
```

Flags: `-t` TCP, `-l` listening, `-n` numeric (don't resolve names), `-p` show process.

**Show all established connections:**

```bash
ss -tn
```

**Show listening UDP sockets:**

```bash
ss -ulnp
```

**Filter by port:**

```bash
ss -tlnp sport = :80
ss -tlnp sport = :22 or sport = :443
```

**Filter by state:**

```bash
ss state established
ss state time-wait
ss state close-wait    # often indicates a problem
```

**Filter by address:**

```bash
ss -tn dst 10.0.0.5
ss -tn src 192.168.1.0/24
```

**Show socket memory usage:**

```bash
ss -tm
```

**Count connections per state:**

```bash
ss -s
```

Output shows total TCP sockets, established, closed, orphaned, time-wait, etc.

**Show timer info (useful for debugging hangs):**

```bash
ss -tn -o
```

**Find what's using a port:**

```bash
ss -tlnp sport = :8080
```

If the `-p` flag shows nothing, run with `sudo` — you can only see processes you own without root.""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 9. curl and wget tricks (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "What are some useful curl and wget tricks?",
    r"""**curl:**

```bash
# GET request with headers shown
curl -i https://example.com

# Only show response headers
curl -I https://example.com

# Send custom headers
curl -H "Authorization: Bearer TOKEN" https://api.example.com/data

# POST JSON
curl -X POST https://api.example.com/items \
    -H "Content-Type: application/json" \
    -d '{"name": "test", "value": 42}'

# POST form data
curl -X POST https://example.com/login \
    -d "user=admin&pass=secret"

# Upload a file
curl -F "file=@photo.jpg" https://example.com/upload

# Follow redirects
curl -L https://example.com/redirect

# Save response to file
curl -o output.html https://example.com
curl -O https://example.com/file.tar.gz   # keep original filename

# Resume a download
curl -C - -O https://example.com/bigfile.iso

# Limit download speed
curl --limit-rate 1M -O https://example.com/file.iso

# Use a proxy
curl -x http://proxy:8080 https://example.com

# Silence progress bar but show errors
curl -sS https://example.com

# Timing info
curl -o /dev/null -s -w "Connect: %{time_connect}s\nTTFB: %{time_starttransfer}s\nTotal: %{time_total}s\n" https://example.com
```

**wget:**

```bash
# Download a file
wget https://example.com/file.tar.gz

# Download to a specific filename
wget -O output.tar.gz https://example.com/file.tar.gz

# Resume interrupted download
wget -c https://example.com/bigfile.iso

# Download an entire directory listing
wget -r -np -nH --cut-dirs=2 https://example.com/pub/files/

# Mirror a website
wget --mirror --convert-links --page-requisites https://example.com

# Download in background
wget -b https://example.com/bigfile.iso
tail -f wget-log

# Limit speed
wget --limit-rate=500k https://example.com/file.iso

# Retry on failure
wget --tries=10 --retry-connrefused https://example.com/file.iso

# Download from a list of URLs
wget -i urls.txt

# Ignore SSL certificate errors (for self-signed certs)
wget --no-check-certificate https://self-signed.example.com
```

**When to use which:**
- `curl` is better for API interaction, custom headers, JSON, and scripting (it outputs to stdout by default)
- `wget` is better for downloading files, recursive downloads, and mirroring sites
- `curl` supports more protocols (dict, gopher, imap, etc.)
- `wget` handles retries and recursive downloads more naturally""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 10. Network bonding/teaming (Debian server)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I set up network bonding on a Debian server?",
    r"""Network bonding combines multiple physical interfaces into one logical interface for redundancy and/or throughput.

Install the bonding module support:

```bash
sudo apt install ifenslave
```

The kernel module loads automatically, but you can verify:

```bash
sudo modprobe bonding
lsmod | grep bonding
```

**Configure in /etc/network/interfaces:**

```bash
sudo nano /etc/network/interfaces
```

```
# Bring up physical interfaces with no IP
auto eth0
iface eth0 inet manual
    bond-master bond0

auto eth1
iface eth1 inet manual
    bond-master bond0

# Bond interface
auto bond0
iface bond0 inet static
    address 192.168.1.100/24
    gateway 192.168.1.1
    dns-nameservers 1.1.1.1
    bond-mode 802.3ad
    bond-miimon 100
    bond-lacp-rate fast
    bond-slaves eth0 eth1
```

**Bonding modes:**

| Mode | Name | Use case |
|---|---|---|
| 0 | balance-rr | Round-robin, simple load balancing |
| 1 | active-backup | Failover only, one active at a time |
| 2 | balance-xor | Hash-based load balancing |
| 3 | broadcast | Send on all interfaces |
| 4 | 802.3ad | LACP — requires switch support |
| 5 | balance-tlb | Adaptive transmit load balancing |
| 6 | balance-alb | Adaptive load balancing (no switch support needed) |

**Most common choices:**
- `active-backup` (mode 1) — simple failover, no switch config needed
- `802.3ad` (mode 4) — best performance, requires LACP on the switch

Apply:

```bash
sudo systemctl restart networking
```

Check status:

```bash
cat /proc/net/bonding/bond0
ip addr show bond0
```

The output shows which slave is active, link status, and MII polling interval.""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 11. VLAN configuration (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I configure VLANs on Debian?",
    r"""Install the VLAN package:

```bash
sudo apt install vlan
```

Load the 8021q kernel module:

```bash
sudo modprobe 8021q
echo "8021q" | sudo tee -a /etc/modules
```

**Configure VLANs in /etc/network/interfaces:**

```bash
sudo nano /etc/network/interfaces
```

```
# Physical interface — no IP, just bring it up
auto eth0
iface eth0 inet manual

# VLAN 10 — Management
auto eth0.10
iface eth0.10 inet static
    address 10.10.10.5/24
    gateway 10.10.10.1
    vlan-raw-device eth0

# VLAN 20 — Production
auto eth0.20
iface eth0.20 inet static
    address 10.20.20.5/24
    vlan-raw-device eth0

# VLAN 30 — Storage (DHCP)
auto eth0.30
iface eth0.30 inet dhcp
    vlan-raw-device eth0
```

Apply:

```bash
sudo systemctl restart networking
```

Verify:

```bash
ip -d link show eth0.10
ip addr show eth0.10
cat /proc/net/vlan/config
```

**Manual (temporary) VLAN creation:**

```bash
# Add VLAN 10 on eth0
sudo ip link add link eth0 name eth0.10 type vlan id 10
sudo ip addr add 10.10.10.5/24 dev eth0.10
sudo ip link set eth0.10 up

# Remove it
sudo ip link delete eth0.10
```

**With NetworkManager (nmcli):**

```bash
nmcli connection add type vlan \
    con-name vlan10 \
    ifname eth0.10 \
    dev eth0 \
    id 10 \
    ipv4.addresses 10.10.10.5/24 \
    ipv4.method manual
```

**Notes:**
- Your switch port must be configured as a trunk carrying the relevant VLANs
- VLAN interface names can be anything (`eth0.10`, `vlan10`, etc.), but `parent.vid` is conventional
- Each VLAN interface gets its own routing table entries, so be mindful of routing with multiple VLANs""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 12. MTU and jumbo frames (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "How do I configure MTU and use jumbo frames?",
    r"""**What is MTU?**

MTU (Maximum Transmission Unit) is the largest packet size an interface will send. Default Ethernet MTU is 1500 bytes. Jumbo frames use 9000 bytes, reducing overhead for large transfers.

**Check current MTU:**

```bash
ip link show eth0 | grep mtu
```

**Set MTU temporarily:**

```bash
sudo ip link set eth0 mtu 9000
```

This resets on reboot.

**Set MTU permanently on Debian (/etc/network/interfaces):**

```
auto eth0
iface eth0 inet static
    address 192.168.1.100/24
    gateway 192.168.1.1
    mtu 9000
```

**On systems using NetworkManager:**

```bash
nmcli connection modify "Wired connection 1" 802-3-ethernet.mtu 9000
nmcli connection up "Wired connection 1"
```

**With systemd-networkd:**

```ini
[Match]
Name=eth0

[Link]
MTUBytes=9000

[Network]
Address=192.168.1.100/24
Gateway=192.168.1.1
```

**Test that jumbo frames work end-to-end:**

```bash
# From the sending machine
ping -M do -s 8972 192.168.1.200
```

The `-M do` flag sets "don't fragment". Packet size 8972 + 28 bytes header = 9000. If this fails, something along the path doesn't support 9000 MTU.

**Important requirements for jumbo frames:**
- Every device in the path must support it — both endpoints AND every switch between them
- Your NIC must support it (most gigabit and faster NICs do)
- Managed switches need jumbo frame support enabled per port
- Do NOT use jumbo frames on interfaces facing the internet — WAN MTU is 1500
- NFS, iSCSI, and storage traffic benefit the most from jumbo frames
- Mixed MTU environments can cause silent packet drops — test carefully

**Find the path MTU between two hosts:**

```bash
tracepath 192.168.1.200
```

This reports the MTU at each hop.""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 13. Diagnose slow network - mtr, iperf3, ping (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "My network feels slow. How do I diagnose it?",
    r"""Start with the basics and work outward.

**1. Basic connectivity test:**

```bash
ping -c 10 1.1.1.1
```

Look at: latency (should be <50ms for local ISP), packet loss (should be 0%), and jitter (variation in latency).

**2. Trace the path with mtr:**

```bash
sudo apt install mtr-tiny
mtr 1.1.1.1
```

`mtr` combines ping and traceroute into a live display. Each hop shows loss% and latency. Look for:
- A hop where latency jumps significantly — that's where the bottleneck is
- Packet loss at intermediate hops is sometimes ICMP rate limiting, not real loss. Only loss at the **final** hop matters.

Send a report:

```bash
mtr -rwc 100 1.1.1.1
```

**3. Test bandwidth with iperf3:**

```bash
sudo apt install iperf3
```

On the remote end (server):

```bash
iperf3 -s
```

On your machine (client):

```bash
# TCP throughput test
iperf3 -c server-ip

# UDP test at specific bitrate
iperf3 -c server-ip -u -b 100M

# Test in both directions
iperf3 -c server-ip --bidir

# Run for 30 seconds
iperf3 -c server-ip -t 30

# Use multiple parallel streams
iperf3 -c server-ip -P 4
```

**4. Check for local issues:**

```bash
# Is an interface dropping packets?
ip -s link show eth0

# Check for errors
ethtool -S eth0 | grep -i error

# Check interface speed/duplex
ethtool eth0 | grep -E "Speed|Duplex"

# Check if something is saturating the link
sudo apt install iftop
sudo iftop -i eth0
```

**5. Check DNS speed:**

```bash
time dig example.com
```

If DNS is slow (>100ms), that makes everything feel slow. Try switching to 1.1.1.1 or 9.9.9.10.

**6. Check for bufferbloat:**

```bash
# Start a continuous download, then check latency
ping -c 20 1.1.1.1
```

If latency spikes during heavy downloads (e.g., from 20ms to 500ms), you have bufferbloat. Fix with SQM/fq_codel on your router.

**Common causes of slow networking:**
- Duplex mismatch (check with `ethtool`)
- DNS resolution delays
- Packet loss on the ISP link
- MTU issues causing fragmentation
- Congested WiFi channel""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 14. Connect to WiFi from command line with nmcli (Fedora)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_FED41,
    "How do I connect to WiFi from the command line?",
    r"""Use `nmcli` — it's installed by default on Fedora.

**Scan for networks:**

```bash
nmcli device wifi list
```

This shows SSIDs, signal strength, security type, and channel.

**Connect to a network:**

```bash
nmcli device wifi connect "MyNetwork" password "MyPassword"
```

**Connect to a hidden SSID:**

```bash
nmcli device wifi connect "HiddenSSID" password "MyPassword" hidden yes
```

**Check connection status:**

```bash
nmcli connection show --active
nmcli device status
```

**Disconnect:**

```bash
nmcli device disconnect wlp2s0
```

**Reconnect to a saved network:**

```bash
nmcli connection up "MyNetwork"
```

**List saved connections:**

```bash
nmcli connection show
```

**Delete a saved connection:**

```bash
nmcli connection delete "MyNetwork"
```

**Connect to WPA Enterprise (802.1X):**

```bash
nmcli connection add type wifi con-name "CorpWifi" \
    ifname wlp2s0 ssid "CorpNetwork" \
    wifi-sec.key-mgmt wpa-eap \
    802-1x.eap peap \
    802-1x.phase2-auth mschapv2 \
    802-1x.identity "user@corp.com" \
    802-1x.password "password"
nmcli connection up "CorpWifi"
```

**Set a static IP on WiFi:**

```bash
nmcli connection modify "MyNetwork" \
    ipv4.addresses 192.168.1.50/24 \
    ipv4.gateway 192.168.1.1 \
    ipv4.dns "1.1.1.1" \
    ipv4.method manual
nmcli connection up "MyNetwork"
```

**Set WiFi power management off (can improve latency):**

```bash
sudo iw dev wlp2s0 set power_save off
```""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 15. wpa_supplicant manual config (Arch)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_ARCH,
    "How do I manually configure WiFi with wpa_supplicant?",
    r"""For a minimal setup without NetworkManager.

**1. Find your wireless interface:**

```bash
ip link show
iw dev
```

Suppose it's `wlan0`.

**2. Create the wpa_supplicant config:**

```bash
sudo nano /etc/wpa_supplicant/wpa_supplicant-wlan0.conf
```

```
ctrl_interface=/run/wpa_supplicant
update_config=1
country=US

network={
    ssid="MyNetwork"
    psk="MyPassword"
    priority=1
}

network={
    ssid="BackupNetwork"
    psk="OtherPassword"
    priority=0
}
```

**Generate a hashed PSK (more secure):**

```bash
wpa_passphrase "MyNetwork" "MyPassword"
```

Output:
```
network={
    ssid="MyNetwork"
    #psk="MyPassword"
    psk=a1b2c3d4...long_hex_hash...
}
```

Remove the commented plaintext line and use the hash in your config.

**3. Enable the interface and service:**

```bash
sudo ip link set wlan0 up
sudo systemctl enable --now wpa_supplicant@wlan0
```

**4. Get an IP address with DHCP:**

Using `dhcpcd`:

```bash
sudo systemctl enable --now dhcpcd@wlan0
```

Or using systemd-networkd, create `/etc/systemd/network/25-wlan.network`:

```ini
[Match]
Name=wlan0

[Network]
DHCP=yes
```

Then:

```bash
sudo systemctl enable --now systemd-networkd
```

**5. Verify:**

```bash
iw dev wlan0 link
ip addr show wlan0
ping -c 3 1.1.1.1
```

**Manual (one-time) connection for testing:**

```bash
sudo wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant/wpa_supplicant-wlan0.conf
sudo dhcpcd wlan0
```

**Troubleshooting:**

```bash
# Check wpa_supplicant status
sudo wpa_cli -i wlan0 status

# Scan for networks
sudo wpa_cli -i wlan0 scan
sudo wpa_cli -i wlan0 scan_results

# Check logs
journalctl -u wpa_supplicant@wlan0 -e
```""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 16. Create a WiFi hotspot (Ubuntu 24.04)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_UBU24,
    "How can I create a WiFi hotspot from the command line?",
    r"""Use `nmcli` to create an AP-mode hotspot.

**Quick one-liner:**

```bash
nmcli device wifi hotspot ifname wlp2s0 ssid "MyHotspot" password "SecurePass123"
```

This immediately starts a hotspot. It uses the default settings (WPA2, auto channel, band 2.4 GHz).

**Check it's running:**

```bash
nmcli connection show --active
```

**Stop the hotspot:**

```bash
nmcli connection down Hotspot
```

**Start it again later:**

```bash
nmcli connection up Hotspot
```

**Create with more control:**

```bash
nmcli connection add type wifi \
    con-name "MyAP" \
    ifname wlp2s0 \
    ssid "MyHotspot" \
    autoconnect no \
    wifi.mode ap \
    wifi.band bg \
    wifi.channel 6 \
    ipv4.method shared \
    ipv4.addresses 10.42.0.1/24 \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "SecurePass123"

nmcli connection up "MyAP"
```

The `ipv4.method shared` setting enables NAT and runs a DHCP server automatically. Clients connecting to your hotspot get 10.42.0.x addresses and share your internet.

**Share a specific wired connection:**

The hotspot automatically NATs traffic from WiFi clients through your default route. If you're connected via Ethernet, clients on the hotspot get internet through that connection.

**Check connected clients:**

```bash
arp -a
# Or check DHCP leases
cat /var/lib/NetworkManager/dnsmasq-*.leases
```

**5 GHz hotspot (if your card supports it):**

```bash
nmcli connection modify "MyAP" wifi.band a wifi.channel 36
nmcli connection up "MyAP"
```

Not all WiFi cards support AP mode. Check:

```bash
iw list | grep -A 5 "Supported interface modes"
```

Look for `AP` in the list.""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 17. Docker basics (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I get started with Docker?",
    r"""Install Docker from the official repository (the Debian `docker.io` package is older):

```bash
sudo apt install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/debian trixie stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

**Add your user to the docker group (to avoid sudo):**

```bash
sudo usermod -aG docker $USER
newgrp docker   # apply without logout
```

**Run a container:**

```bash
docker run hello-world
docker run -it ubuntu:24.04 bash   # interactive shell
docker run -d -p 8080:80 nginx     # detached, port mapping
```

**Manage containers:**

```bash
docker ps              # running containers
docker ps -a           # all (including stopped)
docker stop <id>       # graceful stop
docker rm <id>         # remove
docker logs <id>       # view logs
docker exec -it <id> bash   # shell into running container
```

**Work with images:**

```bash
docker images                   # list local images
docker pull python:3.12-slim    # download image
docker rmi <image-id>           # remove image
docker image prune              # clean up unused images
```

**Build a custom image with a Dockerfile:**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "app.py"]
```

```bash
docker build -t myapp:latest .
docker run -d -p 8000:8000 myapp:latest
```

**Volumes (persistent data):**

```bash
# Named volume
docker volume create mydata
docker run -d -v mydata:/var/lib/mysql mysql:8

# Bind mount (host directory)
docker run -d -v /home/user/html:/usr/share/nginx/html nginx
```

**Networks:**

```bash
docker network create mynet
docker run -d --name db --network mynet postgres:16
docker run -d --name app --network mynet myapp
# app can reach db by hostname "db"
```

**Cleanup:**

```bash
docker system prune        # remove stopped containers, unused networks, dangling images
docker system prune -a     # also remove unused images
docker volume prune        # remove unused volumes
```""",
    "flm_containers"
))

# ---------------------------------------------------------------------------
# 18. Docker Compose setup (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I use Docker Compose for a multi-container application?",
    r"""Docker Compose is bundled with Docker as a plugin (`docker compose`, no hyphen). If you installed `docker-compose-plugin`, you already have it.

```bash
docker compose version
```

**Example: Web app with database and cache**

Create `docker-compose.yml`:

```yaml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://appuser:secret@db:5432/myapp
      REDIS_URL: redis://cache:6379
    depends_on:
      db:
        condition: service_healthy
      cache:
        condition: service_started
    volumes:
      - ./app:/app
    restart: unless-stopped

  db:
    image: postgres:16
    environment:
      POSTGRES_USER: appuser
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: myapp
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U appuser -d myapp"]
      interval: 5s
      timeout: 5s
      retries: 5

  cache:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  pgdata:
  redis_data:
```

**Commands:**

```bash
# Start all services (build if needed)
docker compose up -d

# View logs
docker compose logs -f
docker compose logs app     # specific service

# Rebuild after code changes
docker compose up -d --build

# Stop everything
docker compose down

# Stop and remove volumes (destroys data!)
docker compose down -v

# Scale a service
docker compose up -d --scale app=3

# Run a one-off command
docker compose exec db psql -U appuser -d myapp
docker compose run --rm app python manage.py migrate
```

**Environment variables from a file:**

Create `.env`:

```
POSTGRES_PASSWORD=secret
APP_PORT=8000
```

Reference in `docker-compose.yml`:

```yaml
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
  app:
    ports:
      - "${APP_PORT}:8000"
```

**Override for development:**

Create `docker-compose.override.yml`:

```yaml
services:
  app:
    volumes:
      - ./app:/app
    environment:
      DEBUG: "true"
```

It's automatically merged with the main file. For production, use:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```""",
    "flm_containers"
))

# ---------------------------------------------------------------------------
# 19. Podman as Docker replacement (Fedora)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_FED41,
    "How do I use Podman instead of Docker?",
    r"""Podman is pre-installed on Fedora. It's a drop-in Docker replacement that runs containers rootless by default — no daemon needed.

```bash
podman --version
```

**Running containers (same syntax as Docker):**

```bash
podman run hello-world
podman run -it fedora:41 bash
podman run -d -p 8080:80 nginx
```

**Rootless by default:**

Podman runs as your user, no root needed, no daemon running. Containers are isolated in your user namespace.

```bash
podman ps         # your containers
podman images     # your images
```

**Key differences from Docker:**

1. **No daemon** — each `podman` command is a fork, not a client talking to a daemon
2. **Rootless** — containers run as your UID by default
3. **Pod support** — group containers into pods (like Kubernetes)
4. **Systemd integration** — generate systemd units from containers

**Create a pod (group of containers sharing a network):**

```bash
podman pod create --name myapp -p 8080:80
podman run -d --pod myapp --name web nginx
podman run -d --pod myapp --name api myapp-image
# Both containers share localhost, "web" can reach "api" on 127.0.0.1
```

**Docker Compose compatibility:**

```bash
sudo dnf install podman-compose
podman-compose up -d
```

Or use the official `docker compose` with Podman's Docker-compatible socket:

```bash
systemctl --user enable --now podman.socket
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/podman/podman.sock
docker compose up -d
```

**Generate systemd service from a running container:**

```bash
podman generate systemd --new --name my-nginx > ~/.config/systemd/user/my-nginx.service
systemctl --user daemon-reload
systemctl --user enable --now my-nginx
```

Now your container starts on boot (with user lingering enabled):

```bash
loginctl enable-linger $USER
```

**Build images (same Dockerfile syntax):**

```bash
podman build -t myapp:latest .
podman push myapp:latest registry.example.com/myapp:latest
```

**Volume mounts:**

```bash
podman run -v /home/user/data:/data:Z nginx
```

The `:Z` flag applies SELinux labels automatically, which is important on Fedora.""",
    "flm_containers"
))

# ---------------------------------------------------------------------------
# 20. LXC/LXD containers (Ubuntu 24.04)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_UBU24,
    "How do I use LXC/LXD system containers?",
    r"""LXD provides full system containers — they behave like lightweight VMs running a full OS.

**Install LXD:**

```bash
sudo snap install lxd
sudo lxd init --auto    # or interactive: sudo lxd init
```

Add your user to the lxd group:

```bash
sudo usermod -aG lxd $USER
newgrp lxd
```

**Launch a container:**

```bash
lxc launch ubuntu:24.04 mycontainer
lxc launch images:debian/13 deb-test
lxc launch images:alpine/3.19 alpine-test
```

**Interact with containers:**

```bash
lxc exec mycontainer -- bash            # get a shell
lxc exec mycontainer -- apt update      # run a command
lxc file push local.txt mycontainer/tmp/  # copy file in
lxc file pull mycontainer/etc/hostname .  # copy file out
```

**Manage containers:**

```bash
lxc list                    # show all containers
lxc info mycontainer        # detailed info
lxc stop mycontainer
lxc start mycontainer
lxc restart mycontainer
lxc delete mycontainer      # must be stopped first
lxc delete --force mycontainer  # stop + delete
```

**Snapshots:**

```bash
lxc snapshot mycontainer snap0
lxc restore mycontainer snap0
lxc info mycontainer   # lists snapshots
```

**Resource limits:**

```bash
lxc config set mycontainer limits.cpu 2
lxc config set mycontainer limits.memory 1GB
lxc config set mycontainer limits.memory.enforce hard
```

**Port forwarding:**

```bash
lxc config device add mycontainer myport80 proxy \
    listen=tcp:0.0.0.0:8080 connect=tcp:127.0.0.1:80
```

**Shared directories:**

```bash
lxc config device add mycontainer shared disk \
    source=/home/user/shared path=/mnt/shared
```

**Profiles (reusable config):**

```bash
lxc profile create webserver
lxc profile set webserver limits.cpu 2
lxc profile set webserver limits.memory 2GB
lxc launch ubuntu:24.04 web1 --profile default --profile webserver
```

**LXC vs Docker:**
- LXC/LXD: full system containers, run systemd, feel like VMs. Best for running services that need an init system.
- Docker/Podman: application containers, run a single process. Best for microservices and app deployment.""",
    "flm_containers"
))

# ---------------------------------------------------------------------------
# 21. KVM/QEMU/libvirt setup (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I set up KVM virtual machines on Debian?",
    r"""**Check hardware virtualization support:**

```bash
grep -cE '(vmx|svm)' /proc/cpuinfo
```

If the result is > 0, your CPU supports virtualization. Also check it's enabled in BIOS.

**Install KVM and libvirt:**

```bash
sudo apt install qemu-kvm libvirt-daemon-system \
    libvirt-clients virtinst bridge-utils
```

**Add your user to the libvirt group:**

```bash
sudo usermod -aG libvirt $USER
newgrp libvirt
```

**Verify KVM is working:**

```bash
virsh list --all
sudo systemctl status libvirtd
```

**Create a VM from an ISO:**

```bash
virt-install \
    --name debian13-vm \
    --ram 2048 \
    --vcpus 2 \
    --disk path=/var/lib/libvirt/images/debian13.qcow2,size=20 \
    --os-variant debian12 \
    --network bridge=virbr0 \
    --graphics vnc,listen=0.0.0.0 \
    --cdrom /path/to/debian-13-amd64-netinst.iso \
    --boot cdrom,hd
```

For headless (console install):

```bash
virt-install \
    --name debian13-vm \
    --ram 2048 \
    --vcpus 2 \
    --disk path=/var/lib/libvirt/images/debian13.qcow2,size=20 \
    --os-variant debian12 \
    --network bridge=virbr0 \
    --graphics none \
    --console pty,target_type=serial \
    --extra-args 'console=ttyS0,115200n8' \
    --location https://deb.debian.org/debian/dists/trixie/main/installer-amd64/
```

**Basic VM management with virsh:**

```bash
virsh list --all          # list all VMs
virsh start debian13-vm   # start
virsh shutdown debian13-vm  # graceful shutdown
virsh destroy debian13-vm   # force off (like pulling power)
virsh reboot debian13-vm
virsh console debian13-vm   # serial console
virsh autostart debian13-vm  # start on host boot
```

**Set up a network bridge for VMs to be on the LAN:**

Edit `/etc/network/interfaces`:

```
auto br0
iface br0 inet static
    address 192.168.1.100/24
    gateway 192.168.1.1
    bridge_ports eth0
    bridge_stp off
    bridge_fd 0
```

Then use `--network bridge=br0` in virt-install.

**Resize a disk:**

```bash
virsh shutdown debian13-vm
qemu-img resize /var/lib/libvirt/images/debian13.qcow2 +10G
virsh start debian13-vm
# Then inside the VM: growpart, resize2fs, etc.
```""",
    "flm_containers"
))

# ---------------------------------------------------------------------------
# 22. virt-manager and virsh commands (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "What are the most useful virsh commands for managing VMs?",
    r"""**VM lifecycle:**

```bash
virsh list --all                # list all VMs (running + stopped)
virsh start myvm                # start a VM
virsh shutdown myvm             # graceful ACPI shutdown
virsh destroy myvm              # force power off
virsh reboot myvm               # reboot
virsh suspend myvm              # pause (freeze CPU)
virsh resume myvm               # unpause
virsh undefine myvm             # delete VM definition
virsh undefine --remove-all-storage myvm  # delete VM + disks
```

**Info and monitoring:**

```bash
virsh dominfo myvm              # CPU, memory, state
virsh domblklist myvm           # list disks
virsh domiflist myvm            # list network interfaces
virsh domifaddr myvm            # get IP address
virsh vcpuinfo myvm             # vCPU placement
virsh dommemstat myvm           # memory stats
```

**Snapshots:**

```bash
virsh snapshot-create-as myvm snap1 "Before upgrade"
virsh snapshot-list myvm
virsh snapshot-revert myvm snap1
virsh snapshot-delete myvm snap1
```

**Live modify resources:**

```bash
# Hot-add memory (if VM supports it)
virsh setmem myvm 4G --live

# Change max vCPUs (requires restart)
virsh setvcpus myvm 4 --config --maximum
virsh setvcpus myvm 4 --config
```

**Clone a VM:**

```bash
virsh shutdown myvm
virt-clone --original myvm --name myvm-clone --auto-clone
```

**Migrate a VM to another host:**

```bash
virsh migrate --live myvm qemu+ssh://otherhost/system
```

**Edit VM XML directly:**

```bash
virsh edit myvm    # opens XML in $EDITOR
virsh dumpxml myvm > myvm.xml  # export
virsh define myvm.xml          # import
```

**Attach devices:**

```bash
# Add a disk
virsh attach-disk myvm /path/to/extra.qcow2 vdb --driver qemu --subdriver qcow2 --persistent

# Add a network interface
virsh attach-interface myvm bridge br0 --model virtio --persistent

# Detach
virsh detach-disk myvm vdb --persistent
```

**For a GUI:** install `virt-manager`:

```bash
sudo apt install virt-manager
```

It provides a full graphical interface for creating, managing, and viewing VM consoles. You can also connect to remote libvirtd instances via SSH.""",
    "flm_containers"
))

# ---------------------------------------------------------------------------
# 23. Convert between VM formats (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "How do I convert between VM disk formats like qcow2, vmdk, and vdi?",
    r"""Use `qemu-img` from the QEMU package.

**Convert between formats:**

```bash
# VMDK (VMware) to QCOW2 (KVM)
qemu-img convert -f vmdk -O qcow2 disk.vmdk disk.qcow2

# VDI (VirtualBox) to QCOW2
qemu-img convert -f vdi -O qcow2 disk.vdi disk.qcow2

# QCOW2 to VMDK
qemu-img convert -f qcow2 -O vmdk disk.qcow2 disk.vmdk

# QCOW2 to VDI
qemu-img convert -f qcow2 -O vdi disk.qcow2 disk.vdi

# Raw to QCOW2
qemu-img convert -f raw -O qcow2 disk.img disk.qcow2

# QCOW2 to raw
qemu-img convert -f qcow2 -O raw disk.qcow2 disk.raw
```

**With compression (QCOW2 output only):**

```bash
qemu-img convert -f vmdk -O qcow2 -c disk.vmdk disk.qcow2
```

The `-c` flag compresses the output. Read performance is slightly lower, but disk savings can be significant.

**Check image info:**

```bash
qemu-img info disk.qcow2
```

Shows format, virtual size, actual disk usage, and any backing files.

**Resize during or after conversion:**

```bash
# Resize a QCOW2 image
qemu-img resize disk.qcow2 +10G

# Shrink (dangerous — must shrink filesystem first!)
qemu-img resize --shrink disk.qcow2 20G
```

**Create a new blank image:**

```bash
qemu-img create -f qcow2 newdisk.qcow2 50G
qemu-img create -f vmdk newdisk.vmdk 50G
```

**VirtualBox-specific alternative:**

```bash
VBoxManage clonehd disk.vdi disk.vmdk --format VMDK
VBoxManage clonehd disk.vmdk disk.vdi --format VDI
```

**Format comparison:**

| Format | Thin provisioned | Snapshots | Best for |
|---|---|---|---|
| qcow2 | Yes | Yes (internal) | KVM/QEMU/libvirt |
| vmdk | Yes (sparse) | No (use VMware snapshots) | VMware ESXi/Workstation |
| vdi | Yes | No (use VBox snapshots) | VirtualBox |
| raw | No | No | Maximum I/O performance |

**Tip:** When converting, `qemu-img` only copies allocated blocks, so the output file is often smaller than the virtual disk size.""",
    "flm_containers"
))

# ---------------------------------------------------------------------------
# 24. Git basics for beginners (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "I'm new to Git. Can you explain the basics?",
    r"""Git tracks changes to files in a repository. Here's how it works:

**Initial setup (once):**

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

**Create a new repository:**

```bash
mkdir myproject && cd myproject
git init
```

Or clone an existing one:

```bash
git clone https://github.com/user/repo.git
```

**The three areas:**

1. **Working directory** — your actual files
2. **Staging area (index)** — what will go into the next commit
3. **Repository (.git)** — the commit history

**Basic workflow:**

```bash
# Check what's changed
git status

# Stage specific files
git add file1.py file2.py

# Stage everything
git add .

# Commit staged changes
git commit -m "Add user authentication"

# See commit history
git log
git log --oneline    # compact format
```

**Viewing changes:**

```bash
# What's changed but not staged
git diff

# What's staged and ready to commit
git diff --staged

# Changes in a specific file
git diff file.py
```

**Undoing things:**

```bash
# Unstage a file (keep changes in working dir)
git restore --staged file.py

# Discard changes in working directory
git restore file.py

# Change the last commit message
git commit --amend -m "Better message"
```

**Working with remotes:**

```bash
# See remotes
git remote -v

# Push to remote
git push origin main

# Pull latest changes
git pull origin main

# Fetch without merging
git fetch origin
```

**Ignoring files:**

Create `.gitignore`:

```
__pycache__/
*.pyc
.env
node_modules/
*.log
```

**Key concepts:**
- A **commit** is a snapshot of your project at a point in time
- Each commit has a unique SHA hash (like `a1b2c3d`)
- `HEAD` points to your current commit
- `origin` is the default name for the remote repository
- `main` (or `master`) is the default branch name""",
    "flm_git"
))

# ---------------------------------------------------------------------------
# 25. Git branching workflow (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "How do I work with Git branches?",
    r"""Branches let you work on features or fixes in isolation.

**Create and switch branches:**

```bash
# Create a new branch and switch to it
git checkout -b feature/login

# Same thing, newer syntax
git switch -c feature/login

# Switch to an existing branch
git checkout main
git switch main

# List branches
git branch          # local
git branch -r       # remote
git branch -a       # all
```

**Naming conventions:**
- `feature/description` — new features
- `fix/description` — bug fixes
- `hotfix/description` — urgent production fixes
- `release/v1.2` — release preparation

**Work on a branch:**

```bash
git switch -c feature/user-profile
# ... make changes ...
git add .
git commit -m "Add user profile page"
# ... more changes ...
git add .
git commit -m "Add avatar upload to profile"
```

**Merge a branch:**

```bash
# Switch to the target branch
git switch main

# Merge the feature branch
git merge feature/user-profile
```

If there are no conflicts, Git does a fast-forward or creates a merge commit.

**Delete a branch after merging:**

```bash
git branch -d feature/user-profile        # local
git push origin --delete feature/user-profile  # remote
```

**Push a branch to remote:**

```bash
git push -u origin feature/login
```

The `-u` flag sets the upstream, so future `git push` and `git pull` work without specifying the remote and branch.

**Keep a branch up to date with main:**

```bash
git switch feature/login
git merge main
# or
git rebase main   # cleaner history, but rewrites commits
```

**See branch history:**

```bash
git log --oneline --graph --all
```

This shows all branches as a visual tree.

**Stash changes before switching:**

If you have uncommitted work:

```bash
git stash
git switch other-branch
# ... do work ...
git switch feature/login
git stash pop
```""",
    "flm_git"
))

# ---------------------------------------------------------------------------
# 26. Git rebase vs merge (short Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SHORT_DEB13,
    "What's the difference between git rebase and git merge?",
    r"""Both integrate changes from one branch into another, but they do it differently.

**Merge** creates a new "merge commit" with two parents:

```
main:    A---B---C---M
              \     /
feature:       D---E
```

```bash
git switch main
git merge feature
```

**Rebase** replays your commits on top of the target branch:

```
Before:
main:    A---B---C
              \
feature:       D---E

After rebase:
main:    A---B---C
                  \
feature:           D'---E'
```

```bash
git switch feature
git rebase main
git switch main
git merge feature    # now a fast-forward
```

D' and E' are new commits — same changes but different SHAs.

**When to use each:**

| Merge | Rebase |
|---|---|
| Preserves exact history | Creates linear history |
| Safe for shared/public branches | NEVER rebase public/shared branches |
| Shows when branches diverged | Easier to read `git log` |
| Simple, less risky | Requires understanding of rewriting history |

**Interactive rebase (clean up before merging):**

```bash
git rebase -i main
```

This opens an editor where you can:
- `pick` — keep a commit
- `squash` — combine with previous commit
- `reword` — change commit message
- `drop` — delete a commit
- `edit` — pause to modify

**The golden rule:**
Never rebase commits that have been pushed to a shared remote. Rebasing rewrites history, which breaks other people's branches that are based on the old commits.

**Safe pattern:**
1. Work on your feature branch
2. Before merging, rebase onto main to get a clean linear history
3. Push (or force-push your own branch)
4. Merge into main (fast-forward)

```bash
git switch feature
git rebase main
git push --force-with-lease origin feature   # safe force push
git switch main
git merge feature
```

`--force-with-lease` is safer than `--force` — it refuses to push if someone else has pushed to that branch since you last fetched.""",
    "flm_git"
))

# ---------------------------------------------------------------------------
# 27. Fix common git mistakes (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I fix common git mistakes?",
    r"""**Typo in last commit message:**

```bash
git commit --amend -m "Correct message"
```

**Forgot to add a file to the last commit:**

```bash
git add forgotten_file.py
git commit --amend --no-edit
```

**Committed to the wrong branch:**

```bash
# Move the last commit to the correct branch
git switch correct-branch
git cherry-pick main    # copies the commit
git switch main
git reset --hard HEAD~1  # remove it from main
```

**Undo the last commit but keep changes:**

```bash
git reset --soft HEAD~1    # changes stay staged
git reset HEAD~1           # changes stay in working dir (unstaged)
git reset --hard HEAD~1    # discard everything (dangerous!)
```

**Accidentally ran git reset --hard:**

```bash
# Reflog saves you! Find the lost commit
git reflog
# Find the SHA you want to go back to
git reset --hard abc1234
```

The reflog keeps a record of every HEAD movement for 90 days.

**Discard all uncommitted changes:**

```bash
git checkout -- .          # discard modified files
git clean -fd              # remove untracked files and directories
```

**Revert a commit that's already pushed (safe for shared branches):**

```bash
git revert abc1234
```

This creates a new commit that undoes the changes, without rewriting history.

**Remove a file from Git but keep it locally:**

```bash
git rm --cached secret.env
echo "secret.env" >> .gitignore
git commit -m "Stop tracking secret.env"
```

**Fix a merge gone wrong:**

```bash
# Abort a merge in progress
git merge --abort

# Undo a completed merge (if not yet pushed)
git reset --hard HEAD~1
```

**See what changed in a specific commit:**

```bash
git show abc1234
```

**Find which commit introduced a bug:**

```bash
git bisect start
git bisect bad          # current version is broken
git bisect good v1.0    # this version was fine
# Git checks out a middle commit — test it, then:
git bisect good   # or
git bisect bad
# Repeat until it finds the culprit
git bisect reset   # when done
```""",
    "flm_git"
))

# ---------------------------------------------------------------------------
# 28. Set up a bare git repo on a server (Debian minimal)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEBMIN,
    "How do I set up a bare git repository on a server?",
    r"""A bare repo has no working directory — it's just the `.git` internals. This is what you push to.

**Create the bare repo:**

```bash
sudo mkdir -p /srv/git
sudo git init --bare /srv/git/myproject.git
```

**Set ownership (assuming a `git` user):**

```bash
sudo adduser --system --shell /usr/bin/git-shell --group git
sudo chown -R git:git /srv/git/myproject.git
```

Using `git-shell` as the login shell restricts the user to git operations only — they can't get a regular shell.

**On the client, add the remote and push:**

```bash
cd myproject
git remote add origin git@server:/srv/git/myproject.git
git push -u origin main
```

**Clone from another machine:**

```bash
git clone git@server:/srv/git/myproject.git
```

**Allow multiple users to push (shared repo):**

```bash
sudo git init --bare --shared=group /srv/git/myproject.git
sudo chgrp -R developers /srv/git/myproject.git
```

The `--shared=group` flag sets permissions so any user in the `developers` group can push.

**Organize multiple repos:**

```
/srv/git/
    project-a.git/
    project-b.git/
    website.git/
```

**Alternative: use SSH keys for access control.**

Add public keys to `/home/git/.ssh/authorized_keys`:

```
command="git-shell -c \"$SSH_ORIGINAL_COMMAND\"",no-port-forwarding,no-agent-forwarding ssh-ed25519 AAAA... user@laptop
```

This restricts each key to git operations only.

**Enable post-receive hook for auto-deploy:**

```bash
nano /srv/git/myproject.git/hooks/post-receive
```

```bash
#!/bin/bash
GIT_WORK_TREE=/var/www/mysite git checkout -f main
```

```bash
chmod +x /srv/git/myproject.git/hooks/post-receive
```

Now every push to main automatically deploys to `/var/www/mysite`.""",
    "flm_git"
))

# ---------------------------------------------------------------------------
# 29. Git hooks for automation (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I use git hooks for automation?",
    r"""Git hooks are scripts that run automatically at specific points in the git workflow. They live in `.git/hooks/` in each repository.

**Available hooks (most useful ones):**

| Hook | When it runs | Common use |
|---|---|---|
| `pre-commit` | Before a commit is created | Linting, formatting, tests |
| `commit-msg` | After editing commit message | Enforce message format |
| `pre-push` | Before pushing to remote | Run tests |
| `post-merge` | After a merge completes | Install dependencies |
| `post-checkout` | After switching branches | Install dependencies |
| `post-receive` | On server after push received | Deploy |

**Example: pre-commit hook (lint Python):**

```bash
nano .git/hooks/pre-commit
```

```bash
#!/bin/bash
# Run flake8 on staged Python files
STAGED=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')
if [ -n "$STAGED" ]; then
    echo "Running flake8..."
    flake8 $STAGED
    if [ $? -ne 0 ]; then
        echo "flake8 failed. Fix errors before committing."
        exit 1
    fi
fi
```

```bash
chmod +x .git/hooks/pre-commit
```

**Example: commit-msg hook (enforce format):**

```bash
nano .git/hooks/commit-msg
```

```bash
#!/bin/bash
MSG=$(cat "$1")
if ! echo "$MSG" | grep -qE '^(feat|fix|docs|refactor|test|chore): .+'; then
    echo "Commit message must start with: feat|fix|docs|refactor|test|chore"
    echo "Example: feat: add user login"
    exit 1
fi
```

```bash
chmod +x .git/hooks/commit-msg
```

**Example: pre-push hook (run tests):**

```bash
nano .git/hooks/pre-push
```

```bash
#!/bin/bash
echo "Running tests before push..."
python -m pytest tests/
if [ $? -ne 0 ]; then
    echo "Tests failed. Push aborted."
    exit 1
fi
```

```bash
chmod +x .git/hooks/pre-push
```

**Share hooks with the team:**

Hooks in `.git/hooks/` are not tracked by git. To share them:

1. Create a `hooks/` directory in your repo and put hooks there
2. Configure git to use it:
```bash
git config core.hooksPath hooks
```
3. Or symlink:
```bash
ln -s ../../hooks/pre-commit .git/hooks/pre-commit
```

**Skip a hook temporarily:**

```bash
git commit --no-verify -m "WIP: skip hooks"
git push --no-verify
```

**Server-side hooks (on bare repos):**

The `post-receive` hook is the most common:

```bash
#!/bin/bash
while read oldrev newrev refname; do
    if [ "$refname" = "refs/heads/main" ]; then
        echo "Deploying main..."
        GIT_WORK_TREE=/var/www/app git checkout -f main
        cd /var/www/app && ./deploy.sh
    fi
done
```""",
    "flm_git"
))

# ---------------------------------------------------------------------------
# 30. Python virtual environments (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I use Python virtual environments?",
    r"""Virtual environments give each project its own isolated set of packages.

**Using venv (built into Python 3):**

```bash
# Install venv support (Debian separates it)
sudo apt install python3-venv

# Create a virtual environment
python3 -m venv myproject-env

# Activate it
source myproject-env/bin/activate

# Your prompt changes to show (myproject-env)
# Now pip installs go into the venv only
pip install requests flask

# See what's installed
pip list

# Freeze dependencies
pip freeze > requirements.txt

# Deactivate
deactivate
```

**Install from requirements.txt:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Where to put the venv:**

Convention is to create it inside the project directory:

```
myproject/
    venv/          # virtual environment (don't commit this)
    src/
    requirements.txt
    .gitignore     # should include: venv/
```

**virtualenv (third-party, more features):**

```bash
pip install virtualenv
virtualenv myenv                    # uses default Python
virtualenv -p python3.11 myenv     # specific version
```

Main advantages over venv:
- Faster creation (caches packages)
- Can target different Python versions
- Works with Python 2 (legacy)

**venv vs virtualenv:**

For most purposes, `venv` is sufficient. It's built-in and officially recommended. Use `virtualenv` only if you need its extra features.

**Common mistakes:**
- Don't commit the venv to git — add it to `.gitignore`
- Don't install packages globally with `pip` on Debian — it conflicts with apt-managed packages
- Always activate the venv before running pip or python
- If you move a project, recreate the venv — it contains absolute paths

**Quick tip for scripts:**

```bash
#!/usr/bin/env python3
```

Use this shebang so the script uses the active venv's Python when the venv is activated.""",
    "flm_python"
))

# ---------------------------------------------------------------------------
# 31. Install Python from source on Debian 12
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB12,
    "My Debian 12 has Python 3.11 but I need Python 3.12. How do I install it from source?",
    r"""**Install build dependencies:**

```bash
sudo apt update
sudo apt install build-essential zlib1g-dev libncurses5-dev \
    libgdbm-dev libnss3-dev libssl-dev libreadline-dev \
    libffi-dev libsqlite3-dev wget libbz2-dev liblzma-dev
```

**Download and compile:**

```bash
cd /tmp
wget https://www.python.org/ftp/python/3.12.8/Python-3.12.8.tgz
tar xzf Python-3.12.8.tgz
cd Python-3.12.8

./configure --enable-optimizations --with-lto --prefix=/usr/local

make -j$(nproc)
```

`--enable-optimizations` runs profile-guided optimization (PGO), which makes Python ~10-20% faster. `--with-lto` enables link-time optimization. Both increase compile time but are worth it.

**Install alongside system Python (important!):**

```bash
sudo make altinstall
```

**Use `altinstall`, not `install`!** `make install` would create a `python3` symlink that overwrites the system Python, potentially breaking Debian tools like `apt`.

**Verify:**

```bash
python3.12 --version
python3.12 -m pip --version
```

The system Python remains untouched:

```bash
python3 --version   # still 3.11.2
```

**Create a venv with the new version:**

```bash
python3.12 -m venv myproject-env
source myproject-env/bin/activate
python --version   # 3.12.8
```

**If modules are missing (like _ssl or _sqlite3):**

This means a build dependency was missing. Install it and rebuild:

```bash
# Check what's missing
python3.12 -c "import ssl; print(ssl.OPENSSL_VERSION)"
python3.12 -c "import sqlite3"
```

If either fails, reinstall the dev packages and recompile:

```bash
sudo apt install libssl-dev libsqlite3-dev
cd /tmp/Python-3.12.8
make clean
./configure --enable-optimizations --with-lto --prefix=/usr/local
make -j$(nproc)
sudo make altinstall
```

**To uninstall later:**

There's no `make uninstall`. Remove manually:

```bash
sudo rm /usr/local/bin/python3.12
sudo rm /usr/local/bin/pip3.12
sudo rm -rf /usr/local/lib/python3.12
```""",
    "flm_python"
))

# ---------------------------------------------------------------------------
# 32. pip, pipx, and system packages (Ubuntu 24.04) - PEP 668
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_UBU24,
    "I get 'externally-managed-environment' error when trying to pip install. What's going on?",
    r"""This is PEP 668 in action. Starting with Ubuntu 23.04 (and Debian 12), the system Python is marked as "externally managed" — meaning pip refuses to install packages globally to prevent conflicts with apt-managed packages.

**The error:**
```
error: externally-managed-environment
```

**You have several options:**

**1. Use a virtual environment (recommended):**

```bash
python3 -m venv ~/.venvs/myproject
source ~/.venvs/myproject/bin/activate
pip install requests flask   # works fine inside a venv
```

**2. Use pipx for CLI tools:**

```bash
sudo apt install pipx
pipx ensurepath   # add ~/.local/bin to PATH

# Install CLI tools in isolated environments
pipx install youtube-dl
pipx install black
pipx install httpie

# Run a tool once without installing
pipx run cowsay "hello"

# List installed tools
pipx list

# Upgrade
pipx upgrade black
```

`pipx` creates a separate venv for each tool automatically. It's the right way to install Python command-line applications.

**3. Install the Debian/Ubuntu package (if available):**

```bash
apt search python3-requests
sudo apt install python3-requests
```

Many popular Python packages are available as system packages (`python3-*`).

**4. Force pip install globally (not recommended):**

```bash
pip install --break-system-packages requests
```

This works but can break system tools that depend on specific package versions. Only do this if you know what you're doing.

**5. User-level install:**

```bash
pip install --user requests
```

This installs to `~/.local/lib/python3.12/` and doesn't affect the system. Add `--break-system-packages` if it still complains.

**Best practices:**
- Use **venv** for project development
- Use **pipx** for standalone CLI tools
- Use **apt** (`python3-*`) for system-wide libraries
- Avoid `pip install` outside a venv entirely

**Quick alias for convenience:**

```bash
echo 'alias mkvenv="python3 -m venv .venv && source .venv/bin/activate"' >> ~/.bashrc
source ~/.bashrc
# Now just: cd myproject && mkvenv
```""",
    "flm_python"
))

# ---------------------------------------------------------------------------
# 33. pyenv for managing Python versions (Arch)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_ARCH,
    "How do I use pyenv to manage multiple Python versions?",
    r"""**Install pyenv:**

```bash
# Install dependencies
sudo pacman -S base-devel openssl zlib xz tk

# Install pyenv via the official installer
curl https://pyenv.run | bash
```

**Add to your shell config (~/.zshrc for Arch with zsh):**

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Restart your shell:

```bash
exec zsh
```

**Install Python versions:**

```bash
# List available versions
pyenv install --list | grep '^\s*3\.'

# Install specific versions
pyenv install 3.11.10
pyenv install 3.12.8
pyenv install 3.13.1

# List installed versions
pyenv versions
```

**Set Python version:**

```bash
# Global default
pyenv global 3.12.8

# Per-directory (creates .python-version file)
cd ~/myproject
pyenv local 3.11.10

# Per-shell session
pyenv shell 3.13.1

# Check active version
pyenv version
python --version
```

The priority order is: shell > local > global.

**How it works:**

pyenv inserts shim executables into your PATH. When you run `python`, the shim checks `.python-version`, `PYENV_VERSION`, or the global setting, then runs the correct Python binary.

**Use with virtual environments (pyenv-virtualenv plugin):**

It's included with `pyenv.run` installer:

```bash
# Create a venv with a specific Python
pyenv virtualenv 3.12.8 myproject-env

# Activate it
pyenv activate myproject-env

# Set it as local for a directory
cd ~/myproject
pyenv local myproject-env

# List virtualenvs
pyenv virtualenvs

# Delete
pyenv virtualenv-delete myproject-env
```

**Uninstall a Python version:**

```bash
pyenv uninstall 3.11.10
```

**Build troubleshooting:**

If `pyenv install` fails, it's usually missing build deps:

```bash
# On Arch, make sure you have:
sudo pacman -S base-devel openssl zlib xz tk sqlite libffi
```

**Update pyenv itself:**

```bash
pyenv update
```""",
    "flm_python"
))

# ---------------------------------------------------------------------------
# 34. Multi-turn: Network connectivity troubleshooting (Debian)
# ---------------------------------------------------------------------------
conversations.append(make_multi(
    SYS_DEB13,
    [
        ("user",
         "I can ping machines on my local network but I can't reach the internet. What should I check?"),
        ("assistant",
         r"""Let's work through this step by step.

First, verify you can ping your gateway:

```bash
ip route show default
ping -c 3 <gateway-ip>
```

What's your default gateway, and can you ping it?"""),
        ("user",
         "My gateway is 192.168.1.1 and yes I can ping it fine."),
        ("assistant",
         r"""Good, so the local network is fine. Now let's test if packets can reach the internet at all. Try pinging an external IP (bypassing DNS):

```bash
ping -c 3 1.1.1.1
```

If this works, the problem is DNS. If it fails, it's a routing or firewall issue. What happens?"""),
        ("user",
         "Ping to 1.1.1.1 times out. 100% packet loss."),
        ("assistant",
         r"""So the problem is between your gateway and the internet. Let's check a few things:

**1. Is your gateway actually routing traffic?**

```bash
traceroute 1.1.1.1
```

If it stops at your gateway (hop 1), the router itself isn't forwarding packets.

**2. Check if a local firewall is blocking outgoing traffic:**

```bash
sudo iptables -L -n -v
sudo nft list ruleset    # if using nftables
```

Look for any DROP or REJECT rules on OUTPUT or FORWARD chains.

**3. Check if there's a default route issue:**

```bash
ip route show
```

You should see something like:
```
default via 192.168.1.1 dev eth0
```

If the default route is missing or points to the wrong gateway, that's the problem.

What does traceroute show, and what are your iptables rules?"""),
        ("user",
         "traceroute shows only hop 1 (my gateway) then all asterisks. iptables shows an ACCEPT policy on all chains with no rules. Route looks correct."),
        ("assistant",
         r"""Since your machine's config is clean — correct route, no firewall blocking — the issue is on the router/gateway itself.

Common causes:
1. **Router's WAN connection is down** — check the router admin page (usually http://192.168.1.1)
2. **ISP outage** — check with your ISP
3. **Router needs a reboot** — power cycle it
4. **NAT/masquerade is broken on the router**

To confirm it's the router, try from another device on the same network. If nothing can reach the internet, it's definitely the router or ISP.

**Quick checks from your machine:**

```bash
# Try a different external IP
ping -c 3 8.8.8.8
ping -c 3 9.9.9.9

# Check if your router's WAN interface has an IP
# (access router's admin page or check its status via SSH if possible)
```

If you have access to the router's CLI (e.g., OpenWrt):

```bash
# Check WAN interface
ip addr show eth1
# Check if the router can ping outside
ping 1.1.1.1
# Check NAT rules
iptables -t nat -L
```

Most likely: power cycle the router. If the problem persists, contact your ISP."""),
    ],
    "flm_multiturn_networking"
))

# ---------------------------------------------------------------------------
# 35. Multi-turn: Container won't start (Debian)
# ---------------------------------------------------------------------------
conversations.append(make_multi(
    SYS_DEB13,
    [
        ("user",
         "My Docker container keeps exiting immediately after starting. How do I debug this?"),
        ("assistant",
         r"""Let's investigate. First, check the container status and exit code:

```bash
docker ps -a | grep <container-name>
```

Also grab the logs:

```bash
docker logs <container-name>
```

What's the exit code and what do the logs say?"""),
        ("user",
         "Exit code is 1. The logs say 'Error: bind: address already in use'."),
        ("assistant",
         r"""Port conflict. Something else is already using the port your container needs.

Check what's using it:

```bash
ss -tlnp sport = :8080
```

Replace `8080` with whatever port your container is trying to bind to. What port is it, and what's already using it?"""),
        ("user",
         "It's port 5432. Another container named 'old-postgres' is using it."),
        ("assistant",
         r"""You have two options:

**Option 1: Stop the old container:**

```bash
docker stop old-postgres
docker rm old-postgres    # if you don't need it anymore
```

Then start your container again.

**Option 2: Map to a different host port:**

```bash
docker run -d -p 5433:5432 --name my-postgres postgres:16
```

This maps host port 5433 to the container's 5432. Your app connects to `localhost:5433` instead.

**Option 3: Put them on different Docker networks (if both need port 5432):**

```bash
docker network create newnet
docker run -d --name my-postgres --network newnet -p 5433:5432 postgres:16
```

Which approach works for your situation?"""),
        ("user",
         "I stopped the old container and started mine, but now I get a different error: 'permission denied' when it tries to write to a mounted volume."),
        ("assistant",
         r"""Volume permission issues are common. The container process runs as a specific UID, and if the host directory has different ownership, writes fail.

**Check what user the container runs as:**

```bash
docker inspect <container-name> --format '{{.Config.User}}'
```

For Postgres, the container runs as UID 999 (postgres user inside the container).

**Fix the host directory permissions:**

```bash
# Check current ownership
ls -la /path/to/your/volume/

# Set ownership to match the container's user
sudo chown -R 999:999 /path/to/your/volume/
```

**Or use a named Docker volume instead of a bind mount:**

```bash
docker volume create pgdata
docker run -d -p 5432:5432 \
    -v pgdata:/var/lib/postgresql/data \
    --name my-postgres postgres:16
```

Named volumes handle permissions automatically because Docker manages them.

**If you need a bind mount with a specific user:**

```bash
docker run -d -p 5432:5432 \
    -v /my/data:/var/lib/postgresql/data \
    --user $(id -u):$(id -g) \
    --name my-postgres postgres:16
```

Though for Postgres specifically, this can cause other issues since it expects to run as the `postgres` user. Named volumes are the cleanest solution for databases."""),
    ],
    "flm_multiturn_containers"
))

# ---------------------------------------------------------------------------
# 36. Multi-turn: Git merge conflict resolution (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make_multi(
    None,
    [
        ("user",
         "I tried to merge a branch and got conflicts. How do I resolve them?"),
        ("assistant",
         r"""Don't panic — conflicts are normal. Let's see what happened.

```bash
git status
```

This shows which files have conflicts. They'll be listed as "both modified". Can you share what git status shows?"""),
        ("user",
         "It shows two files with conflicts: src/config.py and src/app.py"),
        ("assistant",
         r"""Open one of the files. Git marks conflicts like this:

```python
<<<<<<< HEAD
database_url = "postgresql://localhost:5432/mydb"
=======
database_url = "postgresql://db-server:5432/production"
>>>>>>> feature/deploy
```

Everything between `<<<<<<< HEAD` and `=======` is your current branch's version. Everything between `=======` and `>>>>>>> feature/deploy` is the incoming branch's version.

**To resolve:**

1. Edit the file to keep the version you want (or combine both):

```python
database_url = "postgresql://db-server:5432/production"
```

2. Remove ALL the conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)

3. After fixing both files, stage them:

```bash
git add src/config.py src/app.py
```

4. Complete the merge:

```bash
git commit
```

Git opens your editor with a pre-filled merge commit message. Save and close.

Go ahead and edit those files. Want me to help decide which changes to keep?"""),
        ("user",
         "I fixed config.py but app.py has a big conflict that's hard to understand. Is there a tool to help?"),
        ("assistant",
         r"""Yes. Try a visual merge tool:

```bash
git mergetool
```

If you don't have one configured, set one up:

```bash
# vimdiff (terminal)
git config --global merge.tool vimdiff

# VS Code
git config --global merge.tool vscode
git config --global mergetool.vscode.cmd 'code --wait --merge $REMOTE $LOCAL $BASE $MERGED'
```

**Other ways to understand the conflict:**

```bash
# See what each side changed (3-way diff)
git diff --merge

# See only what the other branch changed
git diff HEAD...feature/deploy -- src/app.py

# See only what your branch changed
git diff feature/deploy...HEAD -- src/app.py
```

**If you want to just take one side entirely:**

```bash
# Keep your version
git checkout --ours src/app.py

# Keep their version
git checkout --theirs src/app.py
```

**If the merge is a disaster and you want to start over:**

```bash
git merge --abort
```

This resets everything to before the merge attempt. No changes lost.

After resolving, don't forget:

```bash
git add src/app.py
git commit
```"""),
    ],
    "flm_multiturn_git"
))


# -- Write output --------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "a") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    print(f"Appended {len(conversations)} conversations to {OUTPUT}")


if __name__ == "__main__":
    main()
