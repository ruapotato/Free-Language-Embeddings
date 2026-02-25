#!/usr/bin/env python3
"""Write hand-crafted SFT conversations for V4 training -- Batch 8.

Topics: Advanced networking, storage, systemd, config management,
security, database admin, CI/CD, and real-world multi-turn scenarios.

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


# -- Helpers ------------------------------------------------------------------

def make(system, question, answer, source):
    """Build one conversation dict."""
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


# -- Conversations ------------------------------------------------------------

conversations = []

# ===========================================================================
# 1. eBPF and BCC tools for network observability (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How can I use eBPF and BCC tools to observe network activity on my Debian system?",
    r"""eBPF (extended Berkeley Packet Filter) lets you run sandboxed programs inside the Linux kernel for tracing and observability without writing kernel modules. BCC (BPF Compiler Collection) provides pre-built tools that leverage eBPF.

**Install BCC tools on Debian 13:**

```bash
sudo apt update
sudo apt install bpfcc-tools linux-headers-$(uname -r)
```

The tools are installed with a `-bpfcc` suffix on Debian.

**Key networking tools:**

1. **tcpconnect-bpfcc** -- trace outbound TCP connections:
```bash
sudo tcpconnect-bpfcc
# PID    COMM         IP SADDR            DADDR            DPORT
# 12345  curl         4  192.168.1.10     93.184.216.34    443
```

2. **tcpaccept-bpfcc** -- trace inbound TCP connections:
```bash
sudo tcpaccept-bpfcc
# PID    COMM         IP RADDR            RPORT  LADDR            LPORT
# 6789   nginx        4  10.0.0.5         54321  10.0.0.1         80
```

3. **tcpretrans-bpfcc** -- trace TCP retransmissions (great for spotting network issues):
```bash
sudo tcpretrans-bpfcc
# TIME     PID  IP LADDR:LPORT   T> RADDR:RPORT   STATE
```

4. **tcplife-bpfcc** -- trace TCP sessions with duration and throughput:
```bash
sudo tcplife-bpfcc
# PID   COMM    LADDR           LPORT RADDR           RPORT TX_KB RX_KB MS
```

5. **tcptop-bpfcc** -- top-like display of TCP traffic by host/port:
```bash
sudo tcptop-bpfcc
```

6. **biotop-bpfcc** -- top for block I/O (useful when diagnosing if "network slowness" is actually disk):
```bash
sudo biotop-bpfcc
```

**Custom one-liner with bpftrace:**

```bash
sudo apt install bpftrace

# Count packets by destination port
sudo bpftrace -e 'tracepoint:net:net_dev_xmit { @[args->len] = count(); }'

# Trace DNS lookups (udp port 53)
sudo bpftrace -e 'kprobe:udp_sendmsg { @[comm] = count(); }'
```

**Advantages over traditional tools:**
- Near-zero overhead compared to tcpdump on busy servers
- Kernel-level visibility without modifying applications
- Can trace internal kernel networking functions, not just packets on the wire
- No packet capture files to manage""",
    "flm_adv_networking"
))

# ===========================================================================
# 2. Traffic shaping with tc (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How do I use tc for traffic shaping on Linux?",
    r"""`tc` (traffic control) is part of iproute2 and lets you shape, schedule, and police network traffic. It works by attaching queuing disciplines (qdiscs) to network interfaces.

**Core concepts:**
- **qdisc** -- queuing discipline, controls how packets are queued/dequeued
- **class** -- subdivision within a classful qdisc
- **filter** -- rules that classify packets into classes

**Example 1: Rate-limit an interface to 100 Mbit/s:**

```bash
# Add a token bucket filter qdisc
sudo tc qdisc add dev eth0 root tbf rate 100mbit burst 32kbit latency 50ms

# Verify
tc qdisc show dev eth0

# Remove
sudo tc qdisc del dev eth0 root
```

**Example 2: HTB (Hierarchical Token Bucket) -- multiple classes:**

```bash
# Root qdisc
sudo tc qdisc add dev eth0 root handle 1: htb default 30

# Parent class: total bandwidth 1 Gbit
sudo tc class add dev eth0 parent 1: classid 1:1 htb rate 1gbit

# High-priority class: guaranteed 500 Mbit, can burst to 1 Gbit
sudo tc class add dev eth0 parent 1:1 classid 1:10 htb rate 500mbit ceil 1gbit

# Normal class: guaranteed 300 Mbit
sudo tc class add dev eth0 parent 1:1 classid 1:20 htb rate 300mbit ceil 800mbit

# Low-priority (default): 200 Mbit
sudo tc class add dev eth0 parent 1:1 classid 1:30 htb rate 200mbit ceil 500mbit

# Classify traffic: SSH goes to high-priority
sudo tc filter add dev eth0 parent 1: protocol ip prio 1 u32 \
    match ip dport 22 0xffff flowid 1:10

# HTTP/HTTPS to normal
sudo tc filter add dev eth0 parent 1: protocol ip prio 2 u32 \
    match ip dport 80 0xffff flowid 1:20
sudo tc filter add dev eth0 parent 1: protocol ip prio 2 u32 \
    match ip dport 443 0xffff flowid 1:20
```

**Example 3: Simulate bad network (useful for testing):**

```bash
# Add 100ms latency with 20ms jitter
sudo tc qdisc add dev eth0 root netem delay 100ms 20ms

# Add 1% packet loss
sudo tc qdisc add dev eth0 root netem loss 1%

# Combine: latency + loss + bandwidth limit
sudo tc qdisc add dev eth0 root handle 1: netem delay 50ms loss 0.5%
sudo tc qdisc add dev eth0 parent 1: handle 2: tbf rate 10mbit burst 32kbit latency 50ms
```

**Show current configuration:**

```bash
tc -s qdisc show dev eth0    # qdiscs with stats
tc -s class show dev eth0    # classes with stats
tc filter show dev eth0      # filters
```

**Persist across reboots** by adding `tc` commands to a script called from `/etc/network/if-up.d/` or a systemd service.""",
    "flm_adv_networking"
))

# ===========================================================================
# 3. OpenVPN server and client (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How do I set up a VPN with OpenVPN on Debian, both server and client side?",
    r"""Here's a complete OpenVPN setup on Debian 13.

**1. Install on both server and client:**

```bash
sudo apt update
sudo apt install openvpn easy-rsa
```

**2. Set up the CA and certificates (on the server):**

```bash
make-cadir ~/openvpn-ca
cd ~/openvpn-ca

# Initialize the PKI
./easyrsa init-pki
./easyrsa build-ca nopass
# Enter a common name, e.g., "MyVPN-CA"

# Generate server cert
./easyrsa gen-req server nopass
./easyrsa sign-req server server

# Generate client cert
./easyrsa gen-req client1 nopass
./easyrsa sign-req client client1

# Generate Diffie-Hellman parameters
./easyrsa gen-dh

# Generate TLS auth key
openvpn --genkey secret ta.key
```

**3. Server configuration** (`/etc/openvpn/server.conf`):

```
port 1194
proto udp
dev tun

ca /root/openvpn-ca/pki/ca.crt
cert /root/openvpn-ca/pki/issued/server.crt
key /root/openvpn-ca/pki/private/server.key
dh /root/openvpn-ca/pki/dh.pem
tls-auth /root/openvpn-ca/ta.key 0

server 10.8.0.0 255.255.255.0
push "redirect-gateway def1 bypass-dhcp"
push "dhcp-option DNS 1.1.1.1"

keepalive 10 120
cipher AES-256-GCM
auth SHA256

user nobody
group nogroup
persist-key
persist-tun

verb 3
status /var/log/openvpn-status.log
log-append /var/log/openvpn.log
```

**4. Enable IP forwarding on the server:**

```bash
echo 'net.ipv4.ip_forward = 1' | sudo tee /etc/sysctl.d/99-openvpn.conf
sudo sysctl -p /etc/sysctl.d/99-openvpn.conf

# NAT for VPN clients
sudo nft add table nat
sudo nft add chain nat postrouting '{ type nat hook postrouting priority 100 ; }'
sudo nft add rule nat postrouting oifname "eth0" masquerade
```

**5. Start the server:**

```bash
sudo systemctl enable --now openvpn@server
sudo systemctl status openvpn@server
```

**6. Client configuration** (`client1.ovpn`):

```
client
dev tun
proto udp
remote YOUR_SERVER_IP 1194
resolv-retry infinite
nobind
persist-key
persist-tun
remote-cert-tls server
cipher AES-256-GCM
auth SHA256
key-direction 1
verb 3

<ca>
(paste contents of ca.crt)
</ca>
<cert>
(paste contents of client1.crt)
</cert>
<key>
(paste contents of client1.key)
</key>
<tls-auth>
(paste contents of ta.key)
</tls-auth>
```

**7. Connect from the client:**

```bash
sudo openvpn --config client1.ovpn
```

Or install as a systemd service:
```bash
sudo cp client1.ovpn /etc/openvpn/client/client1.conf
sudo systemctl enable --now openvpn-client@client1
```

**Verify the tunnel:**

```bash
ip addr show tun0
ping 10.8.0.1    # ping the server's VPN IP
```""",
    "flm_adv_networking"
))

# ===========================================================================
# 4. HAProxy load balancer setup (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How do I set up HAProxy as a load balancer on Debian?",
    r"""HAProxy is a high-performance TCP/HTTP load balancer.

**Install:**

```bash
sudo apt update
sudo apt install haproxy
```

**Basic HTTP load balancer** (`/etc/haproxy/haproxy.cfg`):

```
global
    log /dev/log local0
    maxconn 4096
    user haproxy
    group haproxy
    daemon

    # TLS tuning
    ssl-default-bind-ciphersuites TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384
    ssl-default-bind-options ssl-min-ver TLSv1.2

defaults
    log     global
    mode    http
    option  httplog
    option  dontlognull
    timeout connect 5s
    timeout client  30s
    timeout server  30s
    retries 3
    option  redispatch

frontend http_front
    bind *:80
    bind *:443 ssl crt /etc/haproxy/certs/site.pem
    http-request redirect scheme https unless { ssl_fc }
    default_backend web_servers

backend web_servers
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200

    server web1 192.168.1.10:8080 check inter 5s fall 3 rise 2
    server web2 192.168.1.11:8080 check inter 5s fall 3 rise 2
    server web3 192.168.1.12:8080 check inter 5s fall 3 rise 2 backup

# Stats page
listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 10s
    stats admin if LOCALHOST
```

**Key configuration elements:**

- **balance roundrobin** -- distributes requests evenly; alternatives: `leastconn`, `source` (sticky by IP)
- **option httpchk** -- active health checks; HAProxy removes failed servers
- **fall 3 / rise 2** -- server marked down after 3 failed checks, up after 2 successful
- **backup** -- web3 only receives traffic when web1 and web2 are both down

**Validate and start:**

```bash
sudo haproxy -c -f /etc/haproxy/haproxy.cfg    # syntax check
sudo systemctl enable --now haproxy
```

**Sticky sessions** (if your app needs them):

```
backend web_servers
    balance roundrobin
    cookie SERVERID insert indirect nocache
    server web1 192.168.1.10:8080 check cookie web1
    server web2 192.168.1.11:8080 check cookie web2
```

**TCP mode** (for databases, mail, etc.):

```
frontend mysql_front
    bind *:3306
    mode tcp
    default_backend mysql_servers

backend mysql_servers
    mode tcp
    balance leastconn
    server db1 192.168.1.20:3306 check
    server db2 192.168.1.21:3306 check backup
```

**Monitor with the stats page** at `http://yourserver:8404/stats` -- it shows per-server connection counts, health status, and throughput in real time.""",
    "flm_adv_networking"
))

# ===========================================================================
# 5. Multicast networking basics (no system prompt)
# ===========================================================================
conversations.append(make(
    None,
    "Can you explain multicast networking basics on Linux?",
    r"""Multicast lets one sender transmit to multiple receivers simultaneously without sending separate copies. The network infrastructure replicates packets as needed.

**Key concepts:**

- **Multicast addresses:** IPv4 range 224.0.0.0 to 239.255.255.255 (Class D)
- **IGMP (Internet Group Management Protocol):** how hosts tell routers they want to join/leave a group
- **Common groups:** 224.0.0.1 (all hosts), 224.0.0.2 (all routers), 239.x.x.x (organization-local scope)

**Check multicast support:**

```bash
ip maddr show                  # list multicast group memberships
cat /proc/net/igmp             # IGMP state
ip link show eth0 | grep -i multicast   # MULTICAST flag should be present
```

**Enable multicast on an interface:**

```bash
sudo ip link set eth0 multicast on

# Add a multicast route if needed
sudo ip route add 224.0.0.0/4 dev eth0
```

**Simple test with socat:**

Receiver (run on one or more hosts):
```bash
socat UDP4-RECVFROM:5000,ip-add-membership=239.1.1.1:eth0,fork -
```

Sender:
```bash
echo "hello multicast" | socat - UDP4-DATAGRAM:239.1.1.1:5000,range=224.0.0.0/4
```

**Test with iperf3:**

```bash
# Receiver
iperf3 -s -B 239.1.1.1 -p 5001

# Sender
iperf3 -c 239.1.1.1 -p 5001 -u --ttl 5 -b 10M
```

**Kernel parameters for multicast tuning:**

```bash
# Maximum number of multicast groups per socket
sysctl net.ipv4.igmp_max_memberships

# Multicast forwarding (if acting as a router)
sysctl net.ipv4.conf.all.mc_forwarding
```

**Troubleshooting:**
- Verify the `MULTICAST` flag is set on the interface (`ip link show`)
- Check that a firewall is not blocking IGMP (protocol 2) or the multicast port
- For cross-subnet multicast, you need a multicast-capable router or use a multicast routing daemon like `pimd`
- TTL must be > 1 for packets to cross routers (default is 1)""",
    "flm_adv_networking"
))

# ===========================================================================
# 6. ZFS on Linux (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How do I use ZFS on Linux? Pool creation, snapshots, send/receive?",
    r"""ZFS is a combined filesystem and volume manager with built-in checksumming, snapshots, and replication.

**Install ZFS on Debian 13:**

```bash
sudo apt install linux-headers-$(uname -r)
sudo apt install zfsutils-linux
sudo modprobe zfs
```

**Create a pool:**

```bash
# Mirror (RAID1) with two disks
sudo zpool create mypool mirror /dev/sdb /dev/sdc

# RAIDZ1 (single parity, like RAID5) with three disks
sudo zpool create datapool raidz1 /dev/sdb /dev/sdc /dev/sdd

# Check pool status
zpool status
zpool list
```

The pool is automatically mounted at `/mypool`.

**Create datasets (filesystems within the pool):**

```bash
sudo zfs create mypool/documents
sudo zfs create mypool/backups
sudo zfs create mypool/vms

# Set properties
sudo zfs set compression=zstd mypool/documents
sudo zfs set quota=100G mypool/backups
sudo zfs set recordsize=64K mypool/vms

# List datasets
zfs list
```

**Snapshots:**

```bash
# Create a snapshot
sudo zfs snapshot mypool/documents@2024-01-15

# List snapshots
zfs list -t snapshot

# Access snapshot contents (read-only)
ls /mypool/documents/.zfs/snapshot/2024-01-15/

# Rollback to a snapshot (destroys changes since then)
sudo zfs rollback mypool/documents@2024-01-15

# Delete a snapshot
sudo zfs destroy mypool/documents@2024-01-15
```

**Send/Receive (replication):**

```bash
# Send a snapshot to a file (backup)
sudo zfs send mypool/documents@snap1 > /backup/documents-snap1.zfs

# Send to another pool (local)
sudo zfs send mypool/documents@snap1 | sudo zfs receive backuppool/documents

# Send over SSH to a remote server
sudo zfs send mypool/documents@snap1 | ssh backup-server sudo zfs receive tank/documents

# Incremental send (only changes between two snapshots)
sudo zfs send -i mypool/documents@snap1 mypool/documents@snap2 | \
    ssh backup-server sudo zfs receive tank/documents
```

**Automated snapshots with a cron job:**

```bash
# /etc/cron.d/zfs-snapshots
0 * * * * root zfs snapshot mypool/documents@hourly-$(date +\%Y\%m\%d-\%H)
0 0 * * * root zfs snapshot mypool/documents@daily-$(date +\%Y\%m\%d)
```

**Monitor health:**

```bash
zpool status -v          # show detailed status and errors
zpool scrub mypool       # start a scrub (checks all data)
zpool events -v          # ZFS event log
```

**Key differences from traditional RAID + ext4:**
- Checksums every block -- detects silent corruption (bit rot)
- Copy-on-write -- snapshots are instant and free until data changes
- Built-in compression saves space and can improve performance
- No need for a separate volume manager (LVM)""",
    "flm_adv_storage"
))

# ===========================================================================
# 7. Ceph basics (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "Can you give me an overview of Ceph and how to do a single-node test setup on Debian?",
    r"""Ceph is a distributed storage system that provides object, block, and filesystem storage in a single unified cluster. It's designed for no single point of failure.

**Core components:**
- **MON (Monitor):** maintains cluster state maps, consensus via Paxos
- **OSD (Object Storage Daemon):** stores data, handles replication, one per disk
- **MGR (Manager):** monitoring, metrics, dashboard
- **MDS (Metadata Server):** only needed for CephFS (the POSIX filesystem layer)

**Single-node test setup with cephadm (Debian 13):**

```bash
# Install prerequisites
sudo apt update
sudo apt install cephadm ceph-common

# Bootstrap a minimal cluster on this node
sudo cephadm bootstrap --mon-ip $(hostname -I | awk '{print $1}') --single-host-defaults
```

This creates a single-node cluster with MON and MGR. It runs containers (podman or docker) under the hood.

**Add OSDs (storage):**

```bash
# List available disks
sudo cephadm shell -- ceph orch device ls

# Add a specific disk as an OSD
sudo cephadm shell -- ceph orch daemon add osd $(hostname):data_devices=/dev/sdb

# Or add all available disks
sudo cephadm shell -- ceph orch apply osd --all-available-devices
```

**Check cluster health:**

```bash
sudo cephadm shell -- ceph status
sudo cephadm shell -- ceph health detail
sudo cephadm shell -- ceph osd tree
```

**Create a block device (RBD):**

```bash
sudo cephadm shell -- ceph osd pool create rbd-pool 32
sudo cephadm shell -- rbd pool init rbd-pool
sudo cephadm shell -- rbd create rbd-pool/myimage --size 10G

# Map it on the host
sudo rbd map rbd-pool/myimage
sudo mkfs.ext4 /dev/rbd0
sudo mount /dev/rbd0 /mnt/ceph-block
```

**Enable the dashboard:**

```bash
sudo cephadm shell -- ceph mgr module enable dashboard
sudo cephadm shell -- ceph dashboard create-self-signed-cert
echo "MyPassword" > /tmp/dash-pw.txt
sudo cephadm shell -- ceph dashboard ac-user-create admin -i /tmp/dash-pw.txt administrator
rm /tmp/dash-pw.txt
```

Access at `https://your-ip:8443`.

**Important notes for production:**
- Minimum 3 nodes for proper replication and availability
- Each OSD should be a dedicated disk (not a partition)
- MON count should be odd (3 or 5) for consensus
- 10 Gbit+ networking recommended for real workloads
- Single-node setup is for testing only -- there's no redundancy""",
    "flm_adv_storage"
))

# ===========================================================================
# 8. mdadm RAID management (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How do I manage software RAID with mdadm on Debian? Create, monitor, replace a failed disk?",
    r"""mdadm is the standard Linux software RAID tool. It supports RAID 0, 1, 5, 6, and 10.

**Install:**

```bash
sudo apt install mdadm
```

**Create a RAID 1 (mirror) array:**

```bash
# Two disks: /dev/sdb and /dev/sdc
sudo mdadm --create /dev/md0 --level=1 --raid-devices=2 /dev/sdb /dev/sdc

# Watch the initial sync
watch cat /proc/mdstat

# Create a filesystem
sudo mkfs.ext4 /dev/md0
sudo mkdir -p /mnt/raid
sudo mount /dev/md0 /mnt/raid
```

**Create a RAID 5 array (three disks + one spare):**

```bash
sudo mdadm --create /dev/md1 --level=5 --raid-devices=3 \
    --spare-devices=1 /dev/sdd /dev/sde /dev/sdf /dev/sdg
```

**Save the configuration (critical!):**

```bash
sudo mdadm --detail --scan >> /etc/mdadm/mdadm.conf
sudo update-initramfs -u
```

Without this, the array may not assemble on reboot.

**Persistent mount in `/etc/fstab`:**

```
/dev/md0  /mnt/raid  ext4  defaults,nofail  0  2
```

**Monitor the array:**

```bash
# Status overview
cat /proc/mdstat

# Detailed info
sudo mdadm --detail /dev/md0

# Enable email alerts on failure
sudo mdadm --monitor --mail=root@localhost --delay=300 /dev/md0 --daemonise
```

mdadm also works with systemd -- the `mdmonitor` service sends alerts:
```bash
sudo systemctl enable --now mdmonitor
```

**Simulate a disk failure:**

```bash
sudo mdadm --manage /dev/md0 --fail /dev/sdc
cat /proc/mdstat   # shows [U_] -- one disk degraded
```

**Replace a failed disk:**

```bash
# Remove the failed disk
sudo mdadm --manage /dev/md0 --remove /dev/sdc

# Physically replace the disk, then add the new one
sudo mdadm --manage /dev/md0 --add /dev/sdh

# Watch the rebuild
watch cat /proc/mdstat
```

If you had a spare disk configured, the rebuild starts automatically when a disk fails.

**Grow an array (add a disk to RAID 5):**

```bash
sudo mdadm --grow /dev/md1 --raid-devices=4 --add /dev/sdh
# Then resize the filesystem
sudo resize2fs /dev/md1
```

**Stop and remove an array:**

```bash
sudo umount /mnt/raid
sudo mdadm --stop /dev/md0
sudo mdadm --zero-superblock /dev/sdb /dev/sdc
# Remove from /etc/mdadm/mdadm.conf and update-initramfs -u
```""",
    "flm_adv_storage"
))

# ===========================================================================
# 9. LVM thin provisioning (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How does LVM thin provisioning work and how do I set it up?",
    r"""LVM thin provisioning lets you allocate more virtual storage than you physically have. Actual disk space is consumed only as data is written -- like overcommitting RAM with swap, but for storage.

**Install prerequisites:**

```bash
sudo apt install lvm2 thin-provisioning-tools
```

**Step 1: Create a volume group (if you don't have one):**

```bash
sudo pvcreate /dev/sdb
sudo vgcreate data_vg /dev/sdb
```

**Step 2: Create a thin pool:**

```bash
# Create a thin pool of 80G in the VG
# The pool needs a data LV and a metadata LV (auto-created)
sudo lvcreate -L 80G -T data_vg/thin_pool
```

Or explicitly control metadata size:
```bash
sudo lvcreate -L 80G --thinpool thin_pool --poolmetadatasize 1G data_vg
```

**Step 3: Create thin volumes (overprovisioned):**

```bash
# Create three 50G volumes from an 80G pool = 150G virtual / 80G physical
sudo lvcreate -V 50G -T data_vg/thin_pool -n vm_disk1
sudo lvcreate -V 50G -T data_vg/thin_pool -n vm_disk2
sudo lvcreate -V 50G -T data_vg/thin_pool -n vm_disk3
```

Each volume thinks it has 50G, but the pool only has 80G total. Space is allocated on write.

**Step 4: Use them like normal LVs:**

```bash
sudo mkfs.ext4 /dev/data_vg/vm_disk1
sudo mount /dev/data_vg/vm_disk1 /mnt/vm1
```

**Monitor pool usage (critical!):**

```bash
# Pool usage
sudo lvs -o+lv_size,pool_lv,data_percent,metadata_percent data_vg

# Detailed thin pool status
sudo lvs -a -o+seg_monitor data_vg/thin_pool

# Set up dmeventd to auto-extend when usage gets high
sudo lvmconfig --type full activation/thin_pool_autoextend_threshold  # default: 70%
sudo lvmconfig --type full activation/thin_pool_autoextend_percent    # default: 20%
```

**Snapshots (instant, thin):**

```bash
# Create a thin snapshot (nearly instant, shares data with origin)
sudo lvcreate -s -n vm_disk1_snap data_vg/vm_disk1

# List snapshots
sudo lvs -o+origin

# Remove snapshot
sudo lvremove data_vg/vm_disk1_snap
```

Thin snapshots are much more efficient than old-style LVM snapshots because both the origin and snapshot only store changed blocks.

**Dangers of overprovisioning:**
- If the pool fills to 100%, all thin volumes freeze (I/O errors)
- Monitor with `dmeventd` and `thin_pool_autoextend`
- Set up alerts at 80% usage
- Always have free space in the VG so auto-extend can work""",
    "flm_adv_storage"
))

# ===========================================================================
# 10. Filesystem benchmarking with fio (no system prompt)
# ===========================================================================
conversations.append(make(
    None,
    "How do I benchmark my filesystem performance with fio?",
    r"""fio (Flexible I/O Tester) is the standard tool for storage benchmarking. Install it from your distro's repos (`apt install fio`, `pacman -S fio`, `dnf install fio`).

**Quick benchmarks from the command line:**

```bash
# Sequential read (like reading a large file)
fio --name=seq-read --rw=read --bs=1M --size=1G --numjobs=1 \
    --ioengine=libaio --direct=1 --runtime=30 --time_based

# Sequential write
fio --name=seq-write --rw=write --bs=1M --size=1G --numjobs=1 \
    --ioengine=libaio --direct=1 --runtime=30 --time_based

# Random 4K read (simulates database workload)
fio --name=rand-read --rw=randread --bs=4k --size=1G --numjobs=4 \
    --ioengine=libaio --iodepth=32 --direct=1 --runtime=30 --time_based

# Random 4K write
fio --name=rand-write --rw=randwrite --bs=4k --size=1G --numjobs=4 \
    --ioengine=libaio --iodepth=32 --direct=1 --runtime=30 --time_based

# Mixed 70/30 read/write (realistic server workload)
fio --name=mixed --rw=randrw --rwmixread=70 --bs=4k --size=1G \
    --numjobs=4 --ioengine=libaio --iodepth=32 --direct=1 --runtime=30 --time_based
```

**Key parameters explained:**

- `--direct=1` -- bypass the OS page cache (tests actual disk speed)
- `--ioengine=libaio` -- Linux native async I/O (most realistic)
- `--iodepth=32` -- queue depth (how many I/Os in flight at once); NVMe drives shine at higher depths
- `--numjobs=4` -- parallel workers
- `--time_based --runtime=30` -- run for 30 seconds regardless of file size

**Using a job file** (`bench.fio`):

```ini
[global]
ioengine=libaio
direct=1
time_based
runtime=60
size=4G
directory=/mnt/test

[seq-read]
rw=read
bs=1M
numjobs=1

[rand-read-4k]
rw=randread
bs=4k
numjobs=4
iodepth=32

[rand-write-4k]
rw=randwrite
bs=4k
numjobs=4
iodepth=32
```

Run it:
```bash
fio bench.fio --output=results.txt --output-format=json
```

**Reading the output:**

The key metrics:
- **IOPS** -- operations per second (matters for random I/O)
- **BW (bandwidth)** -- MB/s (matters for sequential I/O)
- **lat (latency)** -- avg/p99 in usec or msec (matters for responsiveness)
- **clat** -- completion latency (time from submission to completion)

**Comparing filesystems:**

```bash
# Format the test partition with different filesystems and run the same fio job
sudo mkfs.ext4 /dev/sdb1 && sudo mount /dev/sdb1 /mnt/test && fio bench.fio
sudo mkfs.xfs /dev/sdb1 && sudo mount /dev/sdb1 /mnt/test && fio bench.fio
sudo mkfs.btrfs /dev/sdb1 && sudo mount /dev/sdb1 /mnt/test && fio bench.fio
```

**Tips:**
- Always use `--direct=1` to test the storage device, not the page cache
- Run tests for at least 30-60 seconds to get stable numbers
- Drop caches before each test: `echo 3 | sudo tee /proc/sys/vm/drop_caches`
- Test on the actual mount point where your workload runs, not `/tmp`""",
    "flm_adv_storage"
))

# ===========================================================================
# 11. systemd portable services (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "What are systemd portable services and how do I use them on Debian?",
    r"""Portable services are systemd's lightweight application isolation mechanism -- somewhere between a raw systemd service and a full container. They bundle an app with its own filesystem image but integrate directly with the host's systemd, logging, and networking.

**Key idea:** a portable service is a directory or disk image containing a minimal OS tree plus a systemd unit file. When "attached," systemd makes the unit available on the host.

**Create a portable service image:**

```bash
# 1. Build a minimal OS tree
sudo apt install debootstrap
sudo debootstrap --variant=minbase trixie /tmp/myservice

# 2. Install your application inside
sudo chroot /tmp/myservice apt install -y nginx

# 3. Add a systemd unit file inside the image
cat << 'UNIT' | sudo tee /tmp/myservice/usr/lib/systemd/system/myservice.service
[Unit]
Description=My Portable Nginx Service

[Service]
Type=forking
ExecStart=/usr/sbin/nginx
ExecReload=/bin/kill -HUP $MAINPID
PrivateTmp=yes
UNIT

# 4. Create the image file
sudo tar -C /tmp/myservice -czf /var/lib/portables/myservice.raw .
# Or use mkosi / systemd-repart for a proper raw disk image
```

A simpler approach using a directory:
```bash
sudo mv /tmp/myservice /var/lib/portables/myservice
```

**Attach and manage the portable service:**

```bash
# Attach -- makes the unit available to systemd
sudo portablectl attach myservice

# List attached portable services
portablectl list

# Start the service (just like any systemd unit)
sudo systemctl start myservice.service
sudo systemctl status myservice.service

# Enable on boot
sudo systemctl enable myservice.service

# Detach (removes the unit)
sudo portablectl detach myservice
```

**How it differs from containers:**
- Uses the host's systemd PID 1 (no container runtime)
- Logs go to the host's journald
- Shares the host network namespace by default
- Sandboxed via systemd's `PrivateTmp`, `ProtectSystem`, etc.
- No image registry, no orchestrator -- it's just a filesystem image

**When to use portable services:**
- You want app isolation without Docker/Podman complexity
- You need tight integration with systemd (timers, socket activation, journal)
- You're deploying on embedded or minimal systems
- You want reproducible service images that can be attached to any systemd host""",
    "flm_adv_systemd"
))

# ===========================================================================
# 12. systemd user services (Arch)
# ===========================================================================
conversations.append(make(
    SYS_ARCH,
    "How do I create and manage systemd user services?",
    r"""User services run under your user account without root -- managed by a per-user systemd instance.

**Create a user service:**

```bash
mkdir -p ~/.config/systemd/user/
```

Example: a simple web server (`~/.config/systemd/user/myapp.service`):

```ini
[Unit]
Description=My Python Web App
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/myapp
ExecStart=%h/myapp/venv/bin/python app.py
Restart=on-failure
RestartSec=5
Environment=PORT=8080

[Install]
WantedBy=default.target
```

`%h` expands to your home directory.

**Manage the service:**

```bash
# Reload after creating/editing unit files
systemctl --user daemon-reload

# Start / stop / restart
systemctl --user start myapp
systemctl --user stop myapp
systemctl --user restart myapp

# Enable on login
systemctl --user enable myapp

# Check status and logs
systemctl --user status myapp
journalctl --user -u myapp -f
```

**Make user services start at boot (not just at login):**

By default, user services only run while you're logged in. To keep them running:

```bash
sudo loginctl enable-linger $USER
```

This starts your user manager at boot, even without a login session.

**User timers (cron replacement):**

`~/.config/systemd/user/backup.timer`:
```ini
[Unit]
Description=Hourly backup timer

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

`~/.config/systemd/user/backup.service`:
```ini
[Unit]
Description=Backup task

[Service]
Type=oneshot
ExecStart=%h/bin/backup.sh
```

```bash
systemctl --user enable --now backup.timer
systemctl --user list-timers
```

**Useful commands:**

```bash
# List all user services
systemctl --user list-units --type=service

# Show user service file search paths
systemd-analyze --user unit-paths

# Check for errors
systemctl --user --failed
```

**Common paths for user units:**
- `~/.config/systemd/user/` -- your custom units (highest priority)
- `/etc/systemd/user/` -- system-wide user units (admin-installed)
- `/usr/lib/systemd/user/` -- package-installed user units""",
    "flm_adv_systemd"
))

# ===========================================================================
# 13. systemd-homed for portable home directories (Fedora)
# ===========================================================================
conversations.append(make(
    SYS_FED41,
    "How does systemd-homed work and how do I set it up on Fedora?",
    r"""systemd-homed manages user home directories as self-contained, portable, encrypted images. Instead of a traditional `/home/user` directory, your home is a LUKS-encrypted image file that can be moved between machines.

**Enable systemd-homed:**

```bash
sudo systemctl enable --now systemd-homed
```

Fedora 41 ships with systemd-homed ready to use.

**Create a user with homed:**

```bash
# Create with LUKS encryption (default)
sudo homectl create alice --storage=luks --disk-size=50G

# Or with directory-based storage (simpler, no LUKS)
sudo homectl create bob --storage=directory

# With FIDO2 authentication
sudo homectl create carol --storage=luks --fido2-device=auto
```

You'll be prompted to set a password.

**Manage users:**

```bash
# List managed users
homectl list

# Inspect a user
homectl inspect alice

# Change password
homectl passwd alice

# Resize the home image
sudo homectl resize alice 100G

# Lock (unmount home, e.g., on suspend)
sudo homectl lock alice

# Unlock
sudo homectl unlock alice

# Remove a user
sudo homectl remove alice
```

**How it works internally:**
- Home directory is stored as `~/.identity` metadata + a LUKS image
- On login, systemd-homed unlocks the LUKS image and mounts it at `/home/alice`
- On logout, it unmounts and locks
- The image file lives at `/home/alice.home` when inactive

**Portability:**

```bash
# Export the home image to a USB drive
sudo homectl with alice -- cp /home/alice.home /mnt/usb/

# On another systemd-homed machine, import it
sudo cp /mnt/usb/alice.home /home/
sudo homectl activate alice
```

The user record (UID, groups, shell, etc.) is embedded in the LUKS header, so it travels with the image.

**PAM integration:**

Fedora's PAM stack already includes `pam_systemd_homed.so`. If you're setting this up manually, ensure `/etc/pam.d/system-auth` includes the homed module.

**Caveats:**
- Not all display managers fully support homed yet
- NFS/network home directories are not supported
- SSH key-based login requires extra setup (the home isn't mounted until password auth)
- Newer feature -- test before deploying widely""",
    "flm_adv_systemd"
))

# ===========================================================================
# 14. Writing a watchdog service with systemd (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How do I write a watchdog service with systemd?",
    r"""systemd has a built-in software watchdog: your service periodically sends "I'm alive" notifications, and systemd kills and restarts it if the notifications stop.

**Service unit with watchdog** (`/etc/systemd/system/myworker.service`):

```ini
[Unit]
Description=My Worker with Watchdog
After=network.target

[Service]
Type=notify
ExecStart=/usr/local/bin/myworker
WatchdogSec=30
Restart=on-watchdog
# Also restart on other failures
RestartSec=5

# What to do when the watchdog triggers
WatchdogSignal=SIGABRT
# This creates a core dump for debugging

# Limit restart attempts
StartLimitIntervalSec=300
StartLimitBurst=5

[Install]
WantedBy=multi-user.target
```

**Key directives:**
- `WatchdogSec=30` -- service must notify within 30 seconds or it's killed
- `Type=notify` -- service uses sd_notify protocol
- `Restart=on-watchdog` -- restart only on watchdog failures (or use `always`)
- `WatchdogSignal=SIGABRT` -- signal sent on timeout (SIGABRT generates core dump)

**Python example using sd_notify:**

```python
#!/usr/bin/env python3
# Worker that integrates with systemd watchdog.
import os
import socket
import time

def sd_notify(message):
    # Send a notification to systemd.
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    if addr[0] == "@":
        addr = "\0" + addr[1:]
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.connect(addr)
        sock.sendall(message.encode())
    finally:
        sock.close()

def main():
    # Tell systemd we're ready
    sd_notify("READY=1")

    while True:
        try:
            do_work()
            # Tell systemd we're still alive
            sd_notify("WATCHDOG=1")
        except Exception as e:
            # Log the error but keep sending watchdog pings
            # if the exception is recoverable
            print(f"Error: {e}")
            sd_notify("WATCHDOG=1")

        time.sleep(10)

def do_work():
    # Your actual work here.
    pass

if __name__ == "__main__":
    main()
```

Or use the `systemd` Python module:
```bash
sudo apt install python3-systemd
```

```python
from systemd.daemon import notify
notify("READY=1")
# ... in your loop:
notify("WATCHDOG=1")
```

**Shell script with watchdog:**

```bash
#!/bin/bash
systemd-notify --ready
while true; do
    # Do work
    /usr/local/bin/check_something.sh
    systemd-notify WATCHDOG=1
    sleep 10
done
```

**Bash service must use `Type=notify` and `NotifyAccess=all`** since `systemd-notify` runs as a subprocess.

**Hardware watchdog integration:**

systemd can also ping a hardware watchdog (e.g., `/dev/watchdog`):

```ini
# In /etc/systemd/system.conf
RuntimeWatchdogSec=30
RebootWatchdogSec=10min
```

This reboots the entire machine if systemd PID 1 itself hangs -- useful for headless servers.

**Test your watchdog:**

```bash
sudo systemctl start myworker
sudo systemctl status myworker

# Simulate a hang (kill the notify loop but not the process)
# The watchdog should trigger after WatchdogSec
journalctl -u myworker -f
```""",
    "flm_adv_systemd"
))

# ===========================================================================
# 15. Ansible playbook basics (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How do I get started with Ansible playbooks on Debian?",
    r"""Ansible automates system configuration over SSH -- no agent needed on target machines.

**Install Ansible:**

```bash
sudo apt install ansible
# Or in a venv for the latest version:
python3 -m venv ~/ansible-env
source ~/ansible-env/bin/activate
pip install ansible
```

**Set up your inventory** (`/etc/ansible/hosts` or `./inventory.ini`):

```ini
[webservers]
web1 ansible_host=192.168.1.10
web2 ansible_host=192.168.1.11

[dbservers]
db1 ansible_host=192.168.1.20

[all:vars]
ansible_user=deploy
ansible_python_interpreter=/usr/bin/python3
```

**Test connectivity:**

```bash
ansible all -i inventory.ini -m ping
```

**A basic playbook** (`site.yml`):

```yaml
---
- name: Configure web servers
  hosts: webservers
  become: yes

  vars:
    http_port: 80
    domain: example.com

  tasks:
    - name: Update apt cache
      apt:
        update_cache: yes
        cache_valid_time: 3600

    - name: Install packages
      apt:
        name:
          - nginx
          - certbot
          - python3-certbot-nginx
        state: present

    - name: Copy nginx config
      template:
        src: templates/nginx.conf.j2
        dest: /etc/nginx/sites-available/{{ domain }}
        owner: root
        group: root
        mode: '0644'
      notify: Reload nginx

    - name: Enable site
      file:
        src: /etc/nginx/sites-available/{{ domain }}
        dest: /etc/nginx/sites-enabled/{{ domain }}
        state: link
      notify: Reload nginx

    - name: Ensure nginx is running
      service:
        name: nginx
        state: started
        enabled: yes

  handlers:
    - name: Reload nginx
      service:
        name: nginx
        state: reloaded
```

**Jinja2 template** (`templates/nginx.conf.j2`):

```nginx
server {
    listen {{ http_port }};
    server_name {{ domain }};
    root /var/www/{{ domain }};
    index index.html;
}
```

**Run the playbook:**

```bash
# Dry run (check mode)
ansible-playbook -i inventory.ini site.yml --check --diff

# Apply
ansible-playbook -i inventory.ini site.yml

# Limit to specific hosts
ansible-playbook -i inventory.ini site.yml --limit web1

# With extra variables
ansible-playbook -i inventory.ini site.yml -e "http_port=8080"
```

**Common modules you'll use:**
- `apt` / `dnf` / `pacman` -- package management
- `template` -- Jinja2 templates
- `copy` -- static files
- `file` -- create dirs, symlinks, set permissions
- `service` / `systemd` -- manage services
- `user` -- manage users
- `command` / `shell` -- run arbitrary commands (last resort)
- `lineinfile` -- edit a single line in a file

**Best practice:** use `template` and `copy` instead of `lineinfile` where possible -- full file management is more predictable than line-level edits.""",
    "flm_config_mgmt"
))

# ===========================================================================
# 16. Ansible roles and best practices (no system prompt)
# ===========================================================================
conversations.append(make(
    None,
    "What are Ansible roles and what are the best practices for organizing playbooks?",
    r"""Roles are Ansible's way of organizing reusable automation into a standard directory structure.

**Role directory structure:**

```
roles/
  webserver/
    tasks/
      main.yml          # task list
    handlers/
      main.yml          # handlers (e.g., restart nginx)
    templates/
      nginx.conf.j2     # Jinja2 templates
    files/
      index.html        # static files
    vars/
      main.yml          # role variables (high priority)
    defaults/
      main.yml          # default variables (low priority, easily overridden)
    meta/
      main.yml          # role metadata and dependencies
```

**Create a role skeleton:**

```bash
ansible-galaxy init roles/webserver
```

**Example role -- `roles/webserver/tasks/main.yml`:**

```yaml
---
- name: Install nginx
  apt:
    name: nginx
    state: present

- name: Deploy site config
  template:
    src: site.conf.j2
    dest: /etc/nginx/sites-available/{{ site_name }}
  notify: Reload nginx

- name: Enable site
  file:
    src: /etc/nginx/sites-available/{{ site_name }}
    dest: /etc/nginx/sites-enabled/{{ site_name }}
    state: link
  notify: Reload nginx
```

**`roles/webserver/defaults/main.yml`:**

```yaml
---
site_name: default
http_port: 80
```

**Use the role in a playbook:**

```yaml
---
- name: Set up web servers
  hosts: webservers
  become: yes
  roles:
    - role: webserver
      vars:
        site_name: myapp
        http_port: 8080
```

**Best practices:**

1. **Project layout:**
```
ansible-project/
  inventory/
    production.ini
    staging.ini
  group_vars/
    all.yml            # variables for all hosts
    webservers.yml     # variables for webserver group
  host_vars/
    web1.yml           # variables for specific host
  roles/
    common/            # base packages, SSH config, NTP
    webserver/
    database/
  site.yml             # main playbook
  webservers.yml       # playbook for just web servers
```

2. **Use `group_vars` and `host_vars`** instead of putting variables in playbooks.

3. **Keep secrets in Ansible Vault:**
```bash
ansible-vault create group_vars/all/vault.yml
ansible-vault edit group_vars/all/vault.yml
ansible-playbook site.yml --ask-vault-pass
```

4. **Idempotency:** every task should be safe to run multiple times. Avoid `command`/`shell` when a module exists. If you must use them, add `creates:` or `when:` guards.

5. **Use `become: yes` at the play level**, not on every task -- unless only some tasks need root.

6. **Tag your tasks** for selective runs:
```yaml
- name: Install packages
  apt:
    name: nginx
  tags: [packages, nginx]

# Run only tagged tasks
# ansible-playbook site.yml --tags packages
```

7. **Use handlers for service restarts** -- they only fire once, at the end, even if notified multiple times.

8. **Pin role versions** in `requirements.yml`:
```yaml
- src: geerlingguy.nginx
  version: "3.1.0"
```

```bash
ansible-galaxy install -r requirements.yml
```""",
    "flm_config_mgmt"
))

# ===========================================================================
# 17. Terraform basics for local infrastructure (no system prompt)
# ===========================================================================
conversations.append(make(
    None,
    "How do I use Terraform for local infrastructure management?",
    r"""Terraform is an infrastructure-as-code tool. While it's commonly used with cloud providers, it works well for local infrastructure too -- managing libvirt VMs, Docker containers, or even local files.

**Install Terraform:**

Download from releases.hashicorp.com or use your distro's package:
```bash
# Debian/Ubuntu
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform
```

**Example 1: Manage Docker containers:**

```hcl
# main.tf
terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {}

resource "docker_network" "app_network" {
  name = "app-network"
}

resource "docker_image" "nginx" {
  name = "nginx:latest"
}

resource "docker_container" "web" {
  name  = "web-server"
  image = docker_image.nginx.image_id

  ports {
    internal = 80
    external = 8080
  }

  networks_advanced {
    name = docker_network.app_network.name
  }
}

output "web_url" {
  value = "http://localhost:8080"
}
```

**Example 2: Manage libvirt/KVM VMs:**

```hcl
terraform {
  required_providers {
    libvirt = {
      source  = "dmacvicar/libvirt"
      version = "~> 0.7"
    }
  }
}

provider "libvirt" {
  uri = "qemu:///system"
}

resource "libvirt_volume" "debian_base" {
  name   = "debian-base.qcow2"
  pool   = "default"
  source = "/var/lib/libvirt/images/debian-13-generic-amd64.qcow2"
  format = "qcow2"
}

resource "libvirt_domain" "testvm" {
  name   = "test-vm"
  memory = "2048"
  vcpu   = 2

  disk {
    volume_id = libvirt_volume.debian_base.id
  }

  network_interface {
    network_name = "default"
  }
}
```

**Terraform workflow:**

```bash
# Initialize (download providers)
terraform init

# Preview changes
terraform plan

# Apply changes
terraform apply

# Show current state
terraform show

# Destroy everything
terraform destroy
```

**State management:**

Terraform tracks what it manages in `terraform.tfstate`. For local use:
- Keep it in version control (for personal projects)
- Or use a local backend (default)
- Never edit the state file manually

**Variables and modules:**

```hcl
# variables.tf
variable "container_count" {
  type    = number
  default = 3
}

# main.tf
resource "docker_container" "web" {
  count = var.container_count
  name  = "web-${count.index}"
  image = docker_image.nginx.image_id
}
```

```bash
terraform apply -var="container_count=5"
```

**Key concepts:**
- Terraform is **declarative** -- you describe the desired state, it figures out what to create/modify/destroy
- **Plan before apply** -- always review the plan
- **State is the source of truth** -- if you modify resources outside Terraform, run `terraform refresh`
- Resources have **lifecycle** -- create, read, update, delete""",
    "flm_config_mgmt"
))

# ===========================================================================
# 18. Set up a CA and sign certificates with OpenSSL (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How do I set up my own Certificate Authority and sign certificates with OpenSSL?",
    r"""Running your own CA is useful for internal services, development, and mTLS. Here's a complete setup.

**1. Create the CA directory structure:**

```bash
mkdir -p ~/myca/{certs,crl,newcerts,private,csr}
chmod 700 ~/myca/private
touch ~/myca/index.txt
echo 1000 > ~/myca/serial
```

**2. Create the CA configuration** (`~/myca/openssl.cnf`):

```ini
[ca]
default_ca = CA_default

[CA_default]
dir               = /home/user/myca
certs             = $dir/certs
crl_dir           = $dir/crl
new_certs_dir     = $dir/newcerts
database          = $dir/index.txt
serial            = $dir/serial
private_key       = $dir/private/ca.key
certificate       = $dir/certs/ca.crt
default_days      = 365
default_md        = sha256
policy            = policy_loose

[policy_loose]
countryName             = optional
stateOrProvinceName     = optional
organizationName        = optional
commonName              = supplied

[req]
default_bits       = 4096
default_md         = sha256
distinguished_name = req_distinguished_name

[req_distinguished_name]
commonName = Common Name (e.g., your name or hostname)

[v3_ca]
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid:always,issuer
basicConstraints = critical, CA:true
keyUsage = critical, digitalSignature, cRLSign, keyCertSign

[server_cert]
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid,issuer
basicConstraints = CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = example.internal
DNS.2 = *.example.internal
IP.1 = 192.168.1.100
```

**3. Generate the CA key and certificate:**

```bash
# Generate CA private key (keep this safe!)
openssl genrsa -aes256 -out ~/myca/private/ca.key 4096
chmod 400 ~/myca/private/ca.key

# Generate CA certificate (valid 10 years)
openssl req -config ~/myca/openssl.cnf \
    -key ~/myca/private/ca.key \
    -new -x509 -days 3650 -sha256 \
    -extensions v3_ca \
    -out ~/myca/certs/ca.crt \
    -subj "/CN=My Internal CA"
```

**4. Generate a server certificate:**

```bash
# Generate server private key
openssl genrsa -out ~/myca/private/server.key 2048

# Generate CSR (Certificate Signing Request)
openssl req -config ~/myca/openssl.cnf \
    -key ~/myca/private/server.key \
    -new -sha256 \
    -out ~/myca/csr/server.csr \
    -subj "/CN=myserver.example.internal"

# Sign the CSR with your CA
openssl ca -config ~/myca/openssl.cnf \
    -extensions server_cert \
    -days 365 -notext -md sha256 \
    -in ~/myca/csr/server.csr \
    -out ~/myca/certs/server.crt
```

**5. Verify the certificate:**

```bash
openssl x509 -in ~/myca/certs/server.crt -text -noout
openssl verify -CAfile ~/myca/certs/ca.crt ~/myca/certs/server.crt
```

**6. Trust the CA on Debian systems:**

```bash
sudo cp ~/myca/certs/ca.crt /usr/local/share/ca-certificates/myca.crt
sudo update-ca-certificates
```

Now all tools that use the system trust store (curl, wget, Python requests) will trust certificates signed by your CA.

**7. Use with nginx:**

```nginx
server {
    listen 443 ssl;
    server_name myserver.example.internal;
    ssl_certificate     /path/to/server.crt;
    ssl_certificate_key /path/to/server.key;
}
```""",
    "flm_adv_security"
))

# ===========================================================================
# 19. mTLS explained and configured (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "What is mTLS and how do I set it up?",
    r"""mTLS (mutual TLS) means both the server AND the client present certificates to each other. Normal TLS only authenticates the server; mTLS authenticates both sides.

**When to use mTLS:**
- Service-to-service communication (microservices)
- API access control without passwords
- Zero-trust network architectures
- IoT device authentication

**Setup overview:**
1. You need a CA (see my previous answer on setting up a CA)
2. Server gets a certificate signed by the CA
3. Each client gets a certificate signed by the same CA
4. Server is configured to require and verify client certificates

**Generate a client certificate:**

```bash
# Client key
openssl genrsa -out client.key 2048

# Client CSR
openssl req -new -key client.key -out client.csr \
    -subj "/CN=service-a"

# Sign with your CA (use a client_cert extension)
openssl ca -config ~/myca/openssl.cnf \
    -extensions usr_cert \
    -days 365 -notext -md sha256 \
    -in client.csr -out client.crt
```

Add this extension to your `openssl.cnf`:
```ini
[usr_cert]
basicConstraints = CA:FALSE
keyUsage = critical, digitalSignature
extendedKeyUsage = clientAuth
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid,issuer
```

**Configure nginx for mTLS:**

```nginx
server {
    listen 443 ssl;
    server_name api.example.internal;

    ssl_certificate     /etc/nginx/certs/server.crt;
    ssl_certificate_key /etc/nginx/certs/server.key;

    # mTLS: require client certificate
    ssl_client_certificate /etc/nginx/certs/ca.crt;
    ssl_verify_client on;

    # Optional: pass client CN to backend
    location / {
        proxy_pass http://backend:8080;
        proxy_set_header X-Client-CN $ssl_client_s_dn_cn;
        proxy_set_header X-Client-Verify $ssl_client_verify;
    }
}
```

**Test with curl:**

```bash
# This should work (presenting valid client cert)
curl --cacert ca.crt \
     --cert client.crt \
     --key client.key \
     https://api.example.internal/

# This should fail (no client cert)
curl --cacert ca.crt https://api.example.internal/
# Error: SSL peer certificate or SSH remote key was not OK
```

**Configure HAProxy for mTLS:**

```
frontend api
    bind *:443 ssl crt /etc/haproxy/server.pem ca-file /etc/haproxy/ca.crt verify required
    default_backend app_servers

    # Route based on client certificate CN
    acl is_service_a ssl_c_s_dn(cn) -i service-a
    use_backend service_a_backend if is_service_a
```

**Python client with mTLS:**

```python
import requests

response = requests.get(
    "https://api.example.internal/",
    cert=("client.crt", "client.key"),
    verify="ca.crt"
)
```

**Certificate rotation:**
- Issue certificates with short lifetimes (30-90 days)
- Automate renewal with a cron job or systemd timer
- Clients should reload certs without restart (or use a sidecar that handles TLS)""",
    "flm_adv_security"
))

# ===========================================================================
# 20. Kernel hardening - sysctl security parameters (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "What sysctl parameters should I set for kernel hardening on Debian?",
    r"""Here's a practical kernel hardening guide using sysctl. Create `/etc/sysctl.d/99-hardening.conf`:

```ini
# ---- Network hardening ----

# Disable IP forwarding (unless this is a router/gateway)
net.ipv4.ip_forward = 0
net.ipv6.conf.all.forwarding = 0

# Ignore ICMP redirects (prevents MITM routing attacks)
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0

# Don't send ICMP redirects
net.ipv4.conf.all.send_redirects = 0

# Enable reverse path filtering (anti-spoofing)
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# Ignore source-routed packets
net.ipv4.conf.all.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0

# SYN flood protection
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 4096

# Ignore ICMP broadcast requests
net.ipv4.icmp_echo_ignore_broadcasts = 1

# Log martian packets (impossible source addresses)
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1

# ---- Kernel hardening ----

# Restrict dmesg to root
kernel.dmesg_restrict = 1

# Restrict kernel pointer exposure
kernel.kptr_restrict = 2

# Restrict perf_event access
kernel.perf_event_paranoid = 3

# Disable kexec (loading a new kernel at runtime)
kernel.kexec_load_disabled = 1

# Restrict BPF to root
kernel.unprivileged_bpf_disabled = 1
net.core.bpf_jit_harden = 2

# Restrict ptrace (process tracing) to parent processes only
kernel.yama.ptrace_scope = 1

# Disable SysRq (magic key combos)
kernel.sysrq = 0

# ---- Memory hardening ----

# Randomize memory layout (ASLR) -- max randomization
kernel.randomize_va_space = 2

# Restrict core dumps
fs.suid_dumpable = 0

# Protect hardlinks and symlinks
fs.protected_hardlinks = 1
fs.protected_symlinks = 1
fs.protected_fifos = 2
fs.protected_regular = 2
```

**Apply immediately:**

```bash
sudo sysctl -p /etc/sysctl.d/99-hardening.conf
```

**Verify:**

```bash
sysctl kernel.dmesg_restrict
sysctl net.ipv4.conf.all.rp_filter
```

**Additional hardening beyond sysctl:**

1. **Restrict kernel modules:**
```bash
echo "install cramfs /bin/true" | sudo tee /etc/modprobe.d/disable-cramfs.conf
echo "install freevxfs /bin/true" | sudo tee /etc/modprobe.d/disable-freevxfs.conf
echo "install usb-storage /bin/true" | sudo tee /etc/modprobe.d/disable-usb-storage.conf
```

2. **Boot parameters** in `/etc/default/grub`:
```
GRUB_CMDLINE_LINUX="... init_on_alloc=1 init_on_free=1 slab_nomerge page_alloc.shuffle=1"
```
Then `sudo update-grub`.

3. **Check your hardening** with Lynis:
```bash
sudo apt install lynis
sudo lynis audit system
```

**Important:** test all changes in a non-production environment first. Some settings (like disabling `ip_forward`) will break routing; `ptrace_scope` can break debuggers; `kexec_load_disabled` prevents kexec-based reboots.""",
    "flm_adv_security"
))

# ===========================================================================
# 21. Seccomp profiles for containers (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How do seccomp profiles work for containers and how do I create custom ones?",
    r"""Seccomp (Secure Computing Mode) restricts which Linux system calls a process can make. Containers use seccomp profiles to limit the kernel attack surface.

**How it works:**
- A JSON profile lists allowed/denied syscalls
- Docker and Podman apply a default profile that blocks ~44 dangerous syscalls
- You can create custom profiles to be more or less restrictive

**See the default Docker/Podman profile:**

```bash
# Docker's default profile
docker run --rm alpine cat /proc/self/status | grep Seccomp
# Seccomp:  2  (2 = filter mode, meaning seccomp is active)

# Run without seccomp (dangerous, for testing only)
docker run --rm --security-opt seccomp=unconfined alpine cat /proc/self/status | grep Seccomp
# Seccomp:  0
```

**Create a custom seccomp profile** (`custom-seccomp.json`):

```json
{
    "defaultAction": "SCMP_ACT_ERRNO",
    "architectures": [
        "SCMP_ARCH_X86_64",
        "SCMP_ARCH_X86"
    ],
    "syscalls": [
        {
            "names": [
                "read", "write", "open", "close", "stat", "fstat",
                "lstat", "poll", "lseek", "mmap", "mprotect",
                "munmap", "brk", "ioctl", "access", "pipe",
                "select", "sched_yield", "mremap", "msync",
                "mincore", "madvise", "dup", "dup2", "nanosleep",
                "getpid", "socket", "connect", "accept", "sendto",
                "recvfrom", "bind", "listen", "getsockname",
                "getpeername", "clone", "execve", "exit",
                "wait4", "kill", "uname", "fcntl", "flock",
                "fsync", "getcwd", "chdir", "rename", "mkdir",
                "rmdir", "link", "unlink", "chmod", "chown",
                "umask", "gettimeofday", "getuid", "getgid",
                "geteuid", "getegid", "getppid", "setsid",
                "arch_prctl", "exit_group", "openat", "newfstatat",
                "readlinkat", "getrandom", "memfd_create",
                "futex", "set_tid_address", "set_robust_list",
                "clock_gettime", "epoll_create1", "epoll_ctl",
                "epoll_wait", "eventfd2", "pread64", "pwrite64",
                "writev", "readv", "sigaltstack", "rt_sigaction",
                "rt_sigprocmask", "rt_sigreturn", "pipe2",
                "accept4", "prlimit64", "rseq", "close_range"
            ],
            "action": "SCMP_ACT_ALLOW"
        }
    ]
}
```

**Use the custom profile:**

```bash
# Docker
docker run --rm --security-opt seccomp=custom-seccomp.json nginx

# Podman
podman run --rm --security-opt seccomp=custom-seccomp.json nginx
```

**Generate a profile by tracing an application:**

```bash
# Install the OCI seccomp profile generator
sudo apt install golang-github-seccomp-libseccomp-golang-dev

# Use strace to find which syscalls your app uses
strace -c -f docker run --rm --security-opt seccomp=unconfined myapp 2>&1 | \
    tail -n +3 | awk '{print $NF}' | sort -u
```

Or use `oci-seccomp-bpf-hook` with Podman to auto-generate profiles:

```bash
sudo apt install oci-seccomp-bpf-hook
podman run --annotation io.containers.trace-syscall=of:/tmp/profile.json myapp
```

**Profile actions:**
- `SCMP_ACT_ALLOW` -- allow the syscall
- `SCMP_ACT_ERRNO` -- deny with an error code (safe default)
- `SCMP_ACT_KILL` -- kill the process immediately (strict)
- `SCMP_ACT_LOG` -- allow but log (useful for auditing)

**Syscalls you almost always want to block:**
- `kexec_load` -- load a new kernel
- `mount` / `umount` -- filesystem manipulation
- `reboot` -- reboot the host
- `init_module` / `delete_module` -- load kernel modules
- `ptrace` -- process debugging (container escape vector)

**Debugging seccomp denials:**

```bash
# Check audit log for denied syscalls
journalctl -k | grep -i seccomp
# Or
dmesg | grep -i seccomp
```""",
    "flm_adv_security"
))

# ===========================================================================
# 22. Network intrusion detection with Suricata (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How do I set up Suricata for network intrusion detection on Debian?",
    r"""Suricata is an open-source IDS/IPS (Intrusion Detection/Prevention System) that inspects network traffic against rules to detect attacks, malware, and policy violations.

**Install Suricata:**

```bash
sudo apt update
sudo apt install suricata suricata-update
```

**Initial configuration** (`/etc/suricata/suricata.yaml`):

Key settings to adjust:

```yaml
# Set your network
vars:
  address-groups:
    HOME_NET: "[192.168.1.0/24]"      # your internal network
    EXTERNAL_NET: "!$HOME_NET"

# Set the capture interface
af-packet:
  - interface: eth0
    threads: auto
    cluster-type: cluster_flow

# Enable EVE JSON logging (main structured log)
outputs:
  - eve-log:
      enabled: yes
      filename: /var/log/suricata/eve.json
      types:
        - alert
        - http
        - dns
        - tls
        - flow
```

**Update detection rules:**

```bash
# Download and install rules (Emerging Threats open ruleset)
sudo suricata-update

# List available rule sources
sudo suricata-update list-sources

# Enable additional sources
sudo suricata-update enable-source oisf/trafficid

# Update rules (run this daily via cron)
sudo suricata-update
```

Rules are installed to `/var/lib/suricata/rules/suricata.rules`.

**Start Suricata:**

```bash
# Test configuration
sudo suricata -T -c /etc/suricata/suricata.yaml

# Start the service
sudo systemctl enable --now suricata
sudo systemctl status suricata
```

**Monitor alerts:**

```bash
# Real-time alerts (JSON format)
sudo tail -f /var/log/suricata/eve.json | jq 'select(.event_type == "alert")'

# Quick summary of alerts
sudo tail -1000 /var/log/suricata/eve.json | \
    jq -r 'select(.event_type == "alert") | "\(.timestamp) \(.alert.signature) \(.src_ip) -> \(.dest_ip)"'

# Stats
sudo tail -f /var/log/suricata/eve.json | jq 'select(.event_type == "stats")'
```

**Write a custom rule:**

```bash
# /var/lib/suricata/rules/local.rules
alert http any any -> $HOME_NET any (msg:"Possible SQL injection attempt"; \
    content:"UNION"; nocase; content:"SELECT"; nocase; \
    sid:1000001; rev:1; classtype:web-application-attack;)

alert dns any any -> any any (msg:"DNS query for known malware domain"; \
    dns.query; content:"malware.example.com"; nocase; \
    sid:1000002; rev:1; classtype:trojan-activity;)
```

Add to suricata.yaml:
```yaml
rule-files:
  - suricata.rules
  - /var/lib/suricata/rules/local.rules
```

Reload rules without restarting:
```bash
sudo kill -USR2 $(pidof suricata)
```

**Test with a known signature:**

```bash
# This should trigger the ET POLICY test signature
curl http://testmynids.org/uid/index.html

# Check the alert
sudo jq 'select(.alert)' /var/log/suricata/eve.json | tail -1
```

**IPS mode (inline blocking):**

In `/etc/suricata/suricata.yaml`, change from IDS to IPS:
```yaml
af-packet:
  - interface: eth0
    copy-mode: ips
    copy-iface: eth1
```

Or with nftables:
```bash
sudo nft add rule inet filter forward queue num 0
```

And in suricata.yaml:
```yaml
nfq:
  mode: accept
```

**Performance tips:**
- Use multiple threads matching your CPU cores
- Enable hardware offloading: `ethtool -K eth0 gro on`
- For high-traffic links, use AF_PACKET with cluster_flow for load balancing across threads""",
    "flm_adv_security"
))

# ===========================================================================
# 23. PostgreSQL performance tuning (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How do I tune PostgreSQL for better performance on Debian?",
    r"""PostgreSQL's default configuration is very conservative. Here are the key parameters to tune in `/etc/postgresql/16/main/postgresql.conf` (Debian 13 ships PostgreSQL 16).

**Memory settings:**

```ini
# Shared memory for caching data pages
# Rule of thumb: 25% of total RAM (e.g., 4 GB on a 16 GB server)
shared_buffers = 4GB

# Memory per query operation (sorts, joins, etc.)
# Start with 256MB, increase if you see disk sorts
work_mem = 256MB

# Memory for maintenance operations (VACUUM, CREATE INDEX)
maintenance_work_mem = 1GB

# Maximum memory for query plans
effective_cache_size = 12GB
# Not an allocation -- tells the planner how much cache the OS has
# Rule of thumb: 75% of total RAM
```

**Write performance:**

```ini
# WAL (Write-Ahead Log) settings
wal_buffers = 64MB
min_wal_size = 1GB
max_wal_size = 4GB

# Checkpoint tuning (spread I/O over time)
checkpoint_completion_target = 0.9
checkpoint_timeout = 15min

# For SSDs, reduce random I/O cost estimate
random_page_cost = 1.1
# Default is 4.0 (appropriate for spinning disks)
# For NVMe, try 1.0
```

**Connection handling:**

```ini
max_connections = 200
# More connections = more memory. Use connection pooling (PgBouncer) instead of
# raising this to thousands.

# Huge pages (reduces TLB misses for large shared_buffers)
huge_pages = try
```

Enable huge pages on the OS:
```bash
echo 'vm.nr_hugepages = 2200' | sudo tee /etc/sysctl.d/99-hugepages.conf
sudo sysctl -p /etc/sysctl.d/99-hugepages.conf
```

**Query optimization:**

```ini
# Parallel queries (default is 2, increase on multi-core servers)
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_parallel_maintenance_workers = 4

# JIT compilation for complex queries
jit = on
```

**Logging for performance analysis:**

```ini
log_min_duration_statement = 500
# Log queries taking more than 500ms

log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
log_temp_files = 0
# Log any query that uses temp files (disk sorts)
```

**Apply and restart:**

```bash
sudo systemctl restart postgresql

# Check current settings
sudo -u postgres psql -c "SHOW shared_buffers;"
sudo -u postgres psql -c "SHOW work_mem;"
```

**Find slow queries with pg_stat_statements:**

```sql
-- Enable the extension
CREATE EXTENSION pg_stat_statements;

-- Top 10 slowest queries
SELECT query, calls, mean_exec_time, total_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

Add to `postgresql.conf`:
```ini
shared_preload_libraries = 'pg_stat_statements'
```

**Quick wins checklist:**
1. Set `shared_buffers` to 25% RAM
2. Set `effective_cache_size` to 75% RAM
3. Set `random_page_cost = 1.1` if on SSD
4. Enable `pg_stat_statements` to find slow queries
5. Run `ANALYZE` after loading data
6. Use `EXPLAIN ANALYZE` on slow queries to find missing indexes""",
    "flm_databases"
))

# ===========================================================================
# 24. PostgreSQL replication (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How do I set up PostgreSQL streaming replication with a primary and standby on Debian?",
    r"""Streaming replication continuously ships WAL (write-ahead log) records from the primary to one or more standby servers for high availability.

**On the primary server (192.168.1.10):**

1. Create a replication user:
```bash
sudo -u postgres psql
```
```sql
CREATE ROLE replicator WITH REPLICATION LOGIN PASSWORD 'SecureRepPass';
```

2. Configure `/etc/postgresql/16/main/postgresql.conf`:
```ini
listen_addresses = '*'
wal_level = replica
max_wal_senders = 5
wal_keep_size = 1GB
hot_standby = on
```

3. Allow replication in `/etc/postgresql/16/main/pg_hba.conf`:
```
# TYPE  DATABASE        USER          ADDRESS            METHOD
host    replication     replicator    192.168.1.0/24     scram-sha-256
```

4. Restart:
```bash
sudo systemctl restart postgresql
```

**On the standby server (192.168.1.11):**

1. Stop PostgreSQL and clear the data directory:
```bash
sudo systemctl stop postgresql
sudo rm -rf /var/lib/postgresql/16/main/*
```

2. Copy data from the primary using `pg_basebackup`:
```bash
sudo -u postgres pg_basebackup \
    -h 192.168.1.10 \
    -U replicator \
    -D /var/lib/postgresql/16/main \
    -Fp -Xs -P -R
```

The `-R` flag creates `standby.signal` and writes the connection info to `postgresql.auto.conf`.

3. Verify the standby configuration was created:
```bash
sudo -u postgres cat /var/lib/postgresql/16/main/postgresql.auto.conf
# Should contain: primary_conninfo = 'host=192.168.1.10 user=replicator ...'

ls -la /var/lib/postgresql/16/main/standby.signal
# Should exist (empty file that tells PostgreSQL to run as standby)
```

4. Start the standby:
```bash
sudo systemctl start postgresql
```

**Verify replication is working:**

On the primary:
```sql
SELECT client_addr, state, sent_lsn, write_lsn, replay_lsn
FROM pg_stat_replication;
```

On the standby:
```sql
SELECT pg_is_in_recovery();
-- Should return 't' (true)

SELECT pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn(), pg_last_xact_replay_timestamp();
```

**Monitoring replication lag:**

On the primary:
```sql
SELECT client_addr,
       pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS replay_lag_bytes
FROM pg_stat_replication;
```

**Promote the standby to primary** (failover):

```bash
sudo -u postgres pg_ctlcluster 16 main promote
# Or:
sudo -u postgres psql -c "SELECT pg_promote();"
```

The standby removes `standby.signal` and becomes a read-write primary.

**Replication slots** (prevent WAL removal before standby catches up):

On the primary:
```sql
SELECT pg_create_physical_replication_slot('standby1');
```

On the standby, in `postgresql.auto.conf`:
```ini
primary_slot_name = 'standby1'
```

**Important notes:**
- The standby is read-only (perfect for offloading SELECT queries)
- Failover is manual unless you add a tool like Patroni or repmgr
- Always monitor replication lag -- if the standby falls too far behind, you may need to rebuild it""",
    "flm_databases"
))

# ===========================================================================
# 25. Redis setup and basic operations (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How do I set up Redis and use its basic operations on Debian?",
    r"""Redis is an in-memory key-value store used for caching, session storage, queues, and real-time data.

**Install:**

```bash
sudo apt update
sudo apt install redis-server
```

**Configure** (`/etc/redis/redis.conf`):

```ini
# Bind to localhost only (security)
bind 127.0.0.1 ::1

# Set a password
requirepass YourStrongPassword

# Memory limit
maxmemory 256mb
maxmemory-policy allkeys-lru
# Evict least-recently-used keys when memory is full

# Persistence
save 900 1       # snapshot if 1+ keys changed in 900 seconds
save 300 10      # snapshot if 10+ keys changed in 300 seconds
save 60 10000    # snapshot if 10000+ keys changed in 60 seconds

# AOF (Append Only File) for durability
appendonly yes
appendfsync everysec
```

**Start Redis:**

```bash
sudo systemctl enable --now redis-server
sudo systemctl status redis-server
```

**Basic operations with `redis-cli`:**

```bash
redis-cli -a YourStrongPassword
```

**Strings:**
```
SET user:1:name "Alice"
GET user:1:name
# "Alice"

SET counter 0
INCR counter         # 1
INCRBY counter 10    # 11

# Expiring keys (TTL)
SET session:abc123 "user1" EX 3600    # expires in 1 hour
TTL session:abc123                     # seconds remaining
```

**Hashes** (like a mini-document):
```
HSET user:1 name "Alice" email "alice@example.com" age 30
HGET user:1 name
# "Alice"
HGETALL user:1
# name Alice email alice@example.com age 30
```

**Lists** (queues):
```
LPUSH jobs "send-email"
LPUSH jobs "resize-image"
RPOP jobs
# "send-email" (FIFO queue with LPUSH + RPOP)

LRANGE jobs 0 -1    # list all items
LLEN jobs            # length
```

**Sets:**
```
SADD tags:post:1 "linux" "debian" "sysadmin"
SMEMBERS tags:post:1
SISMEMBER tags:post:1 "linux"   # 1 (true)
SINTER tags:post:1 tags:post:2  # intersection
```

**Sorted sets** (leaderboards, priority queues):
```
ZADD scores 100 "alice" 95 "bob" 110 "carol"
ZRANGE scores 0 -1 WITHSCORES    # ascending
ZREVRANGE scores 0 2 WITHSCORES  # top 3
ZRANK scores "alice"             # rank (0-indexed)
```

**Pub/Sub:**
```bash
# Terminal 1 (subscriber)
redis-cli SUBSCRIBE alerts

# Terminal 2 (publisher)
redis-cli PUBLISH alerts "disk-full on web1"
```

**Useful admin commands:**

```
INFO                  # server stats
INFO memory           # memory usage
DBSIZE                # number of keys
KEYS user:*           # find keys (CAUTION: slow on large datasets)
SCAN 0 MATCH user:*   # safer iteration
MONITOR               # real-time command log (debugging)
CONFIG GET maxmemory
```

**Backup:**

```bash
# Trigger a snapshot
redis-cli -a YourStrongPassword BGSAVE

# The dump file
ls -la /var/lib/redis/dump.rdb

# Copy it offsite
cp /var/lib/redis/dump.rdb /backup/redis-$(date +%Y%m%d).rdb
```""",
    "flm_databases"
))

# ===========================================================================
# 26. SQLite tips for small applications (no system prompt)
# ===========================================================================
conversations.append(make(
    None,
    "What are some SQLite tips for small applications?",
    r"""SQLite is an embedded database -- no server, no configuration. The entire database is a single file. It's perfect for small-to-medium applications, CLI tools, and local data storage.

**Enable WAL mode** (Write-Ahead Logging) -- the single biggest performance improvement:

```sql
PRAGMA journal_mode=WAL;
```

WAL allows concurrent readers and writers. Without it, any write locks the entire database. Set this once; it persists.

**Recommended PRAGMAs for applications:**

```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;     -- safe with WAL, faster than FULL
PRAGMA foreign_keys=ON;        -- off by default!
PRAGMA busy_timeout=5000;      -- wait 5s for locks instead of failing immediately
PRAGMA cache_size=-64000;      -- 64MB page cache (negative = KB)
PRAGMA temp_store=MEMORY;      -- temp tables in RAM
```

Set these on every connection open.

**Python example:**

```python
import sqlite3

def get_connection(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA cache_size=-64000")
    conn.row_factory = sqlite3.Row  # access columns by name
    return conn

# Use context manager for automatic commit/rollback
with get_connection("app.db") as conn:
    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
    # auto-commits on exit

# Read query
with get_connection("app.db") as conn:
    users = conn.execute("SELECT * FROM users").fetchall()
    for u in users:
        print(u["name"])
```

**Use parameterized queries** (prevent SQL injection):

```python
# GOOD
conn.execute("SELECT * FROM users WHERE name = ?", (name,))

# BAD -- SQL injection vulnerability
conn.execute(f"SELECT * FROM users WHERE name = '{name}'")
```

**Batch inserts** (massive speed difference):

```python
# Slow: 1000 separate transactions
for item in items:
    conn.execute("INSERT INTO data VALUES (?)", (item,))

# Fast: one transaction with executemany
conn.executemany("INSERT INTO data VALUES (?)", [(i,) for i in items])
conn.commit()
```

**Indexes:**

```sql
-- Index columns you filter or join on
CREATE INDEX idx_users_email ON users(email);

-- Check if your query uses an index
EXPLAIN QUERY PLAN SELECT * FROM users WHERE email = 'alice@example.com';
-- Should show "SEARCH TABLE users USING INDEX idx_users_email"
```

**Full-text search:**

```sql
CREATE VIRTUAL TABLE docs USING fts5(title, body);
INSERT INTO docs VALUES ('Linux Admin', 'How to manage systemd services...');
SELECT * FROM docs WHERE docs MATCH 'systemd';
```

**Backup a live database:**

```python
import sqlite3

source = sqlite3.connect("app.db")
backup = sqlite3.connect("backup.db")
source.backup(backup)
backup.close()
source.close()
```

Or from the command line:
```bash
sqlite3 app.db ".backup backup.db"
```

**When SQLite is NOT the right choice:**
- Multiple processes writing heavily and concurrently
- Database larger than ~100 GB
- You need network access from multiple servers
- You need fine-grained user permissions

For everything else -- config storage, local caches, single-server web apps with moderate traffic, CLI tools -- SQLite is excellent.""",
    "flm_databases"
))

# ===========================================================================
# 27. Set up Gitea (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How do I set up Gitea as a self-hosted Git server on Debian?",
    r"""Gitea is a lightweight, self-hosted Git service -- like a minimal GitHub/GitLab.

**1. Create the git user:**

```bash
sudo adduser --system --shell /bin/bash --group --disabled-password \
    --home /home/git git
```

**2. Install Gitea:**

```bash
# Download the latest binary
GITEA_VERSION=1.21.4
sudo wget -O /usr/local/bin/gitea \
    "https://dl.gitea.io/gitea/${GITEA_VERSION}/gitea-${GITEA_VERSION}-linux-amd64"
sudo chmod +x /usr/local/bin/gitea
```

**3. Create directories:**

```bash
sudo mkdir -p /etc/gitea /var/lib/gitea/{custom,data,log}
sudo chown -R git:git /var/lib/gitea
sudo chown root:git /etc/gitea
sudo chmod 770 /etc/gitea
```

**4. Create systemd service** (`/etc/systemd/system/gitea.service`):

```ini
[Unit]
Description=Gitea (Git with a cup of tea)
After=network.target
Wants=postgresql.service

[Service]
Type=simple
User=git
Group=git
WorkingDirectory=/var/lib/gitea
ExecStart=/usr/local/bin/gitea web --config /etc/gitea/app.ini
Restart=always
Environment=USER=git HOME=/home/git GITEA_WORK_DIR=/var/lib/gitea

[Install]
WantedBy=multi-user.target
```

**5. (Optional) Set up PostgreSQL backend:**

```bash
sudo -u postgres psql
```
```sql
CREATE USER gitea WITH PASSWORD 'giteapass';
CREATE DATABASE giteadb OWNER gitea;
\q
```

SQLite also works fine for small installations.

**6. Start Gitea and run the installer:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gitea
```

Open `http://your-server:3000` in a browser. The web installer lets you configure:
- Database type (SQLite or PostgreSQL)
- Repository root path
- Admin account
- Email settings

**7. Lock down the config after install:**

```bash
sudo chmod 750 /etc/gitea
sudo chmod 640 /etc/gitea/app.ini
```

**8. Reverse proxy with nginx:**

```nginx
server {
    listen 80;
    server_name git.example.com;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 100M;
    }
}
```

**9. Add SSH passthrough** (so `git clone git@git.example.com:user/repo.git` works):

In `/etc/gitea/app.ini`:
```ini
[server]
SSH_DOMAIN       = git.example.com
START_SSH_SERVER = false
SSH_PORT         = 22
```

Gitea uses the system's `git` user for SSH. Keys added in the web UI go to `/home/git/.ssh/authorized_keys`.

**Useful admin commands:**

```bash
# Create a user from CLI
sudo -u git gitea admin user create --username admin --password changeme \
    --email admin@example.com --admin --config /etc/gitea/app.ini

# Dump for backup
sudo -u git gitea dump --config /etc/gitea/app.ini
```""",
    "flm_cicd"
))

# ===========================================================================
# 28. Woodpecker CI setup (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How do I set up Woodpecker CI on Debian alongside Gitea?",
    r"""Woodpecker CI is a lightweight, community-driven CI/CD engine forked from Drone. It integrates directly with Gitea, Forgejo, GitHub, and GitLab.

**Architecture:**
- **Woodpecker Server** -- web UI and API, receives webhooks from Gitea
- **Woodpecker Agent** -- runs the actual CI pipelines (in Docker containers)

**1. Set up an OAuth application in Gitea:**

In Gitea: Site Administration > Applications > Create OAuth2 Application:
- Application Name: `Woodpecker CI`
- Redirect URI: `http://ci.example.com/authorize`

Note the Client ID and Client Secret.

**2. Install using Docker Compose** (`docker-compose.yml`):

```yaml
version: "3"
services:
  woodpecker-server:
    image: woodpeckerci/woodpecker-server:latest
    ports:
      - "8000:8000"
    volumes:
      - woodpecker-data:/var/lib/woodpecker
    environment:
      - WOODPECKER_HOST=http://ci.example.com
      - WOODPECKER_OPEN=true
      - WOODPECKER_GITEA=true
      - WOODPECKER_GITEA_URL=http://git.example.com
      - WOODPECKER_GITEA_CLIENT=${GITEA_CLIENT_ID}
      - WOODPECKER_GITEA_SECRET=${GITEA_CLIENT_SECRET}
      - WOODPECKER_AGENT_SECRET=${AGENT_SECRET}
    restart: always

  woodpecker-agent:
    image: woodpeckerci/woodpecker-agent:latest
    depends_on:
      - woodpecker-server
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - WOODPECKER_SERVER=woodpecker-server:9000
      - WOODPECKER_AGENT_SECRET=${AGENT_SECRET}
      - WOODPECKER_MAX_WORKFLOWS=4
    restart: always

volumes:
  woodpecker-data:
```

Generate a shared secret:
```bash
openssl rand -hex 32
```

Create `.env`:
```bash
GITEA_CLIENT_ID=your-client-id
GITEA_CLIENT_SECRET=your-client-secret
AGENT_SECRET=your-generated-hex-secret
```

**3. Start it:**

```bash
docker compose up -d
```

Access the UI at `http://ci.example.com:8000`. Log in with your Gitea account (OAuth).

**4. Add a pipeline to your repo** (`.woodpecker.yml`):

```yaml
steps:
  - name: test
    image: python:3.12
    commands:
      - pip install -r requirements.txt
      - pytest tests/

  - name: lint
    image: python:3.12
    commands:
      - pip install flake8
      - flake8 src/

  - name: build
    image: docker:latest
    commands:
      - docker build -t myapp:${CI_COMMIT_SHA} .
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    when:
      branch: main
```

**5. Activate the repo:**

In the Woodpecker UI, find your repo and click "Activate." This creates a webhook in Gitea.

Now every push triggers the pipeline.

**Pipeline features:**
- **`when` conditions:** branch, event (push/pull_request/tag), path changes
- **Services:** run sidecar containers (databases for tests)
- **Secrets:** add in the Woodpecker UI, reference as `${SECRET_NAME}`
- **Matrix builds:** test against multiple versions

```yaml
matrix:
  PYTHON_VERSION:
    - "3.10"
    - "3.11"
    - "3.12"

steps:
  - name: test
    image: python:${PYTHON_VERSION}
    commands:
      - pip install -r requirements.txt
      - pytest
```

**Using services (e.g., PostgreSQL in tests):**

```yaml
services:
  - name: db
    image: postgres:16
    environment:
      POSTGRES_DB: testdb
      POSTGRES_PASSWORD: testpass

steps:
  - name: test
    image: python:3.12
    environment:
      DATABASE_URL: postgresql://postgres:testpass@db:5432/testdb
    commands:
      - pip install -r requirements.txt
      - pytest
```""",
    "flm_cicd"
))

# ===========================================================================
# 29. Systemd path units for file watching (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How do I use systemd path units to watch for file changes?",
    r"""systemd path units monitor filesystem paths and trigger a corresponding service when changes are detected. No extra software needed.

**Basic setup -- trigger a script when a file appears in a directory:**

Create the path unit (`/etc/systemd/system/upload-watcher.path`):

```ini
[Unit]
Description=Watch /srv/uploads for new files

[Path]
PathExistsGlob=/srv/uploads/*.csv
MakeDirectory=yes
# Can also use:
# PathChanged=/srv/uploads       -- triggers when anything in the dir changes
# PathModified=/srv/uploads      -- triggers when file content is modified
# DirectoryNotEmpty=/srv/uploads -- triggers when dir becomes non-empty

[Install]
WantedBy=multi-user.target
```

Create the matching service (`/etc/systemd/system/upload-watcher.service`):

```ini
[Unit]
Description=Process uploaded files

[Service]
Type=oneshot
ExecStart=/usr/local/bin/process-uploads.sh
# The service name must match the path unit name (minus the suffix)
```

The processing script (`/usr/local/bin/process-uploads.sh`):

```bash
#!/bin/bash
for f in /srv/uploads/*.csv; do
    [ -f "$f" ] || continue
    echo "Processing $f..."
    # Do your work here
    mv "$f" /srv/uploads/processed/
done
```

```bash
chmod +x /usr/local/bin/process-uploads.sh
```

**Enable and start:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now upload-watcher.path

# Check status
systemctl status upload-watcher.path
systemctl status upload-watcher.service
```

**Path directive types:**

| Directive | Triggers when |
|-----------|--------------|
| `PathExists` | path exists (checked on start + inotify) |
| `PathExistsGlob` | glob pattern matches any file |
| `PathChanged` | file/dir is written to and closed |
| `PathModified` | file is written to (even before close) |
| `DirectoryNotEmpty` | directory becomes non-empty |

**Example: Reload nginx when config changes:**

`/etc/systemd/system/nginx-config-watcher.path`:
```ini
[Unit]
Description=Watch nginx config for changes

[Path]
PathChanged=/etc/nginx/sites-enabled
PathChanged=/etc/nginx/nginx.conf

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/nginx-config-watcher.service`:
```ini
[Unit]
Description=Reload nginx on config change

[Service]
Type=oneshot
ExecStartPre=/usr/sbin/nginx -t
ExecStart=/usr/bin/systemctl reload nginx
```

**Debugging:**

```bash
# Check if the path unit is active and what it's watching
systemctl status upload-watcher.path

# See trigger history
journalctl -u upload-watcher.path
journalctl -u upload-watcher.service
```

**Limitations:**
- Path units use inotify internally, which does not work over NFS or CIFS
- No recursive directory watching (only watches the specified path, not subdirectories)
- Rate limiting is automatic -- rapid changes may be batched into one trigger
- For more complex file watching, consider `inotifywait` or `incron`""",
    "flm_cicd"
))

# ===========================================================================
# 30. inotifywait for filesystem event monitoring (Debian)
# ===========================================================================
conversations.append(make(
    SYS_DEB13,
    "How do I use inotifywait for filesystem event monitoring?",
    r"""`inotifywait` watches files and directories for changes using the Linux kernel's inotify API. It's more flexible than systemd path units for complex monitoring.

**Install:**

```bash
sudo apt install inotify-tools
```

**Basic usage:**

```bash
# Watch a directory for any event (blocks until an event occurs)
inotifywait /srv/data

# Watch continuously (don't exit after first event)
inotifywait -m /srv/data

# Watch recursively (all subdirectories)
inotifywait -mr /srv/data

# Watch for specific events
inotifywait -m -e create -e modify -e delete /srv/data
```

**Common events:**
- `create` -- file/dir created
- `delete` -- file/dir deleted
- `modify` -- file content modified
- `moved_to` / `moved_from` -- file renamed/moved
- `close_write` -- file opened for writing was closed (best for "file upload complete")
- `attrib` -- metadata changed (permissions, timestamps)

**Practical script -- process new files:**

```bash
#!/bin/bash
# Watch for completed uploads and process them
WATCH_DIR="/srv/uploads"
PROCESSED_DIR="/srv/uploads/processed"
mkdir -p "$PROCESSED_DIR"

inotifywait -m -e close_write --format '%w%f' "$WATCH_DIR" | while read FILE; do
    echo "$(date): New file: $FILE"

    case "$FILE" in
        *.csv)
            echo "Processing CSV: $FILE"
            /usr/local/bin/import-csv.sh "$FILE"
            mv "$FILE" "$PROCESSED_DIR/"
            ;;
        *.log)
            echo "Archiving log: $FILE"
            gzip "$FILE"
            mv "${FILE}.gz" /srv/archive/
            ;;
    esac
done
```

**Format options:**

```bash
# Custom output format
inotifywait -m -e create -e delete --format '%T %e %w%f' --timefmt '%Y-%m-%d %H:%M:%S' /srv/data
# Output: 2024-01-15 14:30:22 CREATE /srv/data/newfile.txt
```

**Run as a systemd service:**

`/etc/systemd/system/file-watcher.service`:
```ini
[Unit]
Description=File system event watcher
After=local-fs.target

[Service]
Type=simple
ExecStart=/usr/local/bin/file-watcher.sh
Restart=always
RestartSec=5
User=www-data

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now file-watcher
```

**Sync directory changes to a remote server:**

```bash
#!/bin/bash
inotifywait -mr -e close_write -e create -e delete --format '%w%f' /srv/data | \
while read FILE; do
    rsync -avz --delete /srv/data/ backup-server:/srv/data/
done
```

**Performance tuning:**

```bash
# Check current inotify watch limit
cat /proc/sys/fs/inotify/max_user_watches
# Default: 8192

# Increase for large directory trees
echo 'fs.inotify.max_user_watches = 524288' | sudo tee /etc/sysctl.d/99-inotify.conf
sudo sysctl -p /etc/sysctl.d/99-inotify.conf
```

**inotifywait vs systemd path units:**
- inotifywait: more flexible, custom output, recursive, multiple event types
- systemd path: simpler, integrated with systemd, no script needed for basic cases
- Both use inotify internally, so neither works over NFS""",
    "flm_cicd"
))

# ===========================================================================
# 31. Migrate a web application to a new server (multi-turn, Debian)
# ===========================================================================
conversations.append(make_multi(
    SYS_DEB13,
    [
        ("user", "I need to migrate a web application from an old server to a new Debian 13 server. It's running nginx, a Python Flask app with gunicorn, and PostgreSQL. What's my plan?"),
        ("assistant", r"""Here's a migration plan in phases:

**Phase 1: Inventory the old server**
```bash
# Document the full setup
dpkg --get-selections > ~/old-packages.txt
systemctl list-units --type=service --state=running > ~/old-services.txt
crontab -l > ~/old-crontab.txt
sudo cp -r /etc/nginx ~/old-nginx-config/
pip freeze > ~/old-python-deps.txt    # or from the virtualenv

# Note the PostgreSQL version and databases
sudo -u postgres psql -c "SELECT version();"
sudo -u postgres psql -l
```

**Phase 2: Set up the new server** (do this before any downtime)
- Install packages
- Configure nginx, gunicorn, Python venv
- Set up PostgreSQL
- Test with dummy data

**Phase 3: Data migration** (brief downtime window)
- Stop writes on the old server
- Dump PostgreSQL
- rsync application files
- Restore PostgreSQL on the new server

**Phase 4: Test on the new server**
- Verify the app works
- Check all endpoints

**Phase 5: DNS cutover**
- Update DNS records to point to new server IP
- Monitor for errors

Want me to walk through each phase in detail?"""),
        ("user", "Yes, let's start with Phase 2 -- setting up the new server."),
        ("assistant", r"""**Phase 2: Set up the new Debian 13 server**

**Install required packages:**

```bash
sudo apt update
sudo apt install nginx postgresql python3 python3-venv python3-pip \
    certbot python3-certbot-nginx rsync
```

**Set up PostgreSQL:**

```bash
sudo -u postgres psql
```
```sql
CREATE USER myapp WITH PASSWORD 'secure_password_here';
CREATE DATABASE myappdb OWNER myapp;
\q
```

**Set up the Python application:**

```bash
# Create an application user
sudo adduser --system --group --home /srv/myapp myapp

# Create the app directory and virtualenv
sudo mkdir -p /srv/myapp/app
sudo python3 -m venv /srv/myapp/venv
sudo chown -R myapp:myapp /srv/myapp
```

**Create the gunicorn systemd service** (`/etc/systemd/system/myapp.service`):

```ini
[Unit]
Description=My Flask App (gunicorn)
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=notify
User=myapp
Group=myapp
WorkingDirectory=/srv/myapp/app
Environment=PATH=/srv/myapp/venv/bin
ExecStart=/srv/myapp/venv/bin/gunicorn \
    --workers 4 \
    --bind unix:/run/myapp/gunicorn.sock \
    --access-logfile /var/log/myapp/access.log \
    --error-logfile /var/log/myapp/error.log \
    wsgi:app
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure
RuntimeDirectory=myapp

[Install]
WantedBy=multi-user.target
```

```bash
sudo mkdir -p /var/log/myapp
sudo chown myapp:myapp /var/log/myapp
```

**Configure nginx** (`/etc/nginx/sites-available/myapp`):

```nginx
server {
    listen 80;
    server_name myapp.example.com;

    location / {
        proxy_pass http://unix:/run/myapp/gunicorn.sock;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias /srv/myapp/app/static/;
        expires 30d;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/myapp /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

Don't start the app service yet -- we need to transfer the code and data first. Ready for Phase 3?"""),
        ("user", "Yes, let's do the data migration."),
        ("assistant", r"""**Phase 3: Data migration**

**Step 1: Put the old server in maintenance mode:**

On the old server:
```bash
# Create a maintenance page
echo "<h1>Maintenance in progress</h1>" | sudo tee /var/www/html/maintenance.html

# Update nginx to serve the maintenance page
# Add to the server block:
#   return 503;
#   error_page 503 /maintenance.html;
#   location = /maintenance.html { root /var/www/html; internal; }
sudo systemctl reload nginx
```

**Step 2: Dump the database on the old server:**

```bash
sudo -u postgres pg_dump --format=custom --compress=9 myappdb > /tmp/myappdb.dump
ls -lh /tmp/myappdb.dump
```

**Step 3: Transfer files to the new server:**

```bash
# From the new server, pull everything
rsync -avz --progress old-server:/srv/myapp/app/ /srv/myapp/app/
rsync -avz --progress old-server:/tmp/myappdb.dump /tmp/

# Also grab any uploaded files, media, etc.
rsync -avz --progress old-server:/srv/myapp/uploads/ /srv/myapp/uploads/
```

**Step 4: Restore the database on the new server:**

```bash
sudo -u postgres pg_restore --dbname=myappdb --clean --if-exists /tmp/myappdb.dump

# Verify
sudo -u postgres psql -d myappdb -c "\dt"
sudo -u postgres psql -d myappdb -c "SELECT count(*) FROM users;"  # or whatever your main table is
```

**Step 5: Install Python dependencies and start the app:**

```bash
sudo -u myapp /srv/myapp/venv/bin/pip install -r /srv/myapp/app/requirements.txt
sudo chown -R myapp:myapp /srv/myapp

# Update the app's database connection string to use the new credentials
# (edit .env or config.py as appropriate)

sudo systemctl daemon-reload
sudo systemctl enable --now myapp
sudo systemctl status myapp

# Test locally
curl --unix-socket /run/myapp/gunicorn.sock http://localhost/
```

**Step 6: Test through nginx:**

```bash
curl -H "Host: myapp.example.com" http://localhost/
```

Everything working? Then we move to DNS cutover."""),
        ("user", "It's working. How do I do the DNS cutover safely?"),
        ("assistant", r"""**Phase 5: DNS cutover**

**Step 1: Lower DNS TTL in advance** (do this 24-48 hours before migration):

Change your DNS record's TTL to 60 seconds (from the typical 3600):
```
myapp.example.com.  60  IN  A  NEW_SERVER_IP
```

This ensures clients refresh DNS quickly after the switch.

**Step 2: Update the DNS record:**

Change the A record to point to the new server's IP. With most DNS providers this is done via their web interface.

If you manage your own DNS (BIND):
```
myapp.example.com.  60  IN  A  203.0.113.50  ; new server IP
```
```bash
sudo rndc reload
```

**Step 3: Set up TLS on the new server:**

```bash
sudo certbot --nginx -d myapp.example.com
# Certbot will obtain a certificate and configure nginx for HTTPS
```

**Step 4: Monitor the transition:**

```bash
# Watch nginx access logs on the new server
sudo tail -f /var/log/nginx/access.log

# Watch application logs
sudo journalctl -u myapp -f

# Check that DNS is resolving to the new IP
dig myapp.example.com +short

# Monitor from an external location
curl -I https://myapp.example.com
```

**Step 5: Keep the old server running as a fallback:**

```bash
# On the old server, keep everything running for 48-72 hours
# If something goes wrong, just point DNS back to the old IP
```

**Step 6: After 48-72 hours with no issues:**

1. Restore DNS TTL to 3600 (or your normal value)
2. Decommission the old server
3. Take a full backup of the new server:
```bash
sudo -u postgres pg_dumpall > /backup/full-dump-$(date +%Y%m%d).sql
sudo rsync -a /srv/myapp/ /backup/myapp-$(date +%Y%m%d)/
```

**Post-migration checklist:**
- [ ] All pages/endpoints load correctly
- [ ] TLS certificate is valid
- [ ] Cron jobs transferred (if any)
- [ ] Log rotation configured
- [ ] Monitoring/alerting set up
- [ ] Backup schedule configured on new server
- [ ] Old server decommissioned"""),
    ],
    "flm_multiturn_migration"
))

# ===========================================================================
# 32. Set up high-availability PostgreSQL (multi-turn, Debian)
# ===========================================================================
conversations.append(make_multi(
    SYS_DEB13,
    [
        ("user", "I want to set up a high-availability PostgreSQL cluster with automatic failover. What's the recommended approach?"),
        ("assistant", r"""For automatic failover with PostgreSQL on Debian, the recommended stack is:

- **PostgreSQL** streaming replication (built-in)
- **Patroni** -- manages failover, leader election, and cluster state
- **etcd** -- distributed key-value store for consensus (Patroni uses it)
- **HAProxy** or **PgBouncer** -- connection routing to the current primary

**Architecture:**

```
Client -> HAProxy (port 5432) -> Patroni selects leader
                                   |
                            +------+------+
                            |             |
                        Primary       Standby
                        (pg01)        (pg02)
                            |             |
                            +--- etcd ----+
```

When the primary fails, Patroni promotes the standby and updates HAProxy automatically.

**Servers needed:**
- `pg01` (192.168.1.10) -- PostgreSQL + Patroni + etcd
- `pg02` (192.168.1.11) -- PostgreSQL + Patroni + etcd
- `pg03` (192.168.1.12) -- etcd (tie-breaker) + HAProxy

Want me to walk through the setup step by step?"""),
        ("user", "Yes, let's start with etcd."),
        ("assistant", r"""**Step 1: Install and configure etcd on all three nodes**

```bash
sudo apt install etcd-server etcd-client
```

On each node, edit `/etc/default/etcd`:

**pg01 (192.168.1.10):**
```bash
ETCD_NAME="etcd1"
ETCD_DATA_DIR="/var/lib/etcd/default"
ETCD_LISTEN_PEER_URLS="http://192.168.1.10:2380"
ETCD_LISTEN_CLIENT_URLS="http://192.168.1.10:2379,http://127.0.0.1:2379"
ETCD_INITIAL_ADVERTISE_PEER_URLS="http://192.168.1.10:2380"
ETCD_ADVERTISE_CLIENT_URLS="http://192.168.1.10:2379"
ETCD_INITIAL_CLUSTER="etcd1=http://192.168.1.10:2380,etcd2=http://192.168.1.11:2380,etcd3=http://192.168.1.12:2380"
ETCD_INITIAL_CLUSTER_STATE="new"
ETCD_INITIAL_CLUSTER_TOKEN="pg-cluster-1"
```

**pg02 (192.168.1.11):** same but with `etcd2` and its own IP.

**pg03 (192.168.1.12):** same but with `etcd3` and its own IP.

Start etcd on all three nodes simultaneously:
```bash
sudo systemctl enable --now etcd
```

Verify the cluster:
```bash
etcdctl member list
etcdctl endpoint health --cluster
```

You should see all three members healthy. Now let's set up Patroni."""),
        ("user", "Good, etcd is running. Now Patroni and PostgreSQL."),
        ("assistant", r"""**Step 2: Install Patroni on pg01 and pg02**

```bash
sudo apt install patroni
```

If the Debian package isn't available or too old:
```bash
sudo apt install python3-pip python3-psycopg2
sudo pip3 install patroni[etcd] --break-system-packages
```

**Stop any existing PostgreSQL instance:**
```bash
sudo systemctl stop postgresql
sudo systemctl disable postgresql
```

Patroni will manage PostgreSQL itself.

**Configure Patroni on pg01** (`/etc/patroni/config.yml`):

```yaml
scope: pg-cluster
name: pg01

restapi:
  listen: 192.168.1.10:8008
  connect_address: 192.168.1.10:8008

etcd:
  hosts: 192.168.1.10:2379,192.168.1.11:2379,192.168.1.12:2379

bootstrap:
  dcs:
    ttl: 30
    loop_wait: 10
    retry_timeout: 10
    maximum_lag_on_failover: 1048576    # 1MB
    postgresql:
      use_pg_rewind: true
      parameters:
        max_connections: 200
        shared_buffers: 2GB
        effective_cache_size: 6GB
        work_mem: 64MB
        wal_level: replica
        max_wal_senders: 5
        max_replication_slots: 5
        hot_standby: "on"

  initdb:
    - encoding: UTF8
    - data-checksums

  pg_hba:
    - host replication replicator 192.168.1.0/24 scram-sha-256
    - host all all 192.168.1.0/24 scram-sha-256
    - host all all 127.0.0.1/32 scram-sha-256

  users:
    admin:
      password: "AdminPass"
      options:
        - createrole
        - createdb
    replicator:
      password: "ReplPass"
      options:
        - replication

postgresql:
  listen: "*:5432"
  connect_address: 192.168.1.10:5432
  data_dir: /var/lib/postgresql/16/main
  bin_dir: /usr/lib/postgresql/16/bin
  authentication:
    replication:
      username: replicator
      password: "ReplPass"
    superuser:
      username: postgres
      password: "PostgresPass"
```

**On pg02**, use the same config but change:
- `name: pg02`
- `restapi.listen` and `connect_address` to `192.168.1.11`
- `postgresql.connect_address` to `192.168.1.11`

**Start Patroni:**

```bash
sudo systemctl enable --now patroni

# Check cluster status
patronictl -c /etc/patroni/config.yml list
```

You should see pg01 as the Leader and pg02 as a Replica. Patroni handles the initial `pg_basebackup` for the replica automatically."""),
        ("user", "Patroni shows both nodes. Now how do I set up HAProxy to route connections to the current primary?"),
        ("assistant", r"""**Step 3: Install and configure HAProxy on pg03**

```bash
sudo apt install haproxy
```

Edit `/etc/haproxy/haproxy.cfg`:

```
global
    log /dev/log local0
    maxconn 1000

defaults
    log global
    mode tcp
    timeout connect 5s
    timeout client 30s
    timeout server 30s

# Primary (read-write) -- only routes to the current Patroni leader
listen postgresql_primary
    bind *:5432
    option httpchk GET /primary
    http-check expect status 200
    default-server inter 3s fall 3 rise 2 on-marked-down shutdown-sessions
    server pg01 192.168.1.10:5432 maxconn 100 check port 8008
    server pg02 192.168.1.11:5432 maxconn 100 check port 8008

# Replicas (read-only) -- routes to standbys
listen postgresql_replicas
    bind *:5433
    option httpchk GET /replica
    http-check expect status 200
    default-server inter 3s fall 3 rise 2 on-marked-down shutdown-sessions
    server pg01 192.168.1.10:5432 maxconn 100 check port 8008
    server pg02 192.168.1.11:5432 maxconn 100 check port 8008

# Stats page
listen stats
    bind *:7000
    mode http
    stats enable
    stats uri /
```

**How it works:**
- Patroni exposes a REST API on port 8008
- `/primary` returns HTTP 200 only on the current leader
- `/replica` returns HTTP 200 only on standbys
- HAProxy health-checks these endpoints to route traffic correctly

**Start HAProxy:**

```bash
sudo haproxy -c -f /etc/haproxy/haproxy.cfg
sudo systemctl enable --now haproxy
```

**Test the setup:**

```bash
# Connect to the primary through HAProxy
psql -h 192.168.1.12 -p 5432 -U admin -d postgres -c "SELECT pg_is_in_recovery();"
# Should return 'f' (false -- it's the primary)

# Connect to a replica
psql -h 192.168.1.12 -p 5433 -U admin -d postgres -c "SELECT pg_is_in_recovery();"
# Should return 't' (true -- it's a standby)
```

**Test automatic failover:**

```bash
# On pg01, simulate a crash
sudo systemctl stop patroni

# Watch Patroni promote pg02
patronictl -c /etc/patroni/config.yml list
# pg02 should become the new Leader within ~30 seconds

# HAProxy automatically routes to the new primary
psql -h 192.168.1.12 -p 5432 -U admin -d postgres -c "SELECT inet_server_addr();"
# Should now show 192.168.1.11
```

Your applications connect to `192.168.1.12:5432` for writes and `192.168.1.12:5433` for reads. They never need to know which server is primary -- HAProxy and Patroni handle it."""),
    ],
    "flm_multiturn_ha_postgres"
))

# ===========================================================================
# 33. Debug intermittent network latency (multi-turn, Debian)
# ===========================================================================
conversations.append(make_multi(
    SYS_DEB13,
    [
        ("user", "Our web application has intermittent latency spikes. Users report the site is slow randomly, but sometimes it's fine. How do I systematically investigate?"),
        ("assistant", r"""Intermittent latency is one of the hardest problems to debug because the issue might be in the network, the application, the database, or the OS. Let's work through it layer by layer.

**Layer 1: Is it network or application?**

Start with `mtr` (combines traceroute and ping) to check for network-level issues:

```bash
# Run a continuous trace to the client network
sudo apt install mtr-tiny
mtr -rwzbc 100 client-ip-or-gateway
```

Look at the `Loss%` and `Avg`/`StDev` columns. High standard deviation means inconsistent latency.

**Meanwhile, check if the server itself is slow during spikes:**

```bash
# Quick system overview
vmstat 1 60    # watch for 60 seconds
# Look at: si/so (swap), bi/bo (block I/O), wa (I/O wait), us/sy (CPU)

# Is memory pressure causing I/O?
free -h
cat /proc/meminfo | grep -i swap

# Is the CPU saturated?
mpstat -P ALL 1 10
```

**Check active connections:**

```bash
ss -s                  # connection summary
ss -tnp state established | wc -l    # count established TCP connections
ss -tnp | awk '{print $4}' | sort | uniq -c | sort -rn | head    # connections per port
```

What do you see from these checks?"""),
        ("user", "mtr looks clean, no packet loss. But vmstat shows high 'wa' (I/O wait) values during the spikes, around 30-40%. Memory looks fine though."),
        ("assistant", r"""High I/O wait with clean networking and sufficient memory tells us the disk subsystem is the bottleneck during spikes. Let's dig deeper.

**Identify what's causing the I/O:**

```bash
# Real-time I/O by process
sudo iotop -aoP
# Sort by DISK WRITE or DISK READ to find the culprit
```

```bash
# Detailed block I/O stats per device
iostat -xz 1 10
# Key columns:
#   %util   - how busy the device is (>80% = saturated)
#   await   - average wait time per I/O request (high = bad)
#   r_await / w_await  - read/write wait separately
#   avgqu-sz - average queue length (>1 means I/Os are queuing)
```

**Check for specific processes doing heavy I/O:**

```bash
# Using pidstat
pidstat -d 1 10
# Shows kB_rd/s and kB_wr/s per process
```

**Check if PostgreSQL is the I/O source:**

```bash
# PostgreSQL active queries
sudo -u postgres psql -c "
SELECT pid, now() - pg_stat_activity.query_start AS duration, query, state, wait_event_type, wait_event
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC
LIMIT 10;"
```

```bash
# PostgreSQL I/O stats
sudo -u postgres psql -c "
SELECT relname, seq_scan, seq_tup_read, idx_scan, idx_tup_fetch
FROM pg_stat_user_tables
ORDER BY seq_scan DESC
LIMIT 10;"
```

Large `seq_scan` counts with low `idx_scan` means full table scans -- a major I/O source.

What does iotop show? Is it PostgreSQL, the application, or something else?"""),
        ("user", "iotop shows postgresql is doing heavy writes periodically. iostat shows %util hitting 95% on sda during the spikes. The await goes from 2ms to 200ms."),
        ("assistant", r"""PostgreSQL periodic heavy writes with spiking disk utilization -- this is almost certainly a **checkpoint storm**. PostgreSQL periodically writes all dirty buffers to disk (a "checkpoint"), and if the settings aren't tuned, it tries to flush everything at once, saturating the disk.

**Confirm it's checkpoints:**

```bash
# Check PostgreSQL log for checkpoint warnings
sudo grep -i checkpoint /var/log/postgresql/postgresql-16-main.log | tail -20

# Or query checkpoint stats
sudo -u postgres psql -c "SELECT * FROM pg_stat_bgwriter;"
# Look at checkpoints_timed and checkpoints_req
# High checkpoints_req means checkpoints are being forced (too much WAL)
```

**Fix checkpoint storms -- tune these in `postgresql.conf`:**

```ini
# Spread checkpoint writes over a longer period (default is 0.5)
checkpoint_completion_target = 0.9

# Allow more WAL between checkpoints (default is 1GB)
max_wal_size = 4GB
min_wal_size = 1GB

# Increase checkpoint timeout (default is 5min)
checkpoint_timeout = 15min

# Background writer: flush dirty pages more gradually
bgwriter_lru_maxpages = 400
bgwriter_lru_multiplier = 4.0
bgwriter_delay = 50ms
```

**Explanation of the fix:**
- `checkpoint_completion_target = 0.9` tells PostgreSQL to spread the checkpoint writes across 90% of the checkpoint interval instead of rushing
- `max_wal_size = 4GB` allows more data to accumulate before forcing a checkpoint
- The background writer continuously flushes small amounts of dirty pages, reducing the burst at checkpoint time

**Apply and restart:**

```bash
sudo systemctl restart postgresql
```

**Additional optimizations for I/O:**

```bash
# Check the I/O scheduler (should be 'none' or 'mq-deadline' for SSDs)
cat /sys/block/sda/queue/scheduler

# If it's an SSD, set to 'none' (noop):
echo none | sudo tee /sys/block/sda/queue/scheduler
```

Persist in `/etc/udev/rules.d/60-scheduler.rules`:
```
ACTION=="add|change", KERNEL=="sd*", ATTR{queue/scheduler}="none"
```

**Monitor after the fix:**

```bash
# Watch for future checkpoint performance
sudo -u postgres psql -c "
SELECT checkpoints_timed, checkpoints_req,
       checkpoint_write_time, checkpoint_sync_time
FROM pg_stat_bgwriter;"

# And confirm I/O wait drops
vmstat 1 60
```

Also enable checkpoint logging for ongoing monitoring:
```ini
# In postgresql.conf
log_checkpoints = on
```

This logs timing for every checkpoint so you can verify the spread is working."""),
        ("user", "The checkpoint tuning helped a lot. I/O wait is mostly under 5% now. But we still see occasional small spikes. Any other things to check?"),
        ("assistant", r"""Good progress. For the remaining smaller spikes, let's check a few more things:

**1. Check for autovacuum interference:**

```bash
sudo -u postgres psql -c "
SELECT schemaname, relname, last_autovacuum, last_autoanalyze,
       n_dead_tup, n_live_tup
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000
ORDER BY n_dead_tup DESC;"
```

If autovacuum is running on large tables, it can cause I/O spikes. Tune it:

```ini
# In postgresql.conf -- throttle autovacuum I/O
autovacuum_vacuum_cost_delay = 20ms     # default 2ms, increase to slow it down
autovacuum_vacuum_cost_limit = 200      # default 200, decrease to slow it down
```

**2. Check for heavy log writes:**

```bash
# Are logs being written to the same disk?
df -h /var/log/postgresql/
df -h /var/lib/postgresql/

# If they're on the same disk, consider moving WAL to a separate device
# or at minimum, reduce logging verbosity
```

**3. Use tcpdump to catch slow queries from the application:**

```bash
# Capture PostgreSQL traffic and look for slow responses
sudo tcpdump -i lo port 5432 -w /tmp/pg-traffic.pcap -c 10000

# Analyze with pg_stat_statements
sudo -u postgres psql -c "
SELECT query, calls, mean_exec_time, max_exec_time, stddev_exec_time
FROM pg_stat_statements
ORDER BY max_exec_time DESC
LIMIT 10;"
```

Look at `max_exec_time` vs `mean_exec_time`. A huge gap means some executions of that query are occasionally slow.

**4. Monitor with a continuous script:**

```bash
#!/bin/bash
# Save as /usr/local/bin/latency-monitor.sh
while true; do
    IOWAIT=$(vmstat 1 2 | tail -1 | awk '{print $16}')
    if [ "$IOWAIT" -gt 15 ]; then
        echo "=== HIGH I/O WAIT: ${IOWAIT}% at $(date) ==="
        iotop -b -n 1 -o >> /var/log/iowait-events.log
        sudo -u postgres psql -c "
            SELECT pid, now() - query_start AS duration, query
            FROM pg_stat_activity WHERE state = 'active'
            ORDER BY duration DESC LIMIT 5;" >> /var/log/iowait-events.log
        echo "---" >> /var/log/iowait-events.log
    fi
    sleep 5
done
```

This captures exactly what's happening every time I/O wait spikes, so you can correlate disk activity with specific queries.

**5. Long-term: consider separating WAL onto a different disk:**

```ini
# Stop PostgreSQL, move the WAL directory, create a symlink
sudo systemctl stop postgresql
sudo mv /var/lib/postgresql/16/main/pg_wal /fast-nvme/pg_wal
sudo ln -s /fast-nvme/pg_wal /var/lib/postgresql/16/main/pg_wal
sudo chown -R postgres:postgres /fast-nvme/pg_wal
sudo systemctl start postgresql
```

WAL writes are sequential and latency-sensitive. Putting them on a fast, dedicated device eliminates contention with random reads/writes from queries."""),
    ],
    "flm_multiturn_debug_latency"
))

# ===========================================================================
# 34. Capacity planning for a growing service (multi-turn, Debian)
# ===========================================================================
conversations.append(make_multi(
    SYS_DEB13,
    [
        ("user", "Our service is growing steadily. How do I do capacity planning to figure out when we'll need to upgrade?"),
        ("assistant", r"""Capacity planning starts with measuring current usage, identifying trends, and projecting when you'll hit limits. Let's set up monitoring and analysis.

**Step 1: Establish baselines -- what are your current resource levels?**

```bash
# CPU: how many cores and current utilization
nproc
mpstat -P ALL 1 5

# Memory: total, used, available
free -h
# The "available" column is the real indicator -- not "free"

# Disk: space and I/O
df -h
iostat -xz 1 5

# Network: bandwidth utilization
sudo apt install iftop nload
nload eth0
```

**Step 2: Collect metrics over time**

Install a lightweight metrics collector. The simplest approach on a single server:

```bash
# Create a metrics collection script
cat << 'EOF' > /usr/local/bin/collect-metrics.sh
#!/bin/bash
TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
CPU=$(mpstat 1 1 | awk '/Average:/ && /all/ {print 100-$NF}')
MEM=$(free | awk '/Mem:/ {printf "%.1f", $3/$2*100}')
DISK=$(df / | awk 'NR==2 {print $5}' | tr -d '%')
IOWAIT=$(vmstat 1 2 | tail -1 | awk '{print $16}')
CONNS=$(ss -tn state established | wc -l)
LOAD=$(awk '{print $1}' /proc/loadavg)

echo "$TIMESTAMP,$CPU,$MEM,$DISK,$IOWAIT,$CONNS,$LOAD" >> /var/log/capacity-metrics.csv
EOF
chmod +x /usr/local/bin/collect-metrics.sh

# Initialize the CSV header
echo "timestamp,cpu_pct,mem_pct,disk_pct,iowait_pct,tcp_conns,load1" > /var/log/capacity-metrics.csv
```

Run it every 5 minutes via cron:
```bash
echo '*/5 * * * * root /usr/local/bin/collect-metrics.sh' | sudo tee /etc/cron.d/capacity-metrics
```

Let this run for at least 2-4 weeks to capture daily and weekly patterns.

What specific resources are you most concerned about -- CPU, memory, disk, or all of them?"""),
        ("user", "Mainly disk space and database growth. Our PostgreSQL database is growing about 2 GB per week."),
        ("assistant", r"""Good, disk and database growth are the most predictable things to plan for. Let's measure precisely and project forward.

**Track PostgreSQL database size:**

```bash
# Current database sizes
sudo -u postgres psql -c "
SELECT datname, pg_size_pretty(pg_database_size(datname)) AS size
FROM pg_database
ORDER BY pg_database_size(datname) DESC;"

# Per-table sizes (find what's growing)
sudo -u postgres psql -d yourdb -c "
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
       pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
       pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) AS index_size,
       n_live_tup AS row_count
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 15;"
```

**Automate weekly size tracking:**

```bash
cat << 'SCRIPT' > /usr/local/bin/track-db-size.sh
#!/bin/bash
DATE=$(date +%Y-%m-%d)
SIZE=$(sudo -u postgres psql -t -c "SELECT pg_database_size('yourdb');" | tr -d ' ')
echo "$DATE,$SIZE" >> /var/log/db-size-history.csv
SCRIPT
chmod +x /usr/local/bin/track-db-size.sh

# Initialize
echo "date,size_bytes" > /var/log/db-size-history.csv

# Run weekly
echo '0 0 * * 0 root /usr/local/bin/track-db-size.sh' | sudo tee /etc/cron.d/db-size-track
```

**Quick projection with current data:**

```bash
# Available disk space
df -h /var/lib/postgresql/
# Let's say you have 200 GB free

# At 2 GB/week growth:
# 200 GB / 2 GB/week = 100 weeks ~ 23 months

# But account for:
# - WAL files (typically 2-4 GB)
# - Temporary files during VACUUM
# - pg_dump space if backing up locally
# - Index bloat (indexes grow faster than data sometimes)
# Rule of thumb: plan to act at 70% disk usage
```

**Check for bloat (wasted space you can reclaim):**

```sql
-- Table bloat estimate
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total,
       CASE WHEN n_live_tup > 0
            THEN round(n_dead_tup::numeric / n_live_tup * 100, 1)
            ELSE 0 END AS dead_pct
FROM pg_stat_user_tables
WHERE n_dead_tup > 10000
ORDER BY n_dead_tup DESC;
```

If `dead_pct` is high (>20%), a `VACUUM FULL` on that table can reclaim space (but locks the table).

**Data retention -- the most effective capacity tool:**

Do you keep all historical data? Consider:
```sql
-- Partition large tables by date
-- Then drop old partitions instead of DELETE (instant, no bloat)
CREATE TABLE events (
    id BIGSERIAL,
    created_at TIMESTAMPTZ NOT NULL,
    data JSONB
) PARTITION BY RANGE (created_at);

CREATE TABLE events_2024_01 PARTITION OF events
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- To drop a month of old data:
DROP TABLE events_2023_01;
```

Want me to help set up partitioning or plan the disk expansion?"""),
        ("user", "We probably need to expand the disk. What are my options?"),
        ("assistant", r"""Here are your options from simplest to most involved:

**Option 1: Expand the existing disk (if using LVM or a cloud provider)**

If you're on LVM (check with `lvs`):
```bash
# Add a new physical disk
sudo pvcreate /dev/sdc
sudo vgextend vg_data /dev/sdc

# Extend the logical volume
sudo lvextend -l +100%FREE /dev/vg_data/lv_pg

# Grow the filesystem online (no downtime)
sudo resize2fs /dev/vg_data/lv_pg      # ext4
# or
sudo xfs_growfs /dev/vg_data/lv_pg     # XFS
```

If on a cloud VM, expand the virtual disk through the provider's UI, then:
```bash
# Resize the partition (if not using LVM)
sudo growpart /dev/sda 1    # grow partition 1
sudo resize2fs /dev/sda1     # grow filesystem
```

**Option 2: Move the PostgreSQL data directory to a larger disk**

```bash
# Stop PostgreSQL
sudo systemctl stop postgresql

# Copy data to the new disk
sudo rsync -av /var/lib/postgresql/ /mnt/newdisk/postgresql/

# Update the data directory
sudo nano /etc/postgresql/16/main/postgresql.conf
# Change: data_directory = '/mnt/newdisk/postgresql/16/main'

# Fix permissions
sudo chown -R postgres:postgres /mnt/newdisk/postgresql

# Start PostgreSQL
sudo systemctl start postgresql
sudo -u postgres psql -c "SHOW data_directory;"
```

**Option 3: Move specific tablespaces to another disk**

Keep the main database on the current disk but put large/growing tables on a new disk:

```bash
sudo mkdir -p /mnt/bigdisk/pg_tablespace
sudo chown postgres:postgres /mnt/bigdisk/pg_tablespace
```

```sql
-- Create a tablespace on the new disk
CREATE TABLESPACE bigdata LOCATION '/mnt/bigdisk/pg_tablespace';

-- Move a specific table to it
ALTER TABLE large_events SET TABLESPACE bigdata;

-- Set default tablespace for new tables
ALTER DATABASE yourdb SET default_tablespace = 'bigdata';
```

**Option 4: Archive old data**

```bash
# Dump old data to compressed files
sudo -u postgres pg_dump -t 'events_2023_*' --format=custom --compress=9 \
    yourdb > /backup/events-2023.dump

# Then drop the old partitions
sudo -u postgres psql -d yourdb -c "DROP TABLE events_2023_01, events_2023_02, ...;"
```

**Monitoring after expansion -- set up alerts:**

```bash
# Simple disk space alert in cron
cat << 'EOF' > /usr/local/bin/disk-alert.sh
#!/bin/bash
USAGE=$(df /var/lib/postgresql | awk 'NR==2 {print $5}' | tr -d '%')
if [ "$USAGE" -gt 80 ]; then
    echo "WARNING: PostgreSQL disk at ${USAGE}%" | mail -s "Disk Alert" admin@example.com
fi
EOF
chmod +x /usr/local/bin/disk-alert.sh
echo '0 * * * * root /usr/local/bin/disk-alert.sh' | sudo tee /etc/cron.d/disk-alert
```

**Recommended approach for your situation:**
1. **Short-term:** Add a new disk with Option 1 or 2 -- gives you immediate breathing room
2. **Medium-term:** Implement table partitioning on your largest tables so you can drop old data cleanly
3. **Long-term:** Set up automated monitoring with alerting at 70% and 85% thresholds

The key is to never let a database disk hit 100% -- PostgreSQL will crash and potentially corrupt data."""),
    ],
    "flm_multiturn_capacity"
))

# ===========================================================================
# 35. Disaster recovery drill (multi-turn, Debian minimal)
# ===========================================================================
conversations.append(make_multi(
    SYS_DEBMIN,
    [
        ("user", "I want to run a disaster recovery drill. How do I simulate a failure, restore from backup, and verify everything works?"),
        ("assistant", r"""A proper DR drill proves your backups actually work and documents the recovery procedure. Here's a structured approach.

**Prerequisites:**
- A test environment (VM, spare server, or container) -- never drill on production
- Your actual backup files
- Documentation of your backup procedures

**Phase 1: Document what you're recovering**

```bash
# On your production server, record the current state
sudo -u postgres psql -c "\l"               # list databases
sudo -u postgres psql -d mydb -c "\dt"      # list tables
sudo -u postgres psql -d mydb -c "SELECT count(*) FROM important_table;"
dpkg --get-selections > /tmp/prod-packages.txt
systemctl list-units --type=service --state=running > /tmp/prod-services.txt
```

**Phase 2: Verify your backups exist and are readable**

```bash
# Check your backup files
ls -lh /backup/
# For PostgreSQL custom format:
pg_restore --list /backup/latest/mydb.dump | head -20

# For a filesystem backup (tar):
tar -tzf /backup/latest/srv-backup.tar.gz | head -20

# Check integrity if you have checksums
sha256sum -c /backup/latest/checksums.sha256
```

**Phase 3: Set up the DR test environment**

Since you're on a minimal Debian install, let's use a fresh minimal environment. Do you want to test on a VM, a container, or this machine itself?"""),
        ("user", "I have a spare VM I can use. It's a fresh Debian 13 minimal install. Let me restore there."),
        ("assistant", r"""Perfect. Let's restore your full stack on the spare VM.

**Step 1: Transfer backups to the DR VM:**

```bash
# From your backup location, copy to the DR VM
rsync -avz --progress /backup/latest/ dr-vm:/tmp/restore/

# Or if backups are on a remote backup server:
ssh dr-vm "mkdir -p /tmp/restore"
rsync -avz --progress backup-server:/backup/latest/ dr-vm:/tmp/restore/
```

**Step 2: Install required packages on the DR VM:**

```bash
# On the DR VM (minimal install, so we need everything)
sudo apt update
sudo apt install postgresql nginx python3 python3-venv rsync
```

**Step 3: Restore PostgreSQL:**

```bash
# Start PostgreSQL
sudo systemctl start postgresql

# Create the database and user
sudo -u postgres psql << 'SQL'
CREATE USER myapp WITH PASSWORD 'restore_test_pass';
CREATE DATABASE mydb OWNER myapp;
SQL

# Restore from the custom-format dump
sudo -u postgres pg_restore \
    --dbname=mydb \
    --verbose \
    --exit-on-error \
    /tmp/restore/mydb.dump

# Verify the restore
sudo -u postgres psql -d mydb -c "\dt"
sudo -u postgres psql -d mydb -c "SELECT count(*) FROM important_table;"
```

Compare the row count with what you recorded from production. They should match.

**Step 4: Restore application files:**

```bash
# Extract the filesystem backup
sudo tar -xzf /tmp/restore/srv-backup.tar.gz -C /

# Or if using rsync backup:
sudo rsync -av /tmp/restore/srv/ /srv/

# Fix permissions
sudo chown -R www-data:www-data /srv/myapp
```

**Step 5: Restore configuration:**

```bash
# Restore nginx config
sudo cp /tmp/restore/etc-nginx/* /etc/nginx/sites-available/
sudo ln -sf /etc/nginx/sites-available/myapp /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl start nginx

# Restore any crontabs
sudo crontab -u root /tmp/restore/crontab-root
```

Ready to test if the application actually works?"""),
        ("user", "PostgreSQL restored with the correct row counts. Application files are in place. Let's test."),
        ("assistant", r"""**Step 6: Start the application and verify functionality**

```bash
# Install Python dependencies
cd /srv/myapp
sudo python3 -m venv venv
sudo venv/bin/pip install -r requirements.txt

# Update the database connection string to point to localhost
# (edit your config file -- .env, config.py, etc.)

# Start the application
sudo venv/bin/gunicorn --bind 127.0.0.1:8000 wsgi:app &

# Test basic connectivity
curl -s http://localhost:8000/ | head -20
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/
# Should return 200

# Test through nginx
curl -s -o /dev/null -w "%{http_code}" http://localhost/
```

**Step 7: Run verification tests**

```bash
# Test critical endpoints
for endpoint in "/" "/api/health" "/api/users" "/login"; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost${endpoint}")
    echo "$endpoint -> $STATUS"
done

# Verify data integrity
sudo -u postgres psql -d mydb << 'SQL'
-- Compare key metrics with production records
SELECT 'users' AS table_name, count(*) AS rows FROM users
UNION ALL
SELECT 'orders', count(*) FROM orders
UNION ALL
SELECT 'products', count(*) FROM products;
SQL
```

**Step 8: Document the results**

Create a DR drill report:

```bash
cat << EOF > /tmp/dr-drill-report.txt
DR Drill Report - $(date +%Y-%m-%d)
=====================================

Environment: Spare VM (Debian 13 minimal)
Backup date: $(stat -c %y /tmp/restore/mydb.dump | cut -d' ' -f1)

Recovery Steps:
1. Transfer backups:     $(date) - OK
2. Install packages:     OK (postgresql, nginx, python3)
3. Restore database:     OK - row counts match production
4. Restore app files:    OK
5. Restore configs:      OK
6. Application test:     OK - all endpoints responding
7. Data verification:    OK - counts match

Total recovery time: ___ minutes (fill in)

Issues encountered:
- (document any problems and how you solved them)

Action items:
- (improvements to backup/restore procedures)

RPO achieved: (how old was the backup?)
RTO achieved: (how long did recovery take?)
EOF
```

**Step 9: Clean up the DR VM:**

```bash
sudo systemctl stop postgresql nginx
sudo apt purge postgresql nginx
sudo rm -rf /tmp/restore /srv/myapp
```

**Key metrics to track:**
- **RPO (Recovery Point Objective):** How much data can you afford to lose? If your backup is daily, RPO = 24 hours max
- **RTO (Recovery Time Objective):** How long until service is restored? Your drill just measured this

**Recommendations based on the drill:**
1. Run this drill quarterly
2. If RTO is too long, consider streaming replication (automatic failover, near-zero RPO)
3. If RPO is too high, switch from daily pg_dump to continuous WAL archiving
4. Automate the restore process into a script so any team member can run it
5. Store the DR report alongside your backups"""),
    ],
    "flm_multiturn_dr_drill"
))

# -- Write output -------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "a") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    print(f"Appended {len(conversations)} conversations to {OUTPUT}")


if __name__ == "__main__":
    main()
