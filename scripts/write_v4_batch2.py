#!/usr/bin/env python3
"""Write hand-crafted SFT conversations for V4 training — Batch 2.

Topics: Systemd & Services, Process Management, Logging,
        Hardware & Boot, Power Management, Debian Packaging,
        Multi-turn Troubleshooting.

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
        text = f"<|system|>\n{system}\n<|user|>\n{question}\n<|assistant|>\n{answer}\n"
    return {"text": text, "source": source}


def make_multi(system, turns, source):
    """Build a multi-turn conversation dict.

    turns is a list of (role, content) tuples where role is 'user' or 'assistant'.
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
# 1. Write a custom systemd service file (Debian) - Node.js app
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I write a systemd service file for a Node.js application?",
    r"""Create a unit file at `/etc/systemd/system/myapp.service`:

```ini
[Unit]
Description=My Node.js Application
Documentation=https://example.com/docs
After=network.target
Wants=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/myapp
ExecStart=/usr/bin/node /opt/myapp/server.js
Restart=on-failure
RestartSec=5
StartLimitBurst=5
StartLimitIntervalSec=60

# Environment
Environment=NODE_ENV=production
Environment=PORT=3000
EnvironmentFile=-/opt/myapp/.env

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/myapp/data
PrivateTmp=true

# Logging — stdout/stderr go to journald automatically
StandardOutput=journal
StandardError=journal
SyslogIdentifier=myapp

[Install]
WantedBy=multi-user.target
```

Key points:
- **Type=simple** is correct for Node.js — the process stays in the foreground.
- **Restart=on-failure** restarts on crashes but not on clean exit (code 0).
- **RestartSec=5** waits 5 seconds between restarts to avoid hammering.
- **StartLimitBurst/IntervalSec** prevents infinite restart loops — max 5 restarts per 60 seconds.
- **ProtectSystem=strict** makes the entire filesystem read-only except `ReadWritePaths`.
- **EnvironmentFile=-** (note the `-`) means "don't fail if the file is missing."

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now myapp
sudo systemctl status myapp
sudo journalctl -u myapp -f
```

If you change the service file, always run `daemon-reload` before restarting.""",
    "flm_systemd"
))

# ---------------------------------------------------------------------------
# 2. systemd timer as cron replacement (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I use a systemd timer instead of cron?",
    r"""You need two files: a `.timer` unit and a matching `.service` unit.

**Example: run a backup script every night at 2:30 AM.**

Create `/etc/systemd/system/backup.service`:
```ini
[Unit]
Description=Nightly backup job

[Service]
Type=oneshot
ExecStart=/usr/local/bin/backup.sh
User=root
```

Create `/etc/systemd/system/backup.timer`:
```ini
[Unit]
Description=Run backup every night at 2:30 AM

[Timer]
OnCalendar=*-*-* 02:30:00
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
```

Enable the **timer** (not the service):
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now backup.timer
```

Key differences from cron:
- **Persistent=true** — if the machine was off when the timer should have fired, it runs as soon as possible after boot. Cron just silently skips it.
- **RandomizedDelaySec=300** — adds up to 5 minutes of random delay to avoid the "thundering herd" problem when many machines fire at the same time.
- Logs go to journald instead of cron mail: `journalctl -u backup.service`
- **Type=oneshot** is correct for scripts that run and exit. systemd waits for them to finish.

**Other timer schedules:**
```ini
OnCalendar=hourly              # every hour on the hour
OnCalendar=Mon *-*-* 08:00:00  # every Monday at 8 AM
OnCalendar=*-*-01 00:00:00     # first of every month
OnBootSec=5min                 # 5 minutes after boot
OnUnitActiveSec=1h             # 1 hour after last run
```

**List active timers:**
```bash
systemctl list-timers --all
```

**Test your OnCalendar expression:**
```bash
systemd-analyze calendar "*-*-* 02:30:00"
```

This parses the expression and shows the next trigger time — very useful for debugging.""",
    "flm_systemd"
))

# ---------------------------------------------------------------------------
# 3. Troubleshoot a failed service (multi-turn, Debian)
# ---------------------------------------------------------------------------
conversations.append(make_multi(
    SYS_DEB13,
    [
        ("user", "My service won't start. I created a systemd unit but it just fails immediately."),
        ("assistant", r"""Let's look at the error. Run these:

```bash
sudo systemctl status yourservice
sudo journalctl -u yourservice -n 50 --no-pager
```

The status command shows the current state, and journalctl shows the recent log output. What's the service called and what do you see?"""),
        ("user", "The service is called myapi. Status says \"failed\" with exit code 203/EXEC. Journal says \"Exec format error\"."),
        ("assistant", r"""Exit code 203/EXEC with "Exec format error" means systemd can't execute the binary specified in `ExecStart`. Common causes:

1. **Missing shebang line** — if ExecStart points to a script, it needs `#!/bin/bash` or `#!/usr/bin/env python3` as the first line.
2. **Wrong architecture binary** — trying to run an ARM binary on x86 or vice versa.
3. **File isn't executable** — missing execute permission.

Check your service file:
```bash
systemctl cat myapi
```

Then check the ExecStart path:
```bash
ls -la /path/to/your/binary
file /path/to/your/binary
head -1 /path/to/your/binary
```

What does your ExecStart line look like?"""),
        ("user", "ExecStart is /opt/myapi/run.sh and it's a bash script. head -1 shows just a comment, no shebang."),
        ("assistant", r"""That's the problem. Without a shebang line (`#!/bin/bash`), the kernel doesn't know how to execute the script. systemd calls `execve()` directly, and unlike running from a shell, there's no fallback to try bash.

Fix it — add the shebang as the first line of `/opt/myapi/run.sh`:
```bash
#!/bin/bash
```

Also make sure it's executable:
```bash
chmod +x /opt/myapi/run.sh
```

Then restart:
```bash
sudo systemctl restart myapi
sudo systemctl status myapi
```

Alternatively, you can set `ExecStart=/bin/bash /opt/myapi/run.sh` in the unit file, but adding a proper shebang is the right fix. Every script should have one.

One more thing: if your script uses `ExecStart=/opt/myapi/run.sh arg1 arg2`, the path **must be absolute**. systemd rejects relative paths in ExecStart — another common gotcha."""),
    ],
    "flm_systemd_troubleshoot"
))

# ---------------------------------------------------------------------------
# 4. systemd journal management (Ubuntu 24.04)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_UBU24,
    "How do I manage the systemd journal? It's using a lot of disk space.",
    r"""Check current journal disk usage:
```bash
journalctl --disk-usage
```

**Clean up old entries:**
```bash
# Keep only the last 7 days
sudo journalctl --vacuum-time=7d

# Or limit by size — keep at most 500 MB
sudo journalctl --vacuum-size=500M

# Or keep at most 5 journal files
sudo journalctl --vacuum-files=5
```

**Set permanent limits** in `/etc/systemd/journald.conf`:
```ini
[Journal]
SystemMaxUse=500M
SystemKeepFree=1G
SystemMaxFileSize=50M
MaxRetentionSec=1month
```

Then restart journald:
```bash
sudo systemctl restart systemd-journald
```

**Enable persistent journal** (survives reboots — Ubuntu 24.04 does this by default, but verify):
```bash
sudo mkdir -p /var/log/journal
sudo systemd-tmpfiles --create --prefix /var/log/journal
```

If `/var/log/journal` exists, logs persist across reboots. If only `/run/log/journal` exists, logs are volatile (lost on reboot).

**Useful journalctl tips:**
```bash
# Show boots
journalctl --list-boots

# Logs from previous boot
journalctl -b -1

# Logs from current boot only
journalctl -b

# Follow in real time
journalctl -f

# Show kernel messages only
journalctl -k
```

**Forward journal to syslog** if you also need traditional log files:
```ini
[Journal]
ForwardToSyslog=yes
```""",
    "flm_journal"
))

# ---------------------------------------------------------------------------
# 5. List and manage services (Arch)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_ARCH,
    "What are the most useful systemctl commands for managing services?",
    r"""Here's a comprehensive systemctl reference:

**Service lifecycle:**
```bash
sudo systemctl start nginx        # start now
sudo systemctl stop nginx         # stop now
sudo systemctl restart nginx      # stop then start
sudo systemctl reload nginx       # reload config without restart (if supported)
sudo systemctl reload-or-restart nginx  # reload if possible, restart otherwise
```

**Enable/disable (auto-start at boot):**
```bash
sudo systemctl enable nginx       # start at boot
sudo systemctl disable nginx      # don't start at boot
sudo systemctl enable --now nginx # enable AND start immediately
sudo systemctl is-enabled nginx   # check if enabled
```

**Status and inspection:**
```bash
systemctl status nginx            # current state, recent logs
systemctl show nginx              # all properties (machine-readable)
systemctl cat nginx               # show the unit file contents
systemctl list-dependencies nginx # what it depends on
```

**List services:**
```bash
systemctl list-units --type=service              # running services
systemctl list-units --type=service --all         # all services (including dead)
systemctl list-units --type=service --state=failed # failed services only
systemctl list-unit-files --type=service          # all installed service files
```

**Find failed services:**
```bash
systemctl --failed
```

**Masking — stronger than disable:**
```bash
sudo systemctl mask nginx         # cannot be started at all (even manually)
sudo systemctl unmask nginx       # undo mask
```

Masking symlinks the unit to `/dev/null`. Use this to prevent a service from ever running, even as a dependency of another unit.

**Timers:**
```bash
systemctl list-timers --all       # show all timers and next fire time
```

**System targets (runlevels):**
```bash
systemctl get-default             # current default target
sudo systemctl set-default multi-user.target   # boot to console
sudo systemctl set-default graphical.target    # boot to GUI
sudo systemctl isolate rescue.target           # switch to rescue mode now
```

On Arch, remember that freshly installed packages do NOT auto-enable their services — you must explicitly `enable --now` them.""",
    "flm_systemd"
))

# ---------------------------------------------------------------------------
# 6. systemd-resolved DNS configuration (Ubuntu 24.04)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_UBU24,
    "How do I configure DNS on Ubuntu 24.04? I keep seeing 127.0.0.53 in /etc/resolv.conf.",
    r"""Ubuntu 24.04 uses `systemd-resolved` for DNS resolution. The `127.0.0.53` in `/etc/resolv.conf` is the local stub resolver — it's correct and intentional.

**Check current DNS config:**
```bash
resolvectl status
```

This shows the DNS servers for each network interface and the global config.

**Set custom DNS servers** (e.g., Cloudflare and Google):

Edit `/etc/systemd/resolved.conf`:
```ini
[Resolve]
DNS=1.1.1.1 9.9.9.9
FallbackDNS=8.8.8.8
Domains=~.
DNSSEC=allow-downgrade
DNSOverTLS=opportunistic
```

Then restart:
```bash
sudo systemctl restart systemd-resolved
```

**Per-interface DNS** (useful if you want different DNS for VPN vs. LAN):
```bash
# Set DNS for a specific interface
sudo resolvectl dns eth0 1.1.1.1 1.0.0.1
sudo resolvectl domain eth0 ~.
```

**Debug DNS resolution:**
```bash
resolvectl query example.com      # resolve and show which server answered
resolvectl statistics             # cache hit/miss stats
resolvectl flush-caches           # clear the DNS cache
```

**If you want to bypass systemd-resolved entirely** (not recommended on Ubuntu, but possible):
```bash
sudo systemctl disable --now systemd-resolved
sudo rm /etc/resolv.conf   # it's a symlink
# Create a plain resolv.conf:
echo -e "nameserver 1.1.1.1\nnameserver 9.9.9.9" | sudo tee /etc/resolv.conf
```

**Important:** Do NOT manually edit `/etc/resolv.conf` on Ubuntu 24.04 — it's a symlink to `/run/systemd/resolve/stub-resolv.conf` and will be overwritten. Always configure DNS through `resolved.conf` or `resolvectl`.

**The `Domains=~.` line** means "use these DNS servers for all domains" (the `~.` is a routing domain that matches everything). Without it, the servers are only used as fallback.""",
    "flm_systemd_dns"
))

# ---------------------------------------------------------------------------
# 7. Override vendor systemd units with drop-in files (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I override settings in a systemd service file that came from a package?",
    r"""Never edit files in `/lib/systemd/system/` directly — package updates will overwrite your changes. Use **drop-in files** instead.

**The easy way — `systemctl edit`:**
```bash
sudo systemctl edit nginx
```

This opens an editor and creates a drop-in file at `/etc/systemd/system/nginx.service.d/override.conf`. Any settings you put there override the vendor unit.

**Example — increase the file descriptor limit and add an environment variable:**
```ini
[Service]
LimitNOFILE=65536
Environment=NGINX_WORKER_CONNECTIONS=4096
```

Save and exit. systemd automatically runs `daemon-reload`.

**To replace the entire unit file** (not just override parts):
```bash
sudo systemctl edit --full nginx
```

This copies the vendor file to `/etc/systemd/system/nginx.service` where you can change anything. The copy in `/etc/systemd/system/` takes priority over `/lib/systemd/system/`.

**Priority order** (highest to lowest):
1. `/etc/systemd/system/unit.d/*.conf` — drop-in overrides
2. `/etc/systemd/system/unit` — full admin override
3. `/run/systemd/system/unit` — runtime overrides
4. `/lib/systemd/system/unit` — vendor/package files

**View effective configuration** (merged result of vendor + drop-ins):
```bash
systemctl cat nginx
```

**To revert to the vendor default:**
```bash
sudo rm /etc/systemd/system/nginx.service.d/override.conf
sudo rmdir /etc/systemd/system/nginx.service.d/
sudo systemctl daemon-reload
sudo systemctl restart nginx
```

**Important gotcha:** Some directives are additive (like `Environment=`), while others are last-one-wins. If you need to *clear* a list-type directive before setting it, use an empty assignment first:
```ini
[Service]
ExecStart=
ExecStart=/usr/sbin/nginx -g "daemon off;" -c /my/custom/nginx.conf
```

The empty `ExecStart=` clears the vendor value, then your new line sets the replacement.""",
    "flm_systemd"
))

# ---------------------------------------------------------------------------
# 8. systemd resource limits (Fedora)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_FED41,
    "How do I limit CPU and memory usage for a systemd service?",
    r"""systemd has built-in cgroup resource controls. Add them to the `[Service]` section of the unit file (or use a drop-in override).

**Memory limits:**
```ini
[Service]
MemoryMax=1G          # hard limit — process gets OOM-killed if exceeded
MemoryHigh=800M       # soft limit — kernel throttles the process
MemorySwapMax=0       # prevent swapping entirely
```

**CPU limits:**
```ini
[Service]
CPUQuota=200%         # max 200% CPU (= 2 full cores)
CPUWeight=50          # relative weight (default 100, range 1-10000)
CPUAffinity=0 1       # pin to CPU cores 0 and 1
```

**I/O limits:**
```ini
[Service]
IOWeight=50                          # relative I/O weight (default 100)
IOReadBandwidthMax=/dev/sda 100M     # max read speed
IOWriteBandwidthMax=/dev/sda 50M     # max write speed
IODeviceLatencyTargetSec=/dev/sda 50ms  # latency target
```

**Process limits:**
```ini
[Service]
LimitNOFILE=65536     # max open file descriptors
LimitNPROC=4096       # max processes/threads
TasksMax=512          # max tasks (cgroup-level, includes threads)
```

**Apply to an existing service with a drop-in:**
```bash
sudo systemctl edit myapp
```

Add your limits, save, then:
```bash
sudo systemctl restart myapp
```

**Verify limits are applied:**
```bash
systemctl show myapp -p MemoryMax,CPUQuota,TasksMax
# Or check the cgroup directly:
systemd-cgtop
```

**Slice-level limits** — control a group of services together:
```bash
sudo systemctl set-property myapp.service MemoryMax=1G
```

This creates a persistent drop-in automatically. Or define a custom slice:

Create `/etc/systemd/system/limited.slice`:
```ini
[Unit]
Description=Resource-limited services

[Slice]
MemoryMax=2G
CPUQuota=300%
```

Then assign services to it:
```ini
[Service]
Slice=limited.slice
```

All services in the slice share the 2 GB / 300% limits collectively.

**Note:** CPU and I/O controls require the kernel to have the appropriate cgroup controllers enabled. On Fedora 41 with cgroups v2 (the default), everything should work out of the box.""",
    "flm_systemd_resources"
))

# ---------------------------------------------------------------------------
# 9. Process priorities — nice, renice, ionice (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do nice and renice work? How do I control process priority?",
    r"""Linux has two kinds of scheduling priority: CPU (nice) and I/O (ionice).

**CPU priority — nice values:**

Nice values range from -20 (highest priority) to 19 (lowest). Default is 0.

```bash
# Start a process with low priority (won't steal CPU from other tasks)
nice -n 10 make -j$(nproc)

# Start with high priority (needs root)
sudo nice -n -15 /opt/critical-app

# Check the nice value of a running process
ps -o pid,ni,comm -p 1234
```

**Change priority of a running process — renice:**
```bash
# Lower priority of PID 1234
renice 10 -p 1234

# Raise priority (needs root)
sudo renice -5 -p 1234

# Change priority for all processes of a user
sudo renice 15 -u slowuser

# Change priority for a process group
sudo renice 10 -g 5678
```

**I/O priority — ionice:**

Three classes:
- **1 (realtime)** — highest I/O priority (needs root). Use with caution.
- **2 (best-effort)** — default. Has priority levels 0-7 (0 = highest).
- **3 (idle)** — only gets I/O time when nothing else needs the disk.

```bash
# Start a backup with idle I/O priority
ionice -c3 rsync -a /data /backup/

# Start with best-effort, priority level 6 (low)
ionice -c2 -n6 tar czf archive.tar.gz /bigdata

# Change I/O class of a running process
ionice -c3 -p 1234

# Check current I/O class
ionice -p 1234
```

**Combine both for background tasks:**
```bash
nice -n 19 ionice -c3 find / -type f -name "*.log" -mtime +30 -delete
```

This runs the find command with the lowest CPU and I/O priority — it will only use resources that nothing else wants.

**In systemd service files:**
```ini
[Service]
Nice=10
IOSchedulingClass=idle
IOSchedulingPriority=7
CPUSchedulingPolicy=batch
```

**Important:** Nice values only matter under CPU contention. If nothing else is competing for CPU, a nice-19 process still runs at full speed. Same with ionice — it's about relative priority, not absolute throttling. For absolute CPU limits, use cgroups (systemd's `CPUQuota=`).""",
    "flm_processes"
))

# ---------------------------------------------------------------------------
# 10. Understanding load average (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "What does load average mean in Linux? I see numbers like 2.5, 1.8, 0.9 when I run uptime.",
    r"""Load average represents the average number of processes that are either **running on the CPU** or **waiting for CPU/I/O** over 1, 5, and 15 minute intervals.

From your example: `2.5, 1.8, 0.9`
- 2.5 processes were active/waiting on average over the last 1 minute
- 1.8 over the last 5 minutes
- 0.9 over the last 15 minutes

**How to interpret it depends on your CPU count.** Check how many cores you have:
```bash
nproc
```

If you have 4 cores:
- Load 4.0 = 100% utilized, no queuing
- Load 2.0 = 50% utilized, plenty of headroom
- Load 8.0 = fully utilized + 4 processes waiting in queue

So your `2.5, 1.8, 0.9` on a 4-core system means load is increasing but still well within capacity.

**Rule of thumb:**
- Load < number of cores = system has spare capacity
- Load = number of cores = fully utilized but not overloaded
- Load > number of cores = processes are queuing up, things will feel slow

**The trend matters:**
- `0.9, 1.8, 2.5` (increasing left to right) = load is *decreasing* — the spike was recent but it's settling down
- `2.5, 1.8, 0.9` (decreasing left to right) = load is *increasing* — something is ramping up

**Load includes I/O wait,** which is important. A high load with low CPU usage usually means processes are waiting on disk I/O (slow disk, NFS, heavy reads/writes). Check with:
```bash
# iowait is the %wa column
top -bn1 | head -5
# or
vmstat 1 5
```

**Per-CPU breakdown:**
```bash
mpstat -P ALL 1 5
```

**Common misconception:** Load average is NOT the same as CPU usage percentage. A single process stuck in uninterruptible disk sleep (D state) adds 1.0 to load average even though it uses 0% CPU.""",
    "flm_processes"
))

# ---------------------------------------------------------------------------
# 11. cgroups v2 basics (Debian) — systemd slices
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "Can you explain cgroups v2 and how systemd uses them?",
    r"""cgroups (control groups) v2 is the Linux kernel mechanism for organizing processes into hierarchical groups and applying resource limits, accounting, and isolation.

**systemd and cgroups v2 on Debian 13:**

Debian 13 uses cgroups v2 (unified hierarchy) by default. systemd automatically organizes all processes into a cgroup tree:

```bash
# View the cgroup hierarchy
systemd-cgls

# Real-time cgroup resource usage
systemd-cgtop
```

**The hierarchy uses "slices":**
```
-.slice (root)
├── init.scope (PID 1)
├── system.slice (system services)
│   ├── ssh.service
│   ├── nginx.service
│   └── ...
├── user.slice (user sessions)
│   ├── user-1000.slice
│   │   ├── session-1.scope
│   │   └── user@1000.service
│   └── ...
└── machine.slice (VMs and containers)
```

**Resource control at the slice level:**

Create a custom slice at `/etc/systemd/system/batch.slice`:
```ini
[Unit]
Description=Batch jobs slice

[Slice]
MemoryMax=4G
CPUQuota=200%
IOWeight=20
```

Assign services to it:
```ini
[Service]
Slice=batch.slice
```

All services in `batch.slice` share the 4 GB memory and 200% CPU limits collectively.

**Check what cgroup a process is in:**
```bash
cat /proc/$(pidof nginx)/cgroup
# Output: 0::/system.slice/nginx.service
```

**View cgroup resource limits and usage:**
```bash
# Memory usage of a service
systemctl show nginx -p MemoryCurrent
cat /sys/fs/cgroup/system.slice/nginx.service/memory.current

# CPU usage
cat /sys/fs/cgroup/system.slice/nginx.service/cpu.stat

# I/O stats
cat /sys/fs/cgroup/system.slice/nginx.service/io.stat
```

**Set limits on the fly (persistent):**
```bash
sudo systemctl set-property nginx.service MemoryMax=512M
```

This creates a drop-in file that persists across reboots.

**Set limits on the fly (runtime only):**
```bash
sudo systemctl set-property --runtime nginx.service MemoryMax=512M
```

**User-level cgroups:** Each logged-in user gets their own slice (`user-UID.slice`). You can set per-user limits:
```bash
sudo systemctl set-property user-1000.slice MemoryMax=8G
```

**cgroups v2 controllers:** The available controllers on Debian 13 are `cpu`, `cpuset`, `io`, `memory`, `pids`, and `rdma`. Check which are enabled:
```bash
cat /sys/fs/cgroup/cgroup.controllers
```""",
    "flm_cgroups"
))

# ---------------------------------------------------------------------------
# 12. ulimits — what they are and how to change them (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "What are ulimits and how do I change them?",
    r"""ulimits are per-process resource limits enforced by the kernel. Each process inherits limits from its parent.

**View current limits:**
```bash
ulimit -a          # show all soft limits
ulimit -aH         # show all hard limits
```

Key limits:
- **open files (-n)** — max file descriptors per process (default often 1024)
- **max user processes (-u)** — max processes per user
- **stack size (-s)** — max stack size per thread
- **core file size (-c)** — max core dump size (0 = disabled)
- **virtual memory (-v)** — max address space
- **file size (-f)** — max file size a process can create

**Soft vs. hard limits:**
- **Soft limit** — the effective limit. A process can raise it up to the hard limit.
- **Hard limit** — the ceiling. Only root can raise hard limits.

**Change limits for the current shell session:**
```bash
ulimit -n 65536    # set soft limit for open files
ulimit -Hn 65536   # set hard limit (needs root)
```

**Permanent changes — `/etc/security/limits.conf`:**
```
# <domain>  <type>  <item>  <value>
www-data    soft    nofile  65536
www-data    hard    nofile  65536
*           soft    nproc   4096
*           hard    nproc   8192
@devs       soft    core    unlimited
```

Domain can be a username, `*` (all users), or `@groupname`.

**For systemd services** (limits.conf does NOT apply to systemd services):
```ini
[Service]
LimitNOFILE=65536
LimitNPROC=4096
LimitCORE=infinity
LimitMEMLOCK=infinity
```

Or globally for all services in `/etc/systemd/system.conf`:
```ini
[Manager]
DefaultLimitNOFILE=65536
```

**Check limits of a running process:**
```bash
cat /proc/$(pidof nginx)/limits
```

**Important:** PAM must be configured to apply limits.conf. Verify this line exists in `/etc/pam.d/common-session`:
```
session required pam_limits.so
```

On Debian it's there by default. If you SSH in and limits aren't applied, check `/etc/pam.d/sshd` — it needs `UsePAM yes` in `/etc/ssh/sshd_config` (also the default).""",
    "flm_ulimits"
))

# ---------------------------------------------------------------------------
# 13. strace a running process (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I use strace to debug a running process?",
    r"""strace traces system calls made by a process — it's invaluable for figuring out what a program is actually doing.

**Install:**
```bash
sudo apt install strace
```

**Attach to a running process:**
```bash
sudo strace -p 1234
```

This shows every system call in real-time. Press Ctrl+C to detach (the process continues running normally).

**Useful flags:**
```bash
# Follow child processes (forks)
sudo strace -f -p 1234

# Show timestamps
sudo strace -t -p 1234          # wall clock time
sudo strace -T -p 1234          # time spent in each syscall
sudo strace -r -p 1234          # relative timestamps

# Filter specific system calls
sudo strace -e trace=open,read,write -p 1234
sudo strace -e trace=network -p 1234       # network calls only
sudo strace -e trace=file -p 1234          # file access only
sudo strace -e trace=process -p 1234       # fork, exec, exit

# Save output to a file
sudo strace -o /tmp/trace.log -p 1234

# Show strings fully (default truncates at 32 chars)
sudo strace -s 1024 -p 1234
```

**Trace a command from start:**
```bash
strace ls -la /tmp
strace -e trace=file ls -la /tmp   # only file-related syscalls
```

**Common debugging scenarios:**

"What files is this process opening?"
```bash
sudo strace -e trace=openat -p 1234
```

"Why is this process hanging?"
```bash
sudo strace -p 1234
# If you see it stuck on futex(), read(), poll(), etc. — it's waiting for a lock, data, or event
```

"What network connections is it making?"
```bash
sudo strace -e trace=connect,sendto,recvfrom -p 1234
```

"Where is it writing its output?"
```bash
sudo strace -e trace=write -s 1024 -p 1234
```

**Count system calls (summary mode):**
```bash
sudo strace -c -p 1234
# Let it run for a while, then Ctrl+C
```

This prints a table showing how many times each syscall was called and total time spent — great for finding bottlenecks.

**Performance note:** strace adds significant overhead (can slow a process 10-100x). For production, consider `perf trace` or `bpftrace` which use in-kernel tracing with much less overhead.""",
    "flm_debugging"
))

# ---------------------------------------------------------------------------
# 14. lsof usage (short Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SHORT_DEB13,
    "How do I use lsof to find open files and who's using a port?",
    r"""lsof (list open files) shows what files, sockets, and pipes are open by which processes.

**Find what's using a port:**
```bash
sudo lsof -i :80            # who's listening on port 80
sudo lsof -i :8080          # port 8080
sudo lsof -i TCP:443        # TCP port 443 specifically
sudo lsof -i UDP:53         # UDP port 53 (DNS)
```

**Find all network connections for a process:**
```bash
sudo lsof -i -a -p 1234     # all network activity for PID 1234
sudo lsof -i -a -c nginx    # all network activity for processes named nginx
```

**Find open files for a process:**
```bash
lsof -p 1234                # all files open by PID 1234
lsof -c python3             # all files open by processes named python3
```

**Find who has a file open:**
```bash
lsof /var/log/syslog         # who has this file open
lsof +D /var/log/            # who has files open in this directory
```

**Find deleted files still held open** (freeing disk space):
```bash
sudo lsof +L1
```

This finds files that have been deleted but are still open by a process — the disk space won't be freed until the process closes the file or exits.

**Find what's preventing an unmount:**
```bash
sudo lsof +D /mnt/usb       # who has files open on this mount
```

**List all network connections:**
```bash
sudo lsof -i                 # all network connections
sudo lsof -i -n -P           # numeric IPs/ports (faster, no DNS lookups)
```

**Useful combinations:**
```bash
# Find all files opened by a specific user
lsof -u www-data

# Exclude a user (^ negation)
lsof -u ^root

# Find files open in a directory tree, recursively
lsof +D /opt/myapp/
```""",
    "flm_lsof"
))

# ---------------------------------------------------------------------------
# 15. journalctl advanced usage (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "What are some advanced journalctl filtering techniques?",
    r"""journalctl has powerful filtering that goes far beyond `journalctl -u service`.

**Filter by time:**
```bash
journalctl --since "2024-01-15 08:00:00"
journalctl --since "1 hour ago"
journalctl --since "yesterday" --until "today"
journalctl --since "2024-01-15" --until "2024-01-16"
```

**Filter by priority (severity):**
```bash
journalctl -p err               # errors and above (err, crit, alert, emerg)
journalctl -p warning           # warnings and above
journalctl -p 0..3              # emerg through err (numeric: 0=emerg, 7=debug)
```

Priority levels: emerg(0), alert(1), crit(2), err(3), warning(4), notice(5), info(6), debug(7).

**Filter by unit:**
```bash
journalctl -u nginx                     # single unit
journalctl -u nginx -u php-fpm          # multiple units
journalctl -u "nginx*"                  # wildcard
```

**Filter by boot:**
```bash
journalctl --list-boots                  # list all boots with IDs
journalctl -b                            # current boot only
journalctl -b -1                         # previous boot
journalctl -b abc123                     # specific boot ID
```

**Filter by process, user, or group:**
```bash
journalctl _PID=1234                     # specific PID
journalctl _UID=1000                     # specific user
journalctl _GID=33                       # specific group
journalctl _COMM=nginx                   # process name
journalctl _EXE=/usr/sbin/nginx          # executable path
```

**Combine filters (AND logic):**
```bash
journalctl -u nginx -p err --since "1 hour ago"
```

**Output formats:**
```bash
journalctl -o json                       # JSON (one object per line)
journalctl -o json-pretty               # pretty JSON
journalctl -o verbose                    # all metadata fields
journalctl -o short-iso                  # ISO 8601 timestamps
journalctl -o cat                        # just the message, no metadata
```

**Kernel messages only:**
```bash
journalctl -k                            # same as dmesg but filterable
journalctl -k -p err                     # kernel errors only
```

**Follow with filtering:**
```bash
journalctl -f -u nginx -p warning        # follow nginx warnings in real-time
```

**Show full messages (no truncation):**
```bash
journalctl --no-pager -l
```

**Export for analysis:**
```bash
journalctl -u nginx --since "1 hour ago" -o json > /tmp/nginx_logs.json
```""",
    "flm_logging"
))

# ---------------------------------------------------------------------------
# 16. Configure rsyslog (Debian) — remote logging
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I configure rsyslog for remote logging and separate log files?",
    r"""rsyslog is the traditional syslog daemon on Debian. It receives log messages and routes them to files, remote servers, or databases.

**Logging to separate files based on program name:**

Create `/etc/rsyslog.d/50-myapp.conf`:
```
# Log myapp messages to a dedicated file
if $programname == 'myapp' then /var/log/myapp.log
& stop
```

The `& stop` prevents the message from also going to `/var/log/syslog`.

**Filtering by facility and severity:**
```
# All mail logs to one file
mail.*                          /var/log/mail.log

# Only auth errors and above
auth.err                        /var/log/auth-errors.log

# Everything except debug messages
*.info;mail.none;authpriv.none  /var/log/messages
```

**Remote logging — sending logs to a central server:**

On the **client** (sender), add to `/etc/rsyslog.d/60-remote.conf`:
```
# Send all logs to remote server via TCP (reliable)
*.* @@logserver.example.com:514

# Or via UDP (faster but unreliable)
*.* @logserver.example.com:514

# Send only auth logs remotely
auth,authpriv.* @@logserver.example.com:514
```

`@@` = TCP, `@` = UDP.

On the **server** (receiver), edit `/etc/rsyslog.conf`:
```
# Enable TCP reception
module(load="imtcp")
input(type="imtcp" port="514")

# Enable UDP reception
module(load="imudp")
input(type="imudp" port="514")
```

**Organize remote logs by hostname:**
```
# Template for per-host log files
template(name="RemoteLogs" type="string"
    string="/var/log/remote/%HOSTNAME%/%PROGRAMNAME%.log")

# Apply to all remote messages
if $fromhost-ip != '127.0.0.1' then {
    action(type="omfile" dynaFile="RemoteLogs")
    stop
}
```

**Restart after any changes:**
```bash
sudo rsyslogd -N1              # validate config syntax
sudo systemctl restart rsyslog
```

**Test logging:**
```bash
logger -t myapp "Test message from rsyslog"
tail /var/log/myapp.log
```""",
    "flm_logging"
))

# ---------------------------------------------------------------------------
# 17. logrotate configuration (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEBMIN,
    "How do I configure logrotate for my application's log files?",
    r"""logrotate manages log file rotation — compressing, removing old files, and optionally restarting services to reopen log handles.

**Create a config file** at `/etc/logrotate.d/myapp`:
```
/var/log/myapp/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0640 www-data adm
    sharedscripts
    postrotate
        systemctl reload myapp > /dev/null 2>&1 || true
    endscript
}
```

What each directive does:
- **daily** — rotate once per day (also: weekly, monthly)
- **rotate 14** — keep 14 old versions
- **compress** — gzip old log files
- **delaycompress** — don't compress the most recent rotated file (so the previous log is still readable without decompression)
- **missingok** — don't error if the log file doesn't exist
- **notifempty** — don't rotate empty files
- **create 0640 www-data adm** — create new log file with these permissions
- **sharedscripts** — run postrotate only once even if multiple files match
- **postrotate** — command to run after rotation (often a reload to reopen log files)

**Size-based rotation:**
```
/var/log/myapp/app.log {
    size 100M
    rotate 5
    compress
    copytruncate
}
```

- **size 100M** — rotate when file exceeds 100 MB
- **copytruncate** — copy the log then truncate the original, instead of moving it. Use this when the application can't be told to reopen its log file. Slight risk of losing lines written between copy and truncate.

**Test your configuration:**
```bash
# Dry run — show what would happen
sudo logrotate -d /etc/logrotate.d/myapp

# Force a rotation now (useful for testing)
sudo logrotate -f /etc/logrotate.d/myapp
```

**Check logrotate status:**
```bash
cat /var/lib/logrotate/status
```

This shows when each file was last rotated.

**logrotate runs via cron** (or systemd timer on modern Debian). Check:
```bash
cat /etc/cron.daily/logrotate
# or
systemctl list-timers | grep logrotate
```""",
    "flm_logging"
))

# ---------------------------------------------------------------------------
# 18. Centralized logging with journald (Fedora)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_FED41,
    "How do I set up centralized logging with journald on Fedora?",
    r"""systemd-journal can forward logs to a remote journal using `systemd-journal-remote` and `systemd-journal-upload`.

**On the central log server:**

Install and configure the receiver:
```bash
sudo dnf install systemd-journal-remote
```

Edit `/etc/systemd/journal-remote.conf`:
```ini
[Remote]
Seal=false
SplitMode=host
ServerKeyFile=/etc/ssl/private/journal-remote.key
ServerCertificateFile=/etc/ssl/certs/journal-remote.crt
TrustedCertificateFile=/etc/ssl/ca/ca.crt
```

For quick testing without TLS:
```bash
sudo systemd-journal-remote --listen-http=-3 --output=/var/log/journal/remote/
```

Enable the service:
```bash
sudo systemctl enable --now systemd-journal-remote.socket
```

Logs are stored in `/var/log/journal/remote/` organized by client machine ID.

**On the client machines:**
```bash
sudo dnf install systemd-journal-remote
```

Edit `/etc/systemd/journal-upload.conf`:
```ini
[Upload]
URL=http://logserver.example.com:19532
```

Enable and start:
```bash
sudo systemctl enable --now systemd-journal-upload
```

**Browse remote logs on the server:**
```bash
# List all remote machine IDs
ls /var/log/journal/remote/

# Read logs from a specific machine
journalctl --file /var/log/journal/remote/remote-*.journal

# Filter by hostname
journalctl --file /var/log/journal/remote/remote-*.journal _HOSTNAME=webserver1
```

**Alternative: forward journal to syslog for rsyslog aggregation:**

In `/etc/systemd/journald.conf`:
```ini
[Journal]
ForwardToSyslog=yes
```

Then use rsyslog to forward to a central syslog server. This is simpler and more widely supported than native journal remote.

**The firewall port:**
```bash
sudo firewall-cmd --permanent --add-port=19532/tcp
sudo firewall-cmd --reload
```""",
    "flm_logging"
))

# ---------------------------------------------------------------------------
# 19. lspci, lsusb, dmidecode — identify hardware (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I identify my hardware on Linux?",
    r"""Several tools for hardware identification:

**PCI devices (GPUs, network cards, storage controllers):**
```bash
lspci                        # brief listing
lspci -v                     # verbose — shows drivers, capabilities
lspci -vv                    # very verbose
lspci -nn                    # show vendor:device IDs (useful for driver searches)
lspci -k                     # show kernel drivers in use
lspci | grep -i vga          # find GPU
lspci | grep -i net          # find network cards
```

**USB devices:**
```bash
lsusb                        # brief listing
lsusb -v                     # verbose
lsusb -t                     # tree view showing hub topology
```

**System/motherboard info (needs root):**
```bash
sudo dmidecode                # full SMBIOS/DMI dump
sudo dmidecode -t system      # system manufacturer, model, serial
sudo dmidecode -t baseboard   # motherboard info
sudo dmidecode -t memory      # RAM details (type, speed, slots)
sudo dmidecode -t processor   # CPU socket info
sudo dmidecode -t bios        # BIOS/UEFI version
```

**CPU information:**
```bash
lscpu                        # architecture, cores, threads, caches
cat /proc/cpuinfo            # detailed per-core info
nproc                        # just the number of available cores
```

**Memory:**
```bash
free -h                      # total/used/available
sudo dmidecode -t memory     # physical DIMM details
cat /proc/meminfo            # detailed kernel memory stats
```

**Block devices (disks, partitions):**
```bash
lsblk                        # tree view of disks and partitions
lsblk -f                     # include filesystem type and UUIDs
sudo fdisk -l                # partition table details
sudo blkid                   # filesystem UUIDs and types
```

**All hardware in one command:**
```bash
sudo apt install lshw
sudo lshw -short             # brief summary of all hardware
sudo lshw -html > hw.html    # full report as HTML
sudo lshw -class network     # just network hardware
sudo lshw -class disk        # just disks
```

**Network interfaces:**
```bash
ip link show                 # interface list with MAC addresses
ethtool eth0                 # speed, duplex, driver info
```

**Kernel's hardware view:**
```bash
ls /sys/class/                # sysfs classes (net, block, input, etc.)
cat /sys/class/dmi/id/board_name      # motherboard model
cat /sys/class/dmi/id/product_name    # system model
```""",
    "flm_hardware"
))

# ---------------------------------------------------------------------------
# 20. GRUB configuration and recovery (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I configure GRUB and use it for recovery?",
    r"""GRUB is the bootloader on Debian. Its configuration lives in two places:

**Main config (auto-generated — DO NOT edit directly):**
```
/boot/grub/grub.cfg
```

**Your settings (edit this):**
```
/etc/default/grub
```

**Common `/etc/default/grub` tweaks:**
```bash
# Show the boot menu (default hides it)
GRUB_TIMEOUT_STYLE=menu
GRUB_TIMEOUT=5

# Default boot entry (0 = first, "saved" = last used)
GRUB_DEFAULT=saved
GRUB_SAVEDEFAULT=true

# Kernel command line parameters
GRUB_CMDLINE_LINUX_DEFAULT="quiet"
GRUB_CMDLINE_LINUX=""

# Text mode console (useful for servers)
GRUB_TERMINAL=console

# Higher resolution
GRUB_GFXMODE=1920x1080
```

**After editing, regenerate the config:**
```bash
sudo update-grub
```

This is a wrapper around `grub-mkconfig -o /boot/grub/grub.cfg`.

**Recovery from GRUB menu:**

1. At boot, hold **Shift** (BIOS) or press **Esc** (UEFI) to show the GRUB menu.
2. Select "Advanced options for Debian" and choose the recovery mode entry.
3. Or edit a boot entry: press **e**, then modify the kernel line.

**Boot to single-user/rescue mode** from the GRUB editor:
- Find the line starting with `linux /vmlinuz-...`
- Add `single` or `init=/bin/bash` at the end
- Press **Ctrl+X** or **F10** to boot

**If GRUB is broken — reinstall from a live USB:**
```bash
# Boot a live Debian USB, then:
sudo mount /dev/sda2 /mnt             # your root partition
sudo mount /dev/sda1 /mnt/boot/efi    # EFI partition (UEFI systems)
sudo mount --bind /dev /mnt/dev
sudo mount --bind /proc /mnt/proc
sudo mount --bind /sys /mnt/sys

sudo chroot /mnt
grub-install /dev/sda                  # BIOS systems
# or
grub-install --target=x86_64-efi --efi-directory=/boot/efi  # UEFI
update-grub
exit
sudo umount -R /mnt
```

**List installed kernels:**
```bash
dpkg --list | grep linux-image
```

**Remove old kernels** (frees /boot space):
```bash
sudo apt autoremove --purge
```""",
    "flm_boot"
))

# ---------------------------------------------------------------------------
# 21. initramfs — update, troubleshoot (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "What is initramfs and how do I update or troubleshoot it?",
    r"""The initramfs (initial RAM filesystem) is a small filesystem loaded into RAM during boot, before the real root filesystem is mounted. It contains kernel modules, scripts, and tools needed to find and mount the root filesystem.

**Why it's needed:**
- Load storage drivers (RAID, LVM, NVMe, SCSI) to access the root partition
- Unlock encrypted root partitions (LUKS)
- Assemble RAID arrays
- Handle complex root setups (NFS root, ZFS, btrfs)

**Update initramfs** (do this after changing module configs, crypttab, fstab, etc.):
```bash
# Regenerate for the current kernel
sudo update-initramfs -u

# Regenerate for a specific kernel
sudo update-initramfs -u -k 6.12.6-amd64

# Regenerate for ALL installed kernels
sudo update-initramfs -u -k all
```

**Configuration — what gets included:**

`/etc/initramfs-tools/initramfs.conf`:
```
# MODULES: most, dep, netboot, list
# "most" includes most drivers (safe default)
# "dep" includes only detected hardware (smaller but may miss things)
MODULES=most

# Compression
COMPRESS=zstd
```

**Force-include specific modules:**
```bash
echo "dm-crypt" >> /etc/initramfs-tools/modules
echo "vfat" >> /etc/initramfs-tools/modules
sudo update-initramfs -u
```

**Add custom scripts or files to initramfs:**
Put hooks in `/etc/initramfs-tools/hooks/` and boot scripts in `/etc/initramfs-tools/scripts/`.

**Inspect the contents of an initramfs:**
```bash
lsinitramfs /boot/initrd.img-$(uname -r)

# Or extract it:
mkdir /tmp/initrd
cd /tmp/initrd
unmkinitramfs /boot/initrd.img-$(uname -r) .
ls -la
```

**Troubleshoot boot failures:**

If the system drops to an initramfs shell (`(initramfs)` prompt), common causes:
```
# Check if the root device exists
ls /dev/sd*
ls /dev/mapper/*
cat /proc/partitions

# Check what the kernel was told
cat /proc/cmdline

# Try to mount root manually
mount /dev/sda2 /root

# Check LVM
lvm vgchange -ay

# Check RAID
mdadm --assemble --scan

# Exit to continue boot
exit
```

**If /boot is full** and update-initramfs fails:
```bash
# Check space
df -h /boot

# Remove old kernels
sudo apt autoremove --purge
# or manually
sudo dpkg --purge linux-image-6.1.0-old-amd64
```""",
    "flm_boot"
))

# ---------------------------------------------------------------------------
# 22. UEFI vs BIOS boot (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "What's the difference between UEFI and BIOS boot?",
    r"""BIOS (Basic Input/Output System) and UEFI (Unified Extensible Firmware Interface) are both firmware interfaces between the hardware and the operating system, but they work very differently.

**BIOS (legacy):**
- Uses the **MBR (Master Boot Record)** partitioning scheme
- MBR is stored in the first 512 bytes of the disk
- Maximum disk size: **2 TB** (MBR limitation)
- Maximum **4 primary partitions** (or 3 primary + 1 extended with logical partitions)
- Boot process: firmware reads MBR -> MBR loads bootloader (GRUB stage 1) -> bootloader loads kernel
- 16-bit real mode execution
- No built-in boot manager — relies entirely on the bootloader
- Simple, well-understood, decades old

**UEFI:**
- Uses **GPT (GUID Partition Table)**
- No practical disk size limit (9.4 ZB theoretical max)
- Up to **128 partitions** (no extended partition hack needed)
- Requires an **EFI System Partition (ESP)** — a small FAT32 partition (typically 512 MB) at `/boot/efi`
- Boot process: firmware reads ESP -> loads EFI bootloader directly (e.g., `grubx64.efi`) -> bootloader loads kernel
- 32/64-bit execution environment
- Built-in boot manager with boot entry list
- Supports **Secure Boot** — verifies bootloader signatures

**Check which you're using:**
```bash
# If this directory exists, you're booted in UEFI mode
ls /sys/firmware/efi

# Or check:
[ -d /sys/firmware/efi ] && echo "UEFI" || echo "BIOS"
```

**Partitioning differences:**

BIOS + MBR:
```
/dev/sda1  /boot    ext4    512M
/dev/sda2  /        ext4    rest
```

UEFI + GPT:
```
/dev/sda1  /boot/efi  vfat(ESP)  512M   # EFI System Partition
/dev/sda2  /boot      ext4       512M   # optional separate /boot
/dev/sda3  /          ext4       rest
```

**Managing UEFI boot entries:**
```bash
# List boot entries
efibootmgr -v

# Change boot order
sudo efibootmgr -o 0001,0002,0003

# Add a new boot entry
sudo efibootmgr -c -d /dev/sda -p 1 -L "Debian" -l "\\EFI\\debian\\grubx64.efi"

# Delete a boot entry
sudo efibootmgr -b 0004 -B
```

**Secure Boot:** UEFI can verify cryptographic signatures on bootloaders and kernels. Debian supports Secure Boot out of the box via a signed shim bootloader. Some third-party kernel modules (like NVIDIA drivers) may need manual signing with `mokutil`.""",
    "flm_boot"
))

# ---------------------------------------------------------------------------
# 23. Kernel module management — modprobe, lsmod, blacklist (Arch)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_ARCH,
    "How do I manage kernel modules? Load, unload, blacklist?",
    r"""Kernel modules are loadable pieces of kernel code — drivers, filesystems, network protocols, etc.

**List loaded modules:**
```bash
lsmod
lsmod | grep snd         # find sound-related modules
```

**Info about a module:**
```bash
modinfo iwlwifi           # show description, parameters, depends
modinfo -p iwlwifi        # just the parameters
```

**Load a module:**
```bash
sudo modprobe iwlwifi            # load with dependency resolution
sudo modprobe vfat               # load FAT filesystem support
sudo modprobe bonding mode=1     # load with parameters
```

**Unload a module:**
```bash
sudo modprobe -r iwlwifi         # unload (also removes unused dependencies)
sudo rmmod iwlwifi               # low-level unload (no dependency handling)
```

You can't unload a module that's in use. Check with:
```bash
lsmod | grep iwlwifi
# The "Used by" column shows dependencies
```

**Load modules at boot:**

Create `/etc/modules-load.d/mymodules.conf`:
```
# Load these modules at boot
vhost_net
bonding
```

**Set module parameters at boot:**

Create `/etc/modprobe.d/myparams.conf`:
```
options bonding mode=1 miimon=100
options snd-hda-intel power_save=1
```

**Blacklist a module** (prevent it from loading):

Create `/etc/modprobe.d/blacklist-mymodule.conf`:
```
# Prevent nouveau from loading (e.g., when using NVIDIA proprietary)
blacklist nouveau
# Also prevent it from loading as a dependency
install nouveau /bin/false
```

After blacklisting, rebuild initramfs:
```bash
sudo mkinitcpio -P
```

On Arch, `mkinitcpio` handles the initramfs. The `-P` flag regenerates all presets.

**Check if a module is blacklisted:**
```bash
modprobe --showconfig | grep blacklist | grep nouveau
```

**Find which module a device uses:**
```bash
lspci -k                         # shows kernel driver for each PCI device
lsusb -t                         # shows USB device tree with drivers

# Or find the module for a specific device
lspci -k -s 00:02.0              # specific PCI slot
```

**Module location on disk:**
```bash
find /lib/modules/$(uname -r) -name "*.ko*" | head -20
modinfo -F filename iwlwifi
```""",
    "flm_kernel_modules"
))

# ---------------------------------------------------------------------------
# 24. dmesg and hardware troubleshooting (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I use dmesg for hardware troubleshooting?",
    r"""dmesg shows kernel ring buffer messages — hardware detection, driver loading, errors, and warnings.

**Basic usage:**
```bash
dmesg                         # all kernel messages
dmesg | tail -50              # recent messages
sudo dmesg -w                 # follow in real-time (like tail -f)
```

**Filter by level:**
```bash
dmesg -l err                  # errors only
dmesg -l warn                 # warnings only
dmesg -l err,warn             # errors and warnings
dmesg -l emerg,alert,crit,err # all serious messages
```

Levels: emerg, alert, crit, err, warn, notice, info, debug.

**Human-readable timestamps:**
```bash
dmesg -T                      # human-readable timestamps
dmesg -T -l err               # errors with readable times
```

Note: `-T` timestamps can drift on long-running systems. For precise timing, use the default monotonic timestamps.

**Filter by facility:**
```bash
dmesg -f kern                 # kernel messages only
dmesg -f user                 # user-space messages
dmesg -f daemon               # daemon messages
```

**Common troubleshooting patterns:**

"Is my hardware detected?"
```bash
dmesg | grep -i usb           # USB device detection
dmesg | grep -i eth           # network interfaces
dmesg | grep -i sda           # SATA/SCSI disks
dmesg | grep -i nvme          # NVMe drives
dmesg | grep -i gpu           # GPU (also try "drm" or "amdgpu" or "i915")
```

"Did a device just get plugged in?"
```bash
sudo dmesg -w                 # watch in real-time, plug in the device
```

"Are there any hardware errors?"
```bash
dmesg -l err,warn -T | grep -iE "error|fail|fault|timeout|reset"
```

"Disk problems?"
```bash
dmesg | grep -iE "ata|sata|scsi|i/o error|sector|read error"
```

"Memory issues?"
```bash
dmesg | grep -iE "oom|out of memory|memory"
```

**Clear the ring buffer** (useful to see only new messages after a test):
```bash
sudo dmesg -c > /tmp/old_dmesg.log   # clear and save old messages
# ... do your test ...
dmesg                                # only new messages
```

**Persistent kernel logs via journalctl:**
```bash
journalctl -k                 # kernel log (persistent across reboots)
journalctl -k -b -1           # kernel log from previous boot
journalctl -k -p err          # kernel errors with journalctl filtering
```

Using `journalctl -k` is often better than dmesg because it survives reboots (if persistent journal is enabled) and supports richer filtering.""",
    "flm_hardware"
))

# ---------------------------------------------------------------------------
# 25. TLP for laptop power management (Arch)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_ARCH,
    "How do I set up TLP for laptop power management?",
    r"""TLP is an advanced power management tool for Linux laptops. It applies optimized power settings automatically based on whether you're on AC or battery.

**Install:**
```bash
sudo pacman -S tlp tlp-rdw
# tlp-rdw handles radio device (WiFi/Bluetooth) switching
```

**Enable and start:**
```bash
sudo systemctl enable --now tlp
sudo systemctl enable --now NetworkManager-dispatcher  # needed for tlp-rdw

# Disable conflicting services
sudo systemctl mask systemd-rfkill.service systemd-rfkill.socket
```

**Check current status:**
```bash
sudo tlp-stat -s    # system info and TLP status
sudo tlp-stat -b    # battery info (thresholds, charge level, health)
sudo tlp-stat -t    # temperatures
sudo tlp-stat       # full status report
```

**Configuration** — edit `/etc/tlp.conf`:

Key settings:
```bash
# CPU governor
CPU_SCALING_GOVERNOR_ON_AC=performance
CPU_SCALING_GOVERNOR_ON_BAT=powersave

# CPU frequency limits (optional)
CPU_SCALING_MIN_FREQ_ON_BAT=800000
CPU_SCALING_MAX_FREQ_ON_BAT=2000000

# CPU turbo boost
CPU_BOOST_ON_AC=1
CPU_BOOST_ON_BAT=0

# Disk I/O scheduler
DISK_IOSCHED="mq-deadline"

# SATA link power management
SATA_LINKPWR_ON_AC="med_power_with_dipm"
SATA_LINKPWR_ON_BAT="med_power_with_dipm"

# WiFi power saving
WIFI_PWR_ON_AC=off
WIFI_PWR_ON_BAT=on

# USB autosuspend
USB_AUTOSUSPEND=1

# Battery charge thresholds (ThinkPads with tp-smapi or acpi_call)
START_CHARGE_THRESH_BAT0=75
STOP_CHARGE_THRESH_BAT0=80
```

**Apply changes without reboot:**
```bash
sudo tlp start
```

**Manually switch power profiles:**
```bash
sudo tlp bat      # apply battery settings (even on AC)
sudo tlp ac       # apply AC settings (even on battery)
```

**For ThinkPads** — install additional packages:
```bash
sudo pacman -S acpi_call     # for newer ThinkPads (T480+, X1 Carbon 6th+)
# or
sudo pacman -S tp_smapi      # for older ThinkPads
```

These enable battery charge threshold control, which extends battery lifespan by not charging to 100%.

**Note:** Don't use TLP together with `power-profiles-daemon` (GNOME's default) — they conflict. Disable one or the other.""",
    "flm_power"
))

# ---------------------------------------------------------------------------
# 26. Suspend/hibernate configuration (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I configure suspend and hibernate on Debian?",
    r"""**Suspend (sleep) — saves state to RAM, very fast resume:**
```bash
systemctl suspend
```

This usually works out of the box.

**Hibernate — saves state to swap, survives power loss:**
```bash
systemctl hibernate
```

**Requirements for hibernate:**
1. Swap partition or swap file at least as large as your RAM
2. Kernel must know where to resume from

**Set up hibernate with a swap partition:**

Add the resume parameter to GRUB. Edit `/etc/default/grub`:
```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet resume=UUID=your-swap-uuid"
```

Find your swap UUID:
```bash
blkid | grep swap
```

Update GRUB and initramfs:
```bash
sudo update-grub
echo "RESUME=UUID=your-swap-uuid" | sudo tee /etc/initramfs-tools/conf.d/resume
sudo update-initramfs -u
```

**Set up hibernate with a swap file:**
```bash
# Create a swap file (must be at least your RAM size)
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Add to /etc/fstab
echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab

# Find the physical offset (needed for resume)
sudo filefrag -v /swapfile | head -5
# Note the first physical offset number

# Add to GRUB
# GRUB_CMDLINE_LINUX_DEFAULT="quiet resume=UUID=root-fs-uuid resume_offset=YOUR_OFFSET"
sudo update-grub
sudo update-initramfs -u
```

**Hybrid sleep** (suspend + write to swap as backup):
```bash
systemctl hybrid-sleep
```

**Configure lid close behavior** — edit `/etc/systemd/logind.conf`:
```ini
[Login]
HandleLidSwitch=suspend
HandleLidSwitchExternalPower=ignore
HandleLidSwitchDocked=ignore
HandlePowerKey=poweroff
HandleSuspendKey=suspend
HandleHibernateKey=hibernate
IdleAction=suspend
IdleActionSec=30min
```

Restart logind:
```bash
sudo systemctl restart systemd-logind
```

**Troubleshoot suspend/resume issues:**
```bash
# Check what happened during last suspend/resume
journalctl -b -u systemd-suspend
journalctl -b | grep -i "suspend\|resume\|sleep"

# Test suspend from console
sudo pm-suspend      # needs pm-utils
# or
echo mem | sudo tee /sys/power/state

# List devices that can wake the system
cat /proc/acpi/wakeup
```""",
    "flm_power"
))

# ---------------------------------------------------------------------------
# 27. powertop usage (Ubuntu 24.04)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_UBU24,
    "How do I use powertop to analyze and reduce power consumption?",
    r"""powertop is an Intel tool for analyzing power consumption and applying power-saving tunings.

**Install and run:**
```bash
sudo apt install powertop
sudo powertop
```

The interactive UI has several tabs (switch with Tab key):

1. **Overview** — top power consumers sorted by wakeups per second
2. **Idle stats** — CPU C-state residency (deeper = better power savings)
3. **Frequency stats** — CPU frequency distribution
4. **Device stats** — power usage by hardware device
5. **Tunables** — specific tweaks you can enable/disable

**Calibrate first** (for accurate power estimates, needs to run on battery):
```bash
sudo powertop --calibrate
```

This cycles through brightness levels and device states. Takes a few minutes. Only works on battery.

**Apply all recommended tunings:**
```bash
sudo powertop --auto-tune
```

This sets everything in the "Tunables" tab to "Good." Be careful — some tunings can cause issues:
- USB autosuspend may disconnect mice/keyboards briefly
- SATA power management may increase latency
- Audio codec power saving may cause pops/clicks

**Make tunings persistent** — create a systemd service:

Create `/etc/systemd/system/powertop.service`:
```ini
[Unit]
Description=PowerTOP auto-tune
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/sbin/powertop --auto-tune
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
```

Enable it:
```bash
sudo systemctl enable --now powertop
```

**Generate an HTML report:**
```bash
sudo powertop --html=power-report.html
```

Open in a browser — it contains detailed analysis, graphs, and recommendations.

**Specific tunings you might want individually** (instead of auto-tune):
```bash
# Enable SATA link power management
echo med_power_with_dipm | sudo tee /sys/class/scsi_host/host*/link_power_management_policy

# Enable audio codec power saving
echo 1 | sudo tee /sys/module/snd_hda_intel/parameters/power_save

# Enable WiFi power saving
sudo iw dev wlan0 set power_save on

# Enable USB autosuspend
echo auto | sudo tee /sys/bus/usb/devices/*/power/control
```

**Note:** On desktops, powertop is less useful since you're on AC power. It's primarily designed for laptops where battery life matters.""",
    "flm_power"
))

# ---------------------------------------------------------------------------
# 28. Build a simple .deb package (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I build a simple .deb package from scratch?",
    r"""Here's a minimal example — packaging a custom script called `hello-world`.

**1. Create the directory structure:**
```bash
mkdir -p hello-world-1.0/DEBIAN
mkdir -p hello-world-1.0/usr/local/bin
```

**2. Create your script:**
```bash
cat > hello-world-1.0/usr/local/bin/hello-world << 'SCRIPT'
#!/bin/bash
echo "Hello from hello-world 1.0!"
SCRIPT
chmod 755 hello-world-1.0/usr/local/bin/hello-world
```

**3. Create the control file** (`hello-world-1.0/DEBIAN/control`):
```
Package: hello-world
Version: 1.0
Section: utils
Priority: optional
Architecture: all
Maintainer: Your Name <you@example.com>
Description: A simple hello world program
 This is a longer description.
 Each continuation line starts with a space.
Depends: bash
```

**4. Build the package:**
```bash
dpkg-deb --build hello-world-1.0
```

This creates `hello-world-1.0.deb`.

**5. Install and test:**
```bash
sudo dpkg -i hello-world-1.0.deb
hello-world
# Output: Hello from hello-world 1.0!
```

**Optional: add maintainer scripts:**

Pre/post install and remove scripts go in `DEBIAN/`:
```bash
# DEBIAN/postinst — runs after installation
#!/bin/bash
echo "hello-world installed successfully"
chmod 755 hello-world-1.0/DEBIAN/postinst

# DEBIAN/prerm — runs before removal
#!/bin/bash
echo "Removing hello-world..."
chmod 755 hello-world-1.0/DEBIAN/prerm
```

**Optional: conffiles** — list config files that shouldn't be overwritten on upgrade:
```bash
echo "/etc/hello-world.conf" > hello-world-1.0/DEBIAN/conffiles
```

**Better approach for real packages — use debhelper:**
```bash
sudo apt install build-essential devscripts debhelper
mkdir hello-world-1.0
cd hello-world-1.0
dh_make --createorig -s -p hello-world_1.0
```

This creates the full `debian/` directory with rules, changelog, etc. Then build with:
```bash
dpkg-buildpackage -us -uc -b
```

The `-us -uc` flags skip signing. The `.deb` appears in the parent directory.

**Inspect a .deb file:**
```bash
dpkg-deb -c hello-world-1.0.deb   # list contents
dpkg-deb -I hello-world-1.0.deb   # show control info
```""",
    "flm_packaging"
))

# ---------------------------------------------------------------------------
# 29. Using Debian backports (Debian 12)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB12,
    "How do I use backports on Debian 12 to get newer packages?",
    r"""Debian Backports provides newer versions of packages from testing/unstable, recompiled for the stable release.

**Enable backports:**

Add the backports repository:
```bash
echo "deb http://deb.debian.org/debian bookworm-backports main contrib non-free non-free-firmware" | \
    sudo tee /etc/apt/sources.list.d/backports.list
sudo apt update
```

**Install a package from backports:**
```bash
# Backports are NOT installed by default — you must explicitly request them
sudo apt install -t bookworm-backports package-name

# Examples:
sudo apt install -t bookworm-backports linux-image-amd64   # newer kernel
sudo apt install -t bookworm-backports golang-go           # newer Go
sudo apt install -t bookworm-backports firefox-esr          # newer Firefox
```

**Search for available backports:**
```bash
apt list -a package-name                          # see all available versions
apt-cache policy package-name                     # see versions and priorities
apt search -t bookworm-backports search-term      # search backports
```

**Why backports don't install by default:**

Backports have a lower APT priority (100) than stable packages (500). This means `apt upgrade` will never pull in a backport unless you've already installed it explicitly. Once installed from backports, it will be upgraded from backports on future `apt upgrade` runs.

**Pin a specific package to always use backports:**

Create `/etc/apt/preferences.d/backports`:
```
Package: golang-go
Pin: release a=bookworm-backports
Pin-Priority: 600
```

With priority 600 (higher than stable's 500), `apt upgrade` will automatically upgrade golang-go from backports.

**Important notes:**
- Backports are tested and compiled specifically for stable — they're safer than mixing in testing/unstable
- Security updates for backports come through the backports repo, not the security team
- Not every package has a backport — only those someone has chosen to maintain
- Backports may pull in newer dependencies, but these are also in the backports repo""",
    "flm_packaging"
))

# ---------------------------------------------------------------------------
# 30. Pin package versions with apt preferences (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I pin a package to a specific version with apt?",
    r"""APT pinning controls which version of a package gets installed or upgraded. It works through priority values in `/etc/apt/preferences.d/`.

**Hold a package at its current version** (simplest approach):
```bash
sudo apt-mark hold package-name
# Now apt upgrade won't touch it

# To unhold:
sudo apt-mark unhold package-name

# List held packages:
apt-mark showhold
```

**Pin to a specific version** — create `/etc/apt/preferences.d/pin-nginx`:
```
Package: nginx
Pin: version 1.24.0-2
Pin-Priority: 1001
```

Priority meanings:
- **1001+** — install even if it's a downgrade
- **990** — install unless target release is set
- **500** — default for stable
- **100** — default for backports
- **-1** — never install

**Pin a package to a specific release:**
```
Package: firefox-esr
Pin: release a=trixie-security
Pin-Priority: 900
```

**Block a package from ever being installed:**
```
Package: snapd
Pin: release *
Pin-Priority: -1
```

**Pin by origin (repository):**
```
Package: *
Pin: origin "packages.example.com"
Pin-Priority: 200
```

This gives lower priority to all packages from a third-party repo.

**Check effective pinning:**
```bash
apt-cache policy nginx
```

Output shows all available versions with their priorities:
```
nginx:
  Installed: 1.24.0-2
  Candidate: 1.24.0-2
  Version table:
 *** 1.24.0-2 1001
        500 http://deb.debian.org/debian trixie/main amd64 Packages
        100 /var/lib/dpkg/status
```

**Common use cases:**

Prevent accidental major version upgrades:
```
Package: postgresql-16
Pin: version 16.*
Pin-Priority: 900
```

Prefer security repo over main:
```
Package: *
Pin: release a=trixie-security
Pin-Priority: 600
```

**Simulate to verify pinning works:**
```bash
apt-cache policy package-name
sudo apt install --simulate package-name
```""",
    "flm_packaging"
))

# ---------------------------------------------------------------------------
# 31. Add a third-party repository safely (Ubuntu 24.04)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_UBU24,
    "What's the safe way to add a third-party apt repository?",
    r"""The old `apt-key add` method is deprecated. Here's the modern approach using signed-by.

**Step 1: Download the GPG key and store it properly:**
```bash
# Download and dearmor the key (convert from ASCII to binary)
curl -fsSL https://example.com/repo-key.gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/example.gpg
```

If the key is already in binary format:
```bash
curl -fsSL https://example.com/repo-key.gpg | \
    sudo tee /etc/apt/keyrings/example.gpg > /dev/null
```

**Step 2: Add the repository with signed-by:**
```bash
echo "deb [signed-by=/etc/apt/keyrings/example.gpg arch=amd64] https://repo.example.com/apt stable main" | \
    sudo tee /etc/apt/sources.list.d/example.list
```

The `signed-by=` binds this key to only this repository — it can't be used to sign packages from other repos. Much safer than the old global keyring approach.

**Step 3: Update and install:**
```bash
sudo apt update
sudo apt install example-package
```

**Real-world example — Docker:**
```bash
# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add the repository
echo "deb [signed-by=/etc/apt/keyrings/docker.gpg arch=amd64] https://download.docker.com/linux/ubuntu noble stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list

sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io
```

**Pin third-party repos to lower priority** (prevent them from overriding system packages):

Create `/etc/apt/preferences.d/example`:
```
Package: *
Pin: origin "repo.example.com"
Pin-Priority: 200
```

This ensures system packages from Ubuntu repos are preferred over third-party ones, unless you explicitly request a package from the third-party repo.

**DEB822 format** (newer alternative to one-line format):

Create `/etc/apt/sources.list.d/example.sources`:
```
Types: deb
URIs: https://repo.example.com/apt
Suites: stable
Components: main
Signed-By: /etc/apt/keyrings/example.gpg
Architectures: amd64
```

**Remove a third-party repo:**
```bash
sudo rm /etc/apt/sources.list.d/example.list
sudo rm /etc/apt/keyrings/example.gpg
sudo apt update
```""",
    "flm_packaging"
))

# ---------------------------------------------------------------------------
# 32. update-alternatives explained (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How does update-alternatives work?",
    r"""update-alternatives manages symbolic links for commands that have multiple implementations. For example, `editor` can be vim, nano, or emacs; `python3` can be 3.11 or 3.12.

**List all registered alternatives:**
```bash
update-alternatives --get-selections
```

**See options for a specific command:**
```bash
update-alternatives --list editor
# /usr/bin/nano
# /usr/bin/vim.basic
# /usr/bin/vim.tiny
```

**Detailed info with priorities:**
```bash
update-alternatives --display editor
```

**Change the default interactively:**
```bash
sudo update-alternatives --config editor
```

This shows a numbered menu — pick the one you want.

**Set it non-interactively:**
```bash
sudo update-alternatives --set editor /usr/bin/vim.basic
```

**Register a new alternative:**
```bash
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 120
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 110
```

Format: `--install <link> <name> <path> <priority>`
- **link** — the symlink path (what users run)
- **name** — the group name
- **path** — the actual binary
- **priority** — higher number = preferred in auto mode

**Auto vs. manual mode:**
- **Auto mode** — the highest-priority alternative is used. Installing a new one with higher priority switches automatically.
- **Manual mode** — your explicit choice is preserved, even if a higher-priority alternative is installed.

```bash
# Switch to auto mode
sudo update-alternatives --auto editor
```

**Slave alternatives** — linked alternatives that change together:
```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 120 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-12 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-12

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-13 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-13
```

Now switching `gcc` also switches `g++` and `gcov` together.

**Remove an alternative:**
```bash
sudo update-alternatives --remove python3 /usr/bin/python3.11
```

**Common alternatives on Debian:**
- `editor` — default text editor
- `x-www-browser` — default web browser
- `x-terminal-emulator` — default terminal
- `awk` — gawk vs. mawk
- `java` / `javac` — JDK versions""",
    "flm_packaging"
))

# ---------------------------------------------------------------------------
# 33. System won't boot after kernel update (multi-turn, Debian)
# ---------------------------------------------------------------------------
conversations.append(make_multi(
    SYS_DEB13,
    [
        ("user", "My Debian server won't boot after I ran apt upgrade. It gets stuck at a black screen with a blinking cursor."),
        ("assistant", r"""Sounds like a kernel or initramfs issue. The new kernel likely can't find or mount the root filesystem.

Can you access the GRUB boot menu? Restart the machine and hold **Shift** (BIOS) or press **Esc** (UEFI) right after the firmware splash screen. You should see a GRUB menu with kernel entries.

If you see GRUB, try:
1. Select "Advanced options for Debian"
2. Choose the **previous kernel** (not the newest one)
3. Boot with that

Does it boot with the old kernel?"""),
        ("user", "Yes, I can get to GRUB and the old kernel boots fine. But I need the new kernel to work."),
        ("assistant", r"""Good — the system itself is fine, it's just the new kernel having issues. Boot with the old kernel and let's investigate.

Check what went wrong with the new kernel's initramfs:
```bash
# List installed kernels
dpkg --list | grep linux-image

# Check if initramfs exists for the new kernel
ls -la /boot/initrd.img-*

# Check if /boot has space (a full /boot is the most common cause)
df -h /boot
```

The most common reasons for this:
1. **/boot was full** — the new initramfs didn't get generated properly
2. **Missing firmware or modules** in the initramfs
3. **Changed storage driver** that's not included in the initramfs

What does `df -h /boot` show?"""),
        ("user", "Oh, /boot is 100% full. Only 0 bytes available."),
        ("assistant", r"""That's the problem. The initramfs for the new kernel couldn't be generated because /boot ran out of space. This is extremely common on systems with a small /boot partition.

**Fix it:**

1. Remove old kernels to free space:
```bash
# See what's taking space
du -sh /boot/*

# List installed kernels
dpkg --list | grep linux-image

# Remove old kernels (keep at least the current running one and the new one)
sudo apt purge linux-image-6.12.5-amd64   # example old version
```

Or let apt do it:
```bash
sudo apt autoremove --purge
```

2. After freeing space, regenerate the initramfs for the new kernel:
```bash
sudo update-initramfs -u -k all
```

3. Update GRUB:
```bash
sudo update-grub
```

4. Verify space:
```bash
df -h /boot
```

5. Reboot and select the new kernel.

**Prevent this in the future:**

Add to `/etc/apt/apt.conf.d/01autoremove-kernels`:
```
Unattended-Upgrade::Remove-Unused-Kernel-Packages "true";
```

Or periodically run `sudo apt autoremove` to clean up old kernels. On Debian, the `linux-image-amd64` metapackage depends on the latest kernel, so `autoremove` knows which old ones are safe to delete.

If your /boot partition is consistently too small (under 500 MB), consider resizing it when you next have the opportunity."""),
    ],
    "flm_boot_troubleshoot"
))

# ---------------------------------------------------------------------------
# 34. Service keeps crashing and restarting (multi-turn, Fedora)
# ---------------------------------------------------------------------------
conversations.append(make_multi(
    SYS_FED41,
    [
        ("user", "My web application service keeps crashing and restarting every few minutes. How do I figure out what's wrong?"),
        ("assistant", r"""Let's look at the service status and journal. What's the service name? Run these:

```bash
sudo systemctl status your-service-name
sudo journalctl -u your-service-name --since "30 min ago" --no-pager
```

The status will tell us if it's in a crash loop (look for "start-limit-hit" or rapid active/inactive cycling). The journal will show the actual error output from the application.

What's the service called?"""),
        ("user", "It's called webapp. Status shows it restarted 12 times in the last hour. Journal shows \"Killed\" and then \"Main process exited, code=killed, signal=KILL\"."),
        ("assistant", r"""Signal KILL (SIGKILL, signal 9) means the process was forcefully killed — it didn't crash on its own. On a systemd-managed system, this is almost always the **OOM killer** — the system ran out of memory and the kernel killed your process.

Confirm it:
```bash
# Check for OOM events in kernel log
journalctl -k | grep -i "oom\|out of memory\|killed process"

# Check if systemd's memory limit is triggering it
journalctl -u webapp | grep -i "memory"

# Check the service's memory cgroup
systemctl show webapp -p MemoryCurrent,MemoryMax
```

Also check system memory:
```bash
free -h
```

If it's the OOM killer, the kernel log will show something like:
```
Out of memory: Killed process 1234 (node) total-vm:2048000kB, anon-rss:1800000kB
```

What does the kernel log say?"""),
        ("user", "Yeah, journalctl -k shows \"Out of memory: Killed process\" for the webapp process. MemoryMax shows 512M and MemoryCurrent was 511M right before the kill."),
        ("assistant", r"""There it is. The service has `MemoryMax=512M` set, and your app is hitting that ceiling. When it reaches 512 MB, systemd's cgroup OOM killer sends SIGKILL.

You have two options:

**Option 1: Increase the memory limit** if the server has enough RAM:
```bash
sudo systemctl edit webapp
```

Add:
```ini
[Service]
MemoryMax=1G
MemoryHigh=800M
```

`MemoryHigh` is a soft limit — the kernel will throttle the process (slow it down) when it hits 800 MB, giving it a chance to free memory before hitting the hard `MemoryMax` limit.

```bash
sudo systemctl restart webapp
```

**Option 2: Fix the memory leak in your application.**

Monitor memory growth over time:
```bash
# Watch memory usage
watch -n 5 'systemctl show webapp -p MemoryCurrent'
```

If memory grows continuously without leveling off, you have a memory leak. Common causes:
- Unbounded caches or session stores
- Event listeners not being cleaned up
- Large objects held in closures

For a Node.js app, you can get a heap snapshot:
```bash
kill -USR2 $(systemctl show webapp -p MainPID --value)
```

**Option 3: Allow graceful restart when memory is high** instead of SIGKILL:

Add to the service:
```ini
[Service]
OOMPolicy=stop
```

With `OOMPolicy=stop`, systemd stops the unit cleanly instead of letting the kernel SIGKILL it. Combined with `Restart=on-failure`, this gives the app a chance to clean up.

For now, I'd increase the limit and then investigate the root cause of the high memory usage."""),
    ],
    "flm_systemd_troubleshoot"
))

# ---------------------------------------------------------------------------
# 35. High CPU usage investigation (multi-turn, Debian)
# ---------------------------------------------------------------------------
conversations.append(make_multi(
    SYS_DEB13,
    [
        ("user", "My server is extremely slow and the load average is way too high. How do I figure out what's causing it?"),
        ("assistant", r"""Let's identify the culprit. Run:

```bash
top -bn1 | head -20
```

This gives us a snapshot of the top CPU consumers. Also:

```bash
uptime
nproc
```

This tells us the load average and how many cores you have, so we can judge how overloaded you actually are. What do you see?"""),
        ("user", "Load average is 24.5, 22.1, 18.3 and I have 4 cores. Top shows about 40 php-fpm processes, each using 2-5% CPU."),
        ("assistant", r"""Load 24 on 4 cores means roughly 20 processes are queued up waiting — the system is massively overloaded.

40 php-fpm processes each using a few percent adds up fast. This is a classic "too many workers" problem, often triggered by slow upstream requests (slow database, slow external API) causing worker processes to pile up.

Let's dig deeper:

```bash
# How many php-fpm processes total?
ps aux | grep php-fpm | wc -l

# What are they doing? (D = waiting for I/O, R = running, S = sleeping)
ps aux | grep php-fpm | awk '{print $8}' | sort | uniq -c | sort -rn

# Check if it's actually CPU or I/O wait
vmstat 1 5
```

In the `vmstat` output, look at:
- **us** — user CPU time (actual PHP work)
- **sy** — system CPU time (kernel overhead)
- **wa** — I/O wait (processes waiting on disk)
- **id** — idle

If `wa` is high, the bottleneck is disk I/O, not CPU. What does vmstat show?"""),
        ("user", "vmstat shows wa at about 65% and us at 15%. Most php-fpm processes show state D."),
        ("assistant", r"""65% I/O wait confirms it — your CPU isn't overloaded, your **disk is the bottleneck**. The php-fpm processes are stuck in state D (uninterruptible sleep), meaning they're waiting for disk I/O to complete.

This is why load average is misleading — it counts I/O-waiting processes, not just CPU-hungry ones.

Find what's causing the I/O:

```bash
# Which processes are doing the most I/O?
sudo iotop -o -b -n 3

# If iotop isn't installed:
sudo apt install iotop-c
sudo iotop -o -b -n 3

# Check disk throughput
iostat -x 1 5
```

In `iostat`, look at:
- **%util** — if near 100%, the disk is saturated
- **r/s, w/s** — reads/writes per second
- **avgqu-sz** — queue depth (high = backed up)
- **await** — average I/O wait time in ms

Common causes of high I/O on PHP servers:
1. **Slow database queries** causing MySQL/PostgreSQL to thrash the disk
2. **Missing database indexes** causing full table scans
3. **PHP writing too many session files** or logs
4. **Disk hardware issue** or dying drive

```bash
# Check if MySQL/PostgreSQL is the I/O source
sudo iotop -o -P -b -n 3 | head -20

# Check database slow queries
sudo tail -100 /var/log/mysql/mysql-slow.log   # MySQL
# or
sudo -u postgres psql -c "SELECT pid, now()-query_start AS duration, query FROM pg_stat_activity WHERE state='active' ORDER BY duration DESC LIMIT 10;"
```

Also check disk health:
```bash
sudo smartctl -H /dev/sda
```

A dying disk dramatically increases I/O latency. What does iotop show?"""),
    ],
    "flm_troubleshoot_cpu"
))

# ---------------------------------------------------------------------------
# 36. Enable persistent systemd journal (Debian minimal)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEBMIN,
    "How do I make journald logs persist across reboots?",
    r"""By default on minimal Debian installs, the journal is stored in `/run/log/journal/` which is a tmpfs — logs are lost on reboot.

**Enable persistent logging:**
```bash
sudo mkdir -p /var/log/journal
sudo systemd-tmpfiles --create --prefix /var/log/journal
sudo systemctl restart systemd-journald
```

That's it. When `/var/log/journal/` exists, journald stores logs there automatically and they survive reboots.

**Verify:**
```bash
journalctl --list-boots
```

If you see multiple boots listed, persistence is working.

**Set storage explicitly** in `/etc/systemd/journald.conf`:
```ini
[Journal]
Storage=persistent
```

The `Storage=` options:
- **persistent** — always store in `/var/log/journal/` (creates the directory if needed)
- **volatile** — only in `/run/log/journal/` (lost on reboot)
- **auto** — persistent if `/var/log/journal/` exists, volatile otherwise (default)
- **none** — discard all logs

**Set size limits** (important on minimal installs with small disks):
```ini
[Journal]
Storage=persistent
SystemMaxUse=200M
SystemKeepFree=500M
```

Then:
```bash
sudo systemctl restart systemd-journald
```""",
    "flm_journal"
))

# ---------------------------------------------------------------------------
# 37. Find what service listens on a port (Ubuntu 22.04)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_UBU22,
    "How do I find which process is listening on a specific port?",
    r"""Several methods:

**ss (fastest, modern replacement for netstat):**
```bash
sudo ss -tlnp              # TCP listening sockets with process names
sudo ss -ulnp              # UDP listening sockets
sudo ss -tlnp | grep :80   # who's on port 80
```

Flags: `-t` TCP, `-u` UDP, `-l` listening, `-n` numeric (no DNS), `-p` show process.

**lsof:**
```bash
sudo lsof -i :80           # anything on port 80 (TCP and UDP)
sudo lsof -i TCP:443       # TCP port 443
sudo lsof -i -P -n         # all network connections, numeric
```

**fuser:**
```bash
sudo fuser 80/tcp           # PIDs using TCP port 80
sudo fuser -v 80/tcp        # verbose (shows process name)
```

**netstat (legacy, but still common):**
```bash
sudo netstat -tlnp          # TCP listening with process names
sudo netstat -tlnp | grep :8080
```

netstat is in the `net-tools` package which may not be installed by default on Ubuntu 22.04:
```bash
sudo apt install net-tools
```

**Example output from ss:**
```
State  Recv-Q  Send-Q  Local Address:Port  Peer Address:Port  Process
LISTEN 0       511     0.0.0.0:80          0.0.0.0:*          users:(("nginx",pid=1234,fd=6))
LISTEN 0       128     127.0.0.1:3306      0.0.0.0:*          users:(("mariadbd",pid=5678,fd=21))
```

This tells you nginx is listening on port 80 (all interfaces) and MariaDB on port 3306 (localhost only).

**If nothing shows up** but you know something should be listening, the service might be listening on a different interface or may have crashed:
```bash
sudo systemctl status nginx
sudo journalctl -u nginx -n 20
```""",
    "flm_networking"
))

# ---------------------------------------------------------------------------
# 38. systemd Type= explained (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SHORT_DEB13,
    "What are the different Type= options in a systemd service file?",
    r"""The `Type=` directive tells systemd how the service starts up and when to consider it "ready."

**Type=simple** (default):
```ini
[Service]
Type=simple
ExecStart=/usr/bin/myapp
```
systemd considers the service started as soon as the process is forked. Use for processes that stay in the foreground and don't fork.

**Type=exec**:
```ini
[Service]
Type=exec
ExecStart=/usr/bin/myapp
```
Like simple, but systemd waits until the binary is actually executed (the execve() call succeeds). Catches errors like missing binaries or permission problems before reporting the service as started.

**Type=forking**:
```ini
[Service]
Type=forking
PIDFile=/run/myapp.pid
ExecStart=/usr/bin/myapp --daemon
```
For traditional daemons that fork into the background. systemd waits for the initial process to exit, then tracks the forked child. Use PIDFile so systemd knows which PID to monitor.

**Type=oneshot**:
```ini
[Service]
Type=oneshot
ExecStart=/usr/local/bin/do-something.sh
RemainAfterExit=yes
```
For commands that do their job and exit. systemd waits for them to finish before starting dependent units. `RemainAfterExit=yes` keeps the service "active" after the process exits — useful for setup scripts.

**Type=notify**:
```ini
[Service]
Type=notify
ExecStart=/usr/bin/myapp
```
The service sends a readiness notification to systemd via `sd_notify()`. systemd doesn't consider it started until it receives this signal. Used by services that need initialization time (e.g., database startup).

**Type=dbus**:
```ini
[Service]
Type=dbus
BusName=org.example.MyApp
ExecStart=/usr/bin/myapp
```
Like notify, but readiness is determined by the service acquiring its D-Bus bus name.

**Type=idle**:
```ini
[Service]
Type=idle
ExecStart=/usr/bin/myapp
```
Like simple, but execution is delayed until all active jobs are finished. Rarely used — mainly for console output ordering.

**Most common choices:**
- Background daemon (stays running): `Type=simple` or `Type=exec`
- Legacy daemon that forks: `Type=forking`
- Script that runs once: `Type=oneshot`
- Service with startup time: `Type=notify`""",
    "flm_systemd"
))

# ---------------------------------------------------------------------------
# 39. Socket activation with systemd (Fedora)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_FED41,
    "What is systemd socket activation and when would I use it?",
    r"""Socket activation means systemd opens the listening socket and only starts the actual service when a connection arrives. The service inherits the socket file descriptor from systemd.

**Why use it:**
- **On-demand startup** — services that are rarely used don't consume resources until needed
- **Zero-downtime restarts** — systemd holds the socket during restart, so no connections are dropped
- **Parallelization** — services can start in parallel because their sockets exist before the services do
- **Privilege separation** — systemd (root) opens port 80, the service runs as non-root

**Example: a simple HTTP service with socket activation**

Create `/etc/systemd/system/myhttp.socket`:
```ini
[Unit]
Description=My HTTP Service Socket

[Socket]
ListenStream=8080
Accept=no
ReusePort=true

[Install]
WantedBy=sockets.target
```

Create `/etc/systemd/system/myhttp.service`:
```ini
[Unit]
Description=My HTTP Service
Requires=myhttp.socket

[Service]
Type=simple
ExecStart=/usr/bin/myhttp-server
User=www-data
NonBlocking=true

# Receives socket as fd 3
# Use sd_listen_fds() in C or socket activation libraries
```

Enable the **socket** (not the service):
```bash
sudo systemctl enable --now myhttp.socket
```

The service starts automatically on the first connection to port 8080.

**Accept=no vs Accept=yes:**
- **Accept=no** (default) — one service instance handles all connections (like nginx). The service gets a listening socket.
- **Accept=yes** — systemd spawns a new service instance per connection (like inetd). Good for simple, short-lived handlers.

**For Accept=yes**, the service file must use a template name (`myhttp@.service`):
```ini
[Service]
ExecStart=/usr/bin/handler
StandardInput=socket
```

**Check socket status:**
```bash
systemctl list-sockets                    # all active sockets
systemctl status myhttp.socket            # specific socket status
```

**Real-world examples in Fedora:**
- `sshd.socket` — start SSH on demand
- `cups.socket` — start CUPS printing on demand
- `docker.socket` — start Docker daemon on first `docker` command

```bash
# Use SSH socket activation instead of running sshd all the time
sudo systemctl disable sshd.service
sudo systemctl enable --now sshd.socket
```""",
    "flm_systemd"
))

# ---------------------------------------------------------------------------
# 40. Troubleshooting service dependency ordering (Arch)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_ARCH,
    "My service starts before the database is ready. How do I fix service ordering in systemd?",
    r"""systemd has several ordering and dependency directives. Here's how they work:

**After=/Before= (ordering only):**
```ini
[Unit]
After=postgresql.service
```
This means your service starts **after** postgresql, but only if both are being started. It does NOT pull in postgresql — if postgresql isn't enabled, your service starts without waiting for it.

**Requires= (hard dependency):**
```ini
[Unit]
Requires=postgresql.service
After=postgresql.service
```
If postgresql fails to start, your service also fails. If postgresql is stopped, your service is stopped too. **Always combine with After=** — Requires alone doesn't guarantee ordering.

**Wants= (soft dependency):**
```ini
[Unit]
Wants=postgresql.service
After=postgresql.service
```
Like Requires, but your service starts even if postgresql fails. Best for services that work better with the dependency but can function without it.

**BindsTo= (tight coupling):**
```ini
[Unit]
BindsTo=postgresql.service
After=postgresql.service
```
Like Requires, but also stops your service if postgresql is stopped or restarted.

**The "ready" problem:**

Even with correct ordering, `After=postgresql.service` only waits until systemd considers postgresql "started" — which for `Type=simple` just means the process forked. PostgreSQL might not actually be accepting connections yet.

**Solution 1: Use Type=notify** (PostgreSQL does this by default on modern distros)

Check:
```bash
systemctl show postgresql -p Type
```

If it's `Type=notify`, systemd waits for PostgreSQL to signal readiness. Your `After=` will work correctly.

**Solution 2: Add a readiness check to your service:**
```ini
[Service]
ExecStartPre=/bin/sh -c 'until pg_isready -q; do sleep 1; done'
ExecStart=/usr/bin/myapp
```

This loops until PostgreSQL is actually accepting connections before starting your app.

**Solution 3: Use systemd's built-in socket activation readiness:**

If your service connects to a socket, you can use:
```ini
[Unit]
After=postgresql.service
Requires=postgresql.service

[Service]
Restart=on-failure
RestartSec=3
```

The service restarts automatically if it crashes because the database wasn't ready.

**Debug ordering:**
```bash
systemd-analyze critical-chain myapp.service
systemd-analyze dot myapp.service | dot -Tsvg > deps.svg
```""",
    "flm_systemd"
))

# ---------------------------------------------------------------------------
# 41. Inspect what a process has open (Ubuntu 24.04)
# ---------------------------------------------------------------------------
conversations.append(make(
    SHORT_UBU24,
    "A process is using a lot of file descriptors. How do I see what it has open?",
    r"""You can inspect open file descriptors through `/proc` and `lsof`.

**Using /proc (no extra tools needed):**
```bash
# List all open file descriptors for PID 1234
ls -la /proc/1234/fd/

# Count open file descriptors
ls /proc/1234/fd/ | wc -l

# See the file descriptor limits
cat /proc/1234/limits | grep "open files"
```

The `fd/` directory shows symlinks from file descriptor numbers to the actual files, sockets, or pipes.

**Using lsof (more readable):**
```bash
sudo lsof -p 1234              # all open files for PID 1234
sudo lsof -p 1234 | wc -l     # count
sudo lsof -p 1234 | grep TCP   # just network connections
sudo lsof -p 1234 | grep REG   # just regular files
```

**Find processes with the most open files:**
```bash
# Top 10 processes by open fd count
for pid in /proc/[0-9]*/fd; do
    echo "$(ls "$pid" 2>/dev/null | wc -l) $pid"
done | sort -rn | head -10
```

Or use lsof:
```bash
sudo lsof -n | awk '{print $2}' | sort | uniq -c | sort -rn | head -10
```

**Check if a process is near its limit:**
```bash
# Current count vs limit
PID=1234
echo "Open: $(ls /proc/$PID/fd 2>/dev/null | wc -l)"
grep "open files" /proc/$PID/limits
```

**If a process is running out of file descriptors**, increase the limit:

For a systemd service:
```ini
[Service]
LimitNOFILE=65536
```

For interactive processes, set it in the shell:
```bash
ulimit -n 65536
```

**Common fd leak patterns:**
- Opening files/sockets without closing them in loops
- Forking without closing inherited fds
- Connection pools that grow without bound
- Log files opened repeatedly without closing""",
    "flm_processes"
))

# ---------------------------------------------------------------------------
# 42. Analyze boot performance with systemd-analyze (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I figure out what's making my system boot slowly?",
    r"""systemd-analyze is the built-in boot performance tool.

**Overall boot time:**
```bash
systemd-analyze
```

Output example:
```
Startup finished in 3.456s (firmware) + 1.234s (loader) + 2.345s (kernel) + 8.901s (userspace) = 15.936s
graphical.target reached after 8.500s in userspace
```

**Blame — services sorted by startup time:**
```bash
systemd-analyze blame
```

This shows which services took the longest. Focus on the top entries.

**Critical chain — the actual bottleneck path:**
```bash
systemd-analyze critical-chain
```

This shows the chain of units that determined the total boot time. Look for the longest single dependency path — that's what you need to optimize.

```bash
# Critical chain for a specific target
systemd-analyze critical-chain multi-user.target
```

**SVG plot of the entire boot:**
```bash
systemd-analyze plot > boot.svg
```

Open in a browser. This is a Gantt chart showing every service's start/stop timeline.

**Find slow generators:**
```bash
systemd-analyze generators
```

Generators run early in boot and create dynamic units. Slow generators delay everything.

**Verify unit files for errors:**
```bash
systemd-analyze verify myservice.service
```

**Common slow services and fixes:**

`NetworkManager-wait-online.service` — waits for network (often 10-30s):
```bash
# Disable if you don't need network at boot
sudo systemctl disable NetworkManager-wait-online.service
```

`apt-daily.service` — runs apt update at boot:
```bash
# Change to run later, not at boot
sudo systemctl edit apt-daily.timer
```

`plymouth` — graphical boot splash (adds overhead):
```bash
sudo apt purge plymouth
```

`fstab` entries with slow network mounts — add `nofail,x-systemd.automount` options so they don't block boot.

**General tips:**
- Services with `After=network-online.target` wait for the network, adding significant delay
- Move non-essential services to `Type=idle` or disable them at boot
- Use socket activation for services that aren't needed immediately""",
    "flm_boot"
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
