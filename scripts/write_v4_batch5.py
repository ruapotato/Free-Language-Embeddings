#!/usr/bin/env python3
"""Write hand-crafted SFT conversations for V4 training -- Batch 5.

Topics: Security & Hardening, Monitoring, Backup, Shell Scripting, Multi-turn.

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


# -- Helpers ------------------------------------------------------------------

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


# -- Conversations ------------------------------------------------------------

conversations = []

# ---------------------------------------------------------------------------
# 1. AppArmor basics and troubleshooting (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""How does AppArmor work on Debian and how do I troubleshoot it?""",
    r"""AppArmor is a mandatory access control (MAC) system that confines programs to a limited set of resources. On Debian, it's enabled by default.

**Check AppArmor status:**

```bash
sudo aa-status
```

This shows loaded profiles and which are in enforce vs. complain mode. Enforced profiles block violations; complain-mode profiles only log them.

**Key directories:**

- `/etc/apparmor.d/` -- profile definitions
- `/etc/apparmor.d/local/` -- local overrides (preferred for customization)
- `/etc/apparmor.d/tunables/` -- variables used across profiles

**Common troubleshooting workflow:**

If a program is being blocked by AppArmor, check the logs first:

```bash
sudo journalctl -k | grep apparmor
# or
sudo dmesg | grep apparmor
```

Denied operations show up as `apparmor="DENIED"` with the profile name, the operation, and the path that was blocked.

**Put a profile in complain mode** to stop blocking while you debug:

```bash
sudo aa-complain /etc/apparmor.d/usr.sbin.mysqld
```

**Generate a profile update from the logs:**

```bash
sudo aa-logprof
```

This scans the logs for denied operations and interactively asks if you want to allow each one. It updates the profile accordingly.

**Put the profile back in enforce mode:**

```bash
sudo aa-enforce /etc/apparmor.d/usr.sbin.mysqld
```

**Install the utilities if missing:**

```bash
sudo apt install apparmor-utils
```

This gives you `aa-status`, `aa-complain`, `aa-enforce`, `aa-logprof`, and `aa-genprof`.

**To create a new profile from scratch:**

```bash
sudo aa-genprof /usr/bin/myapp
```

Then run the application in another terminal, exercise its functionality, and come back to `aa-genprof` to approve the detected access patterns.

**Disable a specific profile** (without removing it):

```bash
sudo ln -s /etc/apparmor.d/usr.sbin.mysqld /etc/apparmor.d/disable/
sudo apparmor_parser -R /etc/apparmor.d/usr.sbin.mysqld
```

AppArmor is lighter than SELinux and well-integrated into Debian. Most issues come from profiles that are too restrictive for a custom setup -- `aa-logprof` is usually the fastest fix.""",
    "flm_security"
))

# ---------------------------------------------------------------------------
# 2. SELinux basics (Fedora)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_FED41,
    r"""Can you explain SELinux modes and how to troubleshoot denials on Fedora?""",
    r"""SELinux (Security-Enhanced Linux) is the MAC system used by Fedora and RHEL-family distros. It labels every process and file with a security context and enforces policies about what each context can access.

**Three modes:**

- **Enforcing** -- policy is enforced, violations are blocked and logged.
- **Permissive** -- policy is not enforced, but violations are logged. Good for debugging.
- **Disabled** -- SELinux is completely off (requires reboot to change to/from this).

**Check current mode:**

```bash
getenforce
# or full status:
sestatus
```

**Temporarily switch between enforcing and permissive:**

```bash
sudo setenforce 0   # permissive
sudo setenforce 1   # enforcing
```

This does not survive a reboot. For permanent changes, edit `/etc/selinux/config`:

```
SELINUX=enforcing
```

**Troubleshoot denials:**

The key tool is `ausearch` combined with `audit2why`:

```bash
# Find recent SELinux denials
sudo ausearch -m AVC -ts recent

# Explain why something was denied
sudo ausearch -m AVC -ts recent | audit2why
```

`audit2why` tells you in plain English what happened and often suggests a fix, such as setting a boolean or restoring a file context.

**Common fixes:**

1. **Wrong file context** -- most common issue. If you moved a file instead of copying it, it retains the old context:

```bash
# Check context
ls -Z /var/www/html/index.html

# Restore default context
sudo restorecon -Rv /var/www/html/
```

2. **SELinux boolean needs to be toggled:**

```bash
# List booleans related to httpd
getsebool -a | grep httpd

# Allow httpd to connect to the network
sudo setsebool -P httpd_can_network_connect on
```

The `-P` makes it persistent across reboots.

3. **Custom policy needed:**

```bash
# Generate a policy module from denials
sudo ausearch -m AVC -ts recent | audit2allow -M mypolicy
sudo semodule -i mypolicy.pp
```

**Install troubleshooting tools if missing:**

```bash
sudo dnf install setroubleshoot-server policycoreutils-python-utils
```

The `sealert` tool provides detailed, human-readable explanations:

```bash
sudo sealert -a /var/log/audit/audit.log
```

Rule of thumb: never disable SELinux entirely. Use permissive mode for debugging, fix the root cause, then re-enable enforcing.""",
    "flm_security"
))

# ---------------------------------------------------------------------------
# 3. PAM configuration (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""How does PAM work on Debian? I see files in /etc/pam.d/ but don't understand the format.""",
    r"""PAM (Pluggable Authentication Modules) is the framework that handles authentication, account management, session setup, and password changes on Linux. Every program that needs to authenticate users (login, sshd, sudo, su, etc.) has a config file in `/etc/pam.d/`.

**File format:**

Each line has four fields:

```
type  control  module  [arguments]
```

**Types:**
- `auth` -- verify identity (password, key, fingerprint)
- `account` -- check if account is allowed (expired? locked?)
- `password` -- update credentials (password change)
- `session` -- setup/teardown around login (mount home, set env, logging)

**Controls:**
- `required` -- must succeed, but continue checking other modules
- `requisite` -- must succeed, fail immediately if not
- `sufficient` -- if this succeeds (and nothing required has failed), stop and succeed
- `optional` -- result only matters if it's the only module for this type
- `include` -- include another PAM config file

**Example -- `/etc/pam.d/sshd` (simplified):**

```
auth       required     pam_sepermit.so
auth       include      common-auth
account    required     pam_nologin.so
account    include      common-account
session    include      common-session
password   include      common-password
```

This reads: for SSH auth, first check `pam_sepermit` (SELinux), then include the common auth stack. The common files (`/etc/pam.d/common-auth`, etc.) are shared across services.

**Key modules:**

- `pam_unix.so` -- traditional password auth against `/etc/shadow`
- `pam_permit.so` -- always succeeds (used for passwordless steps)
- `pam_deny.so` -- always fails (safety net)
- `pam_nologin.so` -- blocks non-root logins if `/etc/nologin` exists
- `pam_limits.so` -- applies `/etc/security/limits.conf` (ulimits)
- `pam_pwquality.so` -- password strength checking
- `pam_faillock.so` -- lock accounts after failed attempts
- `pam_motd.so` -- display message of the day

**Common Debian files:**

```bash
ls /etc/pam.d/common-*
```

These are managed by `pam-auth-update` on Debian. Don't edit them directly -- use:

```bash
sudo pam-auth-update
```

This shows an interactive dialog for enabling/disabling PAM modules system-wide.

**Be very careful editing PAM configs.** A misconfigured PAM can lock you out of your system entirely. Always keep a root shell open while testing changes, and test with a separate terminal before closing the working one.""",
    "flm_security"
))

# ---------------------------------------------------------------------------
# 4. Two-factor auth with TOTP for SSH (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""How do I set up two-factor authentication with TOTP for SSH on Debian?""",
    r"""You can use Google Authenticator's PAM module to add TOTP (time-based one-time passwords) as a second factor for SSH.

**Install the PAM module:**

```bash
sudo apt install libpam-google-authenticator
```

**Configure TOTP for your user:**

```bash
google-authenticator
```

Answer the prompts:
- **Time-based tokens** -- yes
- It shows a QR code and a secret key. Scan the QR code with an authenticator app (FreeOTP, Aegis, Google Authenticator, etc.).
- **Update .google_authenticator file** -- yes
- **Disallow multiple uses** -- yes (prevents replay)
- **Allow 30-second window** -- yes (handles clock skew)
- **Rate limiting** -- yes

This creates `~/.google_authenticator` with your secret and settings.

**Configure PAM for SSH:**

Edit `/etc/pam.d/sshd` and add this line at the top of the auth section:

```
auth required pam_google_authenticator.so
```

**Configure sshd to ask for the code:**

Edit `/etc/ssh/sshd_config`:

```
ChallengeResponseAuthentication yes
# Or on newer OpenSSH:
KbdInteractiveAuthentication yes

# If you want key + TOTP (not password + TOTP):
AuthenticationMethods publickey,keyboard-interactive
```

The `AuthenticationMethods` line means: first the user must present a valid SSH key, then they must enter the TOTP code. This gives you two real factors (something you have x2).

If you want password + TOTP instead:

```
AuthenticationMethods keyboard-interactive
```

**Restart sshd:**

```bash
sudo systemctl restart sshd
```

**Test from another terminal before closing your current session:**

```bash
ssh youruser@localhost
```

You should be prompted for your key (or password) and then for a verification code.

**Important notes:**

- Each user needs to run `google-authenticator` individually.
- If a user hasn't configured TOTP and you have `auth required`, they will be locked out. You can use `nullok` to allow users without TOTP to still log in: `auth required pam_google_authenticator.so nullok`
- Keep backup codes in a safe place. The setup generates emergency scratch codes.
- Make sure the server's clock is accurate. TOTP depends on time: `sudo apt install systemd-timesyncd && timedatectl set-ntp true`""",
    "flm_security"
))

# ---------------------------------------------------------------------------
# 5. Audit system with auditd (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""How do I use auditd to monitor system activity on Debian?""",
    r"""auditd is the Linux audit daemon. It monitors system calls and file access based on rules you define, producing detailed logs for security analysis and compliance.

**Install:**

```bash
sudo apt install auditd audispd-plugins
```

**Start and enable:**

```bash
sudo systemctl enable --now auditd
```

**Key files:**

- `/etc/audit/auditd.conf` -- daemon settings (log size, rotation, etc.)
- `/etc/audit/rules.d/` -- persistent rules (loaded at boot)
- `/var/log/audit/audit.log` -- the audit log

**Add audit rules:**

Watch a file for any access:

```bash
sudo auditctl -w /etc/passwd -p rwxa -k passwd_changes
```

- `-w` -- watch this path
- `-p rwxa` -- permissions to watch (read, write, execute, attribute change)
- `-k` -- key tag for searching later

Watch a directory:

```bash
sudo auditctl -w /etc/ssh/ -p wa -k ssh_config
```

Monitor a specific system call:

```bash
# Watch for all execve calls by uid >= 1000 (non-system users)
sudo auditctl -a always,exit -F arch=b64 -S execve -F uid>=1000 -k user_commands
```

**List active rules:**

```bash
sudo auditctl -l
```

**Search the audit log:**

```bash
# Search by key
sudo ausearch -k passwd_changes

# Search by time range
sudo ausearch -k user_commands -ts today -te now

# Search by user
sudo ausearch -ua 1000
```

**Generate reports:**

```bash
# Summary of events
sudo aureport

# Authentication report
sudo aureport -au

# File access report
sudo aureport -f

# Executable report
sudo aureport -x --summary
```

**Make rules persistent:**

Write rules to a file in `/etc/audit/rules.d/`:

```bash
sudo tee /etc/audit/rules.d/custom.rules << 'EOF'
-w /etc/passwd -p rwxa -k passwd_changes
-w /etc/shadow -p rwxa -k shadow_changes
-w /etc/ssh/sshd_config -p wa -k ssh_config
-a always,exit -F arch=b64 -S execve -F uid>=1000 -k user_commands
EOF
```

Then load them:

```bash
sudo augenrules --load
```

**Performance note:** Audit rules for high-frequency system calls (like execve for every user command) can generate large logs. Use specific filters and rotate logs aggressively in `/etc/audit/auditd.conf`.""",
    "flm_security"
))

# ---------------------------------------------------------------------------
# 6. Secure /tmp with noexec (Debian minimal)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEBMIN,
    r"""How do I mount /tmp with noexec for security on a minimal Debian server?""",
    r"""Mounting `/tmp` with `noexec` prevents execution of binaries placed there, which blocks a common attack vector where malware is downloaded to `/tmp` and executed.

**Option 1: /tmp is a separate partition**

If `/tmp` is already its own partition, just add mount options in `/etc/fstab`:

```bash
# Find your current /tmp mount
findmnt /tmp
```

Edit `/etc/fstab` and add `noexec,nosuid,nodev`:

```
/dev/sda3  /tmp  ext4  defaults,noexec,nosuid,nodev  0  2
```

Remount without rebooting:

```bash
sudo mount -o remount /tmp
```

**Option 2: /tmp is on the root filesystem (more common)**

Create a dedicated tmpfs or file-backed mount:

```bash
# Using tmpfs (RAM-backed, fast, limited by RAM)
echo 'tmpfs  /tmp  tmpfs  defaults,noexec,nosuid,nodev,size=512M  0  0' | sudo tee -a /etc/fstab
```

Or use a file-backed loop device for a size-limited /tmp:

```bash
sudo dd if=/dev/zero of=/tmp.img bs=1M count=1024
sudo mkfs.ext4 /tmp.img
echo '/tmp.img  /tmp  ext4  loop,defaults,noexec,nosuid,nodev  0  0' | sudo tee -a /etc/fstab
```

**Option 3: Use systemd's tmp.mount**

Debian supports `tmp.mount` which mounts tmpfs on `/tmp`:

```bash
sudo cp /usr/share/systemd/tmp.mount /etc/systemd/system/
sudo systemctl enable tmp.mount
```

Then edit `/etc/systemd/system/tmp.mount` and add options:

```ini
[Mount]
What=tmpfs
Where=/tmp
Type=tmpfs
Options=mode=1777,strictatime,noexec,nosuid,nodev,size=512M
```

```bash
sudo systemctl daemon-reload
sudo systemctl start tmp.mount
```

**Verify:**

```bash
mount | grep /tmp
# Should show noexec,nosuid,nodev in the options

# Test that execution is blocked
cp /bin/ls /tmp/test_ls
chmod +x /tmp/test_ls
/tmp/test_ls     # Should fail with "Permission denied"
rm /tmp/test_ls
```

**Caveats:**

- Some programs expect to execute from `/tmp` (package builds, some installers). If something breaks, check if it's trying to run from `/tmp`.
- `apt` sometimes uses `/tmp` for scripts. If you hit issues during package installation, temporarily remount: `sudo mount -o remount,exec /tmp`
- Also consider applying the same treatment to `/var/tmp` and `/dev/shm`.""",
    "flm_security"
))

# ---------------------------------------------------------------------------
# 7. Check for rootkits (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""How do I check my Debian system for rootkits?""",
    r"""Two main tools: rkhunter and chkrootkit. They work differently and complement each other, so running both is a good idea.

**chkrootkit:**

```bash
sudo apt install chkrootkit
sudo chkrootkit
```

chkrootkit examines the system for known rootkit signatures -- checking binaries, network interfaces, log files, and other indicators. It's simple and fast.

Look for any lines that say `INFECTED`. Note that false positives are common -- research any findings before panicking.

**rkhunter (Rootkit Hunter):**

```bash
sudo apt install rkhunter
```

Update its database first:

```bash
sudo rkhunter --update
sudo rkhunter --propupd    # save current file properties as baseline
```

Run a scan:

```bash
sudo rkhunter --check
```

It checks for:
- Known rootkit signatures
- Modified system binaries
- Suspicious file permissions
- Hidden files and directories
- Listening network ports
- System startup files

Review results:

```bash
sudo cat /var/log/rkhunter.log | grep -i warning
```

**Configure rkhunter for automated daily scans:**

Edit `/etc/default/rkhunter`:

```
CRON_DAILY_RUN="true"
REPORT_EMAIL="root"
APT_AUTOGEN="yes"
```

The `APT_AUTOGEN` setting is important on Debian -- it tells rkhunter to update its file property database whenever packages are updated via apt, preventing false positives from legitimate package updates.

**Reduce false positives:**

Edit `/etc/rkhunter.conf`:

```bash
# Allow known scripts in /dev
ALLOWDEVDIR=/dev/.udev
ALLOWDEVDIR=/dev/shm

# Allow hidden directories that are normal
ALLOWHIDDENDIR=/etc/.java

# Known OK inode changes
ALLOW_SSH_ROOT_USER=prohibit-password
```

After editing, verify config:

```bash
sudo rkhunter -C
```

**What to do if something is found:**

1. Don't panic -- verify the finding manually.
2. Check if the flagged binary matches the package version: `debsums -c` to verify installed packages.
3. Compare with a known good system.
4. If confirmed, isolate the system from the network, investigate, and plan a reinstall. A compromised system cannot be trusted.""",
    "flm_security"
))

# ---------------------------------------------------------------------------
# 8. GPG key management (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    r"""How do I manage GPG keys for signing and encrypting?""",
    r"""GPG (GNU Privacy Guard) lets you sign, encrypt, and verify data using public-key cryptography.

**Generate a new key pair:**

```bash
gpg --full-generate-key
```

Choose:
- Key type: RSA and RSA (default)
- Key size: 4096 bits
- Expiration: set one (e.g., 2 years). You can always extend it later.
- Enter your name and email.
- Set a strong passphrase.

**List your keys:**

```bash
# Public keys
gpg --list-keys
gpg --list-keys --keyid-format long

# Secret (private) keys
gpg --list-secret-keys
```

**Export your public key** (to share with others):

```bash
gpg --armor --export you@example.com > mykey.pub
```

**Import someone else's public key:**

```bash
gpg --import theirkey.pub
```

**Verify a key's fingerprint** (always verify through a trusted channel):

```bash
gpg --fingerprint you@example.com
```

**Sign a file:**

```bash
# Detached signature (separate .sig file)
gpg --detach-sign --armor myfile.tar.gz

# Cleartext signature (for text files)
gpg --clearsign message.txt
```

**Verify a signature:**

```bash
gpg --verify myfile.tar.gz.asc myfile.tar.gz
```

**Encrypt a file for a recipient:**

```bash
gpg --encrypt --recipient them@example.com secrets.txt
# Produces secrets.txt.gpg
```

**Decrypt:**

```bash
gpg --decrypt secrets.txt.gpg > secrets.txt
```

**Upload your public key to a keyserver:**

```bash
gpg --send-keys --keyserver keys.openpgp.org YOUR_KEY_ID
```

**Key maintenance:**

```bash
# Edit a key (add UIDs, set expiration, etc.)
gpg --edit-key you@example.com

# Revoke a key (generate revocation certificate first!)
gpg --gen-revoke you@example.com > revoke.asc

# Back up your secret key
gpg --export-secret-keys --armor you@example.com > secret-backup.asc
```

**Best practices:**
- Always set an expiration date and extend as needed.
- Keep a revocation certificate in a safe offline location.
- Back up your `~/.gnupg/` directory (or at least export secret keys) to an encrypted offline medium.
- Use subkeys for daily operations and keep the master key offline if possible.""",
    "flm_security"
))

# ---------------------------------------------------------------------------
# 9. File integrity monitoring with AIDE (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""How do I set up AIDE for file integrity monitoring on Debian?""",
    r"""AIDE (Advanced Intrusion Detection Environment) creates a database of file checksums and metadata, then reports any changes. It's the free equivalent of Tripwire.

**Install:**

```bash
sudo apt install aide
```

**Configure what to monitor:**

The main config is `/etc/aide/aide.conf`. On Debian, rules are also split into `/etc/aide/aide.conf.d/`. The defaults monitor system binaries, libraries, and config files.

Key rule definitions in the config:

```
# R = read-only files (detect any change)
R = p+i+n+u+g+s+m+c+md5+sha256

# L = log files (only track deletion, not content changes)
L = p+i+n+u+g

# Custom rule for configs
CONFIG = p+i+n+u+g+s+md5+sha256
```

Where: p=permissions, i=inode, n=link count, u=user, g=group, s=size, m=mtime, c=ctime, md5/sha256=checksums.

**Initialize the database:**

```bash
sudo aideinit
```

This scans the entire filesystem and creates the baseline database. It can take several minutes. On Debian, the database is written to `/var/lib/aide/aide.db.new` and needs to be moved:

```bash
sudo cp /var/lib/aide/aide.db.new /var/lib/aide/aide.db
```

**Run a check:**

```bash
sudo aide --check
```

This compares the current filesystem state against the database. Any added, removed, or modified files are reported.

**After legitimate changes (updates, config edits), update the database:**

```bash
sudo aide --update
sudo cp /var/lib/aide/aide.db.new /var/lib/aide/aide.db
```

**Automate daily checks with cron:**

Debian's AIDE package installs a daily cron job in `/etc/cron.daily/aide`. Enable it by editing `/etc/default/aide`:

```
CRON_DAILY_RUN=yes
MAILTO=root
```

**Custom rules example -- add monitoring for a web root:**

Create `/etc/aide/aide.conf.d/70_webroot`:

```
/var/www R
!/var/www/html/uploads
```

The `!` prefix excludes a path. You don't want to flag every uploaded file as a change.

**Important security note:** Store a copy of the AIDE database offline or on a read-only medium. If an attacker compromises the system, they could modify the AIDE database to hide their changes. A trusted offline copy lets you verify against a known-good state.""",
    "flm_security"
))

# ---------------------------------------------------------------------------
# 10. SSH hardening - certificates, AllowUsers (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""What are the best practices for hardening SSH on a Debian server?""",
    r"""Here's a practical SSH hardening checklist for a Debian server. Edit `/etc/ssh/sshd_config` for each change.

**1. Disable root login:**

```
PermitRootLogin no
```

**2. Disable password authentication (use keys only):**

```
PasswordAuthentication no
```

Make sure you have a working SSH key set up before doing this.

**3. Restrict which users can log in:**

```
AllowUsers deploy admin
# Or restrict by group:
AllowGroups ssh-users
```

Only users listed will be able to SSH in. This is a strong whitelist.

**4. Change the default port** (reduces automated scanning noise):

```
Port 2222
```

**5. Use strong ciphers only:**

```
KexAlgorithms curve25519-sha256,curve25519-sha256@libssh.org
Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com,aes128-gcm@openssh.com
MACs hmac-sha2-512-etm@openssh.com,hmac-sha2-256-etm@openssh.com
HostKeyAlgorithms ssh-ed25519,rsa-sha2-512,rsa-sha2-256
```

**6. Limit authentication attempts and timeouts:**

```
MaxAuthTries 3
LoginGraceTime 30
MaxStartups 3:50:10
```

**7. Disable unused features:**

```
X11Forwarding no
AllowTcpForwarding no
AllowAgentForwarding no
PermitTunnel no
```

Re-enable only what you actually need.

**8. Use SSH certificates (advanced):**

Instead of copying public keys to each server, set up a CA:

```bash
# On your CA machine, generate a CA key
ssh-keygen -t ed25519 -f ca_key -C "SSH CA"

# Sign a user's public key
ssh-keygen -s ca_key -I user@example.com -n deploy -V +52w id_ed25519.pub
```

On the server, trust the CA:

```
TrustedUserCAKeys /etc/ssh/ca_key.pub
```

Now any key signed by the CA is accepted, without adding individual keys to `authorized_keys`. This scales much better for teams.

**9. Log more detail:**

```
LogLevel VERBOSE
```

**10. Apply and test:**

```bash
sudo sshd -t             # test config for syntax errors
sudo systemctl reload sshd
```

Always test in a separate terminal before closing your current session. If the config is broken, you'll lock yourself out.

**Also consider:**
- fail2ban to block brute-force attempts: `sudo apt install fail2ban`
- Port knocking for obscurity
- Two-factor authentication (TOTP) as an additional factor""",
    "flm_security"
))

# ---------------------------------------------------------------------------
# 11. Prometheus node_exporter setup (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""How do I set up Prometheus node_exporter on Debian for system monitoring?""",
    r"""node_exporter exposes system metrics (CPU, memory, disk, network) on an HTTP endpoint that Prometheus can scrape.

**Install from Debian repos:**

```bash
sudo apt install prometheus-node-exporter
```

This installs, enables, and starts the service. It listens on port 9100 by default.

**Verify it's running:**

```bash
sudo systemctl status prometheus-node-exporter
curl -s http://localhost:9100/metrics | head -20
```

You should see metric lines like `node_cpu_seconds_total`, `node_memory_MemTotal_bytes`, etc.

**Configure which collectors are enabled:**

Edit `/etc/default/prometheus-node-exporter`:

```bash
ARGS="--collector.systemd --collector.processes --no-collector.wifi"
```

Common collectors to enable:
- `systemd` -- service states
- `processes` -- process counts
- `textfile` -- custom metrics from files

Common ones to disable on servers:
- `wifi` -- not relevant on servers
- `infiniband` -- unless you have it

Restart after changes:

```bash
sudo systemctl restart prometheus-node-exporter
```

**Configure Prometheus to scrape it:**

On your Prometheus server, add to `/etc/prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'node'
    static_configs:
      - targets:
        - 'server1:9100'
        - 'server2:9100'
    scrape_interval: 15s
```

**Useful PromQL queries for dashboards:**

```
# CPU usage percentage
100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory usage percentage
(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100

# Disk usage percentage
(1 - node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100

# Network traffic rate
irate(node_network_receive_bytes_total{device="eth0"}[5m]) * 8
```

**Security: Restrict access to metrics:**

node_exporter exposes potentially sensitive system info. Limit access:

```bash
# Firewall rule -- only allow Prometheus server
sudo iptables -A INPUT -p tcp --dport 9100 -s PROMETHEUS_IP -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 9100 -j DROP
```

Or use nginx as a reverse proxy with basic auth in front of node_exporter.""",
    "flm_monitoring"
))

# ---------------------------------------------------------------------------
# 12. Simple monitoring with monit (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""How do I use monit for simple service monitoring on Debian?""",
    r"""monit is a lightweight daemon that monitors processes, files, filesystems, and network connections. It can restart crashed services and send alerts.

**Install:**

```bash
sudo apt install monit
```

**Main config:** `/etc/monit/monitrc`

Enable the web interface for status checking:

```
set httpd port 2812
    use address localhost
    allow admin:yourpassword
```

**Configure monitored services in `/etc/monit/conf.d/`:**

Monitor sshd:

```bash
sudo tee /etc/monit/conf.d/sshd << 'EOF'
check process sshd with pidfile /run/sshd.pid
    start program = "/usr/bin/systemctl start sshd"
    stop program  = "/usr/bin/systemctl stop sshd"
    if failed port 22 protocol ssh then restart
    if 3 restarts within 5 cycles then alert
EOF
```

Monitor nginx:

```bash
sudo tee /etc/monit/conf.d/nginx << 'EOF'
check process nginx with pidfile /run/nginx.pid
    start program = "/usr/bin/systemctl start nginx"
    stop program  = "/usr/bin/systemctl stop nginx"
    if failed host localhost port 80 protocol http
        request "/" with timeout 10 seconds
    then restart
    if 3 restarts within 5 cycles then alert
EOF
```

Monitor disk space:

```bash
sudo tee /etc/monit/conf.d/disk << 'EOF'
check filesystem rootfs with path /
    if space usage > 80% then alert
    if space usage > 95% then exec "/usr/local/bin/disk-cleanup.sh"
EOF
```

Monitor system resources:

```bash
sudo tee /etc/monit/conf.d/system << 'EOF'
check system $HOST
    if loadavg (1min) > 8 then alert
    if memory usage > 90% then alert
    if swap usage > 50% then alert
    if cpu usage > 95% for 5 cycles then alert
EOF
```

**Set up email alerts:**

In `/etc/monit/monitrc`:

```
set mailserver smtp.example.com port 587
    username "alerts@example.com"
    password "password"
    using tls

set alert you@example.com
```

**Test the config and start:**

```bash
sudo monit -t            # syntax check
sudo systemctl enable --now monit
```

**Useful commands:**

```bash
sudo monit status        # show all monitored services
sudo monit summary       # quick summary
sudo monit reload        # reload after config changes
sudo monit restart nginx # manually restart a service through monit
```

monit is ideal for small setups where you want automatic restart of crashed services and basic alerting without the complexity of Prometheus or Nagios.""",
    "flm_monitoring"
))

# ---------------------------------------------------------------------------
# 13. Health check script (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""Can you write a health check script that monitors disk, memory, and key services?""",
    r"""Here's a practical health check script you can run from cron:

```bash
#!/bin/bash
set -euo pipefail

# Configuration
ALERT_EMAIL="admin@example.com"
DISK_THRESHOLD=85       # percent
MEM_THRESHOLD=90        # percent
SWAP_THRESHOLD=50       # percent
SERVICES=("sshd" "nginx" "postgresql")
LOG="/var/log/healthcheck.log"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

PROBLEMS=()

# -- Disk check -------------------------------------------------------
while IFS= read -r line; do
    usage=$(echo "$line" | awk '{print $5}' | tr -d '%')
    mount=$(echo "$line" | awk '{print $6}')
    if (( usage > DISK_THRESHOLD )); then
        PROBLEMS+=("DISK: ${mount} is ${usage}% full")
    fi
done < <(df -h --output=pcent,target -x tmpfs -x devtmpfs | tail -n +2)

# -- Memory check -----------------------------------------------------
mem_total=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
mem_avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
mem_pct=$(( (mem_total - mem_avail) * 100 / mem_total ))
if (( mem_pct > MEM_THRESHOLD )); then
    PROBLEMS+=("MEMORY: ${mem_pct}% used ($(( mem_avail / 1024 )) MB available)")
fi

# -- Swap check --------------------------------------------------------
swap_total=$(awk '/SwapTotal/ {print $2}' /proc/meminfo)
if (( swap_total > 0 )); then
    swap_free=$(awk '/SwapFree/ {print $2}' /proc/meminfo)
    swap_pct=$(( (swap_total - swap_free) * 100 / swap_total ))
    if (( swap_pct > SWAP_THRESHOLD )); then
        PROBLEMS+=("SWAP: ${swap_pct}% used")
    fi
fi

# -- Service check -----------------------------------------------------
for svc in "${SERVICES[@]}"; do
    if ! systemctl is-active --quiet "$svc"; then
        PROBLEMS+=("SERVICE: ${svc} is not running")
    fi
done

# -- Load average check ------------------------------------------------
cores=$(nproc)
load=$(awk '{print $1}' /proc/loadavg)
# Compare using bc for floating point
if echo "$load > $cores * 2" | bc -l | grep -q 1; then
    PROBLEMS+=("LOAD: ${load} (${cores} cores)")
fi

# -- Report ------------------------------------------------------------
if (( ${#PROBLEMS[@]} > 0 )); then
    report="Health check ALERT at $(timestamp) on $(hostname):"$'\n'
    for p in "${PROBLEMS[@]}"; do
        report+="  - ${p}"$'\n'
    done

    echo "$report" | tee -a "$LOG"

    # Send email if mail is configured
    if command -v mail &>/dev/null; then
        echo "$report" | mail -s "ALERT: $(hostname) health check" "$ALERT_EMAIL"
    fi

    exit 1
else
    echo "$(timestamp) - All checks passed" >> "$LOG"
    exit 0
fi
```

**Deploy it:**

```bash
sudo install -m 755 healthcheck.sh /usr/local/bin/
```

**Run every 5 minutes via cron:**

```bash
sudo crontab -e
# Add:
*/5 * * * * /usr/local/bin/healthcheck.sh 2>&1 | logger -t healthcheck
```

The script is intentionally simple and has no dependencies beyond coreutils and systemd. Customize the thresholds and service list for your environment.""",
    "flm_monitoring"
))

# ---------------------------------------------------------------------------
# 14. sar/sysstat for historical data (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""How do I use sar and sysstat for historical performance data on Debian?""",
    r"""sysstat collects and stores system performance data on a schedule. The `sar` command lets you query historical data -- CPU, memory, disk I/O, network, and more.

**Install and enable:**

```bash
sudo apt install sysstat
```

Enable data collection by editing `/etc/default/sysstat`:

```
ENABLED="true"
```

Restart the service:

```bash
sudo systemctl enable --now sysstat
sudo systemctl restart sysstat
```

sysstat uses a cron job (or systemd timer) to collect data every 10 minutes and store it in `/var/log/sysstat/`.

**Using sar to query historical data:**

```bash
# CPU usage for today
sar -u

# CPU usage for a specific date (day of month)
sar -u -f /var/log/sysstat/sa15

# CPU between specific times today
sar -u -s 09:00:00 -e 17:00:00

# Memory usage
sar -r

# Swap usage
sar -S

# Disk I/O (all devices)
sar -d

# Network interface traffic
sar -n DEV

# Network errors
sar -n EDEV

# Load average
sar -q

# Context switches and interrupts
sar -w
```

**Real-time monitoring (like vmstat but with sar formatting):**

```bash
# Collect every 2 seconds, 10 samples
sar -u 2 10
sar -r 2 10
```

**Other useful sysstat tools:**

```bash
# Interactive top-like view of disk I/O
iostat -xz 1

# Per-process I/O statistics
pidstat -d 1

# Per-process CPU stats
pidstat -u 1

# Per-process memory stats
pidstat -r 1

# Combined CPU, memory, and I/O in one view
cifsiostat 1    # for CIFS mounts
```

**Generate a report for a full day:**

```bash
sar -A -f /var/log/sysstat/sa15 > full_report.txt
```

**Adjust collection interval:**

The default 10-minute interval is in `/etc/cron.d/sysstat` or the systemd timer. For more granular data:

```bash
sudo systemctl edit sysstat-collect.timer
```

```ini
[Timer]
OnCalendar=*:0/2
```

This collects every 2 minutes. Be aware this increases disk usage for the logs.

**Data retention:**

By default, sysstat keeps about 28 days of data. Adjust in `/etc/sysstat/sysstat`:

```
HISTORY=90
```

sar data is invaluable when investigating performance issues. You can look back at what CPU, memory, and I/O looked like at the time of a reported problem instead of trying to reproduce it.""",
    "flm_monitoring"
))

# ---------------------------------------------------------------------------
# 15. 3-2-1 backup strategy (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    r"""What is the 3-2-1 backup strategy?""",
    r"""The 3-2-1 rule is a widely recommended data backup framework:

- **3** copies of your data (1 primary + 2 backups)
- **2** different storage media or types
- **1** copy offsite (physically separate location)

**Why each number matters:**

**3 copies:** Any single storage device can fail. Having two backups means you can lose one and still recover. The probability of two independent failures at the same time is extremely low.

**2 different media:** If both backups are on the same type of storage (say, two hard drives in the same RAID array), a firmware bug, manufacturing defect, or environmental factor could take both out. Use different types: local disk + external drive, SSD + tape, local NAS + cloud, etc.

**1 offsite:** Protects against site-level disasters -- fire, flood, theft, power surge. If your server room burns down, the backup on the shelf next to it burns too. An offsite copy (cloud storage, a remote server, or a drive at another physical location) survives.

**Practical implementation examples:**

For a server:
1. Primary data on the server's RAID array
2. Daily backup to a local NAS (different device, same site)
3. Weekly encrypted backup to offsite storage (rsync to a remote server, or upload to S3/Backblaze B2)

For a personal workstation:
1. Primary data on your SSD
2. Daily backup to an external USB drive (automated with rsync or borgbackup)
3. Cloud backup (Backblaze, or encrypted rclone to any cloud provider)

**Additional considerations:**

- **Test your restores.** A backup you can't restore is not a backup. Schedule periodic restore tests.
- **Encrypt offsite backups.** Data leaving your control should always be encrypted.
- **Version your backups.** Keep multiple snapshots over time (daily, weekly, monthly). This protects against ransomware or accidental deletion that gets replicated to the backup before you notice.
- **Automate everything.** Manual backups don't happen. Set up cron jobs or systemd timers.
- **Monitor backup success.** If a backup job fails silently for weeks, you won't know until you need it.

The 3-2-1 rule is a minimum. Some extend it to 3-2-1-1-0: the extra "1" is one immutable/air-gapped copy (ransomware protection), and "0" means zero errors in restore testing.""",
    "flm_backup"
))

# ---------------------------------------------------------------------------
# 16. Incremental backups with rsync (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""How do I set up incremental backups with rsync on Debian?""",
    r"""rsync is perfect for incremental backups because it only transfers changed files. Combined with hard links, you can create space-efficient snapshots.

**Basic rsync backup:**

```bash
rsync -avz --delete /home/ /backup/home/
```

- `-a` -- archive mode (preserves permissions, ownership, timestamps, symlinks)
- `-v` -- verbose
- `-z` -- compress during transfer
- `--delete` -- remove files from destination that are no longer in source

**Incremental snapshots with hard links:**

This creates daily snapshots where unchanged files are hard-linked (no extra disk space), and only changed files are new copies:

```bash
#!/bin/bash
set -euo pipefail

SRC="/home/"
DEST="/backup/snapshots"
DATE=$(date +%Y-%m-%d)
LATEST="$DEST/latest"

mkdir -p "$DEST"

rsync -avz --delete \
    --link-dest="$LATEST" \
    "$SRC" "$DEST/$DATE/"

# Update the 'latest' symlink
rm -f "$LATEST"
ln -s "$DEST/$DATE" "$LATEST"

echo "Backup complete: $DEST/$DATE"
```

`--link-dest` is the key option. It tells rsync to hard-link unchanged files from the previous backup instead of copying them. Each snapshot looks like a full backup (you can browse it normally), but only uses disk space for the files that actually changed.

**Remote backups over SSH:**

```bash
rsync -avz --delete \
    --link-dest=/backup/snapshots/latest \
    /home/ user@backup-server:/backup/snapshots/$(date +%Y-%m-%d)/
```

**Useful options for backups:**

```bash
rsync -avz --delete \
    --exclude='.cache' \
    --exclude='*.tmp' \
    --exclude='node_modules' \
    --exclude='.local/share/Trash' \
    --progress \
    --stats \
    /home/ /backup/home/
```

Put excludes in a file for cleanliness:

```bash
# /etc/rsync-excludes.txt
.cache
*.tmp
*.swp
node_modules
__pycache__
.local/share/Trash
```

```bash
rsync -avz --delete --exclude-from=/etc/rsync-excludes.txt /home/ /backup/home/
```

**Retention -- clean up old snapshots:**

```bash
# Keep last 7 daily, 4 weekly, 12 monthly
find /backup/snapshots/ -maxdepth 1 -type d -mtime +7 \
    -not -name '*-01' -not -name '*-08' -not -name '*-15' -not -name '*-22' \
    -exec rm -rf {} +
```

**Automate with cron:**

```bash
sudo crontab -e
# Daily at 2 AM
0 2 * * * /usr/local/bin/rsync-backup.sh >> /var/log/rsync-backup.log 2>&1
```

rsync is reliable, well-tested, and already installed on virtually every Linux system.""",
    "flm_backup"
))

# ---------------------------------------------------------------------------
# 17. Full system backup with tar (Debian minimal)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEBMIN,
    r"""How do I make a full system backup of this minimal Debian server using tar?""",
    r"""tar is the simplest way to back up a whole system. No extra software needed.

**Full system backup:**

```bash
sudo tar czf /backup/system-$(date +%Y%m%d).tar.gz \
    --exclude=/backup \
    --exclude=/proc \
    --exclude=/sys \
    --exclude=/dev \
    --exclude=/run \
    --exclude=/tmp \
    --exclude=/mnt \
    --exclude=/media \
    --exclude=/lost+found \
    --one-file-system \
    /
```

The excluded directories are virtual filesystems or temporary data that shouldn't be backed up. `--one-file-system` prevents tar from crossing into other mounted filesystems.

**For better compression (slower but smaller):**

```bash
sudo tar cJf /backup/system-$(date +%Y%m%d).tar.xz \
    --exclude=/backup \
    --exclude=/proc \
    --exclude=/sys \
    --exclude=/dev \
    --exclude=/run \
    --exclude=/tmp \
    --exclude=/mnt \
    --exclude=/media \
    --exclude=/lost+found \
    --one-file-system \
    /
```

**Verify the archive:**

```bash
sudo tar tzf /backup/system-20250101.tar.gz | head -20
sudo tar tzf /backup/system-20250101.tar.gz | wc -l
```

**Restore the full system:**

Boot from a live USB, mount your target disk, then:

```bash
sudo mount /dev/sda1 /mnt
sudo tar xzf /backup/system-20250101.tar.gz -C /mnt

# Recreate the excluded directories
sudo mkdir -p /mnt/{proc,sys,dev,run,tmp,mnt,media}
sudo chmod 1777 /mnt/tmp
```

Then reinstall GRUB:

```bash
sudo mount --bind /dev /mnt/dev
sudo mount --bind /proc /mnt/proc
sudo mount --bind /sys /mnt/sys
sudo chroot /mnt grub-install /dev/sda
sudo chroot /mnt update-grub
```

**Backup script with logging:**

```bash
#!/bin/bash
set -euo pipefail

BACKUP_DIR="/backup"
FILENAME="system-$(date +%Y%m%d-%H%M).tar.gz"
LOG="$BACKUP_DIR/backup.log"

echo "$(date) - Starting backup" >> "$LOG"

sudo tar czf "$BACKUP_DIR/$FILENAME" \
    --exclude="$BACKUP_DIR" \
    --exclude=/proc --exclude=/sys \
    --exclude=/dev --exclude=/run \
    --exclude=/tmp --exclude=/mnt \
    --exclude=/media --exclude=/lost+found \
    --one-file-system / 2>> "$LOG"

SIZE=$(du -h "$BACKUP_DIR/$FILENAME" | cut -f1)
echo "$(date) - Backup complete: $FILENAME ($SIZE)" >> "$LOG"

# Keep only last 5 backups
ls -t "$BACKUP_DIR"/system-*.tar.gz | tail -n +6 | xargs -r rm
```

Simple, reliable, no dependencies. For a minimal server, this is often all you need.""",
    "flm_backup"
))

# ---------------------------------------------------------------------------
# 18. Database backup automation with cron (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""How do I automate PostgreSQL database backups with cron on Debian?""",
    r"""Here's a complete setup for automated PostgreSQL backups with retention.

**Simple daily dump:**

```bash
sudo -u postgres pg_dumpall | gzip > /backup/pg/all-$(date +%Y%m%d).sql.gz
```

**Better: per-database backups with a script:**

```bash
#!/bin/bash
set -euo pipefail

BACKUP_DIR="/backup/postgresql"
RETENTION_DAYS=30
DATE=$(date +%Y%m%d-%H%M)
LOG="$BACKUP_DIR/backup.log"

mkdir -p "$BACKUP_DIR"

echo "$(date) - Starting PostgreSQL backup" >> "$LOG"

# Get list of databases (excluding templates)
DATABASES=$(sudo -u postgres psql -At -c \
    "SELECT datname FROM pg_database WHERE NOT datistemplate AND datname != 'postgres';")

for DB in $DATABASES; do
    OUTFILE="$BACKUP_DIR/${DB}-${DATE}.sql.gz"
    sudo -u postgres pg_dump "$DB" | gzip > "$OUTFILE"
    SIZE=$(du -h "$OUTFILE" | cut -f1)
    echo "$(date) - Dumped $DB -> $OUTFILE ($SIZE)" >> "$LOG"
done

# Also dump globals (roles, tablespaces)
sudo -u postgres pg_dumpall --globals-only | gzip > "$BACKUP_DIR/globals-${DATE}.sql.gz"

# Clean up old backups
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +"$RETENTION_DAYS" -delete

echo "$(date) - Backup complete" >> "$LOG"
```

**Install the script:**

```bash
sudo install -m 750 -o postgres -g postgres pg-backup.sh /usr/local/bin/
```

**Schedule with cron:**

```bash
sudo crontab -u postgres -e
```

Add:

```
# Daily at 3 AM
0 3 * * * /usr/local/bin/pg-backup.sh 2>&1 | logger -t pg-backup

# Weekly full custom-format dump (supports parallel restore)
0 4 * * 0 pg_dump -Fc mydb > /backup/postgresql/mydb-weekly-$(date +\%Y\%m\%d).dump 2>&1 | logger -t pg-backup
```

**Custom format for faster restores:**

The `-Fc` (custom format) dump supports parallel restore, selective table restore, and compression:

```bash
# Backup
sudo -u postgres pg_dump -Fc mydb > /backup/mydb.dump

# Restore (parallel, 4 jobs)
sudo -u postgres pg_restore -d mydb -j 4 /backup/mydb.dump

# Restore a single table
sudo -u postgres pg_restore -d mydb -t users /backup/mydb.dump
```

**Verify backups regularly:**

Add a weekly restore test to a temp database:

```bash
sudo -u postgres createdb backup_test
zcat /backup/postgresql/mydb-latest.sql.gz | sudo -u postgres psql backup_test
sudo -u postgres dropdb backup_test
```

**Send the backup offsite:**

```bash
# After the backup script runs
rsync -az /backup/postgresql/ user@offsite:/backup/postgresql/
```

Untested backups are not backups. Schedule restore tests.""",
    "flm_backup"
))

# ---------------------------------------------------------------------------
# 19. Bash best practices (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    r"""What are the most important bash scripting best practices?""",
    r"""Here are the practices that prevent the most bugs and headaches:

**1. Start every script with strict mode:**

```bash
#!/bin/bash
set -euo pipefail
```

- `set -e` -- exit on any command failure
- `set -u` -- treat unset variables as errors
- `set -o pipefail` -- a pipeline fails if any command in it fails (not just the last one)

Without these, scripts silently continue after errors, leading to data loss or corruption.

**2. Always quote your variables:**

```bash
# Wrong -- breaks on filenames with spaces
rm $file

# Right
rm "$file"

# Wrong -- glob expansion, word splitting
for f in $files; do

# Right
for f in "${files[@]}"; do
```

Unquoted variables undergo word splitting and glob expansion. This is the #1 source of bash bugs.

**3. Use `[[ ]]` instead of `[ ]` for tests:**

```bash
# Fragile -- breaks if $var is empty
[ $var = "hello" ]

# Robust -- handles empty vars, supports && ||, regex
[[ "$var" = "hello" ]]
[[ "$file" == *.txt ]]
[[ "$input" =~ ^[0-9]+$ ]]
```

**4. Use `$(command)` not backticks:**

```bash
# Hard to read, can't nest easily
result=`command`

# Clear, nestable
result=$(command)
```

**5. Handle errors explicitly:**

```bash
if ! cp "$src" "$dst"; then
    echo "ERROR: copy failed" >&2
    exit 1
fi

# Or with a trap for cleanup
cleanup() { rm -f "$tempfile"; }
trap cleanup EXIT

tempfile=$(mktemp)
```

**6. Use functions for organization:**

```bash
log() { echo "[$(date '+%H:%M:%S')] $*" >&2; }
die() { log "FATAL: $*"; exit 1; }

log "Starting backup"
rsync -a /src/ /dst/ || die "rsync failed"
log "Done"
```

**7. Use `readonly` and `local`:**

```bash
readonly CONFIG_FILE="/etc/myapp.conf"

process_file() {
    local filename="$1"
    local content
    content=$(cat "$filename")
    # $content is scoped to this function
}
```

**8. Check dependencies at the start:**

```bash
for cmd in rsync gzip curl; do
    command -v "$cmd" &>/dev/null || die "$cmd is required but not installed"
done
```

**9. Use `mktemp` for temporary files:**

```bash
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
```

Never use fixed temp paths like `/tmp/myscript.tmp` -- race conditions and security issues.

**10. ShellCheck your scripts:**

```bash
apt install shellcheck
shellcheck myscript.sh
```

ShellCheck catches quoting issues, common pitfalls, and POSIX compatibility problems. It's the single best tool for bash quality.""",
    "flm_shell"
))

# ---------------------------------------------------------------------------
# 20. Bash arrays and loops (short Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SHORT_DEB13,
    r"""How do bash arrays work? Show me arrays with loops.""",
    r"""Bash has indexed arrays and associative arrays.

**Indexed arrays:**

```bash
# Declare and assign
fruits=("apple" "banana" "cherry")

# Or one at a time
fruits[0]="apple"
fruits[1]="banana"
fruits[2]="cherry"

# Append
fruits+=("date" "elderberry")

# Access elements
echo "${fruits[0]}"        # apple
echo "${fruits[-1]}"       # elderberry (last element)

# All elements
echo "${fruits[@]}"        # all elements, separately quoted
echo "${fruits[*]}"        # all elements as single string

# Length
echo "${#fruits[@]}"       # 5

# Slice
echo "${fruits[@]:1:3}"    # banana cherry date
```

**Looping over arrays:**

```bash
# Always quote and use [@]
for fruit in "${fruits[@]}"; do
    echo "Fruit: $fruit"
done

# With index
for i in "${!fruits[@]}"; do
    echo "$i: ${fruits[$i]}"
done
```

**Associative arrays (bash 4+):**

```bash
declare -A colors
colors[red]="#ff0000"
colors[green]="#00ff00"
colors[blue]="#0000ff"

# Or inline
declare -A ports=(
    [ssh]=22
    [http]=80
    [https]=443
)

echo "${ports[ssh]}"       # 22

# Loop over keys
for key in "${!ports[@]}"; do
    echo "$key -> ${ports[$key]}"
done
```

**Practical examples:**

Process a list of servers:

```bash
servers=("web1" "web2" "db1" "db2")
for srv in "${servers[@]}"; do
    echo "Checking $srv..."
    ssh "$srv" uptime
done
```

Build a command with array arguments (avoids quoting nightmares):

```bash
cmd=(rsync -avz --exclude=".cache" --exclude="*.tmp")
cmd+=(--delete)
cmd+=("/home/" "/backup/home/")

# Execute
"${cmd[@]}"
```

Read lines into an array:

```bash
mapfile -t lines < /etc/hosts
echo "Total lines: ${#lines[@]}"
echo "First line: ${lines[0]}"
```

**Common pitfall:** Always use `"${array[@]}"` (double-quoted, `@` not `*`). Without quotes, elements with spaces get split into multiple words.""",
    "flm_shell"
))

# ---------------------------------------------------------------------------
# 21. Bash functions and error handling (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""How do I write proper bash functions with error handling?""",
    r"""**Basic function syntax:**

```bash
# Preferred style
greet() {
    local name="$1"
    echo "Hello, $name"
}

# Also valid (but less common)
function greet {
    local name="$1"
    echo "Hello, $name"
}
```

Always use `local` for variables inside functions to avoid polluting the global scope.

**Return values:**

Bash functions return an exit code (0-255), not a value. To "return" data, use stdout:

```bash
get_ip() {
    hostname -I | awk '{print $1}'
}

my_ip=$(get_ip)
echo "IP is: $my_ip"
```

Use `return` for exit codes:

```bash
is_root() {
    [[ $EUID -eq 0 ]]
}

if is_root; then
    echo "Running as root"
fi
```

**Error handling patterns:**

```bash
# Logging helpers
log()  { echo "[INFO] $*" >&2; }
warn() { echo "[WARN] $*" >&2; }
die()  { echo "[FATAL] $*" >&2; exit 1; }

# Usage
log "Starting process"
some_command || die "some_command failed with code $?"
```

**Trap for cleanup:**

```bash
cleanup() {
    local exit_code=$?
    rm -rf "$tmpdir"
    log "Exiting with code $exit_code"
}

tmpdir=$(mktemp -d)
trap cleanup EXIT   # runs on any exit (normal, error, signal)

# Now do work freely -- cleanup always runs
cp important.dat "$tmpdir/"
process "$tmpdir/important.dat"
```

**Argument validation:**

```bash
backup_file() {
    local src="${1:?ERROR: source file required}"
    local dst="${2:?ERROR: destination required}"

    [[ -f "$src" ]] || { warn "Source not found: $src"; return 1; }
    [[ -d "$(dirname "$dst")" ]] || { warn "Dest dir does not exist"; return 1; }

    cp -a "$src" "$dst"
}
```

The `${1:?message}` syntax exits with an error if the parameter is unset or empty.

**Multiple return values with a subshell array:**

```bash
get_disk_info() {
    local mount="$1"
    local total used pct
    read -r total used pct < <(df -h "$mount" | awk 'NR==2 {print $2, $3, $5}')
    echo "$total $used $pct"
}

read -r total used pct <<< "$(get_disk_info /)"
echo "Root: $total total, $used used ($pct)"
```

**Retry with backoff:**

```bash
retry() {
    local max_attempts="${1:?}"
    shift
    local attempt=1
    while (( attempt <= max_attempts )); do
        if "$@"; then
            return 0
        fi
        warn "Attempt $attempt/$max_attempts failed, retrying in ${attempt}s..."
        sleep "$attempt"
        (( attempt++ ))
    done
    return 1
}

retry 3 curl -sf https://example.com/health || die "Health check failed"
```

These patterns make bash scripts much more robust and maintainable.""",
    "flm_shell"
))

# ---------------------------------------------------------------------------
# 22. getopts for argument parsing (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    r"""How do I use getopts to parse command-line arguments in bash?""",
    r"""getopts is a built-in bash command for parsing short options (-v, -f filename, etc.).

**Basic pattern:**

```bash
#!/bin/bash
set -euo pipefail

usage() {
    cat << 'EOF'
Usage: myscript [-v] [-o output] [-n count] [-h] input_file

Options:
  -v          Verbose mode
  -o FILE     Output file (default: stdout)
  -n COUNT    Number of iterations (default: 1)
  -h          Show this help
EOF
    exit 1
}

# Defaults
verbose=false
output="/dev/stdout"
count=1

while getopts ":vo:n:h" opt; do
    case "$opt" in
        v) verbose=true ;;
        o) output="$OPTARG" ;;
        n) count="$OPTARG" ;;
        h) usage ;;
        :) echo "Error: -$OPTARG requires an argument" >&2; usage ;;
        *) echo "Error: unknown option -$OPTARG" >&2; usage ;;
    esac
done

# Remove parsed options, leaving positional args
shift $((OPTIND - 1))

# Check required positional args
if [[ $# -lt 1 ]]; then
    echo "Error: input_file required" >&2
    usage
fi

input_file="$1"

# Now use the variables
if $verbose; then
    echo "Input: $input_file"
    echo "Output: $output"
    echo "Count: $count"
fi
```

**How the option string works:**

In `":vo:n:h"`:
- Leading `:` -- silent error handling (you handle errors in the `*` and `:` cases)
- `v` -- boolean flag, no argument
- `o:` -- option that takes an argument (the colon after means "requires argument")
- `n:` -- another option with argument
- `h` -- boolean flag

**The key variables:**
- `$opt` -- the current option letter
- `$OPTARG` -- the argument to the current option
- `$OPTIND` -- index of the next argument to process

**Practical example -- a backup script with options:**

```bash
#!/bin/bash
set -euo pipefail

verbose=false
dry_run=false
compress=true
dest="/backup"

while getopts ":vnd:Ch" opt; do
    case "$opt" in
        v) verbose=true ;;
        n) dry_run=true ;;
        d) dest="$OPTARG" ;;
        C) compress=false ;;
        h)
            echo "Usage: $0 [-v] [-n] [-d dest] [-C] source..."
            echo "  -v  verbose  -n  dry run  -d  destination  -C  no compression"
            exit 0
            ;;
        :) echo "Error: -$OPTARG requires an argument" >&2; exit 1 ;;
        *) echo "Error: unknown option -$OPTARG" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

[[ $# -ge 1 ]] || { echo "Error: at least one source required" >&2; exit 1; }

rsync_opts=(-a --delete)
$verbose && rsync_opts+=(-v --progress)
$dry_run && rsync_opts+=(--dry-run)
$compress && rsync_opts+=(-z)

for src in "$@"; do
    echo "Backing up: $src -> $dest/"
    rsync "${rsync_opts[@]}" "$src" "$dest/"
done
```

**Limitations of getopts:**
- Only supports short options (-v, not --verbose)
- For long options, use a `while` loop with `case` manually, or use `getopt` (the external command, with two t's) which supports long options but is less portable.""",
    "flm_shell"
))

# ---------------------------------------------------------------------------
# 23. Useful sysadmin one-liners (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""What are some useful one-liner commands for sysadmins?""",
    r"""Here are practical one-liners I use regularly:

**Disk and files:**

```bash
# Find the 10 largest files on the system
find / -xdev -type f -exec du -h {} + 2>/dev/null | sort -rh | head -10

# Find files modified in the last 24 hours
find /etc -mtime -1 -ls

# Find and delete files older than 30 days
find /var/log -name "*.gz" -mtime +30 -delete

# Disk usage by directory, sorted
du -sh /var/*/ 2>/dev/null | sort -rh
```

**Processes and resources:**

```bash
# Top 10 memory consumers
ps aux --sort=-%mem | head -11

# Top 10 CPU consumers
ps aux --sort=-%cpu | head -11

# Find which process is using a port
ss -tlnp | grep :80

# Watch file descriptors for a process
ls -la /proc/$(pgrep nginx | head -1)/fd | wc -l

# Kill all processes by name
pkill -f "python.*myapp"
```

**Networking:**

```bash
# All listening TCP ports with process names
ss -tlnp

# Check if a remote port is open
timeout 3 bash -c 'echo >/dev/tcp/example.com/443' && echo open || echo closed

# Show bandwidth usage per connection
ss -ti

# Find your public IP
curl -s ifconfig.me
```

**Text processing:**

```bash
# Count occurrences of each HTTP status code in an access log
awk '{print $9}' /var/log/nginx/access.log | sort | uniq -c | sort -rn

# Extract unique IPs from a log
awk '{print $1}' /var/log/nginx/access.log | sort -u

# Show lines that appear in file1 but not file2
comm -23 <(sort file1) <(sort file2)

# Replace text in all files recursively
find . -name "*.conf" -exec sed -i 's/old/new/g' {} +
```

**System information:**

```bash
# Quick system summary
echo "Hostname: $(hostname)" && echo "Uptime: $(uptime -p)" && echo "Kernel: $(uname -r)" && free -h | head -2

# Last 10 users who logged in
last -10

# Failed SSH login attempts
journalctl -u sshd | grep -i "failed" | tail -20

# List all installed packages with sizes
dpkg-query -Wf '${Installed-Size}\t${Package}\n' | sort -rn | head -20
```

**Quick server audit:**

```bash
# All users with login shells
grep -v '/nologin\|/false' /etc/passwd

# All SUID binaries
find / -perm -4000 -type f 2>/dev/null

# All open ports
ss -tulnp

# Cron jobs for all users
for user in $(cut -f1 -d: /etc/passwd); do echo "=== $user ==="; crontab -l -u "$user" 2>/dev/null; done
```

These cover about 80% of daily sysadmin investigation tasks.""",
    "flm_shell"
))

# ---------------------------------------------------------------------------
# 24. Here documents and here strings (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    r"""Explain here documents and here strings in bash.""",
    r"""Here documents (heredocs) and here strings let you pass multi-line or inline text to commands without temporary files.

**Here documents (<<):**

```bash
cat << EOF
Hello, $USER.
Today is $(date).
Your home is $HOME.
EOF
```

This sends the text between `<< EOF` and `EOF` to stdin of the command. Variables and command substitutions are expanded.

**Quoted delimiter -- no expansion:**

```bash
cat << 'EOF'
This is literal: $USER $(date)
No expansion happens here.
EOF
```

Quoting the delimiter (`'EOF'`) prevents variable and command expansion. Use this when you're writing scripts or config files that contain `$` signs.

**Indented heredocs (<<-):**

```bash
if true; then
    cat <<- EOF
	This text can be indented with tabs.
	The <<- strips leading tabs (not spaces).
	EOF
fi
```

Note: only leading **tabs** are stripped, not spaces. This lets you indent heredocs inside functions or if blocks.

**Redirecting heredocs to a file:**

```bash
cat << 'EOF' > /etc/myapp.conf
[server]
port = 8080
host = 0.0.0.0
debug = false
EOF
```

Or with sudo:

```bash
sudo tee /etc/myapp.conf << 'EOF'
[server]
port = 8080
host = 0.0.0.0
EOF
```

**Piping heredocs:**

```bash
cat << 'EOF' | grep "error"
this is fine
this has error in it
this is also fine
EOF
```

**Here strings (<<<):**

A here string passes a single string to stdin:

```bash
grep "pattern" <<< "search in this string"

# Useful for feeding variables to commands
read -r first last <<< "John Doe"
echo "First: $first, Last: $last"

# Convert case
tr '[:lower:]' '[:upper:]' <<< "hello world"
```

**Practical examples:**

Write a systemd unit file:

```bash
sudo tee /etc/systemd/system/myapp.service << 'EOF'
[Unit]
Description=My Application
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/myapp
Restart=always

[Install]
WantedBy=multi-user.target
EOF
```

Embed SQL:

```bash
sudo -u postgres psql << 'EOF'
CREATE DATABASE myapp;
CREATE USER myuser WITH PASSWORD 'secret';
GRANT ALL ON DATABASE myapp TO myuser;
EOF
```

SSH remote commands:

```bash
ssh server1 << 'EOF'
df -h
free -h
uptime
EOF
```

Here documents are one of bash's most useful features for writing config files, running multi-line commands, and avoiding messy echo chains.""",
    "flm_shell"
))

# ---------------------------------------------------------------------------
# 25. Bash parameter expansion (short Arch)
# ---------------------------------------------------------------------------
conversations.append(make(
    SHORT_ARCH,
    r"""What are the most useful bash parameter expansion tricks?""",
    r"""Parameter expansion is powerful and avoids spawning external processes like sed or cut.

**Defaults:**

```bash
# Use default if unset or empty
name="${1:-anonymous}"

# Use default only if unset (empty is OK)
name="${1-anonymous}"

# Assign default if unset or empty
: "${LOG_DIR:=/var/log}"

# Error if unset or empty
filename="${1:?Error: filename required}"
```

**Substrings:**

```bash
str="Hello, World!"
echo "${str:7}"      # World!
echo "${str:7:5}"    # World
echo "${str: -6}"    # orld!  (note the space before -)
```

**String length:**

```bash
echo "${#str}"       # 13
```

**Pattern removal:**

```bash
path="/home/user/docs/file.tar.gz"

# Remove shortest match from front
echo "${path#*/}"    # home/user/docs/file.tar.gz

# Remove longest match from front
echo "${path##*/}"   # file.tar.gz  (basename equivalent)

# Remove shortest match from end
echo "${path%.*}"    # /home/user/docs/file.tar

# Remove longest match from end
echo "${path%%.*}"   # /home/user/docs/file

# Get directory (dirname equivalent)
echo "${path%/*}"    # /home/user/docs

# Get file extension
echo "${path##*.}"   # gz
```

**Substitution:**

```bash
msg="hello world hello"

# Replace first match
echo "${msg/hello/hi}"     # hi world hello

# Replace all matches
echo "${msg//hello/hi}"    # hi world hi

# Replace at beginning
echo "${msg/#hello/hi}"    # hi world hello

# Replace at end
echo "${msg/%hello/hi}"    # hello world hi

# Delete pattern (replace with nothing)
echo "${msg//hello}"       # world
```

**Case conversion (bash 4+):**

```bash
name="hello WORLD"
echo "${name^^}"     # HELLO WORLD (all uppercase)
echo "${name,,}"     # hello world (all lowercase)
echo "${name^}"      # Hello WORLD (first char uppercase)
```

**Practical examples:**

```bash
# Change file extension
file="report.txt"
echo "${file%.txt}.pdf"    # report.pdf

# Strip a prefix
url="https://example.com/page"
echo "${url#https://}"     # example.com/page

# Batch rename files
for f in *.jpeg; do
    mv "$f" "${f%.jpeg}.jpg"
done

# Extract version number
pkg="nginx-1.24.0-1"
echo "${pkg##*-}"          # 1  (oops, too greedy)
# Better:
ver="${pkg#*-}"            # 1.24.0-1
echo "${ver%-*}"           # 1.24.0
```

Parameter expansion is much faster than calling sed, cut, or awk for simple string operations since it runs inside the shell without spawning a subprocess.""",
    "flm_shell"
))

# ---------------------------------------------------------------------------
# 26. Server security audit (multi-turn, Debian minimal)
# ---------------------------------------------------------------------------
conversations.append(make_multi(
    SYS_DEBMIN,
    [
        ("user", r"""I need to do a security audit on this minimal Debian server. Where do I start?"""),
        ("assistant", r"""Good idea. Let's go through a systematic security audit. Start with these checks:

**1. Check for open ports:**

```bash
ss -tulnp
```

This shows all listening TCP and UDP ports with the process names. Look for anything unexpected -- every open port is an attack surface.

**2. Check for unauthorized users:**

```bash
# Users with login shells
grep -v '/nologin\|/false' /etc/passwd

# Users with UID 0 (root privileges)
awk -F: '$3 == 0 {print $1}' /etc/passwd

# Check for empty passwords
sudo awk -F: '($2 == "" || $2 == "!") {print $1}' /etc/shadow
```

**3. Check SSH configuration:**

```bash
sudo sshd -T | grep -E 'permitrootlogin|passwordauthentication|pubkeyauthentication|allowusers|port'
```

Run these and tell me what you see. We'll go through the results and fix any issues."""),
        ("user", r"""OK. ss shows ports 22 (sshd), 80 (nginx), 3306 (mysqld), and 111 (rpcbind). There are two users with login shells: root and admin. No empty passwords. SSH allows root login with password."""),
        ("assistant", r"""Several issues to address:

**Port 111 (rpcbind):** Unless you're using NFS, rpcbind should not be running. It's a common attack vector:

```bash
sudo systemctl stop rpcbind rpcbind.socket
sudo systemctl disable rpcbind rpcbind.socket
sudo apt purge rpcbind
```

**Port 3306 (MySQL) on all interfaces:** MySQL should only listen on localhost unless remote access is explicitly needed:

Check its current binding:

```bash
sudo grep -r bind-address /etc/mysql/
```

Set it to localhost in the config:

```
bind-address = 127.0.0.1
```

**SSH allows root login with password:** Fix this immediately:

```bash
sudo tee -a /etc/ssh/sshd_config.d/hardening.conf << 'EOF'
PermitRootLogin no
PasswordAuthentication no
AllowUsers admin
MaxAuthTries 3
EOF
sudo systemctl reload sshd
```

Make sure the admin user has an SSH key set up before disabling passwords.

Now let's check the firewall and pending updates:

```bash
# Check if a firewall is active
sudo iptables -L -n
sudo nft list ruleset

# Check for pending security updates
sudo apt update && apt list --upgradable 2>/dev/null
```

What do those show?"""),
        ("user", r"""iptables shows all ACCEPT -- no firewall rules at all. There are 12 packages that need updating, including openssl and the kernel."""),
        ("assistant", r"""No firewall and pending security updates -- let's fix both.

**Install and configure a firewall:**

```bash
sudo apt install nftables
sudo systemctl enable nftables
```

Create a basic ruleset:

```bash
sudo tee /etc/nftables.conf << 'EOF'
#!/usr/sbin/nft -f
flush ruleset

table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;

        # Allow established connections
        ct state established,related accept

        # Allow loopback
        iif lo accept

        # Allow SSH
        tcp dport 22 accept

        # Allow HTTP
        tcp dport 80 accept

        # Allow ICMP ping
        ip protocol icmp accept
        ip6 nexthdr icmpv6 accept

        # Log and drop everything else
        log prefix "nftables-drop: " drop
    }

    chain forward {
        type filter hook forward priority 0; policy drop;
    }

    chain output {
        type filter hook output priority 0; policy accept;
    }
}
EOF

sudo nft -f /etc/nftables.conf
```

Notice MySQL (3306) is not in the allowed ports -- it's only accessible from localhost now anyway.

**Apply security updates:**

```bash
sudo apt upgrade -y
```

If the kernel was updated, schedule a reboot.

**Install fail2ban for brute-force protection:**

```bash
sudo apt install fail2ban
sudo systemctl enable --now fail2ban
```

The default config already protects SSH. Check its status:

```bash
sudo fail2ban-client status sshd
```

**Summary of what we fixed:**
1. Removed rpcbind (unnecessary service)
2. Restricted MySQL to localhost
3. Hardened SSH (no root login, key-only auth)
4. Set up nftables firewall (default deny)
5. Applied pending security updates
6. Installed fail2ban

For ongoing security, set up unattended-upgrades for automatic security patches:

```bash
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```"""),
    ],
    "flm_security"
))

# ---------------------------------------------------------------------------
# 27. Set up automated borgbackup (multi-turn, Debian)
# ---------------------------------------------------------------------------
conversations.append(make_multi(
    SYS_DEB13,
    [
        ("user", r"""I want to set up automated backups with borgbackup on Debian. Can you walk me through it?"""),
        ("assistant", r"""Sure. BorgBackup (borg) is excellent -- it does deduplication, compression, and encryption. Let's set it up step by step.

**Install borg:**

```bash
sudo apt install borgbackup
```

**Initialize a repository:**

For a local backup:

```bash
borg init --encryption=repokey /backup/borg-repo
```

For a remote backup (via SSH):

```bash
borg init --encryption=repokey ssh://user@backup-server/backup/borg-repo
```

Choose `repokey` encryption -- the key is stored in the repo but protected by your passphrase. You'll be asked to set a passphrase. **Write it down and store it safely.** Without it, the backups are irrecoverable.

**Export the key for safekeeping:**

```bash
borg key export /backup/borg-repo /root/borg-key-backup.txt
```

Store this file offline (USB drive, printed paper, etc.).

Did you want local or remote backups? And what directories do you want to back up?"""),
        ("user", r"""Local for now, backing up /etc, /home, and /var/www. The repo is at /backup/borg-repo. What's next?"""),
        ("assistant", r"""Good. Let's create a backup script and automate it.

**Create the backup script:**

```bash
sudo tee /usr/local/bin/borg-backup.sh << 'SCRIPT'
#!/bin/bash
set -euo pipefail

export BORG_REPO="/backup/borg-repo"
export BORG_PASSPHRASE="YOUR_PASSPHRASE_HERE"

LOG="/var/log/borg-backup.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Borg backup started: $(date) ==="

# Create the backup
borg create \
    --verbose \
    --stats \
    --compression zstd,3 \
    --exclude-caches \
    --exclude '/home/*/.cache' \
    --exclude '/home/*/.local/share/Trash' \
    --exclude '*.pyc' \
    --exclude '__pycache__' \
    "::$(hostname)-{now:%Y-%m-%d_%H%M}" \
    /etc \
    /home \
    /var/www

# Prune old backups (keep 7 daily, 4 weekly, 6 monthly)
borg prune \
    --verbose \
    --list \
    --keep-daily=7 \
    --keep-weekly=4 \
    --keep-monthly=6

# Free space from pruned archives
borg compact

echo "=== Borg backup finished: $(date) ==="
SCRIPT

sudo chmod 700 /usr/local/bin/borg-backup.sh
```

**Important:** Replace `YOUR_PASSPHRASE_HERE` with your actual passphrase. Set the script to mode 700 so only root can read it (since it contains the passphrase).

A more secure alternative is to use a key file:

```bash
echo "your-passphrase" > /root/.borg-passphrase
chmod 600 /root/.borg-passphrase
```

And in the script use `export BORG_PASSPHRASE=$(cat /root/.borg-passphrase)`.

**Automate with cron:**

```bash
sudo crontab -e
# Add:
0 2 * * * /usr/local/bin/borg-backup.sh 2>&1 | logger -t borg-backup
```

This runs at 2 AM daily.

Now let's test it. Run the script manually:

```bash
sudo /usr/local/bin/borg-backup.sh
```

Then verify the backup exists:

```bash
borg list /backup/borg-repo
```

What does the output look like?"""),
        ("user", r"""It worked. I see one archive listed. How do I restore files from it?"""),
        ("assistant", r"""Great. Restoring is straightforward.

**List archives:**

```bash
export BORG_REPO="/backup/borg-repo"
export BORG_PASSPHRASE="your-passphrase"

borg list
```

**List files in a specific archive:**

```bash
borg list ::myhost-2025-01-15_0200
```

**Restore a single file:**

```bash
cd /tmp
borg extract ::myhost-2025-01-15_0200 home/user/important.txt
# File is extracted to /tmp/home/user/important.txt
```

Note: paths in borg are relative (no leading slash).

**Restore a directory:**

```bash
cd /
borg extract ::myhost-2025-01-15_0200 etc/nginx/
```

**Restore everything:**

```bash
cd /
borg extract ::myhost-2025-01-15_0200
```

**Mount an archive as a filesystem (for browsing):**

```bash
mkdir /mnt/borg
borg mount ::myhost-2025-01-15_0200 /mnt/borg
ls /mnt/borg/
# Browse and copy files normally
cp /mnt/borg/home/user/file.txt /home/user/file.txt
borg umount /mnt/borg
```

**Check archive integrity:**

```bash
borg check /backup/borg-repo
```

Run this periodically (weekly via cron) to verify data integrity.

**Test a full restore** to make sure everything works:

```bash
mkdir /tmp/restore-test
cd /tmp/restore-test
borg extract ::myhost-2025-01-15_0200
diff -r /tmp/restore-test/etc/nginx /etc/nginx
```

**Key things to remember:**
- Back up the borg key (`borg key export`) and store it separately
- Remember your passphrase -- without it the encrypted repo is useless
- Test restores regularly
- The `borg compact` step in the script is important -- without it, deleted archives still use disk space"""),
    ],
    "flm_backup"
))

# ---------------------------------------------------------------------------
# 28. Debug broken bash script (multi-turn, no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make_multi(
    None,
    [
        ("user", r"""My bash script keeps failing. Can you help me debug it?

```bash
#!/bin/bash
for file in $(ls *.log); do
    if [ $file = "error.log" ]; then
        count=`grep -c ERROR $file`
        echo "Found $count errors"
    fi
    lines=$(wc -l $file)
    echo $file has $lines lines
done
```"""),
        ("assistant", r"""This script has several common bash bugs. Let me go through each one:

**Bug 1: Parsing `ls` output**

```bash
# Broken -- fails on filenames with spaces, special chars
for file in $(ls *.log); do

# Fixed -- let the shell glob directly
for file in *.log; do
```

Never parse `ls`. The shell glob `*.log` handles filenames correctly.

**Bug 2: Unquoted variable in test**

```bash
# Broken -- if $file is empty or has spaces, syntax error
if [ $file = "error.log" ]; then

# Fixed -- quote the variable, use [[ ]]
if [[ "$file" = "error.log" ]]; then
```

**Bug 3: Backticks instead of `$()`**

```bash
# Old style -- hard to nest, easy to confuse with quotes
count=`grep -c ERROR $file`

# Modern -- clear, nestable
count=$(grep -c ERROR "$file")
```

Also: `$file` is unquoted inside the backticks. Always quote.

**Bug 4: `wc -l` includes the filename in output**

```bash
# wc -l file.log outputs: "42 file.log"
lines=$(wc -l $file)
echo $file has $lines lines
# Prints: file.log has 42 file.log lines

# Fixed -- pipe to wc so it doesn't print the filename
lines=$(wc -l < "$file")
```

**Bug 5: Unquoted echo with variables**

```bash
# Could break if values contain glob characters
echo $file has $lines lines

# Fixed
echo "$file has $lines lines"
```

Want me to put the fixed version together?"""),
        ("user", r"""Yes please. Also add error handling."""),
        ("assistant", r"""Here's the corrected and improved version:

```bash
#!/bin/bash
set -euo pipefail

# Check that log files exist
shopt -s nullglob
logs=(*.log)
shopt -u nullglob

if [[ ${#logs[@]} -eq 0 ]]; then
    echo "No .log files found in current directory" >&2
    exit 1
fi

for file in "${logs[@]}"; do
    if [[ ! -r "$file" ]]; then
        echo "Warning: cannot read $file, skipping" >&2
        continue
    fi

    if [[ "$file" = "error.log" ]]; then
        count=$(grep -c "ERROR" "$file" || true)
        echo "Found $count errors in $file"
    fi

    lines=$(wc -l < "$file")
    echo "$file has $lines lines"
done
```

**Changes explained:**

1. `set -euo pipefail` -- strict mode, catch errors early.

2. `shopt -s nullglob` -- if no `.log` files exist, the glob expands to nothing instead of the literal string `*.log`. We check for that and exit cleanly.

3. Collect files into an array first, then iterate with proper quoting.

4. `grep -c ... || true` -- with `set -e`, `grep` returning no matches (exit code 1) would kill the script. The `|| true` prevents that.

5. `-r` test -- check the file is readable before processing.

6. All variables are quoted everywhere.

7. `wc -l < "$file"` -- redirect input so wc only outputs the count.

**To debug bash scripts in general:**

```bash
# Run with tracing -- shows every command before execution
bash -x myscript.sh

# Or add to the script temporarily
set -x    # start tracing
# ... problematic section ...
set +x    # stop tracing
```

The trace output shows exactly what each line expands to after variable substitution and globbing, which makes quoting bugs obvious."""),
    ],
    "flm_shell"
))

# ---------------------------------------------------------------------------
# 29. Harden a fresh server (multi-turn, Debian minimal)
# ---------------------------------------------------------------------------
conversations.append(make_multi(
    SYS_DEBMIN,
    [
        ("user", r"""I just installed a fresh minimal Debian server. What should I do to harden it before putting it on the internet?"""),
        ("assistant", r"""Good that you're thinking about security before going live. Here's the priority order:

**Step 1: Update everything**

```bash
sudo apt update && sudo apt upgrade -y
```

If a kernel update is included, plan a reboot.

**Step 2: Create a non-root user and set up SSH keys**

```bash
# Create a user
sudo adduser deploy
sudo usermod -aG sudo deploy

# Set up SSH key (from your local machine)
ssh-copy-id deploy@server-ip

# Test the key login in a new terminal before proceeding
```

**Step 3: Harden SSH**

```bash
sudo tee /etc/ssh/sshd_config.d/hardening.conf << 'EOF'
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AllowUsers deploy
MaxAuthTries 3
LoginGraceTime 30
X11Forwarding no
AllowTcpForwarding no
ClientAliveInterval 300
ClientAliveCountMax 2
EOF

sudo sshd -t && sudo systemctl reload sshd
```

**Test SSH in a new terminal before closing your current session.** If the config is wrong, you'll lock yourself out.

Done with these three? Let me know and we'll continue with firewall and fail2ban."""),
        ("user", r"""Done. SSH is working with keys only. What's next?"""),
        ("assistant", r"""Great. Now firewall and intrusion protection.

**Step 4: Set up nftables firewall**

```bash
sudo apt install nftables
sudo systemctl enable nftables

sudo tee /etc/nftables.conf << 'EOF'
#!/usr/sbin/nft -f
flush ruleset

table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;
        ct state established,related accept
        iif lo accept
        tcp dport 22 accept
        ip protocol icmp accept
        ip6 nexthdr icmpv6 accept
        log prefix "nft-drop: " counter drop
    }
    chain forward {
        type filter hook forward priority 0; policy drop;
    }
    chain output {
        type filter hook output priority 0; policy accept;
    }
}
EOF

sudo nft -f /etc/nftables.conf
```

Add more `tcp dport` lines when you deploy services (80 for HTTP, 443 for HTTPS, etc.).

**Step 5: Install fail2ban**

```bash
sudo apt install fail2ban

sudo tee /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 1h
findtime = 10m
maxretry = 3
banaction = nftables-multiport

[sshd]
enabled = true
port = ssh
EOF

sudo systemctl enable --now fail2ban
sudo fail2ban-client status sshd
```

**Step 6: Automatic security updates**

```bash
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

This automatically installs security patches daily.

Ready for the remaining hardening steps?"""),
        ("user", r"""Yes, keep going."""),
        ("assistant", r"""**Step 7: Harden kernel parameters**

```bash
sudo tee /etc/sysctl.d/99-hardening.conf << 'EOF'
# Disable IP forwarding (unless this is a router)
net.ipv4.ip_forward = 0

# Ignore ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0

# Ignore source-routed packets
net.ipv4.conf.all.accept_source_route = 0

# Enable SYN flood protection
net.ipv4.tcp_syncookies = 1

# Log martian packets
net.ipv4.conf.all.log_martians = 1

# Disable SUID dumpable
fs.suid_dumpable = 0

# Restrict kernel pointer leaks
kernel.kptr_restrict = 2

# Restrict dmesg to root
kernel.dmesg_restrict = 1
EOF

sudo sysctl --system
```

**Step 8: Remove unnecessary packages and services**

```bash
# List running services
systemctl list-units --type=service --state=running

# Disable anything you don't need
sudo systemctl disable --now rpcbind.socket rpcbind 2>/dev/null || true
sudo apt purge rpcbind 2>/dev/null || true
```

**Step 9: Set up basic logging and monitoring**

```bash
# Ensure rsyslog is running
sudo apt install rsyslog
sudo systemctl enable --now rsyslog

# Install logwatch for daily log summaries
sudo apt install logwatch
```

Logwatch emails a daily summary of log activity.

**Step 10: Restrict permissions**

```bash
# Secure cron
sudo chmod 700 /etc/cron.d /etc/cron.daily /etc/cron.hourly /etc/cron.weekly /etc/cron.monthly

# Restrict su to the sudo group
sudo dpkg-statoverride --update --add root sudo 4750 /usr/bin/su

# Set a login banner warning
sudo tee /etc/issue.net << 'EOF'
Authorized access only. All activity is monitored and logged.
EOF
```

**Summary checklist:**
1. System updated
2. Non-root user with SSH key
3. SSH hardened (no root, no passwords)
4. Firewall with default-deny policy
5. fail2ban for brute-force protection
6. Automatic security updates
7. Kernel parameters hardened
8. Unnecessary services removed
9. Logging configured
10. Permissions tightened

This is a solid baseline. Add additional hardening based on what services you deploy (AppArmor profiles, dedicated service users, etc.)."""),
    ],
    "flm_security"
))

# ---------------------------------------------------------------------------
# 30. Investigate security breach (multi-turn, Debian)
# ---------------------------------------------------------------------------
conversations.append(make_multi(
    SYS_DEB13,
    [
        ("user", r"""I think my Debian server might be compromised. Something is using a lot of CPU and I didn't start it. How do I investigate?"""),
        ("assistant", r"""Stay calm and let's investigate methodically. Do NOT reboot yet -- that could destroy evidence.

**Step 1: Identify the suspicious process**

```bash
# What's using CPU?
top -bn1 | head -20

# Full process list with command lines
ps auxww --sort=-%cpu | head -20

# Check for processes with deleted binaries (common for malware)
ls -la /proc/*/exe 2>/dev/null | grep '(deleted)'
```

**Step 2: Check network connections**

```bash
# Active connections -- look for unknown outbound connections
ss -tunap

# Specifically established connections to remote hosts
ss -tunap state established
```

**Step 3: Check for recent logins**

```bash
# Recent logins
last -20

# Failed login attempts
lastb -20 2>/dev/null

# Currently logged in
who
```

Run these and tell me what stands out. Look for:
- Processes you don't recognize
- Connections to unknown IPs
- Logins you didn't make"""),
        ("user", r"""There's a process called kworker2 using 95% CPU running as www-data. It's connecting to 185.234.xx.xx on port 4444. I also see a login from an unknown IP yesterday at 3 AM."""),
        ("assistant", r"""That's a clear compromise. A real kworker is a kernel thread (shown in brackets as [kworker/...]). A process named `kworker2` running as www-data with a remote connection on port 4444 (common reverse shell port) is malware.

**Immediate containment -- block the C2 server:**

```bash
sudo iptables -A OUTPUT -d 185.234.0.0/16 -j DROP
```

**Gather evidence before killing the process:**

```bash
# Get the PID
PID=$(pgrep -f kworker2)

# Capture the binary
sudo cp /proc/$PID/exe /root/evidence/malware_binary 2>/dev/null

# Capture its environment and command line
sudo cat /proc/$PID/cmdline | tr '\0' ' ' > /root/evidence/cmdline.txt
sudo cat /proc/$PID/environ | tr '\0' '\n' > /root/evidence/environ.txt

# Capture open files and network connections
sudo ls -la /proc/$PID/fd/ > /root/evidence/open_fds.txt
sudo cat /proc/$PID/net/tcp > /root/evidence/network.txt

# Capture the memory map
sudo cat /proc/$PID/maps > /root/evidence/maps.txt
```

**Now kill it:**

```bash
sudo kill -9 "$PID"
```

**Check how it persists -- it may come back:**

```bash
# Crontabs
sudo crontab -l
sudo crontab -u www-data -l
sudo ls -la /etc/cron.d/ /etc/cron.daily/ /var/spool/cron/

# Systemd services
sudo find /etc/systemd/ /run/systemd/ ~/.config/systemd/ -name "*.service" -newer /etc/hostname

# Check rc.local and profile scripts
sudo cat /etc/rc.local 2>/dev/null
sudo find /etc/profile.d/ -mtime -7

# Check for modified binaries
sudo debsums -c 2>/dev/null | head -20

# Recently modified files
sudo find / -mtime -2 -not -path '/proc/*' -not -path '/sys/*' -not -path '/run/*' 2>/dev/null | head -50
```

What do you find in the crontabs and recently modified files?"""),
        ("user", r"""Found a crontab entry under www-data that downloads and runs a script every 5 minutes. Also found a PHP webshell in /var/www/html/uploads/."""),
        ("assistant", r"""That's the persistence mechanism and the initial entry point. Let's clean up and secure:

**Remove the persistence:**

```bash
# Remove malicious crontab
sudo crontab -r -u www-data

# Remove the webshell
sudo rm -f /var/www/html/uploads/*.php
# Check for other webshells
sudo find /var/www/ -name "*.php" -newer /var/www/html/index.html -exec grep -l "eval\|base64_decode\|system\|exec\|passthru\|shell_exec" {} \;
```

**Check for backdoor users or SSH keys:**

```bash
# Check for new users
grep -v '/nologin\|/false' /etc/passwd

# Check for unauthorized SSH keys
sudo find / -name "authorized_keys" -exec echo "=== {} ===" \; -exec cat {} \; 2>/dev/null

# Check for SUID binaries that shouldn't be there
sudo find / -perm -4000 -type f 2>/dev/null | sort > /tmp/suid_current.txt
```

**Determine the entry point:**

The webshell in `/uploads/` suggests an unrestricted file upload vulnerability in the web application. Check nginx/Apache logs:

```bash
# Find requests to the webshell
sudo grep "uploads/" /var/log/nginx/access.log* | head -20

# Find the attacker's IP
sudo grep "uploads/.*\.php" /var/log/nginx/access.log* | awk '{print $1}' | sort -u

# Check auth logs for the 3 AM login
sudo journalctl -u sshd --since "yesterday 02:00" --until "yesterday 04:00"
```

**Prevent reinfection:**

```bash
# Block PHP execution in uploads directory (nginx config)
# In your server block:
location ~* /uploads/.*\.php$ {
    deny all;
}

# Or with .htaccess for Apache:
# <Directory /var/www/html/uploads>
#     php_admin_flag engine Off
# </Directory>

# Change www-data's shell to nologin
sudo usermod -s /usr/sbin/nologin www-data

# Update all packages
sudo apt update && sudo apt upgrade -y
```

**Post-incident recommendations:**

1. **Change all passwords** -- server accounts, database, application.
2. **Rotate SSH keys** -- the attacker may have copied private keys.
3. **Check other servers** -- if this server had SSH access to others, assume those are compromised too.
4. **Review application code** -- fix the upload vulnerability.
5. **Consider a full reinstall** -- a compromised system can never be fully trusted. Restore from a clean backup from before the breach date.
6. **Set up monitoring** -- install AIDE for file integrity monitoring, configure auditd, set up log forwarding to a separate server.

The safest course is to rebuild the server from scratch and restore data (not binaries) from a pre-compromise backup."""),
    ],
    "flm_security"
))

# ---------------------------------------------------------------------------
# 31. Password policy with pam_pwquality (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""How do I enforce a strong password policy on Debian?""",
    r"""Use `pam_pwquality` to enforce password complexity and `pam_pwhistory` to prevent reuse.

**Install:**

```bash
sudo apt install libpam-pwquality
```

**Configure password quality in `/etc/security/pwquality.conf`:**

```
# Minimum length
minlen = 12

# Require at least 1 digit
dcredit = -1

# Require at least 1 uppercase
ucredit = -1

# Require at least 1 lowercase
lcredit = -1

# Require at least 1 special character
ocredit = -1

# Maximum consecutive identical characters
maxrepeat = 3

# Maximum consecutive characters from same class
maxclassrepeat = 4

# Reject passwords containing the username
usercheck = 1

# Minimum number of character classes (digits, upper, lower, special)
minclass = 3

# Check against dictionary words
dictcheck = 1

# Check against cracklib dictionary
gecoscheck = 1
```

**Enable in PAM:**

On Debian, `pam_pwquality` is typically already configured in `/etc/pam.d/common-password`. Verify:

```bash
grep pwquality /etc/pam.d/common-password
```

You should see something like:

```
password requisite pam_pwquality.so retry=3
```

If not, add it before the `pam_unix.so` line.

**Prevent password reuse:**

Add to `/etc/pam.d/common-password`, before the `pam_unix.so` line:

```
password required pam_pwhistory.so remember=12 use_authtok
```

This remembers the last 12 passwords and prevents reuse.

**Set password aging:**

Edit `/etc/login.defs`:

```
PASS_MAX_DAYS   90
PASS_MIN_DAYS   1
PASS_WARN_AGE   14
```

For existing users, apply individually:

```bash
sudo chage -M 90 -m 1 -W 14 username

# Check a user's password aging
sudo chage -l username
```

**Lock accounts after failed attempts:**

Add to `/etc/pam.d/common-auth`:

```
auth required pam_faillock.so preauth deny=5 unlock_time=900
auth required pam_faillock.so authfail deny=5 unlock_time=900
```

This locks an account for 15 minutes (900 seconds) after 5 failed attempts.

**Unlock a locked account:**

```bash
sudo faillock --user username --reset
```

**Test your policy:**

```bash
# Try to set a weak password
sudo passwd testuser
# Should be rejected with a message explaining why
```""",
    "flm_security"
))

# ---------------------------------------------------------------------------
# 32. Restrict su access (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""How do I restrict who can use su on Debian?""",
    r"""By default, any user who knows root's password can `su` to root. You can restrict this to specific users.

**Method 1: Require membership in the sudo (or wheel) group**

Edit `/etc/pam.d/su` and uncomment or add this line:

```
auth required pam_wheel.so group=sudo
```

Now only members of the `sudo` group can use `su`:

```bash
# Add a user to the sudo group
sudo usermod -aG sudo username

# Verify group membership
groups username
```

Users not in the sudo group will get "Permission denied" even if they know root's password.

**Method 2: Restrict the su binary itself**

Change the permissions so only the sudo group can execute it:

```bash
sudo dpkg-statoverride --update --add root sudo 4750 /usr/bin/su
```

This sets su as owned by root:sudo with setuid and group-execute only (rwsr-x---). Users not in the sudo group can't even run the binary.

To verify:

```bash
ls -la /usr/bin/su
# Should show: -rwsr-x--- 1 root sudo ... /usr/bin/su
```

**Method 3: Limit su to specific users (stricter)**

Create a dedicated group:

```bash
sudo groupadd suusers
sudo usermod -aG suusers admin
```

Then in `/etc/pam.d/su`:

```
auth required pam_wheel.so group=suusers
```

And restrict the binary:

```bash
sudo dpkg-statoverride --update --add root suusers 4750 /usr/bin/su
```

**Why restrict su?**

- If an attacker gains access to any user account and knows (or cracks) root's password, they can escalate to root.
- With `pam_wheel`, they also need to be in the right group.
- Using `sudo` with per-user policies is generally better than `su` for privilege escalation, because it provides logging and granular control.

**Verify it works:**

```bash
# As a user NOT in the allowed group
su -
# Should fail with "Permission denied"

# As a user IN the allowed group
su -
# Should prompt for root's password and succeed
```""",
    "flm_security"
))

# ---------------------------------------------------------------------------
# 33. Disable unnecessary services (Debian minimal)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEBMIN,
    r"""How do I find and disable unnecessary services on a minimal Debian server?""",
    r"""Every running service is a potential attack surface. On a minimal server, you should only run what you need.

**List all running services:**

```bash
systemctl list-units --type=service --state=running
```

**List services enabled at boot:**

```bash
systemctl list-unit-files --type=service --state=enabled
```

**List all listening ports and their services:**

```bash
ss -tulnp
```

**Common services to disable on a minimal server:**

```bash
# rpcbind -- only needed for NFS
sudo systemctl disable --now rpcbind.socket rpcbind 2>/dev/null
sudo apt purge rpcbind 2>/dev/null

# avahi-daemon -- mDNS/Bonjour, not needed on a server
sudo systemctl disable --now avahi-daemon.socket avahi-daemon 2>/dev/null
sudo apt purge avahi-daemon 2>/dev/null

# cups / cups-browsed -- printing, not needed on a server
sudo systemctl disable --now cups cups-browsed 2>/dev/null
sudo apt purge cups 2>/dev/null

# ModemManager -- modem support
sudo systemctl disable --now ModemManager 2>/dev/null
sudo apt purge modemmanager 2>/dev/null

# bluetooth
sudo systemctl disable --now bluetooth 2>/dev/null

# wpa_supplicant -- WiFi, not needed on a wired server
sudo systemctl disable --now wpa_supplicant 2>/dev/null
```

**Be careful with these -- only disable if you're sure:**

```bash
# systemd-resolved -- DNS resolver. Disable only if you configure /etc/resolv.conf manually
# NetworkManager -- disable on servers using systemd-networkd or /etc/network/interfaces
# systemd-timesyncd -- keep this for NTP (clock accuracy)
```

**Check for unnecessary packages:**

```bash
# Show installed packages by size
dpkg-query -Wf '${Installed-Size}\t${Package}\n' | sort -rn | head -30

# Find packages that aren't depended on by anything
apt-mark showmanual | sort
```

**After disabling services, verify:**

```bash
# Only expected ports should be open
ss -tulnp

# Only expected services should be running
systemctl list-units --type=service --state=running
```

**Document what you keep and why:**

A good practice is to create a file listing your intended services:

```bash
cat << 'EOF' > /root/expected-services.txt
# Services that should be running on this server
sshd        - remote administration
nginx       - web server
postgresql  - database
cron        - scheduled tasks
fail2ban    - intrusion prevention
nftables    - firewall
EOF
```

This helps during future audits -- you can compare against the expected list.""",
    "flm_security"
))

# ---------------------------------------------------------------------------
# 34. Verify package integrity with debsums (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    r"""How do I check if any installed packages have been tampered with on Debian?""",
    r"""`debsums` verifies installed files against their MD5 checksums from the package metadata.

**Install:**

```bash
sudo apt install debsums
```

**Check all installed packages:**

```bash
sudo debsums -c
```

The `-c` flag only reports files that have changed. If nothing is tampered with, there's no output.

**Check a specific package:**

```bash
debsums openssh-server
```

Output shows OK for matching files or FAILED for mismatches.

**What changed files mean:**

Not all changes indicate tampering. Configuration files are expected to change:

```bash
# Show only non-config file changes (more suspicious)
sudo debsums -c --no-md5sums 2>/dev/null

# Or explicitly check only non-config files
sudo debsums -e
```

- `/etc/` files changed: usually legitimate (you edited the config)
- `/usr/bin/` or `/usr/lib/` files changed: suspicious, could indicate compromise

**For files without stored checksums:**

Some packages don't ship md5sums. Generate them:

```bash
sudo debsums --generate=all
```

**Verify a specific binary against the package:**

If you suspect a specific binary has been replaced:

```bash
# Which package provides this file?
dpkg -S /usr/bin/ssh

# Reinstall that package to restore the original
sudo apt install --reinstall openssh-client

# Verify again
debsums openssh-client
```

**Compare with the online package:**

For extra paranoia, download the package from the repo and compare:

```bash
apt download openssh-server
dpkg-deb -c openssh-server_*.deb | grep usr/sbin/sshd
# Extract and compare checksums
```

**Automate regular checks:**

```bash
sudo tee /etc/cron.weekly/debsums-check << 'SCRIPT'
#!/bin/bash
output=$(debsums -c 2>&1)
if [[ -n "$output" ]]; then
    echo "debsums found changed files on $(hostname):"
    echo "$output"
fi | mail -s "debsums alert: $(hostname)" root
SCRIPT
sudo chmod 755 /etc/cron.weekly/debsums-check
```

**Limitations:**
- debsums uses MD5, which is cryptographically weak. A sophisticated attacker could forge matching MD5s.
- For stronger integrity checking, use AIDE (which supports SHA-256) as a complement.
- debsums can't detect new files added by an attacker -- only modifications to existing package files.""",
    "flm_security"
))

# ---------------------------------------------------------------------------
# 35. Network segmentation basics (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    r"""Explain network segmentation basics for server security.""",
    r"""Network segmentation divides your network into isolated zones, limiting what an attacker can reach if they compromise one system.

**Why segment?**

Without segmentation, a compromised web server can directly access your database server, internal services, and everything else on the network. Segmentation limits blast radius.

**Common zones:**

- **DMZ (Demilitarized Zone)** -- public-facing servers (web servers, mail gateways). Can be reached from the internet but has limited access to internal networks.
- **Internal / Application** -- application servers, APIs. Not directly internet-accessible. Only the DMZ can talk to them on specific ports.
- **Database** -- database servers. Only application servers can reach them.
- **Management** -- SSH jump hosts, monitoring, backup servers. Restricted access.

**Implementation approaches:**

**1. VLANs (physical or virtual network separation):**

```
VLAN 10: DMZ         (192.168.10.0/24)
VLAN 20: Application (192.168.20.0/24)
VLAN 30: Database    (192.168.30.0/24)
VLAN 40: Management  (192.168.40.0/24)
```

Traffic between VLANs goes through a firewall/router where you control the rules.

**2. Host-based firewalls (nftables/iptables on each server):**

On the database server, only allow connections from the application network:

```bash
# Database server nftables
nft add rule inet filter input ip saddr 192.168.20.0/24 tcp dport 5432 accept
nft add rule inet filter input tcp dport 5432 drop
```

On the web server, only allow outbound to the application tier:

```bash
# Web server -- restrict outbound
nft add rule inet filter output ip daddr 192.168.20.0/24 tcp dport 8080 accept
nft add rule inet filter output tcp dport 5432 drop  # no direct DB access
```

**3. For cloud environments** -- use security groups or VPC network policies:

```
Web SG:     inbound 80/443 from 0.0.0.0/0, outbound to App SG on 8080
App SG:     inbound 8080 from Web SG only, outbound to DB SG on 5432
DB SG:      inbound 5432 from App SG only
Mgmt SG:    inbound 22 from VPN only
```

**Key principles:**

1. **Default deny** -- block everything, then explicitly allow only needed traffic.
2. **Least privilege** -- each zone can only reach what it needs.
3. **No direct internet to internal** -- public traffic hits the DMZ, which proxies to internal services.
4. **Separate management access** -- SSH/admin interfaces on a separate network or through a jump host.
5. **Monitor inter-zone traffic** -- log firewall denials to detect lateral movement attempts.

**Simple example for a small setup:**

Even with just two servers (web + database), you can segment:

```
Internet -> [Firewall] -> Web Server (public IP, port 80/443)
                              |
                              v (private network, port 5432 only)
                          Database Server (no public IP)
```

The database has no public IP and only accepts PostgreSQL connections from the web server's private IP.

Network segmentation is one of the most effective security measures. It turns a single compromise from a catastrophe into a contained incident.""",
    "flm_security"
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
