#!/usr/bin/env python3
"""Write hand-crafted SFT conversations for V4 training -- Batch 7.

Topic: OS-SPECIFIC VARIANTS of common tasks.
The SAME question asked on different OSes gets DIFFERENT answers.
This teaches flm to be OS-aware and adapt its answers to the system prompt.

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


# -- Conversations ------------------------------------------------------------

conversations = []

# ===========================================================================
# SET 1 -- "How do I install Python 3 and pip?"
# ===========================================================================

# 1. Debian 13
conversations.append(make(
    SYS_DEB13,
    "How do I install Python 3 and pip?",
    r"""Python 3 is already installed on Debian 13 (it's a core dependency), but pip and venv may not be:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

Check the versions:
```bash
python3 --version    # 3.12.8
pip3 --version
```

**Important:** On Debian 13, PEP 668 marks the system Python as "externally managed." You can't do `pip install` globally without `--break-system-packages`. The right approach is to use a virtual environment:

```bash
python3 -m venv ~/myproject/venv
source ~/myproject/venv/bin/activate
pip install requests flask  # works freely inside the venv
```

This keeps your system Python clean and avoids conflicts with apt-managed packages.""",
    "flm_os_variants"
))

# 2. Arch
conversations.append(make(
    SYS_ARCH,
    "How do I install Python 3 and pip?",
    r"""On Arch, Python 3 is just called `python` (there's no Python 2 in the repos anymore):

```bash
sudo pacman -S python python-pip
```

Check the versions:
```bash
python --version    # 3.12.7
pip --version
```

Arch also enforces PEP 668, so use a virtual environment for project dependencies:

```bash
python -m venv ~/myproject/venv
source ~/myproject/venv/bin/activate
pip install requests flask
```

Note: on Arch, the binary is `python` and `pip`, not `python3` and `pip3`. Both names work, but `python` is the convention here.""",
    "flm_os_variants"
))

# 3. Fedora 41
conversations.append(make(
    SYS_FED41,
    "How do I install Python 3 and pip?",
    r"""Python 3 comes preinstalled on Fedora 41. Install pip if it's not already there:

```bash
sudo dnf install python3 python3-pip
```

Check versions:
```bash
python3 --version    # 3.13.0
pip3 --version
```

Fedora also enforces PEP 668. Use a virtual environment for project work:

```bash
python3 -m venv ~/myproject/venv
source ~/myproject/venv/bin/activate
pip install requests flask
```

Fedora ships Python 3.13, which is quite new. If you need an older version for compatibility, you can install it alongside:

```bash
sudo dnf install python3.12
python3.12 -m venv ~/myproject/venv
```""",
    "flm_os_variants"
))

# 4. Ubuntu 22.04
conversations.append(make(
    SYS_UBU22,
    "How do I install Python 3 and pip?",
    r"""Python 3 is preinstalled on Ubuntu 22.04:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

Check versions:
```bash
python3 --version    # 3.10.12
pip3 --version
```

Ubuntu 22.04 introduced PEP 668, which marks the system Python as externally managed. If you try `pip install` globally you'll get an error. You have two options:

**Option 1 (recommended): Use a virtual environment:**
```bash
python3 -m venv ~/myproject/venv
source ~/myproject/venv/bin/activate
pip install requests flask
```

**Option 2 (not recommended): Override the restriction:**
```bash
pip install --break-system-packages requests
```

This bypasses the protection but can break system tools that depend on specific Python package versions. Stick with venvs.

If you need a newer Python than 3.10, use the deadsnakes PPA:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv
```""",
    "flm_os_variants"
))

# ===========================================================================
# SET 2 -- "How do I install and start a web server?"
# ===========================================================================

# 5. Debian 13
conversations.append(make(
    SYS_DEB13,
    "How do I install and start a web server?",
    r"""**Install nginx:**
```bash
sudo apt update
sudo apt install nginx
```

**Start and enable it:**
```bash
sudo systemctl start nginx
sudo systemctl enable nginx
```

**Verify it's running:**
```bash
systemctl status nginx
curl http://localhost
```

You should see the default nginx welcome page. The document root is `/var/www/html/`.

**Configuration:**
- Main config: `/etc/nginx/nginx.conf`
- Site configs: `/etc/nginx/sites-available/` (symlinked to `sites-enabled/`)
- Default site: `/etc/nginx/sites-enabled/default`

To add a new site:
```bash
sudo nano /etc/nginx/sites-available/mysite
sudo ln -s /etc/nginx/sites-available/mysite /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

Debian uses the `sites-available`/`sites-enabled` pattern with symlinks. This is a Debian-specific layout -- upstream nginx doesn't use it.""",
    "flm_os_variants"
))

# 6. Arch
conversations.append(make(
    SYS_ARCH,
    "How do I install and start a web server?",
    r"""**Install nginx:**
```bash
sudo pacman -S nginx
```

**Start and enable it:**
```bash
sudo systemctl start nginx
sudo systemctl enable nginx
```

**Verify:**
```bash
systemctl status nginx
curl http://localhost
```

**Configuration:**
- Main config: `/etc/nginx/nginx.conf`
- Arch uses a single config file by default -- there's no `sites-available`/`sites-enabled` directory structure
- Document root: `/usr/share/nginx/html/`

To serve your own site, edit `/etc/nginx/nginx.conf` directly or create include files:

```bash
# Add to the http block in /etc/nginx/nginx.conf:
include /etc/nginx/conf.d/*.conf;
```

Then create `/etc/nginx/conf.d/mysite.conf`:
```nginx
server {
    listen 80;
    server_name mysite.example.com;
    root /srv/http/mysite;
    index index.html;
}
```

```bash
sudo mkdir -p /srv/http/mysite
sudo nginx -t
sudo systemctl reload nginx
```

On Arch, the default document root is `/usr/share/nginx/html/`, not `/var/www/html/` like on Debian.""",
    "flm_os_variants"
))

# 7. Fedora 41
conversations.append(make(
    SYS_FED41,
    "How do I install and start a web server?",
    r"""**Install nginx:**
```bash
sudo dnf install nginx
```

**Start and enable it:**
```bash
sudo systemctl start nginx
sudo systemctl enable nginx
```

**Open the firewall:**
Fedora enables firewalld by default, so you need to allow HTTP traffic:
```bash
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

**Verify:**
```bash
systemctl status nginx
curl http://localhost
```

**Configuration:**
- Main config: `/etc/nginx/nginx.conf`
- Additional configs: `/etc/nginx/conf.d/*.conf`
- Document root: `/usr/share/nginx/html/`
- Fedora uses the `conf.d/` directory pattern (no `sites-available`)

**SELinux note:** Fedora has SELinux enforcing by default. If nginx can't read files from a custom directory:
```bash
sudo semanage fcontext -a -t httpd_sys_content_t "/srv/www(/.*)?"
sudo restorecon -Rv /srv/www
```

Or if nginx needs to proxy to a backend:
```bash
sudo setsebool -P httpd_can_network_connect 1
```""",
    "flm_os_variants"
))

# 8. Ubuntu 24.04
conversations.append(make(
    SYS_UBU24,
    "How do I install and start a web server?",
    r"""**Install nginx:**
```bash
sudo apt update
sudo apt install nginx
```

**It starts automatically on Ubuntu.** Verify:
```bash
systemctl status nginx
curl http://localhost
```

**Open the firewall (if ufw is active):**
```bash
sudo ufw allow 'Nginx Full'
sudo ufw status
```

`Nginx Full` allows both HTTP (80) and HTTPS (443). Use `Nginx HTTP` for port 80 only.

**Configuration:**
- Main config: `/etc/nginx/nginx.conf`
- Site configs: `/etc/nginx/sites-available/` (symlinked to `sites-enabled/`)
- Document root: `/var/www/html/`

Ubuntu uses the same `sites-available`/`sites-enabled` layout as Debian:
```bash
sudo nano /etc/nginx/sites-available/mysite
sudo ln -s /etc/nginx/sites-available/mysite /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

Ubuntu may also have Apache installed by default on some server images. Check with:
```bash
systemctl status apache2
```
If it's running and you want nginx instead, stop and disable it first:
```bash
sudo systemctl stop apache2
sudo systemctl disable apache2
```""",
    "flm_os_variants"
))

# ===========================================================================
# SET 3 -- "How do I check which packages are installed?"
# ===========================================================================

# 9. Debian
conversations.append(make(
    SYS_DEB13,
    "How do I check which packages are installed?",
    r"""**List all installed packages:**
```bash
dpkg -l
```

This shows the full list with status, version, and description. The `ii` at the start means "installed."

**Search for a specific package:**
```bash
dpkg -l | grep nginx
# or
apt list --installed 2>/dev/null | grep nginx
```

**Check if a specific package is installed:**
```bash
dpkg -s nginx
```

**Count installed packages:**
```bash
dpkg -l | grep '^ii' | wc -l
```

**List with apt (cleaner output):**
```bash
apt list --installed
```

**Show package details:**
```bash
apt show nginx
```

**List files installed by a package:**
```bash
dpkg -L nginx
```""",
    "flm_os_variants"
))

# 10. Arch
conversations.append(make(
    SYS_ARCH,
    "How do I check which packages are installed?",
    r"""**List all installed packages:**
```bash
pacman -Q
```

**Search installed packages by name:**
```bash
pacman -Qs nginx
```

**Check if a specific package is installed:**
```bash
pacman -Qi nginx
```

This shows detailed info: version, size, dependencies, install reason, etc.

**List only explicitly installed packages (not dependencies):**
```bash
pacman -Qe
```

**List only packages installed as dependencies:**
```bash
pacman -Qd
```

**Count installed packages:**
```bash
pacman -Q | wc -l
```

**List foreign packages (AUR or manually installed):**
```bash
pacman -Qm
```

**List files owned by a package:**
```bash
pacman -Ql nginx
```

**List packages that need updating:**
```bash
checkupdates
```""",
    "flm_os_variants"
))

# 11. Fedora
conversations.append(make(
    SYS_FED41,
    "How do I check which packages are installed?",
    r"""**List all installed packages:**
```bash
dnf list installed
```

**Search for a specific installed package:**
```bash
dnf list installed | grep nginx
# or
rpm -qa | grep nginx
```

**Check if a specific package is installed:**
```bash
rpm -qi nginx
```

**Count installed packages:**
```bash
rpm -qa | wc -l
```

**List with just names and versions:**
```bash
rpm -qa --qf '%{NAME}-%{VERSION}\n' | sort
```

**List files installed by a package:**
```bash
rpm -ql nginx
```

**Show package info with dnf:**
```bash
dnf info nginx
```

**List recently installed packages:**
```bash
dnf history list
dnf history info last
```""",
    "flm_os_variants"
))

# ===========================================================================
# SET 4 -- "How do I find which package provides a file?"
# ===========================================================================

# 12. Debian
conversations.append(make(
    SYS_DEB13,
    "How do I find which package provides a file?",
    r"""**For files already on the system** (from an installed package):
```bash
dpkg -S /usr/bin/curl
# Output: curl: /usr/bin/curl
```

You can also search by partial path:
```bash
dpkg -S bin/curl
```

**For files NOT yet installed** (search all packages in the repos):
```bash
sudo apt install apt-file
sudo apt-file update
apt-file search /usr/bin/curl
```

`apt-file` downloads the full file list from the repos, so it can find files in packages you haven't installed.

**Search by filename pattern:**
```bash
apt-file search bin/convert
dpkg -S '*.so' | grep libssl
```

This is useful when you get "command not found" and need to know which package to install.""",
    "flm_os_variants"
))

# 13. Arch
conversations.append(make(
    SYS_ARCH,
    "How do I find which package provides a file?",
    r"""**For files from installed packages:**
```bash
pacman -Qo /usr/bin/curl
# Output: /usr/bin/curl is owned by curl 8.x.x-1
```

**For files from any package in the repos (including uninstalled):**

Install `pkgfile`:
```bash
sudo pacman -S pkgfile
sudo pkgfile --update
```

Then search:
```bash
pkgfile curl
# Shows which packages contain a file named "curl"

pkgfile -l curl
# Lists all files in the curl package
```

**Using pacman's built-in file database (since pacman 5.1):**
```bash
# Update the file database
sudo pacman -Fy

# Search for a file
pacman -F curl
pacman -F /usr/bin/curl
```

`pacman -F` is built-in and doesn't need extra packages, but `pkgfile` is often faster and more flexible.""",
    "flm_os_variants"
))

# 14. Fedora
conversations.append(make(
    SYS_FED41,
    "How do I find which package provides a file?",
    r"""**For installed files:**
```bash
rpm -qf /usr/bin/curl
# Output: curl-8.x.x-1.fc41.x86_64
```

**For any file (including uninstalled packages):**
```bash
dnf provides /usr/bin/curl
dnf provides '*/bin/curl'
```

`dnf provides` searches the repo metadata, so it works for packages you haven't installed.

**Search by command name:**
```bash
dnf provides curl
```

**Wildcard search:**
```bash
dnf provides '*/libssl.so*'
```

**List files in an installed package:**
```bash
rpm -ql curl
```

**List files in an uninstalled package:**
```bash
dnf repoquery -l curl
```

`dnf provides` is one of the more convenient package search tools across distros -- no extra packages or database updates needed.""",
    "flm_os_variants"
))

# ===========================================================================
# SET 5 -- "How do I clean up old packages and cache?"
# ===========================================================================

# 15. Debian
conversations.append(make(
    SYS_DEB13,
    "How do I clean up old packages and cache?",
    r"""**Remove packages that were installed as dependencies but are no longer needed:**
```bash
sudo apt autoremove
```

**Clean the local package cache:**
```bash
# Remove all cached .deb files
sudo apt clean

# Remove only outdated cached .deb files (keep current versions)
sudo apt autoclean
```

The cache lives in `/var/cache/apt/archives/`. Check how much space it's using:
```bash
du -sh /var/cache/apt/archives/
```

**Remove old kernels (if not auto-removed):**
```bash
dpkg -l 'linux-image-*' | grep '^ii'
sudo apt autoremove --purge
```

**Remove config files from removed packages:**
```bash
dpkg -l | grep '^rc' | awk '{print $2}' | xargs sudo dpkg --purge
```

Packages with status `rc` are removed but have leftover config files. The command above purges them.

**Clean up old logs:**
```bash
sudo journalctl --vacuum-time=7d
sudo journalctl --vacuum-size=100M
```""",
    "flm_os_variants"
))

# 16. Arch
conversations.append(make(
    SYS_ARCH,
    "How do I clean up old packages and cache?",
    r"""**Clean the pacman cache:**
```bash
# Remove all cached packages except the 3 most recent versions
sudo paccache -r

# Remove all cached packages except the latest version
sudo paccache -rk1

# Remove all cached versions of uninstalled packages
sudo pacman -Sc
```

`paccache` is from the `pacman-contrib` package:
```bash
sudo pacman -S pacman-contrib
```

**Find and remove orphan packages** (installed as dependencies but no longer needed):
```bash
# List orphans
pacman -Qdtq

# Remove orphans
sudo pacman -Rns $(pacman -Qdtq)
```

If there are no orphans, `pacman -Qdtq` outputs nothing and the remove command will fail harmlessly.

**Check cache size:**
```bash
du -sh /var/cache/pacman/pkg/
```

**Set up automatic cache cleaning** with a pacman hook:

Create `/etc/pacman.d/hooks/clean_cache.hook`:
```ini
[Trigger]
Operation = Upgrade
Operation = Install
Operation = Remove
Type = Package
Target = *

[Action]
Description = Cleaning pacman cache...
When = PostTransaction
Exec = /usr/bin/paccache -r
```

This automatically trims the cache after every pacman transaction.

**Clean up old journal logs:**
```bash
sudo journalctl --vacuum-time=7d
```""",
    "flm_os_variants"
))

# 17. Fedora
conversations.append(make(
    SYS_FED41,
    "How do I clean up old packages and cache?",
    r"""**Remove unneeded dependencies:**
```bash
sudo dnf autoremove
```

**Clean the package cache:**
```bash
# Remove all cached data (packages, metadata, etc.)
sudo dnf clean all

# Remove only cached packages
sudo dnf clean packages

# Remove only metadata cache
sudo dnf clean metadata
```

The cache lives in `/var/cache/dnf/`. Check its size:
```bash
du -sh /var/cache/dnf/
```

**Remove old kernels:**
Fedora keeps the 3 most recent kernels by default. To keep fewer:

Edit `/etc/dnf/dnf.conf`:
```ini
installonly_limit=2
```

Or manually remove old kernels:
```bash
dnf repoquery --installonly --latest-limit=-1 | xargs sudo dnf remove -y
```

**List duplicate and leftover packages:**
```bash
dnf repoquery --duplicates
```

**Clean journal logs:**
```bash
sudo journalctl --vacuum-time=7d
sudo journalctl --vacuum-size=200M
```

**Check disk usage by package:**
```bash
rpm -qa --qf '%{SIZE}\t%{NAME}\n' | sort -rn | head -20
```""",
    "flm_os_variants"
))

# 18. Ubuntu 24.04
conversations.append(make(
    SYS_UBU24,
    "How do I clean up old packages and cache?",
    r"""**Remove unneeded dependencies:**
```bash
sudo apt autoremove
```

**Clean the apt cache:**
```bash
sudo apt clean       # remove all cached .deb files
sudo apt autoclean   # remove only outdated cached .deb files
```

**Remove residual config files:**
```bash
dpkg -l | grep '^rc' | awk '{print $2}' | xargs sudo dpkg --purge
```

**Clean up old kernels:**
```bash
sudo apt autoremove --purge
```

**Clean up snap packages:**
Ubuntu uses snaps heavily. Old snap revisions accumulate:
```bash
# List snap revisions
snap list --all

# Remove disabled (old) snap revisions
snap list --all | awk '/disabled/{print $1, $3}' | while read snapname revision; do
    sudo snap remove "$snapname" --revision="$revision"
done
```

**Reduce snap retention** (default keeps 3 revisions):
```bash
sudo snap set system refresh.retain=2
```

**Clean snap cache:**
```bash
sudo rm -rf /var/lib/snapd/cache/*
```

**Clean journal logs:**
```bash
sudo journalctl --vacuum-time=7d
```

**Check overall disk usage:**
```bash
df -h
du -sh /var/cache/apt/archives/ /var/lib/snapd/ /var/log/
```

On Ubuntu, snap cleanup is often the biggest space saver after the initial apt cleanup.""",
    "flm_os_variants"
))

# ===========================================================================
# SET 6 -- "How do I add a user to the sudo/admin group?"
# ===========================================================================

# 19. Debian
conversations.append(make(
    SYS_DEB13,
    "How do I add a user to the sudo group?",
    r"""On Debian, the sudo group is called `sudo`:

```bash
sudo usermod -aG sudo alice
```

The `-aG` means "append to group" -- without `-a`, usermod would replace all secondary groups.

The user needs to log out and back in for the new group membership to take effect. Verify with:
```bash
groups alice
# or after they log in:
id alice
```

**Check that sudo is configured for the group:**
```bash
sudo grep '%sudo' /etc/sudoers
# Should show: %sudo   ALL=(ALL:ALL) ALL
```

This is already set up by default on Debian if you installed sudo during system setup.

**If sudo isn't installed** (e.g., minimal install):
```bash
su -
apt install sudo
usermod -aG sudo alice
```""",
    "flm_os_variants"
))

# 20. Arch
conversations.append(make(
    SYS_ARCH,
    "How do I add a user to the sudo group?",
    r"""On Arch, the admin group is called `wheel`, not `sudo`:

```bash
sudo usermod -aG wheel alice
```

**Important:** Arch doesn't enable the wheel group in sudoers by default. You need to uncomment it:

```bash
sudo EDITOR=vim visudo
```

Find and uncomment this line:
```
# %wheel ALL=(ALL:ALL) ALL
```

Remove the `#` so it reads:
```
%wheel ALL=(ALL:ALL) ALL
```

**Always use `visudo`** to edit sudoers -- it validates the syntax before saving and prevents you from locking yourself out.

After the change, the user needs to log out and back in:
```bash
groups alice
```

**Alternative: allow wheel group without password (not recommended):**
Uncomment this line instead:
```
# %wheel ALL=(ALL:ALL) NOPASSWD: ALL
```

This is convenient but less secure -- use it only on personal machines, not servers.""",
    "flm_os_variants"
))

# 21. Fedora
conversations.append(make(
    SYS_FED41,
    "How do I add a user to the sudo group?",
    r"""On Fedora, the admin group is called `wheel`:

```bash
sudo usermod -aG wheel alice
```

Fedora enables the wheel group in sudoers by default, so no extra configuration is needed.

The user needs to log out and back in for the change to take effect:
```bash
groups alice
id alice
```

**Verify the sudoers configuration:**
```bash
sudo grep '%wheel' /etc/sudoers
# Should show: %wheel  ALL=(ALL)       ALL
```

**If you're using the Fedora Workstation** with GNOME, the user will also appear in Settings > Users as an administrator.

**To remove sudo access:**
```bash
sudo gpasswd -d alice wheel
```""",
    "flm_os_variants"
))

# ===========================================================================
# SET 7 -- "How do I set up automatic updates?"
# ===========================================================================

# 22. Debian
conversations.append(make(
    SYS_DEB13,
    "How do I set up automatic updates?",
    r"""**Install unattended-upgrades:**
```bash
sudo apt install unattended-upgrades apt-listchanges
```

**Enable it:**
```bash
sudo dpkg-reconfigure -plow unattended-upgrades
```

Select "Yes" when asked.

**Configure** in `/etc/apt/apt.conf.d/50unattended-upgrades`:
```
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
    "${distro_id}:${distro_codename}";
};

Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "true";
Unattended-Upgrade::Automatic-Reboot-Time "03:00";
```

**The update schedule** is in `/etc/apt/apt.conf.d/20auto-upgrades`:
```
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
```

The "1" means daily.

**Test with a dry run:**
```bash
sudo unattended-upgrades --dry-run --debug
```

**Check logs:**
```bash
cat /var/log/unattended-upgrades/unattended-upgrades.log
```

Best practice: enable auto security updates but test full upgrades manually first.""",
    "flm_os_variants"
))

# 23. Arch
conversations.append(make(
    SYS_ARCH,
    "How do I set up automatic updates?",
    r"""**Short answer: you probably shouldn't.** Arch is a rolling release distro. Updates can require manual intervention (reading Arch news, handling .pacnew files, etc.). Unattended upgrades on Arch can break your system.

**What you should do instead: check for updates regularly.**

Install `checkupdates` (from `pacman-contrib`):
```bash
sudo pacman -S pacman-contrib
```

**Check for available updates without installing:**
```bash
checkupdates
```

**Set up a notification** with a systemd timer:

Create `/etc/systemd/system/checkupdates.service`:
```ini
[Unit]
Description=Check for pacman updates

[Service]
Type=oneshot
ExecStart=/bin/bash -c '/usr/bin/checkupdates > /tmp/pacman-updates.txt 2>&1 || true'
User=nobody
```

Create `/etc/systemd/system/checkupdates.timer`:
```ini
[Unit]
Description=Check for pacman updates daily

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl enable --now checkupdates.timer
```

Then check `/tmp/pacman-updates.txt` or add a script that sends you a notification.

**When you do update, always:**
1. Check https://archlinux.org for news
2. Run `sudo pacman -Syu` (never `pacman -Sy package` -- always full upgrade)
3. Handle any `.pacnew` files: `sudo pacdiff`""",
    "flm_os_variants"
))

# 24. Fedora
conversations.append(make(
    SYS_FED41,
    "How do I set up automatic updates?",
    r"""**Install dnf-automatic:**
```bash
sudo dnf install dnf-automatic
```

**Configure** in `/etc/dnf/automatic.conf`:
```ini
[commands]
upgrade_type = security
apply_updates = yes
download_updates = yes

[emitters]
emit_via = stdio

[base]
debuglevel = 1
```

Set `upgrade_type = security` for security patches only, or `upgrade_type = default` for all updates.

**Enable the timer:**
```bash
sudo systemctl enable --now dnf-automatic.timer
```

**Check the timer:**
```bash
systemctl list-timers dnf-automatic.timer
```

**Test it manually:**
```bash
sudo dnf-automatic
```

**Check what ran:**
```bash
sudo journalctl -u dnf-automatic.service
```

**Alternative: only download, review, then apply:**
```ini
[commands]
apply_updates = no
download_updates = yes
```

This pre-downloads updates so `sudo dnf upgrade` is fast when you're ready.""",
    "flm_os_variants"
))

# 25. Ubuntu 24.04
conversations.append(make(
    SYS_UBU24,
    "How do I set up automatic updates?",
    r"""**apt updates -- unattended-upgrades:**

It's usually pre-installed on Ubuntu, but if not:
```bash
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

Configure in `/etc/apt/apt.conf.d/50unattended-upgrades`:
```
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
    "${distro_id}ESMApps:${distro_codename}-apps-security";
    "${distro_id}ESM:${distro_codename}-infra-security";
};
```

**Snap updates:**
Snaps auto-update by default on Ubuntu. Check the schedule:
```bash
snap refresh --time
```

**Change the snap refresh schedule:**
```bash
sudo snap set system refresh.timer=sat,02:00
```

This sets snap updates to Saturday at 2am.

**Hold a specific snap from updating:**
```bash
sudo snap refresh --hold=72h firefox
```

**Disable snap auto-refresh entirely (not recommended):**
```bash
sudo snap set system refresh.metered=hold
```

**Check unattended-upgrades logs:**
```bash
cat /var/log/unattended-upgrades/unattended-upgrades.log
```

On Ubuntu, you need to think about both apt AND snap updates. The apt side uses unattended-upgrades just like Debian, but snap has its own separate refresh mechanism.""",
    "flm_os_variants"
))

# ===========================================================================
# SET 8 -- "How do I check system logs?"
# ===========================================================================

# 26. Debian
conversations.append(make(
    SYS_DEB13,
    "How do I check system logs?",
    r"""**journalctl (systemd journal):**
```bash
# All logs from this boot
journalctl -b

# Follow logs in real time
journalctl -f

# Logs from a specific unit
journalctl -u nginx.service

# Logs from the last hour
journalctl --since "1 hour ago"

# Kernel messages
journalctl -k

# Show only errors and above
journalctl -p err
```

**Traditional log files** (Debian still ships rsyslog by default):
```bash
# General system log
less /var/log/syslog

# Authentication log
less /var/log/auth.log

# Kernel ring buffer
dmesg

# Apt history
less /var/log/apt/history.log
less /var/log/dpkg.log
```

**Key log locations on Debian:**
- `/var/log/syslog` -- general system messages
- `/var/log/auth.log` -- login attempts, sudo usage
- `/var/log/kern.log` -- kernel messages
- `/var/log/apt/` -- package manager history
- `/var/log/nginx/` -- web server logs (if installed)

Debian runs both journald and rsyslog, so logs appear in both places. journalctl is the modern way; the files in `/var/log/` are the traditional way.""",
    "flm_os_variants"
))

# 27. Arch
conversations.append(make(
    SYS_ARCH,
    "How do I check system logs?",
    r"""**journalctl (the primary log system):**
```bash
# All logs from this boot
journalctl -b

# Follow logs in real time
journalctl -f

# Logs from a specific service
journalctl -u nginx.service

# Logs since a time
journalctl --since "2025-01-15 10:00:00"

# Kernel messages
journalctl -k

# Errors only
journalctl -p err

# Last 50 lines
journalctl -n 50
```

**Important:** Arch does NOT ship a syslog daemon by default. There's no `/var/log/syslog` or `/var/log/messages` unless you install one. The systemd journal is the only log source out of the box.

**Make journals persistent across reboots:**

By default on Arch, journal storage may be set to `auto`. To ensure persistence:
```bash
sudo mkdir -p /var/log/journal
sudo systemd-tmpfiles --create --prefix /var/log/journal
```

Or set it explicitly in `/etc/systemd/journald.conf`:
```ini
[Journal]
Storage=persistent
```

Then restart:
```bash
sudo systemctl restart systemd-journald
```

**Limit journal size:**
```bash
sudo journalctl --vacuum-size=200M
sudo journalctl --vacuum-time=14d
```

**View logs from previous boots:**
```bash
journalctl --list-boots
journalctl -b -1    # previous boot
```

If you want traditional syslog files, install `syslog-ng` or `rsyslog`:
```bash
sudo pacman -S syslog-ng
sudo systemctl enable --now syslog-ng
```""",
    "flm_os_variants"
))

# 28. Fedora
conversations.append(make(
    SYS_FED41,
    "How do I check system logs?",
    r"""**journalctl is the primary log system on Fedora:**
```bash
# All logs from current boot
journalctl -b

# Follow in real time
journalctl -f

# Specific service
journalctl -u sshd.service

# Only errors
journalctl -p err

# Kernel messages
journalctl -k

# Since a timestamp
journalctl --since "1 hour ago"

# Show disk usage of journal
journalctl --disk-usage
```

Fedora uses systemd journal as the primary logging system. Traditional syslog is NOT installed by default.

**Useful filters:**
```bash
# All SSH login attempts
journalctl -u sshd --since today

# All sudo usage
journalctl _COMM=sudo

# Logs from a specific PID
journalctl _PID=1234

# SELinux denials (important on Fedora!)
journalctl -t setroubleshoot
ausearch -m AVC --start today
```

**SELinux audit log** (Fedora-specific):
```bash
sudo ausearch -m AVC -ts recent
sudo sealert -a /var/log/audit/audit.log
```

SELinux denials are a common source of "why isn't this working" on Fedora. Always check the audit log.

**Journal persistence is enabled by default on Fedora.** Check previous boots:
```bash
journalctl --list-boots
journalctl -b -1
```""",
    "flm_os_variants"
))

# ===========================================================================
# SET 9 -- "How do I compile software from source?"
# ===========================================================================

# 29. Debian
conversations.append(make(
    SYS_DEB13,
    "How do I compile software from source?",
    r"""**Install build tools:**
```bash
sudo apt update
sudo apt install build-essential
```

`build-essential` pulls in gcc, g++, make, and libc development headers -- everything you need for standard C/C++ compilation.

**Standard workflow:**
```bash
tar xf software-1.0.tar.gz
cd software-1.0
./configure
make
sudo make install
```

**If it needs extra libraries**, the `./configure` step will tell you. Install the `-dev` packages:
```bash
sudo apt install libssl-dev libcurl4-openssl-dev zlib1g-dev
```

On Debian, development headers are in separate `-dev` packages. The library itself (`libssl3`) doesn't include headers -- you need `libssl-dev`.

**Install to a custom prefix** (recommended to avoid conflicts with apt):
```bash
./configure --prefix=/usr/local
make
sudo make install
```

`/usr/local` is the standard location for locally compiled software. It's already in `$PATH`.

**Build a Debian package instead** (cleaner):
```bash
sudo apt install checkinstall
./configure && make
sudo checkinstall
```

`checkinstall` creates a .deb package from `make install`, so you can later remove it cleanly with `dpkg -r`.""",
    "flm_os_variants"
))

# 30. Arch
conversations.append(make(
    SYS_ARCH,
    "How do I compile software from source?",
    r"""**Install build tools:**
```bash
sudo pacman -S base-devel
```

`base-devel` includes gcc, make, autoconf, automake, binutils, and other essentials.

**Standard manual compile:**
```bash
tar xf software-1.0.tar.gz
cd software-1.0
./configure
make
sudo make install
```

**But on Arch, the preferred method is makepkg** (especially for AUR packages):

If you found a package on the AUR:
```bash
git clone https://aur.archlinux.org/some-package.git
cd some-package
makepkg -si
```

`-s` installs missing dependencies, `-i` installs the built package.

**Read the PKGBUILD first!** It's just a shell script -- always review it before building:
```bash
cat PKGBUILD
```

**If you need development headers**, install the corresponding package. On Arch, headers are usually included in the main package (unlike Debian's `-dev` split):
```bash
sudo pacman -S openssl    # includes headers on Arch
```

Some packages do have separate headers though, like `linux-headers`:
```bash
sudo pacman -S linux-headers
```

**AUR helpers** like `yay` or `paru` automate the AUR workflow:
```bash
# Install yay (from AUR, bootstrapped manually)
git clone https://aur.archlinux.org/yay.git
cd yay
makepkg -si

# Then use it like pacman
yay -S some-aur-package
```""",
    "flm_os_variants"
))

# 31. Fedora
conversations.append(make(
    SYS_FED41,
    "How do I compile software from source?",
    r"""**Install build tools:**
```bash
sudo dnf groupinstall "Development Tools"
```

Or install specific packages:
```bash
sudo dnf install gcc gcc-c++ make autoconf automake
```

**Standard workflow:**
```bash
tar xf software-1.0.tar.gz
cd software-1.0
./configure
make
sudo make install
```

**Install development headers** as needed. On Fedora, they're in `-devel` packages:
```bash
sudo dnf install openssl-devel libcurl-devel zlib-devel
```

Note: Fedora uses `-devel` (not `-dev` like Debian).

**Search for devel packages:**
```bash
dnf search openssl-devel
```

**Install to /usr/local** (recommended):
```bash
./configure --prefix=/usr/local
make
sudo make install
```

**Build an RPM instead** (cleaner):
```bash
sudo dnf install rpm-build rpmdevtools
rpmdev-setuptree
```

Then create a spec file in `~/rpmbuild/SPECS/` and build with `rpmbuild -ba mypackage.spec`. This is more involved than Debian's checkinstall but produces a proper RPM.

**SELinux note:** If the compiled software installs files in non-standard locations, SELinux may block it. Fix with:
```bash
sudo restorecon -Rv /usr/local/bin/myapp
```""",
    "flm_os_variants"
))

# ===========================================================================
# SET 10 -- "How do I check what's using a port?"
# ===========================================================================

# 32. Debian
conversations.append(make(
    SYS_DEB13,
    "How do I check what's using a port?",
    r"""**Using `ss` (the modern tool):**
```bash
# Show all listening TCP ports with process names
sudo ss -tlnp

# Show all listening UDP ports
sudo ss -ulnp

# Check a specific port
sudo ss -tlnp | grep :80
sudo ss -tlnp sport = :443
```

The flags: `-t` TCP, `-l` listening, `-n` numeric (don't resolve names), `-p` show process.

**Using `lsof`:**
```bash
sudo lsof -i :80
sudo lsof -i TCP:443
```

Install if needed: `sudo apt install lsof`

**Using `fuser`:**
```bash
sudo fuser 80/tcp
sudo fuser -v 80/tcp    # verbose -- shows process name
```

**Using `netstat`** (older, from net-tools):
```bash
sudo netstat -tlnp | grep :80
```

Install if needed: `sudo apt install net-tools`

`ss` is the recommended tool -- it's faster than netstat and installed by default on all modern Debian systems.""",
    "flm_os_variants"
))

# 33. Fedora
conversations.append(make(
    SYS_FED41,
    "How do I check what's using a port?",
    r"""**Using `ss`:**
```bash
sudo ss -tlnp
sudo ss -tlnp | grep :80
sudo ss -tlnp sport = :443
```

**Using `lsof`:**
```bash
sudo lsof -i :80
```

`lsof` is usually preinstalled on Fedora. If not: `sudo dnf install lsof`

**Also check the firewall** -- Fedora enables firewalld by default:
```bash
# See which ports are open in the firewall
sudo firewall-cmd --list-ports
sudo firewall-cmd --list-services

# Check if a specific port is allowed
sudo firewall-cmd --query-port=80/tcp

# A service can be listening on a port but blocked by the firewall
# Check both:
sudo ss -tlnp sport = :8080
sudo firewall-cmd --query-port=8080/tcp
```

**Open a port in firewalld:**
```bash
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload
```

**SELinux port labeling:**
If a service tries to bind to a non-standard port, SELinux may block it:
```bash
# Check which ports SELinux allows for HTTP
sudo semanage port -l | grep http

# Allow a custom port for httpd
sudo semanage port -a -t http_port_t -p tcp 8080
```

On Fedora, "why can't I reach my service?" is often a firewalld or SELinux issue, not a service issue.""",
    "flm_os_variants"
))

# ===========================================================================
# SET 11 -- "Where are the config files for X?"
# ===========================================================================

# 34. SSH config -- Debian
conversations.append(make(
    SYS_DEB13,
    "Where is the SSH config?",
    r"""**SSH server config:**
```
/etc/ssh/sshd_config
```

This controls the SSH daemon (sshd). Key settings:
```bash
# View current config (ignoring comments)
grep -v '^#' /etc/ssh/sshd_config | grep -v '^$'
```

Common settings to change:
```
Port 22
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
```

After changes:
```bash
sudo sshd -t                    # test config syntax
sudo systemctl reload sshd      # apply changes
```

**SSH client config:**
```
~/.ssh/config          # per-user
/etc/ssh/ssh_config    # system-wide
```

Example `~/.ssh/config`:
```
Host myserver
    HostName 192.168.1.100
    User alice
    IdentityFile ~/.ssh/id_ed25519
    Port 2222
```

**SSH key locations:**
```
~/.ssh/id_ed25519       # private key
~/.ssh/id_ed25519.pub   # public key
~/.ssh/authorized_keys  # keys allowed to log in as this user
~/.ssh/known_hosts      # fingerprints of servers you've connected to
```

**Permissions matter:**
```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
chmod 600 ~/.ssh/authorized_keys
```""",
    "flm_os_variants"
))

# 35. SSH config -- Fedora
conversations.append(make(
    SYS_FED41,
    "Where is the SSH config?",
    r"""**SSH server config:**
```
/etc/ssh/sshd_config
/etc/ssh/sshd_config.d/*.conf    # drop-in overrides
```

Fedora uses drop-in config files. Check for overrides:
```bash
ls /etc/ssh/sshd_config.d/
```

Common settings:
```bash
sudo grep -v '^#' /etc/ssh/sshd_config | grep -v '^$'
```

After changes:
```bash
sudo sshd -t
sudo systemctl reload sshd
```

**SSH client config:**
```
~/.ssh/config
/etc/ssh/ssh_config
/etc/ssh/ssh_config.d/*.conf
```

**SELinux context (Fedora-specific):**
SSH files must have the correct SELinux labels or sshd will refuse to read them:
```bash
ls -Z ~/.ssh/
# Should show: unconfined_u:object_r:ssh_home_t:s0

# Fix SELinux labels if they're wrong
restorecon -Rv ~/.ssh/
```

**If SSH uses a non-standard port**, tell SELinux:
```bash
sudo semanage port -a -t ssh_port_t -p tcp 2222
```

**And open it in firewalld:**
```bash
sudo firewall-cmd --permanent --add-port=2222/tcp
sudo firewall-cmd --reload
```

On Fedora, SSH issues are often caused by SELinux labels or firewalld rules, not the SSH config itself.""",
    "flm_os_variants"
))

# 36. Nginx config -- Debian
conversations.append(make(
    SYS_DEB13,
    "Where is the nginx config?",
    r"""**Main config:**
```
/etc/nginx/nginx.conf
```

**Site configs (Debian-specific layout):**
```
/etc/nginx/sites-available/    # all site configs
/etc/nginx/sites-enabled/      # symlinks to active configs
```

This is a Debian/Ubuntu convention. To add a site:
```bash
sudo nano /etc/nginx/sites-available/mysite
sudo ln -s /etc/nginx/sites-available/mysite /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

To disable a site:
```bash
sudo rm /etc/nginx/sites-enabled/mysite
sudo systemctl reload nginx
```

The config in `sites-available` is preserved -- only the symlink is removed.

**Other config directories:**
```
/etc/nginx/conf.d/         # additional configs (also included)
/etc/nginx/snippets/       # reusable config fragments
/etc/nginx/mime.types       # MIME type mappings
```

**Default document root:**
```
/var/www/html/
```

**Log files:**
```
/var/log/nginx/access.log
/var/log/nginx/error.log
```

**Test config before reloading:**
```bash
sudo nginx -t
```""",
    "flm_os_variants"
))

# 37. Nginx config -- Fedora
conversations.append(make(
    SYS_FED41,
    "Where is the nginx config?",
    r"""**Main config:**
```
/etc/nginx/nginx.conf
```

**Site configs:**
```
/etc/nginx/conf.d/*.conf
```

Fedora uses the `conf.d/` directory pattern -- there's no `sites-available`/`sites-enabled`. Put your server blocks directly in `conf.d/`:

```bash
sudo nano /etc/nginx/conf.d/mysite.conf
sudo nginx -t
sudo systemctl reload nginx
```

To disable a site, rename the file to not end in `.conf`:
```bash
sudo mv /etc/nginx/conf.d/mysite.conf /etc/nginx/conf.d/mysite.conf.disabled
sudo systemctl reload nginx
```

**Default document root:**
```
/usr/share/nginx/html/
```

**Log files:**
```
/var/log/nginx/access.log
/var/log/nginx/error.log
```

**SELinux considerations:**
If nginx can't access files in a custom directory:
```bash
# Set the correct SELinux context
sudo semanage fcontext -a -t httpd_sys_content_t "/srv/www(/.*)?"
sudo restorecon -Rv /srv/www

# If nginx needs to connect to a backend
sudo setsebool -P httpd_can_network_connect 1

# Check for SELinux denials
sudo ausearch -m AVC -ts recent | grep nginx
```

**Test config:**
```bash
sudo nginx -t
```""",
    "flm_os_variants"
))

# 38. Nginx config -- Arch
conversations.append(make(
    SYS_ARCH,
    "Where is the nginx config?",
    r"""**Main config:**
```
/etc/nginx/nginx.conf
```

Arch ships nginx with a minimal, single-file configuration. There's no `sites-available` or `conf.d` directory by default.

**To use a multi-file setup**, add an include directive to the `http` block in `/etc/nginx/nginx.conf`:
```nginx
http {
    # ... existing config ...
    include /etc/nginx/conf.d/*.conf;
}
```

Then create the directory and add your site:
```bash
sudo mkdir -p /etc/nginx/conf.d
sudo nano /etc/nginx/conf.d/mysite.conf
sudo nginx -t
sudo systemctl reload nginx
```

**Default document root:**
```
/usr/share/nginx/html/
```

Note: this is `/usr/share/nginx/html/`, not `/var/www/html/` like on Debian/Ubuntu.

**Log files:**
```
/var/log/nginx/access.log
/var/log/nginx/error.log
```

**MIME types:**
```
/etc/nginx/mime.types
```

**Test and reload:**
```bash
sudo nginx -t
sudo systemctl reload nginx
```

Arch's minimalist approach means you set up the directory structure yourself. Most people add `conf.d/` or create their own `sites-enabled/` with includes.""",
    "flm_os_variants"
))

# ===========================================================================
# SET 12 -- "How do I install Docker?"
# ===========================================================================

# 39. Debian
conversations.append(make(
    SYS_DEB13,
    "How do I install Docker?",
    r"""**Don't use the Debian repo version** (`docker.io`) -- it's outdated. Use the official Docker repo:

**Add Docker's official GPG key and repo:**
```bash
sudo apt update
sudo apt install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
```

**Add the repository:**
```bash
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

**Install Docker:**
```bash
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

**Start and enable:**
```bash
sudo systemctl enable --now docker
```

**Run Docker without sudo:**
```bash
sudo usermod -aG docker $USER
```

Log out and back in, then verify:
```bash
docker run hello-world
```

**Verify installation:**
```bash
docker --version
docker compose version
```""",
    "flm_os_variants"
))

# 40. Fedora
conversations.append(make(
    SYS_FED41,
    "How do I install Docker?",
    r"""Fedora has two options: **Docker CE** or **Podman** (which comes preinstalled).

**Option 1: Podman (built-in, daemonless)**

Podman is already installed on Fedora and is Docker-compatible:
```bash
podman run hello-world
podman ps
podman images
```

Most Docker commands work with Podman by just replacing `docker` with `podman`. You can even alias it:
```bash
alias docker=podman
```

Podman runs rootless by default -- no daemon, no root access needed.

**Option 2: Docker CE (official Docker)**

Add the Docker repo:
```bash
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
```

Install:
```bash
sudo dnf install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Start and enable:
```bash
sudo systemctl enable --now docker
```

Run without sudo:
```bash
sudo usermod -aG docker $USER
```

Log out and back in, then:
```bash
docker run hello-world
```

**Which to choose?**
- Podman: no daemon, rootless, SELinux-friendly, built into Fedora
- Docker: wider ecosystem, better compose support, needed for some CI/CD tools

For most use cases on Fedora, Podman is the better default.""",
    "flm_os_variants"
))

# 41. Arch
conversations.append(make(
    SYS_ARCH,
    "How do I install Docker?",
    r"""**Install Docker from the official Arch repos:**
```bash
sudo pacman -S docker
```

That's it. No third-party repos needed -- Arch ships Docker in the official `extra` repository.

**Start and enable:**
```bash
sudo systemctl enable --now docker
```

**Run without sudo:**
```bash
sudo usermod -aG docker $USER
```

Log out and back in, then:
```bash
docker run hello-world
```

**Install Docker Compose:**
```bash
sudo pacman -S docker-compose
```

**Optional: install Buildx:**
```bash
sudo pacman -S docker-buildx
```

**Storage driver:**
Arch defaults to overlay2, which is correct for most setups. Check with:
```bash
docker info | grep "Storage Driver"
```

**If you prefer Podman:**
```bash
sudo pacman -S podman
podman run hello-world
```

Podman is also in the Arch repos and works as a daemonless Docker alternative.""",
    "flm_os_variants"
))

# 42. Ubuntu 24.04
conversations.append(make(
    SYS_UBU24,
    "How do I install Docker?",
    r"""**Don't use `docker.io` from the Ubuntu repos** -- it's outdated. Use Docker's official repository:

**Remove old versions (if any):**
```bash
sudo apt remove docker docker-engine docker.io containerd runc 2>/dev/null
```

**Add Docker's GPG key:**
```bash
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

**Add the repository:**
```bash
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

**Install Docker:**
```bash
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

**Start and enable:**
```bash
sudo systemctl enable --now docker
```

**Run without sudo:**
```bash
sudo usermod -aG docker $USER
```

Log out and back in, then test:
```bash
docker run hello-world
docker compose version
```

**Open firewall (if ufw is active):**
Docker manipulates iptables directly and can bypass ufw. If you need to restrict Docker's network access through ufw, see the Docker documentation on iptables integration.

**Verify:**
```bash
docker --version
docker compose version
systemctl status docker
```""",
    "flm_os_variants"
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
