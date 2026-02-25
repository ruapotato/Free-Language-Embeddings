#!/usr/bin/env python3
"""Write hand-crafted SFT conversations for V4 training -- Batch 9.

Topic: BEGINNER-FRIENDLY Linux questions.
Absolute basics, navigation, package management, common tasks, common problems,
and conceptual explanations. Patient tone, no jargon without explanation.

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


# -- Helpers ------------------------------------------------------------------

def make(system, question, answer, source):
    """Build one conversation dict."""
    if system is None:
        text = f"<|user|>\n{question}\n<|assistant|>\n{answer}\n"
    else:
        text = f"<|system|>\n{system}\n<|user|>\n{question}\n<|assistant|>\n{answer}\n"
    return {"text": text, "source": source}


# -- Conversations ------------------------------------------------------------

conversations = []

# ---------------------------------------------------------------------------
# 1. What is Linux? (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "What is Linux?",
    r"""Linux is a free and open-source operating system, similar in purpose to Windows or macOS. An operating system is the core software that manages your computer's hardware and lets you run applications.

Here is what makes Linux special:

**It is free.** You can download, use, copy, and share Linux without paying anything. "Free" here means both free of cost and free to modify — you get the full source code.

**It is open source.** Anyone can read, modify, and distribute the code. Thousands of developers around the world contribute to it.

**It powers most of the internet.** The vast majority of web servers, cloud platforms, and supercomputers run Linux. Android phones also use the Linux kernel.

**A bit of history:** Linus Torvalds created the Linux kernel in 1991 as a hobby project. The "kernel" is the core part that talks to your hardware. The rest of the operating system (command-line tools, file manager, desktop environment) comes from many different projects, most notably the GNU project started by Richard Stallman. That is why some people call it "GNU/Linux."

**Distributions (distros):** Linux comes in many flavors called distributions. Popular ones include:
- **Debian** — stable, community-driven, very widely used
- **Ubuntu** — based on Debian, beginner-friendly
- **Fedora** — cutting-edge features, backed by Red Hat
- **Arch Linux** — minimalist, rolling-release, for more experienced users

Each distro packages the Linux kernel with different default software, desktop environments, and package managers, but they all share the same core.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 2. What is a terminal/shell? (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "What is a terminal? What is a shell?",
    r"""These two terms are related but slightly different. Let me explain both.

**Terminal (terminal emulator):**
A terminal is the window or program where you type commands. Think of it as a text-based interface to your computer. On a modern desktop, you open a terminal application (like GNOME Terminal, Konsole, or xterm) and it gives you a place to type.

**Shell:**
The shell is the program running *inside* the terminal that actually interprets your commands. When you type `ls` and press Enter, it is the shell that understands what `ls` means, runs the command, and shows you the output.

Common shells include:
- **bash** (Bourne Again Shell) — the default on most Linux systems
- **zsh** — popular alternative with extra features, default on macOS and Arch Linux
- **fish** — user-friendly shell with nice auto-completions
- **sh** — the original Bourne shell, very basic

**A simple analogy:** The terminal is like a phone handset — it is the device you hold and speak into. The shell is like the person on the other end who listens to what you say and responds.

**Why use a terminal?**
- It is often faster than clicking through menus
- You can automate tasks by writing scripts
- Some tasks can only be done from the command line
- Remote servers usually only have a terminal interface (no desktop)

**Try it:** Open a terminal on your system and type `echo "Hello, world!"` — the shell will print the message back to you.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 3. What is root? (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "What is root in Linux?",
    r"""The word "root" has two meanings in Linux:

**1. The root user (superuser)**
This is the administrator account with full control over the entire system. The root user can:
- Install and remove software
- Create and delete any user account
- Read, modify, or delete any file on the system
- Start and stop services
- Change system settings

The root user's username is literally `root` and its home directory is `/root`.

**Why you should be careful with root:**
There are no safety nets. If you run `rm -rf /` as root, the system will happily try to delete everything. That is why you should do everyday tasks as a regular user and only use root access when you need it (via `sudo`).

**2. The root directory ( / )**
This is the top-level directory in the Linux file system — every other directory branches off from it. When someone says "the root of the filesystem," they mean `/`.

For example:
```
/           <- root directory
├── home/   <- user home directories
├── etc/    <- configuration files
├── var/    <- variable data (logs, databases)
└── usr/    <- user programs and libraries
```

**On your Debian system,** you can see who you are logged in as by running:
```bash
whoami
```

If it says `root`, you are the superuser. If it says your username (like `david`), you are a regular user.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 4. What is sudo? (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "What is sudo and how do I use it?",
    r"""`sudo` stands for "superuser do." It lets you run a single command with root (administrator) privileges without logging in as root.

**Why sudo exists:**
Running everything as root is dangerous — one typo could break your system. With `sudo`, you stay logged in as your normal user and only elevate privileges for the specific commands that need it.

**Basic usage:**
```bash
# This will fail — regular users cannot install software
apt install htop

# This works — sudo gives you temporary root access
sudo apt install htop
```

When you run `sudo`, it asks for *your* password (not the root password). After you enter it, sudo remembers your authentication for a few minutes so you do not have to type it again right away.

**Common examples on Debian:**
```bash
sudo apt update              # refresh package lists
sudo apt install vim         # install a package
sudo systemctl restart ssh   # restart a service
sudo nano /etc/hostname      # edit a system config file
```

**Who can use sudo?**
Not every user has sudo access. On Debian, your user needs to be in the `sudo` group. You can check with:
```bash
groups
```

If you see `sudo` in the list, you have access.

**To add a user to the sudo group** (you need to already be root or have sudo):
```bash
sudo usermod -aG sudo username
```

**Important tips:**
- Only use `sudo` when the command actually needs root access
- Never run `sudo` on commands you do not understand
- Double-check before running destructive commands like `rm` with `sudo`""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 5. What is a package manager? (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "What is a package manager?",
    r"""A package manager is a tool that installs, updates, and removes software on your system. Think of it like an app store, but for the command line.

**What it does:**
- **Installs software** from official repositories (online collections of software)
- **Resolves dependencies** — if program A needs library B to work, the package manager automatically installs library B too
- **Updates everything** — one command can update all installed software at once
- **Removes software** cleanly, including any files it installed

**Why it matters:**
On Windows, you typically download installers from websites. On Linux, you almost always install software through the package manager. This is safer (packages are verified), easier (one command), and keeps everything up to date.

**Common package managers by distribution:**

| Distro | Package manager | Example install command |
|--------|----------------|----------------------|
| Debian, Ubuntu | apt | `sudo apt install firefox` |
| Fedora | dnf | `sudo dnf install firefox` |
| Arch Linux | pacman | `sudo pacman -S firefox` |
| openSUSE | zypper | `sudo zypper install firefox` |

**A "package"** is just a bundle containing the software files, information about dependencies, and instructions for where to put everything.

**Repositories (repos)** are online servers that host thousands of packages. When you run an install command, the package manager downloads the package from these repos.

**Example workflow (Debian/Ubuntu):**
```bash
sudo apt update          # refresh the list of available packages
sudo apt install htop    # install the htop process viewer
sudo apt remove htop     # remove it later if you want
sudo apt upgrade         # update all installed packages to latest versions
```""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 6. What is a file system? (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "What is a file system?",
    r"""A file system is the way your operating system organizes and stores files on a disk. It is like a filing cabinet system — it determines how data is arranged, named, and retrieved.

**What a file system does:**
- Keeps track of where each file is physically stored on the disk
- Manages file names, sizes, and modification dates
- Controls permissions (who can read, write, or execute files)
- Organizes files into directories (folders)

**Common Linux file systems:**
- **ext4** — the most common Linux file system. Reliable, well-tested, and the default on most distributions
- **btrfs** — newer, supports snapshots and compression. Default on Fedora and openSUSE
- **xfs** — good for very large files, used in enterprise settings
- **fat32 / exfat** — used on USB drives for compatibility with Windows and macOS
- **ntfs** — the Windows file system. Linux can read and write to it

**How Linux organizes files:**
Unlike Windows (which uses C:\, D:\, etc.), Linux has a single directory tree starting at `/` (the root). Everything is under this one tree, even other drives and devices:

```
/              <- the root, top of the tree
├── home/      <- your personal files
├── etc/       <- system configuration
├── var/       <- logs, databases, variable data
├── tmp/       <- temporary files
└── mnt/       <- where extra drives get attached
```

When you plug in a USB drive, it gets "mounted" somewhere in this tree (like `/mnt/usb` or `/media/username/USBDRIVE`), rather than showing up as a separate drive letter.

**To see what file systems are in use on your machine:**
```bash
df -Th
```

This shows each mounted file system, its type, total size, and how much space is used.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 7. What is a process? (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "What is a process in Linux?",
    r"""A process is a running instance of a program. Every time you start a program, the operating system creates a process for it.

**Key things about processes:**
- Every process gets a unique number called a **PID** (Process ID)
- Processes have an **owner** — the user who started them
- Processes can be in different **states**: running, sleeping (waiting for something), stopped, etc.
- Processes can create other processes (called **child processes**)

**Examples:**
- When you open a terminal, that is a process
- When you type `ls`, a short-lived process runs and exits
- A web browser might be one main process that spawns many child processes (one per tab)
- A web server like nginx runs as a process (usually multiple) in the background

**Seeing your processes:**
```bash
# List your own processes
ps

# List all processes on the system
ps aux

# Interactive, real-time view (press q to quit)
top
```

**Every process has a parent.** The very first process (PID 1) is started by the kernel at boot and is usually `systemd` or `init`. Every other process descends from it.

**Foreground vs. background:**
- A **foreground** process takes over your terminal — you cannot type other commands until it finishes
- A **background** process runs without blocking your terminal

```bash
# Run something in the background by adding & at the end
sleep 60 &

# See your background jobs
jobs

# Bring a background job back to the foreground
fg %1
```

**Stopping a process:**
- Press **Ctrl+C** to stop a foreground process
- Use `kill PID` to stop a process by its PID
- Use `kill -9 PID` as a last resort to force-kill a stubborn process""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 8. What is a daemon/service? (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "What is a daemon or service in Linux?",
    r"""A daemon (pronounced "dee-mon") is a program that runs in the background, usually starting at boot time and running continuously. The terms "daemon" and "service" are often used interchangeably.

**Examples of daemons:**
- **sshd** — the SSH server, listens for remote login connections
- **cron** — runs scheduled tasks at specific times
- **nginx** or **apache2** — web servers
- **systemd-journald** — collects and stores log messages
- **NetworkManager** — manages network connections

**How to tell if something is a daemon:** Daemon names often end in "d" — `sshd`, `httpd`, `crond`, `systemd`.

**Managing services on Debian with systemctl:**

Your Debian system uses `systemd` to manage services. The main command is `systemctl`:

```bash
# Check the status of a service
sudo systemctl status ssh

# Start a service
sudo systemctl start ssh

# Stop a service
sudo systemctl stop ssh

# Restart a service (stop then start)
sudo systemctl restart ssh

# Enable a service to start automatically at boot
sudo systemctl enable ssh

# Disable auto-start at boot
sudo systemctl disable ssh
```

**List all running services:**
```bash
systemctl list-units --type=service --state=running
```

**Reading service logs:**
```bash
# Logs for a specific service
journalctl -u ssh

# Follow logs in real time (like tail -f)
journalctl -u ssh -f
```

The key idea: daemons are background workers that keep your system running. You do not interact with them directly — they just quietly do their jobs (serving web pages, accepting SSH connections, scheduling tasks, etc.).""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 9. What is SSH? (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "What is SSH?",
    r"""SSH stands for **Secure Shell**. It lets you connect to another computer over the network and use its terminal as if you were sitting in front of it — but all the communication is encrypted.

**What you can do with SSH:**
- Log in to a remote server and run commands
- Transfer files securely
- Set up encrypted tunnels for other traffic
- Manage servers that have no monitor or keyboard attached

**Basic usage:**
```bash
ssh username@hostname
```

For example, to connect to a server at 192.168.1.50 as user "david":
```bash
ssh david@192.168.1.50
```

You will be asked for your password (or it will use your SSH key if you have set one up). Once connected, you get a terminal on the remote machine.

**SSH keys (recommended):**
Instead of typing a password every time, you can set up SSH keys. This is both more secure and more convenient:

```bash
# Generate a key pair (press Enter to accept defaults)
ssh-keygen

# Copy your public key to the remote server
ssh-copy-id username@hostname
```

After that, you can log in without a password.

**Common SSH options:**
```bash
# Connect on a non-standard port
ssh -p 2222 user@hostname

# Run a single command on the remote machine without opening an interactive session
ssh user@hostname "df -h"
```

**The other side:** For SSH to work, the remote machine needs to be running an SSH server (usually `sshd` or `openssh-server`). Most Linux servers have this enabled by default.

**Security tip:** SSH encrypts everything — your password, your commands, and all data transferred. This is why it replaced older tools like telnet, which sent everything in plain text.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 10. What are file permissions? (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do file permissions work in Linux?",
    r"""Every file and directory in Linux has permissions that control who can do what with it. There are three types of access and three categories of users.

**Three types of access:**
- **r** (read) — view the contents of a file or list a directory
- **w** (write) — modify a file or create/delete files in a directory
- **x** (execute) — run a file as a program, or enter a directory with `cd`

**Three categories of users:**
- **Owner** — the user who owns the file
- **Group** — users in the file's group
- **Others** — everyone else

**Viewing permissions:**
```bash
ls -l
```

Output looks like:
```
-rw-r--r-- 1 david david 1234 Jan 15 10:30 myfile.txt
```

The `-rw-r--r--` part breaks down as:
```
-    rw-    r--    r--
|    |      |      |
|    |      |      └── others: read only
|    |      └── group: read only
|    └── owner: read + write
└── file type (- = regular file, d = directory)
```

**Changing permissions with chmod:**
```bash
# Give the owner execute permission
chmod u+x script.sh

# Remove write permission for others
chmod o-w myfile.txt

# Give everyone read and execute permission
chmod a+rx script.sh
```

**Numeric (octal) notation** is another way:
- r=4, w=2, x=1. Add them up for each category.

```bash
# rwxr-xr-x = 755 (owner: all, group: read+exec, others: read+exec)
chmod 755 script.sh

# rw-r--r-- = 644 (owner: read+write, group: read, others: read)
chmod 644 myfile.txt

# rwx------ = 700 (owner: all, nobody else gets access)
chmod 700 private_script.sh
```

**Changing ownership:**
```bash
# Change the owner
sudo chown alice myfile.txt

# Change owner and group
sudo chown alice:developers myfile.txt
```

**Why this matters:** Permissions keep your system secure. Regular users cannot modify system files, and your personal files are protected from other users.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 11. Basic terminal commands (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "What are the most basic terminal commands I should know?",
    r"""Here are the essential commands for navigating and managing files. These work on any Linux system.

**Navigating:**
```bash
pwd                  # Print Working Directory — shows where you are
ls                   # List files in the current directory
ls -la               # List all files (including hidden) with details
cd /etc              # Change Directory — go to /etc
cd ~                 # Go to your home directory
cd ..                # Go up one directory
cd -                 # Go back to the previous directory
```

**Creating and deleting:**
```bash
mkdir photos                 # Create a directory called "photos"
mkdir -p projects/web/css    # Create nested directories all at once
touch newfile.txt            # Create an empty file (or update its timestamp)
rm myfile.txt                # Delete a file (no undo!)
rm -r myfolder               # Delete a directory and everything inside it
rmdir emptyfolder            # Delete an empty directory only
```

**Copying and moving:**
```bash
cp file.txt backup.txt           # Copy a file
cp -r photos/ photos_backup/     # Copy a directory and its contents
mv file.txt documents/           # Move a file into a directory
mv oldname.txt newname.txt       # Rename a file
```

**Viewing files:**
```bash
cat file.txt           # Print the entire file to the terminal
less file.txt          # View a file page by page (press q to quit)
head file.txt          # Show the first 10 lines
tail file.txt          # Show the last 10 lines
tail -f /var/log/syslog  # Follow a log file in real time
```

**Getting help:**
```bash
man ls          # Read the manual page for any command
ls --help       # Quick help for most commands
```

**Tip:** The terminal is case-sensitive. `Documents` and `documents` are different names. And be very careful with `rm` — there is no trash can on the command line. Deleted files are gone for good.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 12. How do I create and edit a text file? (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I create and edit a text file from the terminal?",
    r"""There are several ways to create and edit text files. Here are the most common options, from simplest to most powerful.

**1. Create a file quickly:**
```bash
# Create an empty file
touch myfile.txt

# Create a file with some content
echo "Hello, world!" > myfile.txt

# Append text to an existing file
echo "Another line" >> myfile.txt
```

**2. nano — the beginner-friendly editor:**
```bash
nano myfile.txt
```

nano is the easiest terminal text editor. It shows keyboard shortcuts at the bottom of the screen:
- **Ctrl+O** — save the file (it will ask for the filename, press Enter to confirm)
- **Ctrl+X** — exit nano
- **Ctrl+K** — cut a line
- **Ctrl+U** — paste a line
- **Ctrl+W** — search for text

nano comes pre-installed on Debian. Just type, use the shortcuts, and you are good to go.

**3. vim — the powerful editor:**
```bash
vim myfile.txt
```

vim is more powerful but has a learning curve. The key thing to know: vim has *modes*.

- When you open vim, you are in **Normal mode** (for navigation and commands)
- Press **i** to enter **Insert mode** (now you can type text)
- Press **Esc** to go back to Normal mode
- In Normal mode, type **:w** and press Enter to save
- Type **:q** and press Enter to quit
- Type **:wq** to save and quit
- Type **:q!** to quit without saving

**4. Quick edit with sed (for scripting):**
```bash
# Replace "old" with "new" in a file
sed -i 's/old/new/g' myfile.txt
```

**My recommendation for beginners:** Start with `nano`. It works, it is simple, and it is always available. You can learn vim later if you want its extra power.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 13. Absolute vs relative paths (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "What is the difference between absolute and relative paths?",
    r"""A **path** is the address of a file or directory. There are two ways to write a path.

**Absolute path** — starts from the root `/` and gives the full location:
```
/home/david/documents/report.txt
```

This always points to the same file, no matter where you are in the file system. It always starts with `/`.

**Relative path** — starts from your current directory:
```
documents/report.txt
```

This points to different files depending on where you are. If you are in `/home/david`, it points to `/home/david/documents/report.txt`. If you are in `/tmp`, it would try to find `/tmp/documents/report.txt`.

**Special shortcuts for relative paths:**
```
.      current directory       ./script.sh = script.sh in this directory
..     parent directory        ../file.txt = file.txt one level up
~      home directory          ~/photos = /home/yourname/photos
```

**Examples:**

Say you are in `/home/david`:
```bash
# These two commands do the same thing:
cat /home/david/documents/report.txt    # absolute
cat documents/report.txt                # relative

# Go up one directory and into another user's home:
cd ../alice                              # relative -> /home/alice

# These are the same:
cd /home/david                           # absolute
cd ~                                     # shortcut for home
```

**When to use which:**
- **Absolute paths** are best in scripts and configuration files — they always work regardless of where the script runs
- **Relative paths** are convenient for everyday use in the terminal — less typing
- **Use `~`** when referring to files in your home directory — it works for any user""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 14. How do I find a file? (Ubuntu 24.04)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_UBU24,
    "How do I find a file on my system?",
    r"""There are several commands for finding files. Here are the most useful ones.

**1. find — the most powerful search tool:**
```bash
# Find a file by name (searches everywhere)
sudo find / -name "report.txt"

# Find by name, case-insensitive
find /home -iname "readme*"

# Find all .log files in /var
find /var -name "*.log"

# Find files larger than 100MB
find / -size +100M

# Find files modified in the last 24 hours
find /home -mtime -1

# Find empty files
find /home -empty -type f
```

**2. locate — instant search (uses a database):**
```bash
# Install it first
sudo apt install plocate

# Update the file database (runs automatically once a day too)
sudo updatedb

# Search for a file
locate report.txt
```

`locate` is much faster than `find` because it searches a pre-built database instead of scanning the disk. The tradeoff is that newly created files will not appear until the database is updated.

**3. which — find where a command lives:**
```bash
which python3
# /usr/bin/python3

which nano
# /usr/bin/nano
```

**4. Use wildcards with ls:**
```bash
# List all .txt files in the current directory
ls *.txt

# List files starting with "report"
ls report*
```

**5. Search for files containing specific text:**
```bash
# Find files that contain the word "error" in /var/log
grep -rl "error" /var/log/

# Same but case-insensitive
grep -ril "error" /var/log/
```

**Quick reference:**
| What you want | Command |
|---|---|
| Find file by name | `find /path -name "filename"` |
| Fast filename search | `locate filename` |
| Find a command's location | `which commandname` |
| Search file contents | `grep -r "text" /path/` |""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 15. How do I read a log file? (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I read log files?",
    r"""Log files are plain text files that record what is happening on your system. They are essential for troubleshooting. On Debian, most logs live in `/var/log/`.

**Viewing a log file:**

```bash
# Read the whole file (okay for small files)
cat /var/log/auth.log

# Page through a large file (press q to quit, space for next page)
less /var/log/syslog

# See the last 20 lines
tail -20 /var/log/syslog

# Watch a log file in real time (new lines appear as they are written)
tail -f /var/log/syslog
```

`tail -f` is extremely useful for debugging — run it in one terminal while you reproduce a problem in another, and you can see exactly what the system logs.

**Searching inside log files:**
```bash
# Find lines containing "error"
grep "error" /var/log/syslog

# Case-insensitive search
grep -i "error" /var/log/syslog

# Show 3 lines before and after each match for context
grep -C 3 "error" /var/log/syslog

# Search across all log files
grep -ri "error" /var/log/
```

**Using journalctl (systemd journal):**

On modern Debian, `systemd` collects logs with its journal. This is often easier than reading raw files:

```bash
# View recent system logs
journalctl

# Logs from the current boot only
journalctl -b

# Logs from a specific service
journalctl -u ssh

# Follow new log entries in real time
journalctl -f

# Logs from the last hour
journalctl --since "1 hour ago"

# Logs between two times
journalctl --since "2025-01-15 09:00" --until "2025-01-15 10:00"
```

**Important log files on Debian:**
| File | What it contains |
|---|---|
| `/var/log/syslog` | General system messages |
| `/var/log/auth.log` | Authentication events (logins, sudo usage) |
| `/var/log/kern.log` | Kernel messages |
| `/var/log/apt/history.log` | Package installation history |
| `/var/log/dpkg.log` | Low-level package manager log |""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 16. How do I install software? (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I install software on Debian?",
    r"""On Debian, you use the `apt` package manager to install software.

**Step 1: Update the package list**
```bash
sudo apt update
```
This downloads the latest list of available packages from the repositories. Always do this before installing something new.

**Step 2: Install a package**
```bash
sudo apt install package-name
```

For example:
```bash
sudo apt install htop       # a nice process viewer
sudo apt install git        # version control system
sudo apt install curl       # tool for downloading from URLs
sudo apt install vim        # powerful text editor
```

You can install multiple packages at once:
```bash
sudo apt install git curl wget vim
```

**Step 3: Confirm**
apt will show you what it plans to install (including any dependencies) and ask you to confirm. Press **Y** and Enter.

**To skip the confirmation prompt:**
```bash
sudo apt install -y htop
```

**Searching for packages:**
```bash
# Search for packages related to "image"
apt search image editor

# Show details about a package before installing
apt show gimp
```

**Updating all installed software:**
```bash
sudo apt update          # refresh the package list
sudo apt upgrade         # upgrade all installed packages
```

**Tip:** Almost all the software you need is available through apt. Only download software from external websites as a last resort — packages from apt are tested, verified, and easy to update.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 17. How do I install software? (Arch)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_ARCH,
    "How do I install software on Arch Linux?",
    r"""On Arch Linux, you use `pacman` to install software from the official repositories.

**Basic install:**
```bash
sudo pacman -S package-name
```

For example:
```bash
sudo pacman -S htop         # process viewer
sudo pacman -S git          # version control
sudo pacman -S firefox      # web browser
```

**What the flags mean:**
- `-S` stands for "sync" — install from repositories
- `-R` stands for "remove"
- `-Q` stands for "query" — search installed packages
- `-Ss` means "sync search" — search available packages

**Update the system:**
Arch is a rolling-release distro, so you update everything at once:
```bash
sudo pacman -Syu
```
- `-S` = sync
- `-y` = refresh the package database
- `-u` = upgrade all outdated packages

Run this regularly to keep your system up to date.

**Search for packages:**
```bash
# Search available packages
pacman -Ss image editor

# Search installed packages
pacman -Qs vim

# Show info about a package
pacman -Si firefox
```

**Remove a package:**
```bash
# Remove a package
sudo pacman -R firefox

# Remove a package and its unused dependencies
sudo pacman -Rs firefox
```

**AUR (Arch User Repository):**
The official repos do not have everything. The AUR is a community-maintained collection of build scripts for additional software. To use it, you need an AUR helper like `yay`:
```bash
# If you have yay installed:
yay -S spotify
yay -S visual-studio-code-bin
```

**Important:** On Arch, always do a full system update (`pacman -Syu`) rather than installing individual packages without updating. Partial upgrades can break things on a rolling-release distro.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 18. How do I install software? (Fedora)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_FED41,
    "How do I install software on Fedora?",
    r"""On Fedora, you use `dnf` (Dandified Yum) to install software.

**Basic install:**
```bash
sudo dnf install package-name
```

For example:
```bash
sudo dnf install htop        # process viewer
sudo dnf install git         # version control
sudo dnf install vim         # text editor
```

**Search for packages:**
```bash
# Search available packages
dnf search image editor

# Show info about a package
dnf info gimp
```

**Update all packages:**
```bash
sudo dnf upgrade
```

This updates all installed packages to their latest versions. You can also use `sudo dnf upgrade --refresh` to force-refresh the metadata first.

**Remove a package:**
```bash
sudo dnf remove firefox
```

**List installed packages:**
```bash
dnf list installed
dnf list installed | grep python
```

**Group installs:**
Fedora lets you install groups of related packages:
```bash
# See available groups
dnf group list

# Install a group (e.g., development tools)
sudo dnf group install "Development Tools"
```

**Enable RPM Fusion (for extra software):**
Some packages (like multimedia codecs) are not in the default Fedora repos due to licensing. RPM Fusion adds them:
```bash
sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
```

**History:**
```bash
# See what you have installed or updated recently
dnf history
dnf history info last
```""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 19. How do I remove software? (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I uninstall or remove software on Debian?",
    r"""There are a few ways to remove software on Debian, depending on how thorough you want to be.

**Basic removal:**
```bash
sudo apt remove package-name
```

This removes the program but keeps its configuration files. Useful if you might reinstall it later and want your settings preserved.

**Complete removal (including config files):**
```bash
sudo apt purge package-name
```

This removes both the program and its system-wide configuration files.

**Remove a package and its unused dependencies:**
```bash
sudo apt autoremove
```

When you install a package, apt often installs extra libraries (dependencies) that it needs. When you remove the package, those libraries are left behind. `autoremove` cleans them up.

**Combine them:**
```bash
sudo apt purge package-name
sudo apt autoremove
```

Or all at once:
```bash
sudo apt purge --autoremove package-name
```

**Examples:**
```bash
# Remove firefox and clean up
sudo apt purge --autoremove firefox

# Remove multiple packages at once
sudo apt remove vim nano
```

**Check what is installed:**
```bash
# List all installed packages
dpkg -l

# Check if a specific package is installed
dpkg -l | grep nginx

# Or use apt
apt list --installed | grep nginx
```

**Tip:** `remove` is safe for most cases. Use `purge` when you want a completely clean removal and do not care about the old configuration. `autoremove` is good to run periodically to free up disk space from leftover dependencies.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 20. How do I search for a package? (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I search for a package in Debian?",
    r"""There are several ways to find packages on Debian.

**Search by name or description:**
```bash
apt search keyword
```

For example:
```bash
apt search image editor
apt search "pdf viewer"
apt search python3
```

This searches both package names and descriptions. It often returns a lot of results.

**Show detailed info about a specific package:**
```bash
apt show package-name
```

For example:
```bash
apt show gimp
```

This tells you the version, size, dependencies, and a description.

**List packages matching a pattern:**
```bash
apt list "python3-*"
apt list --installed         # only show installed packages
apt list --upgradable        # only show packages with updates available
```

**Find which package provides a specific file:**
```bash
# If you know you need a command but do not know the package name:
apt-file search bin/convert

# You may need to install apt-file first:
sudo apt install apt-file
sudo apt-file update
```

**Find which package owns an installed file:**
```bash
dpkg -S /usr/bin/python3
# output: python3-minimal: /usr/bin/python3
```

**Quick reference:**
| What you want | Command |
|---|---|
| Search for a package | `apt search keyword` |
| Package details | `apt show package-name` |
| Is it installed? | `dpkg -l package-name` |
| What file comes from what package? | `dpkg -S /path/to/file` |
| What package provides a file? | `apt-file search filename` |
| List installed packages | `apt list --installed` |""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 21. apt update vs apt upgrade (Ubuntu 24.04)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_UBU24,
    "What is the difference between apt update and apt upgrade?",
    r"""These two commands do very different things, even though they sound similar.

**`sudo apt update`** — refreshes the package list.
It downloads the latest list of available packages and their versions from the repositories. It does **not** install or change any software. Think of it as checking what updates are available.

```bash
sudo apt update
```

**`sudo apt upgrade`** — installs available updates.
It actually downloads and installs newer versions of the packages you already have. It reads the list that `update` downloaded and upgrades everything it can.

```bash
sudo apt upgrade
```

**Always run them in this order:**
```bash
sudo apt update    # first, check what is available
sudo apt upgrade   # then, install the updates
```

If you run `upgrade` without `update` first, you might miss the latest updates because your package list is outdated.

**There is also `sudo apt full-upgrade`:**
Regular `upgrade` will never remove an existing package, even if removing it is needed to complete the upgrade. `full-upgrade` will remove packages if necessary. This is mostly relevant during major distribution upgrades.

**Summary:**
| Command | What it does |
|---|---|
| `apt update` | Refreshes the list of available packages (no installs) |
| `apt upgrade` | Installs newer versions of your current packages |
| `apt full-upgrade` | Like upgrade, but can also remove packages if needed |

**An analogy:** `apt update` is like refreshing a shopping website to see current prices. `apt upgrade` is actually clicking "buy" to get the new versions.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 22. How do I change my password? (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "How do I change my password in Linux?",
    r"""Use the `passwd` command.

**Change your own password:**
```bash
passwd
```

It will ask you for your current password, then your new password twice. Note: when you type passwords in the terminal, nothing appears on screen — no dots, no asterisks, nothing. This is normal. Just type and press Enter.

**Change another user's password (requires root):**
```bash
sudo passwd username
```

For example:
```bash
sudo passwd alice
```

This skips the "current password" step and just asks for the new one.

**Force a user to change their password at next login:**
```bash
sudo passwd -e username
```

**Check password status:**
```bash
passwd -S
```

This shows when your password was last changed and when it expires.

**Tips for a good password:**
- Make it at least 12 characters long
- Mix uppercase, lowercase, numbers, and symbols
- Do not use dictionary words, birthdays, or common patterns
- Consider using a passphrase — a random sequence of words like "correct horse battery staple" is both strong and memorable

**If you forgot the root password:**
This requires rebooting into recovery mode, which varies by distribution. The general approach is to boot into single-user mode and run `passwd root` to set a new one.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 23. How do I see how much disk space I have? (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I check how much disk space I have left?",
    r"""There are a few commands that show disk usage.

**`df` — disk free space:**
```bash
df -h
```

The `-h` flag means "human-readable" — it shows sizes in GB and MB instead of raw bytes.

Example output:
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1       234G  156G   66G  71% /
tmpfs           7.8G     0  7.8G   0% /dev/shm
/dev/sda2       976M  130M  780M  15% /boot
```

This tells you: the main partition is 234 GB, 156 GB is used, 66 GB is free, and it is 71% full.

**`du` — disk usage per directory:**
```bash
# How much space does a directory use?
du -sh /home/david

# Break it down by subdirectory
du -h --max-depth=1 /home/david

# Find the biggest directories
du -h --max-depth=1 /home/david | sort -hr | head -20
```

The last command is very useful — it shows your top 20 biggest directories, sorted from largest to smallest.

**`ncdu` — interactive disk usage viewer:**
```bash
sudo apt install ncdu
ncdu /home
```

ncdu gives you an interactive view where you can navigate through directories and see what is taking up space. Very handy for cleanup. Press `q` to quit.

**Quick cleanup tips if you are running low on space:**
```bash
# Remove cached package files
sudo apt clean

# Remove packages that are no longer needed
sudo apt autoremove

# See how much apt cache is using
du -sh /var/cache/apt/archives
```""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 24. How do I see running processes? (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I see what processes are running?",
    r"""There are several tools, from simple to powerful.

**`ps` — snapshot of current processes:**
```bash
# Show your own processes
ps

# Show all processes on the system
ps aux
```

The `ps aux` output shows:
```
USER       PID %CPU %MEM    VSZ   RSS TTY  STAT START   TIME COMMAND
root         1  0.0  0.1 169536 13120 ?    Ss   Jan15   0:07 /lib/systemd/systemd
david     1234  0.5  1.2 432100 98000 ?    Sl   10:00   0:30 firefox
```

- **PID** — process ID (the number you use to kill it)
- **%CPU** — how much CPU it is using
- **%MEM** — how much memory it is using
- **COMMAND** — what program it is

**`top` — real-time process viewer:**
```bash
top
```

This shows a live, updating view of processes sorted by CPU usage. Useful keys:
- **q** — quit
- **M** — sort by memory
- **P** — sort by CPU (default)
- **k** — kill a process (type the PID)

**`htop` — a better `top`:**
```bash
sudo apt install htop
htop
```

htop is an improved version with colors, mouse support, and easier navigation. You can scroll through processes, search, filter, and kill them with function keys.

**Find a specific process:**
```bash
# Search for a process by name
ps aux | grep firefox

# Or use pgrep
pgrep -a firefox
```

**See a tree of parent/child processes:**
```bash
pstree
```

**Monitor a single process:**
```bash
# Watch a process by PID
top -p 1234
```""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 25. How do I stop a hung program? (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "How do I stop a program that is frozen or not responding?",
    r"""There are several ways to stop a stuck program, from gentle to forceful.

**1. Ctrl+C — the first thing to try**
If the program is running in your terminal, press **Ctrl+C**. This sends an interrupt signal telling the program to stop. Most programs will exit cleanly.

**2. Ctrl+Z — suspend, then kill**
If Ctrl+C does not work, try **Ctrl+Z**. This suspends (pauses) the program and gives you your terminal back. Then you can kill it:
```bash
kill %1
```

**3. Find and kill by PID:**
If the program is not in your terminal (or you have closed that terminal):
```bash
# Find the process
ps aux | grep program-name

# Kill it using the PID (the number in the second column)
kill 12345
```

**4. Kill by name:**
```bash
# Kill all processes with this name
killall firefox

# Or use pkill
pkill firefox
```

**5. Force kill (last resort):**
A regular `kill` sends a polite request to exit. If the process ignores it, force it:
```bash
kill -9 12345
# or
killall -9 program-name
```

`kill -9` (SIGKILL) cannot be ignored — the kernel terminates the process immediately. Use this only when a regular kill does not work, because the process gets no chance to save data or clean up.

**6. If your entire desktop is frozen:**
- Try **Ctrl+Alt+F2** to switch to a text terminal, log in, and kill the problem process
- As a last resort, press and hold the power button (but you may lose unsaved work)

**Signal reference:**
| Signal | Number | Meaning |
|---|---|---|
| SIGTERM | 15 | Polite "please exit" (default for `kill`) |
| SIGKILL | 9 | Forced immediate termination |
| SIGINT | 2 | Interrupt (what Ctrl+C sends) |
| SIGSTOP | 19 | Pause (what Ctrl+Z sends) |""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 26. How do I restart a service? (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I restart a service?",
    r"""On Debian, services are managed with `systemctl`, which is part of systemd.

**Restart a service:**
```bash
sudo systemctl restart service-name
```

For example:
```bash
sudo systemctl restart ssh
sudo systemctl restart nginx
sudo systemctl restart networking
```

**Other service operations:**
```bash
# Start a stopped service
sudo systemctl start ssh

# Stop a running service
sudo systemctl stop ssh

# Check the current status of a service
sudo systemctl status ssh

# Reload configuration without fully restarting (not all services support this)
sudo systemctl reload nginx
```

**The difference between restart and reload:**
- `restart` — completely stops the service and starts it again. Causes a brief interruption.
- `reload` — tells the service to re-read its configuration files without stopping. No interruption, but not all services support it.

If you are not sure, use `reload-or-restart`:
```bash
sudo systemctl reload-or-restart nginx
```

**Check if a service is enabled at boot:**
```bash
systemctl is-enabled ssh
```

**Enable or disable boot start:**
```bash
sudo systemctl enable ssh      # start at boot
sudo systemctl disable ssh     # do not start at boot
```

**See the service's logs if something goes wrong:**
```bash
journalctl -u ssh -n 50        # last 50 log lines
journalctl -u ssh -f           # follow logs in real time
```

**List all services:**
```bash
systemctl list-units --type=service
systemctl list-units --type=service --state=running   # only running ones
```""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 27. How do I connect to a remote server? (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "How do I connect to a remote server?",
    r"""The most common way is using **SSH** (Secure Shell).

**Basic connection:**
```bash
ssh username@server-address
```

For example:
```bash
ssh david@192.168.1.100       # connect using an IP address
ssh david@myserver.example.com  # connect using a hostname
```

You will be asked for your password. Type it and press Enter. (Nothing appears on screen while you type — that is normal.)

Once connected, you get a terminal on the remote machine. Everything you type runs on the remote server. To disconnect, type `exit` or press **Ctrl+D**.

**Connect on a different port:**
The default SSH port is 22. If the server uses a different port:
```bash
ssh -p 2222 david@192.168.1.100
```

**Using SSH keys (recommended):**
Password authentication works but keys are more secure and convenient:

```bash
# Step 1: Generate a key pair (do this once, on your local machine)
ssh-keygen -t ed25519

# Step 2: Copy your public key to the server
ssh-copy-id david@192.168.1.100

# Step 3: Now you can connect without a password
ssh david@192.168.1.100
```

**Save connection details in a config file:**
Edit `~/.ssh/config` to avoid typing long commands:

```
Host myserver
    HostName 192.168.1.100
    User david
    Port 22
```

Now you can just type:
```bash
ssh myserver
```

**Run a command without opening an interactive session:**
```bash
ssh david@192.168.1.100 "df -h"
```

This runs `df -h` on the remote machine, shows the output, and disconnects.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 28. How do I copy files to/from a remote server? (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I copy files to or from a remote server?",
    r"""There are two main tools: `scp` (simple) and `rsync` (powerful).

**scp — Secure Copy:**
```bash
# Copy a file from local to remote
scp myfile.txt david@192.168.1.100:/home/david/

# Copy a file from remote to local
scp david@192.168.1.100:/home/david/report.txt .

# Copy a whole directory (use -r for recursive)
scp -r myfolder/ david@192.168.1.100:/home/david/

# Copy with a non-standard SSH port
scp -P 2222 myfile.txt david@192.168.1.100:/home/david/
```

The syntax is: `scp source destination`. Either source or destination can be remote (`user@host:path`), the other is local.

**rsync — smarter and faster:**
```bash
# Sync a directory to a remote server
rsync -avz myfolder/ david@192.168.1.100:/home/david/myfolder/

# Pull from remote to local
rsync -avz david@192.168.1.100:/home/david/data/ ./data/

# Dry run — see what would be transferred without actually doing it
rsync -avzn myfolder/ david@192.168.1.100:/home/david/myfolder/
```

**Why rsync is better than scp for most tasks:**
- It only transfers files that have changed (much faster for repeated backups)
- It can resume interrupted transfers
- It compresses data during transfer with `-z`
- It preserves permissions, timestamps, and symlinks with `-a`

**The flags explained:**
- `-a` — archive mode (preserves permissions, timestamps, symlinks, etc.)
- `-v` — verbose (shows what is being transferred)
- `-z` — compress during transfer
- `-n` — dry run (preview without copying)
- `--progress` — show transfer progress
- `--delete` — delete files on the destination that do not exist on the source

**Install rsync if needed:**
```bash
sudo apt install rsync
```

rsync needs to be installed on both the local and remote machine.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 29. How do I extract a .tar.gz file? (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "How do I extract a .tar.gz file?",
    r"""The `tar` command handles archive files. A `.tar.gz` (or `.tgz`) is a tar archive compressed with gzip.

**Extract a .tar.gz file:**
```bash
tar -xzf archive.tar.gz
```

That is it. The files are extracted into the current directory.

**What the flags mean:**
- `-x` — extract
- `-z` — decompress gzip (the .gz part)
- `-f` — the next argument is the filename

**Extract to a specific directory:**
```bash
tar -xzf archive.tar.gz -C /path/to/destination/
```

**List contents without extracting (peek inside):**
```bash
tar -tzf archive.tar.gz
```

**Other archive formats:**

| File extension | Extract command |
|---|---|
| `.tar.gz` or `.tgz` | `tar -xzf file.tar.gz` |
| `.tar.bz2` | `tar -xjf file.tar.bz2` |
| `.tar.xz` | `tar -xJf file.tar.xz` |
| `.tar` | `tar -xf file.tar` |
| `.zip` | `unzip file.zip` |
| `.gz` (single file) | `gunzip file.gz` |
| `.7z` | `7z x file.7z` |

**Create a .tar.gz archive:**
```bash
# Archive a directory
tar -czf archive.tar.gz myfolder/

# Archive specific files
tar -czf backup.tar.gz file1.txt file2.txt file3.txt
```

**Tip:** Modern versions of `tar` can auto-detect the compression format, so `tar -xf archive.tar.gz` (without the `-z`) usually works too. But being explicit is clearer.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 30. How do I check my IP address? (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "How do I check my IP address?",
    r"""There are two kinds of IP addresses you might want to know.

**Your local/private IP address** (used on your local network):
```bash
ip addr
```

Look for the `inet` line under your network interface. For example:
```
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> ...
    inet 192.168.1.50/24 brd 192.168.1.255 scope global eth0
```

Here, `192.168.1.50` is your local IP address.

**Shorter version — just the IP:**
```bash
hostname -I
```

This prints your IP addresses without all the extra detail.

**Your public IP address** (how the internet sees you):
```bash
curl ifconfig.me
```

Or:
```bash
curl icanhazip.com
```

This contacts an external service that tells you your public IP.

**More detailed network info:**
```bash
# Show all interfaces with details
ip addr show

# Show just one interface
ip addr show eth0

# Show routing table (which gateway you use)
ip route

# Show DNS servers
cat /etc/resolv.conf
```

**Common network interfaces:**
- `lo` — loopback (127.0.0.1, the machine talking to itself)
- `eth0` or `enp0s3` — wired Ethernet
- `wlan0` or `wlp2s0` — wireless
- `docker0` — Docker bridge network (if Docker is installed)

**Note:** The older `ifconfig` command still works on some systems but `ip` is the modern replacement on Debian. If you see tutorials using `ifconfig`, just use `ip addr` instead.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 31. "Permission denied" (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "I got a 'Permission denied' error. What does it mean and how do I fix it?",
    r"""This error means you tried to do something that your user account does not have permission to do.

**Common causes and fixes:**

**1. You need root access for a system command:**
```bash
# This fails:
apt update
# bash: Permission denied

# Fix: use sudo
sudo apt update
```

Many system tasks require `sudo` — installing packages, editing files in `/etc`, restarting services, etc.

**2. A file is not set as executable:**
```bash
# This fails:
./myscript.sh
# bash: ./myscript.sh: Permission denied

# Fix: make it executable
chmod +x myscript.sh
./myscript.sh
```

**3. You do not own the file:**
```bash
# Check who owns it
ls -la thefile.txt

# If it belongs to another user, you can change ownership (as root):
sudo chown yourname:yourname thefile.txt

# Or adjust permissions:
sudo chmod 644 thefile.txt
```

**4. A directory is not accessible:**
```bash
# You need execute permission on a directory to enter it
ls -la /some/directory/..

# Fix:
sudo chmod +x /some/directory
```

**5. An SSH key has wrong permissions:**
```bash
# SSH refuses keys that are too open
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
chmod 700 ~/.ssh
```

**How to investigate:**
```bash
# Check the permissions
ls -la /path/to/file

# Check who you are
whoami

# Check your groups
groups
```

**Rule of thumb:** If the error is about a system file or system command, try `sudo`. If it is about your own file, check and fix the permissions with `chmod`.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 32. "Command not found" (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "I got 'command not found'. What do I do?",
    r"""This error means the shell cannot find the program you are trying to run. Here is how to troubleshoot it.

**1. Check for typos:**
The most common cause is a simple typo. Linux is case-sensitive:
```bash
Python3    # wrong (capital P)
python3    # correct
```

**2. The program is not installed:**
```bash
# Search for it in the repositories
apt search program-name

# Or check if a package provides the command
sudo apt install apt-file
sudo apt-file update
apt-file search bin/program-name
```

Then install it:
```bash
sudo apt install package-name
```

**3. The program is installed but not in your PATH:**
The shell looks for programs in specific directories (your PATH). Check if the program exists somewhere:
```bash
find / -name "program-name" -type f 2>/dev/null
```

If you find it, you can either:
- Run it with the full path: `/opt/myapp/bin/program-name`
- Add its directory to your PATH (see below)

**4. You need to refresh your shell:**
If you just installed something, you might need to reload:
```bash
hash -r            # clear the command cache
source ~/.bashrc   # reload your shell config
```

Or close and reopen your terminal.

**5. Common packages for common commands:**
| Command | Package to install |
|---|---|
| `curl` | `curl` |
| `wget` | `wget` |
| `git` | `git` |
| `make` | `build-essential` |
| `gcc` | `build-essential` |
| `pip` | `python3-pip` |
| `netstat` | `net-tools` |
| `ifconfig` | `net-tools` |
| `htop` | `htop` |
| `tree` | `tree` |

**6. You typed a bash built-in wrong:**
Some commands are built into the shell and are not programs:
```bash
type cd       # cd is a shell builtin
type echo     # echo is a shell builtin
```

These always exist and cannot be "not found" — if you get this error for them, something more fundamental is wrong.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 33. "No space left on device" (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "I got 'No space left on device'. How do I free up space?",
    r"""This means your disk is full. Here is how to find what is using space and clean it up.

**Step 1: Check disk usage**
```bash
df -h
```

Look for a partition that is at 100% (or close to it).

**Step 2: Find what is using the most space**
```bash
sudo du -h --max-depth=1 / | sort -hr | head -20
```

This shows the 20 largest top-level directories. Drill down into the biggest one:
```bash
sudo du -h --max-depth=1 /var | sort -hr | head -10
```

**Common space hogs and how to clean them:**

**1. APT package cache:**
```bash
# See how much space it uses
du -sh /var/cache/apt/archives

# Clean it
sudo apt clean
```

**2. Old packages no longer needed:**
```bash
sudo apt autoremove
```

**3. Old kernel versions:**
```bash
# See installed kernels
dpkg -l | grep linux-image

# Remove old ones (keep the current one!)
sudo apt purge linux-image-6.1.0-old-version
sudo apt autoremove
```

**4. Log files:**
```bash
# See log directory size
du -sh /var/log

# Rotate and compress old logs
sudo journalctl --vacuum-size=100M

# Or limit journal to 2 weeks
sudo journalctl --vacuum-time=2weeks
```

**5. Large files in your home directory:**
```bash
find /home -size +100M -type f 2>/dev/null
```

**6. /tmp or /var/tmp:**
```bash
sudo rm -rf /tmp/*
```

**7. Docker leftovers (if you use Docker):**
```bash
docker system prune -a
```

**Interactive tool for exploring disk usage:**
```bash
sudo apt install ncdu
sudo ncdu /
```

ncdu lets you browse the file system and see exactly what is taking up space.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 34. "apt: unable to locate package" (Ubuntu 24.04)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_UBU24,
    "I got 'E: Unable to locate package'. What do I do?",
    r"""This error means apt cannot find the package you are trying to install. Here are the common causes and fixes.

**1. Run apt update first:**
The most common fix. Your package list might be outdated:
```bash
sudo apt update
sudo apt install package-name
```

**2. Check the package name:**
Package names are case-sensitive and sometimes not what you expect:
```bash
# Search for the right name
apt search keyword
```

Common naming issues:
- Python packages: `python3-requests` not `requests`
- Development libraries: `libssl-dev` not `openssl-dev`
- The command name is not always the package name: the `ifconfig` command is in the `net-tools` package

**3. The package might be in a different repository:**
Some packages are not in the default repos. Ubuntu has several repository sections:

```bash
# Check if "universe" and "multiverse" repos are enabled
grep -r "universe\|multiverse" /etc/apt/sources.list /etc/apt/sources.list.d/

# Enable universe repo
sudo add-apt-repository universe
sudo apt update
```

**4. The package might not exist for your Ubuntu version:**
Some older packages get dropped from newer releases. Check on packages.ubuntu.com if the package exists for your version (Noble 24.04).

**5. You might need a PPA (Personal Package Archive):**
Some software is distributed via PPAs:
```bash
sudo add-apt-repository ppa:some/ppa
sudo apt update
sudo apt install package-name
```

Only add PPAs from sources you trust.

**6. The package exists but has a different name on Ubuntu:**
```bash
# Use apt-file to find which package provides a specific file or command
sudo apt install apt-file
sudo apt-file update
apt-file search bin/commandname
```

**Quick checklist:**
1. Did you run `sudo apt update` recently?
2. Is the package name spelled correctly?
3. Is the package in the universe/multiverse repos?
4. Does the package exist for Ubuntu 24.04?""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 35. I accidentally deleted a file (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "I accidentally deleted a file. Can I get it back?",
    r"""Unfortunately, on a standard Linux system, **deleted files are usually gone for good**. The command line does not have a recycle bin like Windows or macOS.

When you run `rm myfile.txt`, the file is immediately unlinked from the filesystem. There is no built-in "undo."

**That said, here are some options depending on your situation:**

**1. Check for backups first:**
- Do you have a backup system running? (rsync, borgbackup, timeshift, etc.)
- Is the file in a cloud sync folder? (Nextcloud, Dropbox, etc.)
- Did you commit it to a git repository?
  ```bash
  git log --all -- path/to/file
  git checkout HEAD -- path/to/file
  ```

**2. If the file was just deleted and is still open by a process:**
A file that is open by a program still exists on disk until the program closes it:
```bash
# Find the open file handle
sudo lsof | grep deleted
```

You may be able to recover it from `/proc/PID/fd/`.

**3. Data recovery tools (low success rate):**
If the file was truly important, immediately stop writing to the disk to avoid overwriting the data:
```bash
sudo apt install extundelete   # for ext3/ext4
sudo extundelete /dev/sda1 --restore-file path/to/file
```

Or use `testdisk` / `photorec` for a more thorough scan. Success is not guaranteed.

**4. If you are using btrfs or ZFS:**
These file systems support snapshots. You might have automatic snapshots you can restore from:
```bash
# btrfs snapshots
sudo btrfs subvolume list /

# Check for timeshift snapshots
sudo timeshift --list
```

**Prevention is better than recovery:**
- Use `rm -i` to get a confirmation prompt before deleting
- Set up regular backups (rsync, borgbackup, or restic)
- Use version control (git) for code and documents
- Some people add `alias rm='rm -i'` to their `.bashrc` (though this creates a dangerous habit of relying on it)""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 36. My terminal is messed up/frozen (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "My terminal is messed up or frozen. What do I do?",
    r"""Do not panic. Here are the most common terminal problems and how to fix them.

**Terminal is frozen (not accepting input):**

You probably pressed **Ctrl+S** by accident. This pauses terminal output (it is an old feature called XOFF flow control).

**Fix: Press Ctrl+Q** to resume. That is it.

**Terminal shows garbled text or weird characters:**

This happens when you accidentally `cat` a binary file or a program messes up the terminal encoding.

**Fix: Type `reset` and press Enter.**
```bash
reset
```

You might not be able to see what you type — just type it blindly and press Enter. The `reset` command reinitializes the terminal.

An alternative:
```bash
tput reset
```

Or press **Ctrl+L** to just clear the screen (this does not fix encoding issues though).

**A command is running and you want to stop it:**
- Press **Ctrl+C** to interrupt it
- Press **Ctrl+Z** to suspend it (then `kill %1` to kill it)
- Press **Ctrl+D** to send end-of-file (useful when a program is waiting for input)

**Your prompt disappeared or looks wrong:**
```bash
# Reload your shell configuration
source ~/.bashrc
```

Or close and reopen the terminal window.

**Terminal is showing input on the same line, not wrapping properly:**
```bash
stty sane
```

This resets all terminal line settings to sane defaults.

**If nothing else works:**
- Close the terminal window and open a new one
- If using SSH, the connection may have dropped — close it and reconnect
- If you are in `screen` or `tmux`, detach with **Ctrl+A D** (screen) or **Ctrl+B D** (tmux) and reattach

**Quick reference:**
| Problem | Fix |
|---|---|
| Frozen terminal | Ctrl+Q |
| Garbled characters | Type `reset` and Enter |
| Running command you want to stop | Ctrl+C |
| Prompt looks wrong | `source ~/.bashrc` |
| Line settings messed up | `stty sane` |""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 37. How do I exit vim? (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "How do I exit vim?",
    r"""This is famously one of the most common questions from new Linux users. Here is the answer.

**The quick answer:**
Press **Esc**, then type **:q** and press **Enter**.

**If you made changes and want to save them:**
Press **Esc**, then type **:wq** and press **Enter**.

**If you made changes and want to discard them:**
Press **Esc**, then type **:q!** and press **Enter**.

**Why is this confusing?**
vim has different modes. When you open it, you are in **Normal mode**, where typing letters executes commands rather than inserting text. The **Esc** key always brings you back to Normal mode, and **:** lets you type a command.

**Here is the full picture:**

| Keys | What it does |
|---|---|
| Esc, then `:q` Enter | Quit (fails if you have unsaved changes) |
| Esc, then `:q!` Enter | Quit and discard changes |
| Esc, then `:w` Enter | Save (write) without quitting |
| Esc, then `:wq` Enter | Save and quit |
| Esc, then `ZZ` | Save and quit (shortcut, no colon needed) |
| Esc, then `ZQ` | Quit without saving (shortcut) |

**If you accidentally entered vim** and just want to leave, type:
```
Esc :q! Enter
```

Press Esc first (in case you are in insert mode), then type the colon, q, exclamation mark, and press Enter.

**How to avoid vim if you do not want it:**
Some commands (like `git commit`) open vim by default. You can change the default editor:
```bash
# Set nano as the default editor (much simpler)
export EDITOR=nano

# Make it permanent by adding it to your shell config
echo 'export EDITOR=nano' >> ~/.bashrc
```""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 38. Linux directory structure (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "Can you explain the Linux directory structure?",
    r"""Linux organizes everything in a single tree starting from `/` (root). Here is what the main directories are for.

```
/                       The root — everything starts here
├── /bin                Essential command binaries (ls, cp, mv, bash)
├── /sbin               Essential system binaries (fdisk, init, ip)
├── /boot               Kernel and bootloader files
├── /dev                Device files (your hard drive, USB devices, etc.)
├── /etc                System configuration files
├── /home               User home directories (/home/david, /home/alice)
├── /lib                Essential shared libraries
├── /media              Mount points for removable media (USB, CD)
├── /mnt                Temporary mount point for manual mounts
├── /opt                Optional/third-party software
├── /proc               Virtual filesystem with process and kernel info
├── /root               Home directory of the root user
├── /run                Runtime variable data (PIDs, sockets)
├── /srv                Data for services (web server files, etc.)
├── /sys                Virtual filesystem for kernel/hardware info
├── /tmp                Temporary files (cleared on reboot)
├── /usr                User programs and data
│   ├── /usr/bin        Most user commands
│   ├── /usr/lib        Libraries for /usr/bin programs
│   ├── /usr/local      Locally installed software (compiled from source)
│   └── /usr/share      Architecture-independent data (docs, icons)
└── /var                Variable data that changes during operation
    ├── /var/log        Log files
    ├── /var/cache      Application caches (apt cache, etc.)
    └── /var/tmp        Temporary files preserved between reboots
```

**The most important ones for everyday use:**

- **`/home`** — your personal files are here. `~` is shorthand for your home directory.
- **`/etc`** — system settings. Want to change the hostname? Edit `/etc/hostname`. DNS settings? `/etc/resolv.conf`. Almost all configuration lives here.
- **`/var/log`** — system and application logs. The first place to look when troubleshooting.
- **`/tmp`** — a place for temporary files. Anyone can write here, and it is cleaned regularly.
- **`/usr/bin`** — where most of the programs you run live.

**On modern Debian (including trixie)**, `/bin` is actually a symlink to `/usr/bin`, and `/sbin` links to `/usr/sbin`. This is called the "merged /usr" layout. Everything ends up under `/usr`.

**How this differs from Windows:**
- No drive letters (C:\, D:\) — just one tree
- Configuration is in text files under `/etc`, not in a registry
- Extra drives are "mounted" somewhere in the tree (like `/mnt/data`)""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 39. stdin, stdout, stderr and redirection (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "Can you explain stdin, stdout, stderr and redirection?",
    r"""Every program in Linux has three standard data streams. Understanding them is key to using the command line effectively.

**The three streams:**

1. **stdin (standard input)** — where a program reads its input from. By default, this is your keyboard.
2. **stdout (standard output)** — where a program writes its normal output. By default, this is your terminal screen.
3. **stderr (standard error)** — where a program writes error messages. Also goes to your terminal screen by default.

**Redirection — sending these streams somewhere else:**

**Redirect stdout to a file:**
```bash
# Write output to a file (overwrites the file)
ls /home > filelist.txt

# Append output to a file
ls /home >> filelist.txt
```

**Redirect stderr to a file:**
```bash
# Only errors go to the file, normal output still appears on screen
find / -name "*.conf" 2> errors.txt

# Discard all errors
find / -name "*.conf" 2>/dev/null
```

`2>/dev/null` is very common — `/dev/null` is a "black hole" that discards anything written to it.

**Redirect both stdout and stderr:**
```bash
# Both to the same file
command > output.txt 2>&1

# Or the shorter version (bash)
command &> output.txt
```

**Redirect stdin (feed a file as input):**
```bash
# Instead of typing input, read it from a file
sort < unsorted_list.txt

# Combine input and output redirection
sort < unsorted.txt > sorted.txt
```

**A practical example:**
```bash
# Save command output AND errors to separate files
make 1>build_output.log 2>build_errors.log

# Run a backup and log everything
rsync -av /data /backup > backup.log 2>&1
```

**The numbers explained:**
- `0` = stdin
- `1` = stdout (the `>` shorthand defaults to `1>`)
- `2` = stderr

**Here document (heredoc) — multi-line input:**
```bash
cat << EOF
This is line one
This is line two
EOF
```

This feeds multiple lines to a command's stdin.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 40. What are pipes? (no system prompt)
# ---------------------------------------------------------------------------
conversations.append(make(
    None,
    "What are pipes in Linux and how do they work?",
    r"""A pipe (`|`) connects the output of one command to the input of another. It lets you chain commands together to build powerful one-liners.

**Basic syntax:**
```bash
command1 | command2
```

The stdout of command1 becomes the stdin of command2.

**Simple examples:**
```bash
# List files and search for a specific one
ls -la | grep ".txt"

# Count how many files are in a directory
ls | wc -l

# Sort a file and remove duplicates
sort names.txt | uniq

# Show the 10 largest files in a directory
du -sh * | sort -hr | head -10
```

**How to think about pipes:**
Think of it like an assembly line. Each command does one small job and passes its result to the next:

```bash
cat /var/log/syslog | grep "error" | sort | uniq -c | sort -rn | head -5
```

This pipeline:
1. `cat` reads the log file
2. `grep` filters for lines containing "error"
3. `sort` sorts the matching lines alphabetically
4. `uniq -c` counts consecutive duplicates
5. `sort -rn` sorts numerically, highest first
6. `head -5` shows only the top 5 results

The result: the 5 most common error messages in your system log.

**More practical examples:**
```bash
# Find running processes using lots of memory
ps aux | sort -k4 -rn | head -10

# See who is logged in
who | wc -l

# Show disk usage sorted by size
df -h | sort -k5 -rn

# Extract unique IP addresses from a log
cat access.log | awk '{print $1}' | sort -u

# Search command history
history | grep "apt install"
```

**Important detail:** Pipes only connect **stdout**. Error messages (stderr) are not piped — they still appear on your screen. If you want to pipe both:
```bash
command1 2>&1 | command2
```

**The Unix philosophy:** This is the heart of how Linux command-line tools are designed. Each tool does one thing well, and pipes let you combine them. Instead of one giant program, you build solutions from small, composable pieces.""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 41. What is the PATH variable? (Debian)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_DEB13,
    "What is the PATH variable?",
    r"""PATH is an environment variable that tells the shell where to look for programs when you type a command.

**When you type a command like `ls`:**
The shell does not search the entire filesystem for a program called `ls`. Instead, it looks in a specific list of directories — that list is your PATH.

**See your current PATH:**
```bash
echo $PATH
```

Output looks something like:
```
/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin
```

The directories are separated by colons. The shell searches them in order, left to right. The first match wins.

**Why this matters:**
If you install a program but it is not in any PATH directory, the shell will say "command not found."

**Find where a command lives:**
```bash
which python3
# /usr/bin/python3

which ls
# /usr/bin/ls
```

**Adding a directory to your PATH:**

Temporarily (for this session only):
```bash
export PATH="$PATH:/opt/myapp/bin"
```

Permanently (add to your shell config):
```bash
echo 'export PATH="$PATH:/opt/myapp/bin"' >> ~/.bashrc
source ~/.bashrc
```

**Common PATH directories on Debian:**
| Directory | What is in it |
|---|---|
| `/usr/bin` | Most user commands |
| `/usr/sbin` | System administration commands |
| `/usr/local/bin` | Locally compiled or custom programs |
| `/home/user/.local/bin` | Per-user scripts and programs |

**The order matters:**
If two programs have the same name, the one in the earlier PATH directory wins. This is how you can override a system program with your own version by putting it in `/usr/local/bin`.

**A common mistake:**
Overwriting PATH instead of appending to it:
```bash
# WRONG — this loses all other directories:
export PATH="/my/new/path"

# CORRECT — this adds to existing PATH:
export PATH="$PATH:/my/new/path"
```""",
    "flm_beginner"
))

# ---------------------------------------------------------------------------
# 42. Difference between apt and apt-get? (Ubuntu 24.04)
# ---------------------------------------------------------------------------
conversations.append(make(
    SYS_UBU24,
    "What is the difference between apt and apt-get?",
    r"""Both `apt` and `apt-get` are command-line tools for managing packages on Debian-based systems. They do the same job, but `apt` is newer and nicer to use.

**`apt-get`** — the older tool:
- Has been around since 1998
- Stable, well-documented, and very reliable
- Output is designed for scripts and automation
- Will never change its output format (important for scripts)

**`apt`** — the newer tool (introduced around 2014):
- Combines the most common features of `apt-get` and `apt-cache` into one command
- Has a nicer, more user-friendly output (colors, progress bars)
- Designed for humans typing at a terminal

**Comparison of common commands:**

| Task | apt (use this) | apt-get (older) |
|---|---|---|
| Update package list | `sudo apt update` | `sudo apt-get update` |
| Install a package | `sudo apt install vim` | `sudo apt-get install vim` |
| Remove a package | `sudo apt remove vim` | `sudo apt-get remove vim` |
| Upgrade all packages | `sudo apt upgrade` | `sudo apt-get upgrade` |
| Full upgrade | `sudo apt full-upgrade` | `sudo apt-get dist-upgrade` |
| Search for a package | `apt search vim` | `apt-cache search vim` |
| Show package info | `apt show vim` | `apt-cache show vim` |
| Remove unused deps | `sudo apt autoremove` | `sudo apt-get autoremove` |
| List installed | `apt list --installed` | `dpkg -l` |

**Which should you use?**

- **For everyday use in the terminal:** Use `apt`. It is simpler and gives better output.
- **In scripts and automation:** Use `apt-get`. Its output is stable and machine-parseable. `apt` may warn that its output format is not guaranteed to stay the same.

For example, in a Dockerfile or a bash script:
```bash
# In scripts, use apt-get
apt-get update && apt-get install -y curl
```

At your terminal:
```bash
# Interactive use, use apt
sudo apt update
sudo apt install curl
```

**Bottom line:** They do the same thing. `apt` is the modern way for interactive use. You will see both in tutorials — either works fine.""",
    "flm_beginner"
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
