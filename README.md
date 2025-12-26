# Tiny Cheetah
```
░░      ░░░  ░░░░  ░░        ░░        ░░        ░░░      ░░░  ░░░░  ░
▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒
▓  ▓▓▓▓▓▓▓▓        ▓▓      ▓▓▓▓      ▓▓▓▓▓▓▓  ▓▓▓▓▓  ▓▓▓▓  ▓▓        ▓
█  ████  ██  ████  ██  ████████  ███████████  █████        ██  ████  █
██      ███  ████  ██        ██        █████  █████  ████  ██  ████  █
```
Fast local training and inference using Tinygrad.

![main interface 10272025](media/main_10272025.png "main 10272025")

![chat screen 10272025](media/chat_screen10272025.png "chat screen 10272025")

![train screen 10272025](media/train_screen10272025.png "train screen 10272025")

## About
WIP local TUI based model conversation and training at home interface with distributed network based inference.

TLDR: Chat and train models at home with all your devices and online with other users

## Progress
### 12/25/2025
- Chat: fast log restore, silent model clearing, auto-scroll, and streamlined escape/back navigation.
- Train: stable UI with network node counter; training path editor separated into its own screen.
- Network/Orchestration: free, password-optional compute sharing (no billing). Host/peer hardware reports (CPU/RAM/GPU, TC_DEVICE), LAN discovery, manual connect, peer directory with HL-style server list (lock/name/CPU+RAM/GPU/ping), ASCII host map with status colors, and a host manage screen with GPU inventory + server details.
- Scheduler: remote-first tensor dispatch with local fallback serialization, basic ping measurement on connect.
- Identity: lightweight username/fingerprint, PGP removed.


## Network Orchestration
Tiny Cheetah now includes an experimental orchestration layer so users can lend or borrow compute from each other.

* **Host/peer compute** – each host advertises hardware (CPU/RAM/GPU), optional password, and MOTD;
* **Identity** – usernames with lightweight fingerprints for trust without centralized auth.
* **Socket execution** – the host listens for tensor jobs and executes them as if they were local model calls (currently a stub that echoes requests).
* **LAN discovery** – UDP broadcasts let neighbouring machines advertise themselves as part of a pooled “local cluster”.
* **TUI integration** – the `Network` button in the main menu, chat, and train screens opens the orchestration console; node counters show active peers.
* **Host map** – new ASCII host map renders your node and connected peers with status colors (green=online, yellow=busy, red=offline) so you can see the topology at a glance.
