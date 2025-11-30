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

TLDR: Chat and train models at home with all your devices and MMO with other users

## Progress
### 10/27/2025
Working chat interface and training interface but more UX work and optimization needed. We are finding there is this freezing or delay when loading a model into memory while in the chat interface. For training, better stop before running out of memory points will be needed as causing it to run unmanaged will thrash swap memory. Integrating tinyjit and quantization model usage for faster tokens/sec

## Network Orchestration
Tiny Cheetah now includes an experimental orchestration layer so users can lend or borrow compute from each other.

* **Offer catalog** – each host advertises a description, currency, and hourly rate for their compute resources (think lightweight VM marketplace).
* **PGP identity** – usernames are bound to lightweight PGP-style fingerprints for trust without centralized auth.
* **Socket execution** – the host listens for tensor jobs and executes them as if they were local model calls (currently a stub that echoes requests).
* **LAN discovery** – UDP broadcasts let neighbouring machines advertise themselves as part of a pooled “local cluster”.
* **TUI integration** – the new `Network` button in the main menu, chat, and train screens opens the orchestration console, while a node counter is shown on each screen header.
* **Billing automation** – renters can start/stop timed sessions against any peer, and hosts can edit their offer (rate, currency, payment instructions) from within the Billing view.
