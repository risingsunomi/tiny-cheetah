# Cheetah
```
░░      ░░░  ░░░░  ░░        ░░        ░░        ░░░      ░░░  ░░░░  ░
▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒
▓  ▓▓▓▓▓▓▓▓        ▓▓      ▓▓▓▓      ▓▓▓▓▓▓▓  ▓▓▓▓▓  ▓▓▓▓  ▓▓        ▓
█  ████  ██  ████  ██  ████████  ███████████  █████        ██  ████  █
██      ███  ████  ██        ██        █████  █████  ████  ██  ████  █
```
[tinygrad](https://tinygrad.org/) and [pytorch](https://pytorch.org/) based distributed and local machine learning model training, agent orchestration and chat inference

## install

Default CPU/MPS install:

```bash
pip install -e .
```

CUDA and ROCm PyTorch wheels need the matching PyTorch wheel index. Examples:

```bash
TC_TORCH_VARIANT=cu130 pip install -e . --extra-index-url https://download.pytorch.org/whl/cu130
TC_TORCH_VARIANT=rocm7.1 pip install -e . --extra-index-url https://download.pytorch.org/whl/rocm7.1
```

For tinygrad on AMD/ROCm, use `TC_LLM_BACKEND=tinygrad` with `TC_TINYGRAD_DEVICE=AMD`. To force tinygrad's HIP interface, also set `DEV=AMD:HIP`.

![main interface](media/main_03162026.png "main 03162026")

## chat
![chat screen](media/chat_03162026.png "chat screen 03162026")

## train
![train screen](media/train_03182026.png "train screen 03182026")

![train screen settings](media/train_settings_03182026.png "train screen settings 03182026")

![train screen train path](media/train_path_03182026.png "train screen train path 03182026")

## agent
![agent screen](media/agent_screen_03182026.png "chat screen 03182026")

![agent screen config](media/agent_config_03162026.png "chat screen config 03162026")

## networking
![network screen](media/network_03182026.png "network screen 03182026")

## settings
![settings screen](media/settings_03182026.png "settings screen 03182026")

