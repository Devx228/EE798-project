# Setup on Windows (RTX 4050 laptop)

You have two options. Everything except the Triton kernel works in both.

| | Native Windows | WSL2 (recommended) |
|---|---|---|
| PyTorch + CUDA training | yes | yes |
| Triton kernel | via the community `triton-windows` package | yes (ships with PyTorch) |
| `torch.compile` | limited | yes |
| Closest to how industry runs it | no | yes (Linux) |

Before either option, update the **NVIDIA Game Ready / Studio driver** on Windows. WSL2 uses the
Windows driver, so **do not** install an NVIDIA driver inside Linux.

## Option A: WSL2 (recommended)

In PowerShell (as administrator):

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot, open "Ubuntu" from the Start menu, create a user, then:

```bash
nvidia-smi                                   # should list the RTX 4050
sudo apt update && sudo apt install -y python3-venv python3-pip git
git clone https://github.com/Devx228/<repo-name>.git && cd <repo-name>
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130   # or the command from pytorch.org/get-started
pip install -e ".[dev]"
pytest                                       # includes the Triton kernel test on the GPU
```

Keep the repo inside the Linux filesystem (`~/...`), not under `/mnt/c/...`. File access across
the boundary is very slow.

VS Code: install the "WSL" extension, then run `code .` from the Ubuntu terminal.

## Option B: native Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu130   # or the command from pytorch.org/get-started
pip install -e ".[dev]"
pip install triton-windows    # optional, only needed for the Triton kernel / benchmark
pytest
```

To test the Triton kernel without a GPU (slow, CPU interpreter):

```powershell
$env:TRITON_INTERPRET=1; pytest tests/test_triton_scan.py
```

## Laptop tips for reproducible numbers

- Plug the charger in and set the Windows power mode to **Best performance**. On battery the GPU
  clocks drop a lot and benchmark numbers become meaningless.
- Close browsers and games while benchmarking. Check that `nvidia-smi` shows no other GPU
  processes.
- 6 GB of VRAM: if a sweep runs out of memory, lower `train.batch_size` (e.g. `train.batch_size=32`);
  the benchmark already records out-of-memory as an empty cell instead of crashing.
- Record the driver version, CUDA version and PyTorch version for the report
  (`python -c "import torch; print(torch.__version__, torch.version.cuda)"` and `nvidia-smi`).
