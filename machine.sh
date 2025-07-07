#!/bin/bash
echo "=============================================="
echo "       COMPLETE SYSTEM CONFIGURATION"
echo "=============================================="

echo ""
echo "=== GPU INFORMATION ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
    echo "GPU Details:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap --format=csv
else
    echo "NVIDIA GPU not found or nvidia-smi not available"
fi

echo ""
echo "=== CUDA INFORMATION ==="
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "CUDA compiler not found"
fi

echo ""
echo "CUDA Environment Variables:"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_PATH: $CUDA_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

echo ""
echo "=== PYTORCH CUDA CHECK ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('CUDA not available in PyTorch')
"

echo ""
echo "=== CPU INFORMATION ==="
echo "CPU Model:"
lscpu | grep "Model name"
echo "CPU Architecture:"
lscpu | grep "Architecture"
echo "CPU Cores:"
lscpu | grep -E "CPU\(s\)|Thread|Core"

echo ""
echo "=== MEMORY INFORMATION ==="
free -h
echo ""
echo "Memory Details:"
cat /proc/meminfo | grep -E "MemTotal|MemAvailable|SwapTotal"

echo ""
echo "=== DISK INFORMATION ==="
df -h

echo ""
echo "=== SYSTEM INFORMATION ==="
echo "Hostname: $(hostname)"
echo "Kernel: $(uname -r)"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "Uptime: $(uptime -p)"

echo ""
echo "=== PYTHON ENVIRONMENT ==="
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo ""
echo "Key packages:"
pip list | grep -E "(torch|torchvision|gsplat|cuda|numpy|pillow)" 2>/dev/null || echo "Package info unavailable"

echo ""
echo "=== NETWORK INFORMATION ==="
echo "Network interfaces:"
ip addr show | grep -E "inet.*scope global" | head -3

echo ""
echo "=== ENVIRONMENT SUMMARY ==="
echo "User: $(whoami)"
echo "Working directory: $(pwd)"
echo "Shell: $SHELL"
echo "PATH (CUDA related): $(echo $PATH | tr ':' '\n' | grep -i cuda | head -3)"

echo ""
echo "=== GSPLAT CHECK ==="
python -c "
try:
    import gsplat
    print(f'gsplat version: {gsplat.__version__}')
    print('gsplat successfully imported')
except ImportError as e:
    print(f'gsplat not available: {e}')
" 2>/dev/null

echo ""
echo "=============================================="
echo "         SYSTEM CHECK COMPLETE"
echo "=============================================="
