#!/usr/bin/env bash
# =============================================================================
# Project Environment Setup Script
# Python 3.10 | PyTorch 2.8.0 | PyTorch Geometric 2.5.3
#
# ⚠️  CUDA MISMATCH — READ BEFORE RUNNING:
#     Your old working environment used PyTorch built for CUDA 12.8 (cu128).
#     Your current nvidia-smi reports CUDA 12.3.
#     A cu128 binary requires a driver that supports CUDA 12.8+.
#
#     OPTION A — Update your driver first (recommended, keeps exact old versions):
#       sudo apt update && sudo apt install nvidia-driver-570 && sudo reboot
#       # After reboot, nvidia-smi should show CUDA Version: 12.8
#       bash setup_env.sh --cuda 128
#
#     OPTION B — Keep CUDA 12.3, use PyTorch 2.1.0+cu121 instead:
#       bash setup_env.sh --cuda 121
#       # All your project code will work identically.
# =============================================================================

set -e

# --- Argument parsing ---
CUDA_OPT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda) CUDA_OPT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$CUDA_OPT" == "128" ]]; then
    CUDA_TAG="cu128"
    TORCH_VERSION="2.8.0"
    TORCH_INSTALL="torch==2.8.0"
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    PYG_WHEEL="https://data.pyg.org/whl/torch-2.8.0+cu128.html"
elif [[ "$CUDA_OPT" == "121" ]]; then
    CUDA_TAG="cu121"
    TORCH_VERSION="2.1.0"
    TORCH_INSTALL="torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121"
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    PYG_WHEEL="https://data.pyg.org/whl/torch-2.1.0+cu121.html"
else
    echo ""
    echo "  ERROR: Specify --cuda 128 (after driver update) or --cuda 121"
    echo "  Examples:"
    echo "    bash setup_env.sh --cuda 128"
    echo "    bash setup_env.sh --cuda 121"
    echo ""
    exit 1
fi

ENV_NAME="project_env"
PYTHON_VERSION="3.10"

echo "=============================================="
echo " Environment : $ENV_NAME"
echo " Python      : $PYTHON_VERSION"
echo " PyTorch     : $TORCH_VERSION ($CUDA_TAG)"
echo "=============================================="

# ------------------------------------------------------------------------------
# 1. Create conda environment
# ------------------------------------------------------------------------------
echo ""
echo "[1/6] Creating conda environment..."
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "      ✓ '$ENV_NAME' activated."

# ------------------------------------------------------------------------------
# 2. Install PyTorch
# ------------------------------------------------------------------------------
echo ""
echo "[2/6] Installing PyTorch $TORCH_VERSION ($CUDA_TAG)..."
pip install $TORCH_INSTALL --index-url "$TORCH_INDEX"

# ------------------------------------------------------------------------------
# 3. Verify GPU
# ------------------------------------------------------------------------------
echo ""
echo "[3/6] Verifying GPU access..."
python - <<'PYEOF'
import torch, sys
if torch.cuda.is_available():
    print(f"  ✓ CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}")
else:
    print("  ✗ CUDA not available — check driver installation and reboot if needed.")
    sys.exit(1)
PYEOF

# ------------------------------------------------------------------------------
# 4. Install PyG C++ sparse dependencies (must match torch+cuda tag exactly)
# ------------------------------------------------------------------------------
echo ""
echo "[4/6] Installing PyG sparse deps (torch-scatter, torch-sparse, etc.)..."
pip install \
    torch-scatter==2.1.2 \
    torch-sparse==0.6.18 \
    torch-cluster \
    torch-spline-conv \
    -f "$PYG_WHEEL"

# ------------------------------------------------------------------------------
# 5. Install torch-geometric
# ------------------------------------------------------------------------------
echo ""
echo "[5/6] Installing torch-geometric==2.5.3..."
pip install torch-geometric==2.5.3

# ------------------------------------------------------------------------------
# 6. Install remaining stack at pinned versions from old environment
# ------------------------------------------------------------------------------
echo ""
echo "[6/6] Installing data/ML stack (pinned versions)..."
pip install \
    numpy==1.22.4 \
    pandas==1.4.3 \
    scipy==1.8.1 \
    joblib==1.5.2 \
    networkx==2.8.4 \
    matplotlib==3.5.2 \
    seaborn==0.13.2 \
    pillow==11.3.0 \
    scikit-learn==1.1.1 \
    tqdm==4.67.1 \
    sympy==1.14.0

pip freeze > requirements_pinned.txt
echo "      ✓ Full pinned snapshot → requirements_pinned.txt"

# ------------------------------------------------------------------------------
echo ""
echo "=============================================="
echo " ✓ Setup complete!"
echo " Activate : conda activate $ENV_NAME"
echo " Verify   : python -c \"import torch; print(torch.cuda.get_device_name(0))\""
echo "=============================================="
