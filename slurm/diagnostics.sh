echo "=========================================="
echo "ENVIRONMENT VERIFICATION"
echo "=========================================="

# Check if activation actually worked
echo "1. CONDA/MAMBA ENVIRONMENT:"
echo "Active env: $CONDA_DEFAULT_ENV"
echo "Conda prefix: $CONDA_PREFIX"
echo "Conda exe: $CONDA_EXE"
echo ""

# Python location and version
echo "2. PYTHON VERIFICATION:"
which python
python --version
echo "Python path:"
python -c "import sys; print(sys.executable)"
echo "Python sys.path:"
python -c "import sys; print('\n'.join(sys.path))"
echo ""

# Check PyTorch installation
echo "3. PYTORCH PACKAGE INFO:"
mamba list | grep -E "pytorch|cuda|cudnn|cudatoolkit" || echo "No pytorch/cuda packages found"
echo ""
python -c "import importlib.util; spec = importlib.util.find_spec('torch'); print(f'Torch location: {spec.origin if spec else \"NOT FOUND\"}')"
echo ""

# Environment variables
echo "4. CUDA-RELATED ENVIRONMENT VARIABLES:"
env | grep -E "CUDA|cuda" | sort
echo ""
echo "5. LD_LIBRARY_PATH:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n'
echo ""

# Check what libraries would be loaded
echo "6. TORCH SHARED LIBRARIES:"
find $CONDA_PREFIX -name "*torch*.so" -type f 2>/dev/null | head -5
echo ""
echo "7. CUDA LIBRARIES IN ENV:"
find $CONDA_PREFIX -name "libcuda*.so*" -type f 2>/dev/null | head -10
echo ""

# Check for conflicting libraries
echo "8. CHECKING FOR DUPLICATE CUDA LIBRARIES:"
echo "In conda env:"
ls -la $CONDA_PREFIX/lib/libcudart.so* 2>/dev/null || echo "No libcudart in conda env"
echo "In system:"
ldconfig -p 2>/dev/null | grep libcudart || echo "No libcudart in system ldconfig"
echo ""

# Test Python imports step by step
echo "9. STEP-BY-STEP IMPORT TEST:"
echo "Testing: import sys"
python -c "import sys; print('✓ sys imported')"

echo "Testing: import numpy"
python -c "import numpy; print('✓ numpy imported')"

echo "Testing: import torch (THIS IS WHERE IT MIGHT HANG)"
timeout 5 python -c "import torch; print('✓ torch imported')" || echo "✗ Torch import timed out after 5 seconds"

# If torch imported successfully, get more info
if [ $? -eq 0 ]; then
    echo ""
    echo "10. TORCH CONFIGURATION:"
    python -c "
import torch
print(f'Torch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'cuDNN version: {torch.backends.cudnn.version()}')
"
fi

# Check process limits that might affect CUDA
echo ""
echo "11. PROCESS LIMITS:"
ulimit -a | grep -E "stack|open files"

# Check NVIDIA libraries that Python would load
echo ""
echo "12. LDD CHECK FOR TORCH:"
if [ -f "$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib/libtorch_python.so" ]; then
    echo "First 10 libraries that torch would load:"
    ldd $CONDA_PREFIX/lib/python3.11/site-packages/torch/lib/libtorch_python.so 2>/dev/null | head -10
else
    echo "Torch library not found at expected location"
fi

# Final check for library conflicts
echo ""
echo "13. POTENTIAL LIBRARY CONFLICTS:"
echo "Checking for mixed library versions:"
find $CONDA_PREFIX/lib -name "*.so*" -type f 2>/dev/null | xargs -I {} ldd {} 2>/dev/null | grep -E "not found|cuda" | sort -u | head -10

echo ""
echo "=========================================="
echo "DIAGNOSTICS COMPLETE"
echo "=========================================="