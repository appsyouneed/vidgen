#!/bin/bash

echo "=== Verification Script ==="
echo ""

# Check Python syntax
echo "1. Checking Python syntax..."
python3 -c "import ast; ast.parse(open('app.py').read())" && echo "   ✓ app.py syntax valid" || echo "   ✗ app.py has syntax errors"
echo ""

# Check for ZeroGPU references
echo "2. Checking for ZeroGPU references..."
if grep -q "import spaces\|@spaces\|IS_ZERO_GPU\|aoti\." app.py; then
    echo "   ✗ Found ZeroGPU references"
    grep -n "import spaces\|@spaces\|IS_ZERO_GPU\|aoti\." app.py
else
    echo "   ✓ No ZeroGPU references found"
fi
echo ""

# Check required files
echo "3. Checking required files..."
for file in app.py requirements.txt setup.sh packages.txt; do
    if [ -f "$file" ]; then
        echo "   ✓ $file exists"
    else
        echo "   ✗ $file missing"
    fi
done
echo ""

# Check aoti.py is removed
echo "4. Checking aoti.py is removed..."
if [ -f "aoti.py" ]; then
    echo "   ✗ aoti.py still exists (should be removed)"
else
    echo "   ✓ aoti.py removed"
fi
echo ""

# Check network binding
echo "5. Checking network binding..."
if grep -q 'server_name="0.0.0.0"' app.py; then
    echo "   ✓ Configured for network access (0.0.0.0)"
else
    echo "   ✗ Not configured for network access"
fi
echo ""

# Check VAE optimizations
echo "6. Checking VAE optimizations..."
if grep -q "pipe.vae.enable_slicing()" app.py && grep -q "pipe.vae.enable_tiling()" app.py; then
    if ! grep -q "# pipe.vae.enable_slicing()\|# pipe.vae.enable_tiling()" app.py; then
        echo "   ✓ VAE optimizations enabled"
    else
        echo "   ✗ VAE optimizations commented out"
    fi
else
    echo "   ✗ VAE optimizations not found"
fi
echo ""

# Check cache persistence
echo "7. Checking cache persistence..."
if grep -q "# if os.path.exists(CACHE_DIR):" app.py; then
    echo "   ✓ Cache deletion disabled (models will persist)"
else
    echo "   ⚠ Cache deletion code not commented out"
fi
echo ""

echo "=== Verification Complete ==="
