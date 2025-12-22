#!/bin/bash
# diagnose-malloc-trim.sh
# Check why malloc_trim is returning 0

set -euo pipefail

echo "========================================"
echo "malloc_trim Diagnostic Script"
echo "========================================"
echo ""

# Get a pod
POD=$(kubectl get pods -l app=hpa-flask-app -o jsonpath='{.items[0].metadata.name}')
echo "Testing pod: $POD"
echo ""

# Check 1: PYTHONMALLOC environment variable
echo "1. Checking PYTHONMALLOC setting:"
kubectl exec "$POD" -- env | grep PYTHON || echo "  ❌ PYTHONMALLOC not set!"
echo ""

# Check 2: Python memory allocator in use
echo "2. Checking active Python allocator:"
kubectl exec "$POD" -- python3 -c "
import sys
import os
print(f'  PYTHONMALLOC env: {os.environ.get(\"PYTHONMALLOC\", \"NOT SET\")}')
print(f'  sys.implementation: {sys.implementation.name}')
try:
    import _testcapi
    print(f'  Allocator: {_testcapi.pymem_getallocator(\"mem\")}')
except:
    print('  (Cannot determine allocator - _testcapi not available)')
"
echo ""

# Check 3: Test malloc_trim with some allocated memory
echo "3. Testing malloc_trim with allocated memory:"
kubectl exec "$POD" -- python3 -c "
import ctypes
import gc

# Allocate 50MB
data = bytearray(50 * 1024 * 1024)
print(f'  Allocated 50MB')

# Free it
del data
gc.collect()
print(f'  Freed and GC collected')

# Try malloc_trim
libc = ctypes.CDLL('libc.so.6')
result = libc.malloc_trim(0)
print(f'  malloc_trim result: {result}')

if result == 0:
    print('  ⚠️  malloc_trim returned 0 (no memory released)')
    print('  This usually means:')
    print('    - PYTHONMALLOC=malloc is not set')
    print('    - Or Python is using pymalloc (default)')
else:
    print('  ✅ malloc_trim successfully released memory')
"
echo ""

# Check 4: Check Docker image environment
echo "4. Checking Docker image configuration:"
IMAGE=$(kubectl get pod "$POD" -o jsonpath='{.spec.containers[0].image}')
echo "  Image: $IMAGE"
echo ""
echo "  Checking if image has PYTHONMALLOC..."
if docker inspect "$IMAGE" 2>/dev/null | grep -q "PYTHONMALLOC"; then
    echo "  ✅ Image has PYTHONMALLOC in ENV"
    docker inspect "$IMAGE" 2>/dev/null | grep -A2 "PYTHONMALLOC"
else
    echo "  ❌ Image does NOT have PYTHONMALLOC"
    echo "  You need to rebuild the image!"
fi
echo ""

# Check 5: ConfigMap settings
echo "5. Checking ConfigMap:"
kubectl get configmap hpa-flask-app-config -o yaml | grep -E "PYTHON|MALLOC" || echo "  No PYTHONMALLOC in ConfigMap"
echo ""

echo "========================================"
echo "Summary"
echo "========================================"
echo ""
echo "If malloc_trim returns 0, the most common causes are:"
echo ""
echo "1. ❌ PYTHONMALLOC=malloc not set in container"
echo "   Fix: Rebuild image with updated Dockerfile"
echo ""
echo "2. ❌ ConfigMap overriding environment variables"
echo "   Fix: Add PYTHONMALLOC=malloc to ConfigMap"
echo ""
echo "3. ✅ No free memory to release (rare)"
echo "   This is OK if memory usage is already low"
echo ""

echo "Your current situation:"
if kubectl exec "$POD" -- env | grep -q "PYTHONMALLOC=malloc"; then
    echo "  ✅ PYTHONMALLOC is set correctly"
    echo "  If malloc_trim still returns 0, it might be because:"
    echo "    - Memory is already released (RSS is low)"
    echo "    - Or glibc decided not to release memory"
else
    echo "  ❌ PYTHONMALLOC is NOT set"
    echo "  ACTION REQUIRED: Rebuild Docker image with:"
    echo "    ENV PYTHONMALLOC=malloc"
fi