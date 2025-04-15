#!/bin/bash

# Title
echo "🧪 SYSTEM DIAGNOSTIC TEST SCRIPT FOR AI/ML ENVIRONMENT"

##########################
# Function: Test OpenCV
##########################
test_opencv() {
    echo "-------------------Testing OpenCV installation (START)---------------------"

    echo "📦 Checking OpenCV installation and version..."
    python3 -c "import cv2; print(f'  OpenCV version: {cv2.__version__}')"

    echo "⚙️  Checking OpenCV build flags for CUDA..."
    python3 -c "import cv2; print(cv2.getBuildInformation())" | grep "CUDA"

    echo "🚀 Checking if OpenCV can access CUDA-enabled GPU..."
    python3 -c "import cv2; print('  ✅ CUDA is enabled for OpenCV') if cv2.cuda.getCudaEnabledDeviceCount() else print('  ❌ CUDA is NOT enabled for OpenCV')"

    echo "-------------------Testing OpenCV installation (END)---------------------"
    echo "/"
}

##########################
# Function: Test PyTorch
##########################
test_torch() {
    echo "-------------------Testing PyTorch installation (START)---------------------"

    echo "📦 Checking PyTorch installation and version..."
    python3 -c "import torch; print(f'  Torch version: {torch.__version__}')"

    echo "🚀 Checking if PyTorch can access CUDA-enabled GPU..."
    python3 -c "import torch; print('  ✅ CUDA is enabled for Torch') if torch.cuda.is_available() else print('  ❌ CUDA is NOT enabled for Torch')"

    echo "-------------------Testing PyTorch installation (END)---------------------"
    echo "/"
}

##########################
# Function: Test NVCC (CUDA Compiler)
##########################
test_nvcc() {
    echo "-------------------Testing nvcc installation (START)---------------------"

    echo "📦 Checking nvcc (NVIDIA CUDA Compiler) version..."
    output=$(nvcc --version)

    compiler=$(echo "$output" | grep "nvcc:" | cut -d ':' -f2- | sed 's/^/ /')
    build_date=$(echo "$output" | grep "Built on" | sed 's/Built on //; s/_/ /g')
    cuda_version=$(echo "$output" | grep "Cuda compilation tools" | sed -E 's/.*release ([^,]+), V([0-9\.]+).*/\1 (\2)/')
    build_id=$(echo "$output" | grep "Build cuda")

    echo "🧠 CUDA Compiler Information:"
    echo "- Compiler: $compiler"
    echo "- CUDA Version: $cuda_version"
    echo "- Build Date: $build_date"
    echo "- Internal Build ID: $build_id"

    echo "-------------------Testing nvcc installation (END)---------------------"
    echo "/"
}

##########################
# Function: Test NVIDIA-SMI (Driver & GPU)
##########################
test_nvidia_smi() {
    echo "-------------------Testing nvidia-smi installation (START)---------------------"

    echo "🖥️  GPU and driver status:"
    nvidia-smi

    echo "-------------------Testing nvidia-smi installation (END)---------------------"
}

##########################
# Main: Run All Tests
##########################
run_all_tests() {
    test_opencv
    test_torch
    test_nvcc
    test_nvidia_smi
}

# Execute
run_all_tests
echo "-------------------All tests completed!---------------------"
