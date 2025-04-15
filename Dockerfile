# 📦 STEP 1: Use NVIDIA's Official PyTorch Image
FROM nvcr.io/nvidia/pytorch:23.08-py3
# 🔥 Comes pre-installed with:
# - Ubuntu 22.04 including Python 3.10
# - NVIDIA CUDA® 12.2.1
# - NVIDIA cuBLAS 12.2.5.1
# - NVIDIA cuDNN 8.9.4
# - NVIDIA NCCL 2.18.3
# - NVIDIA RAPIDS™ 23.06
# - Apex
# - rdma-core 39.0
# - NVIDIA HPC-X 2.15
# - OpenMPI 4.1.4+
# - GDRCopy 2.3
# - TensorBoard 2.9.0
# - Nsight Compute 2023.2.1.3
# - Nsight Systems 2023.2.3.1001
# - NVIDIA TensorRT™ 8.6.1.6
# - Torch-TensorRT 2.0.0.dev0
# - NVIDIA DALI® 1.28.0
# - MAGMA 2.6.2
# - JupyterLab 2.3.2 including Jupyter-TensorBoard
# - TransformerEngine 0.11.0++3f01b4f
# - PyTorch quantization wheel 2.1.2

# Saves time & ensures GPU compatibility!

# 🛠 STEP 2: Install Basic Linux Packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \                   
    git \                   
    zip unzip \             
    wget curl \             
    htop \                  
    libgl1 \                
    libglib2.0-0 \          
    libpython3-dev \        
    gnupg \                 
    g++ \                   
    libusb-1.0-0 \          
    libsm6 \                
    && rm -rf /var/lib/apt/lists/*
# 🚿 Clean-up: Shrinks the image size by removing cached lists

# 🔐 STEP 3: Apply Security Updates for Core Packages
RUN apt upgrade --no-install-recommends -y openssl tar
# 🚨 Patch critical vulnerabilities in OpenSSL (network encryption) & tar (archives)

# 📁 STEP 4: Set Working Directory
WORKDIR /app
# All following commands will execute in /app inside the container

# 🧪 STEP 5: Install Python Requirements
COPY requirements.txt .
RUN pip install -r requirements.txt
# 📦 Installs YOLOv7 dependencies (torchvision, opencv-python, numpy, etc.)

# 📂 STEP 6: Copy Source Code to the Container
COPY . .
# ✅ Adds your whole project directory into the container at /app

# 🧹 Clean Up Pre-existing Workspace (Optional)
RUN rm -rf ./workspace
# 💡 Ensures the build is fresh & avoids legacy artifacts

# ⚙️ STEP 7: Build OpenCV with CUDA Support
RUN bash scripts/build_opencv.sh
# 🚀 This script compiles OpenCV from source with GPU acceleration for video/image ops
# 😎 Perfect for fast inference on large datasets

# ✅ STEP 8: Run Diagnostic Tests
RUN bash scripts/test-cmds.sh
# 🧪 Verifies:
# - OpenCV is properly installed
# - CUDA is working
# - Everything’s talking to each other correctly

# ⬇️ STEP 9: Download YOLOv7 Pretrained Models
# ⚠️ Choose the one that matches your hardware & use case

# 🏃‍♂️ Lightweight Model – Fast AF
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
# Use this for: 🔋 embedded systems, Raspberry Pi, Jetson Nano
# Speed ✅ Accuracy ❌

# 🧠 Balanced Model – Best for Most
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt
# Use this for: 🧪 general object detection on mid-range GPUs
# Speed ⚖️ Accuracy ⚖️

# 🔧 Training-Ready – For Custom Datasets
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt
# Use this for: 🏋️‍♂️ training with `train_aux.py`
# Comes with extra layers for learning better

# 🎯 Precision Model – Accuracy First
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt
# Use this for: 📸 detection where precision matters
# Speed ❌ Accuracy ✅✅

# 🧬 Deep Variant – Alternate Heavyweight
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt
# Use this if: You’re experimenting or need d6-specific layers

# 👑 Elite Model – Max Accuracy, Max Power
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
# Use this for: 🖥️ high-end workstations, servers
# Speed ❌❌ Accuracy 💯

