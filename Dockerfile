# Use a CUDA 12.1 base image with Ubuntu 20.04
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

# Set non-interactive mode for apt-get
ARG DEBIAN_FRONTEND=noninteractive

# Set the timezone to Central European Time (CET / MEZ)
ENV TZ=Europe/Berlin
ENV TMPDIR /tmp  

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    liblzma-dev \
    tk-dev \
    locales \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone and generate locale
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8

# Download and install Python 3.12.4 from source
RUN wget https://www.python.org/ftp/python/3.12.4/Python-3.12.4.tgz && \
    tar -xf Python-3.12.4.tgz && \
    cd Python-3.12.4 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.12.4 Python-3.12.4.tgz

# Create symlinks for Python 3.12.4
RUN ln -s /usr/local/bin/python3.12 /usr/bin/python && \
    ln -s /usr/local/bin/pip3.12 /usr/bin/pip

# Set the working directory
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install PyTorch with CUDA 12.1 support
#RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Install additional dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run the target Python file
CMD ["python", "/app/minimalexamples/ridgelet_quaternion_fusion2.py"]
