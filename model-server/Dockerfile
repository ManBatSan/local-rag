FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3 \
    ninja-build \
    python3-pip \
    wget \
    curl \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/Kitware/CMake/releases/download/v4.1.0-rc1/cmake-4.1.0-rc1-linux-x86_64.sh \
    && chmod +x cmake-4.1.0-rc1-linux-x86_64.sh \
    && ./cmake-4.1.0-rc1-linux-x86_64.sh --skip-license --prefix=/usr/local \
    && rm cmake-4.1.0-rc1-linux-x86_64.sh


RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt