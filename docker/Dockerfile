FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip git && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
COPY engine engine
COPY tests tests
COPY README.md README.md
RUN useradd --create-home --uid 10001 engineuser && chown -R engineuser:engineuser /workspace
USER engineuser
EXPOSE 8080
