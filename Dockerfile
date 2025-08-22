FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Configure the build environment.  We avoid caching apt lists and remove
# unnecessary files to keep the image lean.  Python is installed from the
# system packages to ensure compatibility with the pre-built CUDA libraries.
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
        wget \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements separately so Docker can cache pip installs when only
# application code changes.
COPY requirements.txt /requirements.txt

RUN /usr/bin/python3 -m pip install --upgrade pip

# Install torch with CUDA 12.1 support.  Pinning a specific version ensures
# deterministic builds.  Keep this in sync with the underlying CUDA version.
RUN /usr/bin/python3 -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install the remaining Python dependencies.  The --no-cache-dir flag
# prevents pip from caching wheels inside the image.
RUN /usr/bin/python3 -m pip install --no-cache-dir -r /requirements.txt

WORKDIR /

# Copy the application source into the image.  We place it under /src so
# Python imports resolve correctly regardless of the working directory.
COPY src/ /src/

# Run the RunPod handler directly.  RunPod will call this script to start
# processing jobs.  The -u flag forces stdout/stderr to be unbuffered so
# logs appear immediately.
CMD ["/usr/bin/python3", "-u", "/src/handler.py"]
