FROM nvidia/cuda:12.6.3-devel-ubuntu22.04
WORKDIR /workspace

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    bash-completion \
    build-essential \
    patchelf \
    curl \
    git \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    htop \
    tmux \
    pipx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN rm /etc/localtime
RUN ln -s /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN pipx install poetry
COPY pyproject.toml pyproject.toml
# RUN /root/.local/bin/poetry config virtualenvs.in-project true
RUN /root/.local/bin/poetry install 
RUN /root/.cache/pypoetry/virtualenvs/ocrm-**/bin/python -m pip install --upgrade setuptools packaging wheel pip ninja
RUN /root/.cache/pypoetry/virtualenvs/ocrm-**/bin/python -m pip install deepspeed==0.14.4
RUN /root/.cache/pypoetry/virtualenvs/ocrm-**/bin/python -m pip install --no-build-isolation --no-deps flash-attn
# ENV MAX_JOBS=64
# COPY poetry_flash_attn.sh poetry_flash_attn.sh
# RUN /workspace/poetry_flash_attn.sh
