FROM nvidia/cuda:12.9.0-base-ubuntu20.04 AS base

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.12-full python-is-python3

RUN update-alternatives  --install /usr/bin/python3 python3 /usr/bin/python3.12 0 && \
    python -m ensurepip --upgrade && \
    pip3 install --upgrade pip && \
    pip uninstall -y six && pip install six

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

FROM base AS pip-build
WORKDIR /wheels

RUN apt-get update && apt-get -y install git

COPY requirements.txt .
RUN pip install -U pip  \
    && pip wheel -r requirements.txt

FROM base AS eval
COPY --from=pip-build /wheels /wheels
WORKDIR /src

ENV TZ=Europe/Berlin
ENV PYTHONPATH=/src/2025-challenge

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install ffmpeg libsm6 libxext6 git && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

RUN pip install -U pip  \
    && pip install --no-cache-dir \
    --no-index \
    -r /wheels/requirements.txt \
    -f /wheels \
    && rm -rf /wheels

COPY . 2025-challenge/

CMD ["python", "2025-challenge/run.py", "-e", "tournament"]

FROM eval AS dev
# For nvidia GPU
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# libgl1-mesa-glx libgl1-mesa-dri for non-nvidia GPU
RUN apt-get update && apt-get -y install xauth tzdata libgl1-mesa-glx libgl1-mesa-dri && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*
