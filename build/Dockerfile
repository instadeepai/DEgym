# syntax=docker/dockerfile:1

# This Dockerfile is used to build three images:
# - 'degym-aichor': Base image for running with scipy integrators on AIChor.
# - 'degym-ci': Base image for running with scipy integrators on the CI.
# - 'degym-diffeqpy': Image for running with diffeqpy integrators.
# (NOTE: The `degym-diffeqpy` image is not stable, cannot build locally or on the CI.)

ARG UV_VERSION=0.7.14

# We encounter issues with using a ray base image for the CI
# We encounter issues with NOT using a ray base image for AIChor
ARG BASE_AICHOR_IMAGE=rayproject/ray:2.39.0-py310-cpu
ARG BASE_CI_IMAGE=python:3.10.15-slim

# Stage: uv-base
# Contains uv install to use across all stages
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-base

# Stage: base-image
# Contains the base image for all other images.
FROM ${BASE_AICHOR_IMAGE} AS base-image
COPY --from=uv-base /uv /bin/

ARG PYTHON_VERSION
USER root

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Europe/Paris \
    UV_PYTHON=python${PYTHON_VERSION} \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_LOCKED=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    UV_TORCH_BACKEND=cpu \
    UV_PROJECT_ENVIRONMENT="/app/.venv"

# Install build-essential
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential wget && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

# Update PYTHONPATH to make modules in "degym_tutorials" discoverable.
WORKDIR /app

ARG PWD
ARG PYTHONPATH

ENV PWD=${PWD} \
    PYTHONPATH=${PWD}:${PYTHONPATH}

# Stage: 'degym-base'
# This image contains the project and all the dependencies, for running with scipy integrators.
FROM base-image AS degym-aichor

# Files required to install dependencies. Package build with hatchling requires README.
COPY uv.lock pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --group test --no-install-project

COPY . .
RUN uv sync --group test
ENTRYPOINT ["bash", "-c"]

# Stage: 'degym-diffeqpy'
# This image contains the project and all the dependencies, for running with diffeqpy integrators.
### NOTE: This image is currently not stable - cannot build locally or on CI,
FROM base-aichor-image AS degym-diffeqpy

# Install gpg/gnupg for julia install
RUN apt-get update && \
    apt-get install -y --no-install-recommends gpg gnupg && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

# Files required to install dependencies. Package build with hatchling requires README.
COPY uv.lock pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --group diffeqpy --no-install-project

# Julia installation with curl doesn't work, use script to install Julia and Julia packages.
COPY ./build/install_julia.py build/install_julia.py
RUN uv run --no-reinstall python build/install_julia.py

COPY . .
RUN uv sync --group diffeqpy
ENTRYPOINT ["bash", "-c"]

# Stage: 'degym-ci'
# Image for the CI, with all project dependencies and git for pre-commit hooks
FROM ${BASE_CI_IMAGE} AS degym-ci
COPY --from=uv-base /uv /bin/

ARG PYTHON_VERSION
USER root

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Europe/Paris \
    UV_PYTHON=python${PYTHON_VERSION} \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_LOCKED=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    UV_TORCH_BACKEND=cpu

ENV UV_PROJECT_ENVIRONMENT="/app/.venv"

# Install build-essential, and git for pre-commit hooks
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential wget git && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

# Update PYTHONPATH to make modules in "degym_tutorials" discoverable.
WORKDIR /app

ARG PWD
ARG PYTHONPATH

ENV PWD=${PWD} \
    PYTHONPATH=${PWD}:${PYTHONPATH}

# Files required to install dependencies. Package build with hatchling requires README.
COPY uv.lock pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --group test --no-install-project

COPY . .
RUN uv sync --group test
ENTRYPOINT ["bash", "-c"]

# Install pre-commit hooks
RUN git init && pre-commit install-hooks -c .pre-commit-config.yaml
