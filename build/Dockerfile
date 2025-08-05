# syntax=docker/dockerfile:1

# This Dockerfile is used to build three images:
# - 'degym-minimal': minimal image for running with scipy integrators on without access to tutorials.
# - 'degym-main': Base image for running with scipy integrators with access to tutorials.
# - 'degym-diffeqpy': Image for running with diffeqpy integrators with access to tutorials.
# (NOTE: The `degym-diffeqpy` image is not stable, cannot build locally or on the CI.)

ARG UV_VERSION=0.7.14

# ARG BASE_AICHOR_IMAGE=rayproject/ray:2.39.0-py310-cpu
ARG BASE_IMAGE=python:3.10.15-slim

# Stage: uv-base
# Contains uv install to use across all stages
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-base

# Stage: base-image
# Contains the base image for all other images.
FROM ${BASE_IMAGE} AS base-image
COPY --from=uv-base /uv /bin/

ARG PYTHON_VERSION

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
    apt-get install -y --no-install-recommends build-essential wget git && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

# Update PYTHONPATH to make modules in "degym_tutorials" discoverable.
WORKDIR /app

# Stage: 'degym-minimal'
# This image contains the project and all the dependencies, for running with scipy integrators.
FROM base-image AS degym-minimal

# Files required to install dependencies. Package build with hatchling requires README.
COPY uv.lock pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

COPY . .
RUN uv sync
ENTRYPOINT ["bash", "-c"]

# Stage: 'degym-main'
# This image contains the project and all the dependencies, for running with scipy integrators with access to tutorials.
FROM degym-minimal AS degym-main

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --group test --no-install-project

COPY . .
RUN uv sync --group test
ENTRYPOINT ["bash", "-c"]

# Stage: 'degym-diffeqpy'
# Image for running with diffeqpy integrators with access to tutorials.
FROM degym-main AS degym-diffeqpy

# Install git for pre-commit hooks (build-essential and wget already installed in base-image)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

ARG PWD
ARG PYTHONPATH

ENV PWD=${PWD} \
    PYTHONPATH=${PWD}:${PYTHONPATH}

# Files required to install dependencies. Package build with hatchling requires README.
COPY uv.lock pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --group diffeqpy --no-install-project

COPY . .
RUN uv sync --group diffeqpy
ENTRYPOINT ["bash", "-c"]

# Install pre-commit hooks
# Pre-commit should run in your development environment or CI
