# Installation Guide

This guide provides step-by-step instructions for installing DEgym and its dependencies for different use cases.

## Table of Contents

1. [Package Manager Setup](#package-manager-setup)
2. [Installation Options](#installation-options)
3. [Docker Installation](#docker-installation)
4. [Environment Setup](#environment-setup)
5. [Troubleshooting](#troubleshooting)

## Package Manager Setup

DEgym uses `uv` for fast and reliable package management. This is the recommended approach for all installations.

### Installing uv

Follow the installation guide in the [uv documentation](https://docs.astral.sh/uv/) to install `uv` on your system.

**Quick installation:**
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Getting the Repository

Clone the DEgym repository:
```bash
git clone git@github.com:instadeepai/degym.git
cd degym
```

## Installation Options

Choose the installation option that best fits your use case:

### 1. Basic Installation (Recommended for most users)

This installs the core DEgym framework with SciPy-based integrators, suitable for most chemical reactor simulations.

```bash
uv sync
```

**What's included:**
- Core DEgym framework.
- SciPy integrators.
- Basic RL environment functionality.
- Essential dependencies for building and running reactor simulations.

### 2. Development Installation

For developers contributing to DEgym or those who need testing and linting tools.

```bash
uv sync --group test
uv pip install -e .
```

**What's included:**
- Everything from Basic Installation.
- Testing frameworks (pytest, etc.).
- Code linting tools.
- Development utilities.

### 4. DiffEqPy Installation (Experimental)

For advanced users who need Julia-based differential equation solvers.

```bash
uv sync --group diffeqpy
uv run build/install_julia.py
```

**What's included:**
- Everything from Basic Installation.
- Julia-based DiffEqPy integrators.
- Advanced numerical solvers.

> [!WARNING]
> The DiffEqPy installation requires Julia and is currently not stable for local or CI builds. Use with caution in production environments.

### Dependency Groups Summary

| Group | Purpose | Use Case |
|-------|---------|----------|
| **Core** | Basic DEgym functionality | Building and running RL environments |
| **test** | Development tools | Contributing to DEgym, testing |
| **diffeqpy** | Advanced solvers | High-performance numerical integration |

## Docker Installation

For containerized deployments, DEgym provides several Docker images optimized for different use cases. The Dockerfile uses multi-stage builds for efficiency.

### Available Docker Images

- **degym-min**: Minimal production image for running with SciPy integrators (without access to tutorials).
- **degym-ci**: CI/CD image with testing capabilities, includes all dependencies for continuous integration.
- **degym-diffeqpy**: Experimental image with Julia-based DiffEqPy integrators and Git support for development.

### Building Docker Images

Build from the project root directory using the Dockerfile located in `build/Dockerfile`:

```bash
# Build the minimal production image
docker build -f build/Dockerfile --target degym-min -t degym-min .

# Build the CI/testing image
docker build -f build/Dockerfile --target degym-ci -t degym-ci .

# Build the DiffEqPy image (experimental)
docker build -f build/Dockerfile --target degym-diffeqpy -t degym-diffeqpy .
```

### Docker Image Features

**Best Practices:**
- Multi-stage builds for optimal layer caching and image size
- Efficient package installation with UV package manager
- Build cache mounts for faster subsequent builds
- Proper cleanup of package manager caches

**Development vs Production:**
- **degym-min**: Optimized for production workloads, minimal dependencies
- **degym-ci**: Includes testing tools and development dependencies
- **degym-diffeqpy**: Full development environment with Git support (experimental)

### Using Docker Images

```bash
# Run with the minimal image
docker run -it degym-min bash -c "python -c 'import degym; print(\"DEgym loaded successfully!\")'"

# Run with volume mounting for development
docker run -v $(pwd):/app/workspace -it degym-ci bash -c "cd workspace && uv run pytest"

# Interactive development with the CI image
docker run -v $(pwd):/app/workspace -it degym-ci bash
```

### Docker Development Workflow

For development with Docker:

```bash
# Build the CI image
docker build -f build/Dockerfile --target degym-ci -t degym-ci .

# Run tests
docker run -v $(pwd):/app/workspace degym-ci bash -c "cd workspace && uv run pytest"

# Run with interactive shell for debugging
docker run -v $(pwd):/app/workspace -it degym-ci bash
```

## Environment Setup

### Running Commands

Use `uv run` to execute commands within the virtual environment:

```bash
# Run Python scripts
uv run python path_to_script.py

# Run tests
uv run pytest

# Install additional packages
uv add package_name
```

### Python Path Configuration

If you encounter `ModuleNotFoundError` when running scripts, make sure you have installed the package in editable mode:

```bash
uv pip install -e .
```

## Verification

### Test Your Installation

Verify your installation by running a simple test:

```bash
# Run basic tests
uv run pytest tests/

# Test a simple environment
uv run python -c "import degym; print('DEgym imported successfully!')"
```


## Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
# Solution: Update PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH
```

**2. uv command not found**
```bash
# Solution: Ensure uv is in your PATH or reinstall
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal
```

**3. Docker Build Issues**
```bash
# Solution: Ensure you're building from the project root
docker build -f build/Dockerfile --target degym-ci -t degym-ci .

# Clear Docker cache if needed
docker builder prune
```

## Next Steps

After successful installation:

1. **Read Documentation**: Read the [DEgym Essentials](degym_essentials.md) to understand the framework architecture and basic usage.
2. **Read the Tutorial**: Follow the [comprehensive tutorial](how_to_build_new_env.md) to create your first custom environment, and check out the CSTR examples in the `degym_tutorials/` directory.
3. **Start Implementing Your Env** 🧑‍🍳⚗️
> [!TIP]
> For the best development experience, we recommend using the Development Installation with your preferred IDE configured for Python development.
