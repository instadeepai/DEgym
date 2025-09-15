# DEgym


<p align="center">
  <img src="docs/images/startpage.gif" width="800">
  <br>
  <em>A framework for converting dynamical systems to RL environments.</em>

</p>

## ⚗️ What is DEgym?

DEgym addresses the challenge of creating RL environments for process optimization in systems which are described by dynamical systems, in particular differential algebraic equations (DAEs) or ordinary differential equations (ODEs). Traditional RL frameworks require significant domain expertise to bridge the gap between domain specific knowledge (e.g. chemical engineering) and RL implementation. DEgym provides:

### 🎯 **Key Features**

- **Unified Architecture**: Separates RL-specific logic (common across domains) from use-case-specific logic
- **DAE/ODE Integration**: Native support for differential equations that govern the dynamics of the environment
- **Modular Design**: Extensible framework with well-defined interfaces for easy customization
- **AI-Agent Ready**: Rich context and clear interfaces designed for [Software 3.0](https://www.youtube.com/watch?v=LCEmiRjPEtQ) development
- **Production Ready**: Gymnasium-compatible environments for seamless integration with RL libraries

### 🏗️ **Architecture Overview**

DEgym implements a clean separation of concerns:

- **RL-Related Logic** (implemented once): Data flow, interfaces, `step` function structure, observation/reward handling
- **Use-Case Logic** (user implements): Specific reactor dynamics, actions, states, and domain constraints

This separation enables both human developers and AI agents to create new environments systematically by focusing only on the domain-specific aspects.

### 🚀 **Use Cases**

Perfect for optimizing:

- **Chemical Reactors**: CSTR, batch reactors, flow reactors.
- **Biological Systems**: Fermentation processes, metabolic [networks](https://en.wikipedia.org/wiki/Biochemical_systems_theory), population dynamics [model](https://en.wikipedia.org/wiki/Population_dynamics).
- **Environmental and Earth Sciences**: Energy balance climate [models](https://en.wikipedia.org/wiki/Earth%27s_energy_budget), water reservoir systems, groundwater level dynamics, carbon cycle models.
- **Sociology and Psychology**: Opinion dynamics models (e.g., DeGroot [model](https://en.wikipedia.org/wiki/DeGroot_learning)), adoption of innovation ([Bass model](https://en.wikipedia.org/wiki/Bass_diffusion_model)), population-level behavior change models.
- **Economics and Financial Systems**:  Solow-Swan [model](https://en.wikipedia.org/wiki/Solow%E2%80%93Swan_model), dynamics of financial market (e.g., simplified Black-Scholes-Merton [model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)).

### 💡 **Why DEgym?**

Traditional approaches require implementing RL environments from scratch for each reactor type. DEgym provides:

- **Faster Development**: Focus on dynamics of the system, not RL boilerplate
- **Consistent Quality**: Proven RL patterns and best practices built-in
- **Easy Maintenance**: Modular architecture scales across different use-cases
- **Research Acceleration**: Rapid prototyping for optimization research

## Quick Start

### 📦 **Installation**

DEgym uses `uv` for package management. Install `uv` following the [uv documentation](https://docs.astral.sh/uv/), then:

```bash
# Clone the repository
git clone git@github.com:instadeepai/degym.git
cd degym

# Basic installation (recommended for most users)
uv sync

# Development installation (includes testing and linting tools)
uv sync --group test
```

> [!TIP]
> For detailed installation instructions, Docker setup, troubleshooting, and advanced options, see the [Installation Guide](docs/installation.md).

### 🧑‍🔬⚗️💥 Usage

To create a new environment with DEgym:

1. **Define your environment's DAE/ODE system** (mass/energy balances)
2. **Subclass the `Environment` class** and implement abstract components
3. **Get a gymnasium-compatible environment** ready for RL optimization

```python
# Example: CSTR environment usage
from degym_tutorials.cstr_tutorial.make_env import make_cstr_environment

env = make_cstr_environment(config)
obs, info = env.reset()

done = False
while not done:
    action = agent.get_action(obs)  # Your RL agent
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

>[!TIP]
> For a comprehensive step-by-step guide, refer to the **[Tutorial](docs/how_to_build_new_env.md)** which walks through creating a complete CSTR environment from scratch.

## 📖 Documentation

- **[Documentation home page](https://instadeepai.github.io/DEgym/)**: Full documentation website.
- **[Installation Guide](docs/installation.md)**: Comprehensive setup instructions for all use cases
- **[DEgym Essentials](docs/degym_essentials.md)**: Core architecture and design principles
- **[Tutorial](docs/how_to_build_new_env.md)**: Step-by-step guide to creating custom environments

## 🤝 Contributing

If you wish to contribute to DEGym, please refer to the [contribution guidelines](/docs/contributing.md) and follow the [development installation](/docs/installation.md#2-development-installation) for setting up the development environment with testing and linting tools. This is the [list of maintainers](/docs/maintainers.md).

## ↩️ Citation

If you use DEgym in your work, please cite:

```
@misc{degym,
    title={DEgym: A framework for developing reinforcement learning environments for dynamical systems},
    author={Nima H. Siboni, Marco Carobene, Frédéric Renard, Alind Gupta, Miguel Arbesú},
    year={2025},
    url={https://github.com/instadeepai/degym/},
}
```
