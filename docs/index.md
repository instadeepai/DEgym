# DEgym Documentation

Welcome to the DEgym framework documentation! DEgym is a comprehensive framework for creating environments for reinforcement learning which is focused on systems governed by ODEs/DAEs

## Quick Navigation

- **[Installation](installation.md)** - Get started with installing DEgym.
- **[DEgym Essentials](degym_essentials.md)** - Core design principles and usage patterns.
- **[Terminology](terminology.md)** - Key concepts and definitions.
- **[Tutorial](how_to_build_new_env.md)** - Complete guide to building new environments.

## What is DEgym?

DEgym separates environment logic into two categories:

- **RL-related logic**: Core functionality shared by all environments including data handling and system interfaces.
- **Use-case-related logic**: Concrete implementations of abstract classes that define reactor-specific actions, states, dynamics, etc.

The framework provides the RL-related infrastructure, while users implement only the use-case-specific components by inheriting from abstract classes.

## Getting Started

1. Start with the [Installation Guide](installation.md).
2. Read [DEgym Essentials](degym_essentials.md) to understand the core concepts.
3. Review the [Terminology](terminology.md) to familiarize yourself with key terms.
4. Follow the [Comprehensive Tutorial](how_to_build_new_env.md) to build your first environment.

## Contributing

If you wish to contribute to DEGym, please refer to the [contribution guidelines](/docs/contributing.md) and follow the [development installation](/docs/installation.md#2-development-installation) for setting up the development environment with testing and linting tools.
