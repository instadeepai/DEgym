# Inside DEgym: Core Design and Usage

## Table of Contents

- [Inside DEgym: Core Design and Usage](#inside-degym-core-design-and-usage)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Understanding the DEgym Architecture](#understanding-the-degym-architecture)
  - [How to use DEgym](#how-to-use-degym)

## Overview

DEgym is a framework for developing RL environment for systems governed by dynamical systems. It provides a structured approach to implementing environments that model complex systems using Differential-Algebraic Equations (DAEs) or Ordinary Differential Equations (ODEs). Additionally it offers:

* a unified structure across different use cases while remaining flexible, modular, extensible, and maintainable; and
* a rich context and output format for AI agents to complete the implementation. This is aligned with the new generation of softwares, known as [Software 3.0](https://www.youtube.com/watch?v=LCEmiRjPEtQ), which are designed to be developed by AI-agents.


## Understanding the DEgym Architecture

To achieve the above goals, it is essential that DEgym is built with the understanding that every RL environment has components/logic that are either: (i) RL-specific, or (ii) use-case-specific which are explained below.

>[!TIP]
> For a refresher on basics of a RL environment, refer to Gymnasium's [Basic Usage](https://gymnasium.farama.org/introduction/basic_usage/) and [Env API](https://gymnasium.farama.org/api/env/).

* **RL-related logic** refers to the shared structure across use cases, implemented once to define the data flow and interfaces between components. For instance, every RL environment includes a step function that takes an action, preprocesses it, passes it to an integrator to update the environment state, and then returns the next observation, reward, done flags, and info. This data flow constitutes the RL-related logic and is independent of domain-specific details. Elements like action are instances of abstract classes that must be concretely defined per use case.
* **Use-case-related logic**, for example what the `action` actually entails, is implemented by inheriting from the abstract classes mentioned above.

In DEgym, we implemented all RL-related logic, leaving only the use-case-specific logic for the user or AI agent to define. The fixed data flow, well-defined interfaces, and documentation provide the agent with rich context and clear output formats, guiding the development of concrete methods. Without these guides, maintaining a unified structure across use cases --and enabling agent success-- would be much harder.

The main RL-related logics belong to `__init__` and `step` functions. In the figure below, we have the implemented information flow in those two functions. The concrete implementation of the data types, e.g. `State`, and the abstract methods of the components like `RewardExtractor` requires knowledge the use-case.

<p align="center">
  <img src="images/diagrams.png" width="800">
</p>

> [!NOTE]
> For easier visualization, the above diagrams does not show the data classes which are passed between the components, nor it indicates where the information is saved.

## How to use DEgym
To create a new environment using DEgym, one needs to subclass the `Environment` class and implement all the required abstract classes. The `Environment.__init__()` method requires the following components:

```python
def __init__(
    self,
    physical_parameters_generator: PhysicalParametersGenerator,
    initial_state_generator: InitialStateGenerator,
    integrator: Integrator,
    action_preprocessor: ActionPreprocessor,
    state_preprocessor: StatePreprocessor,
    state_postprocessor: StatePostprocessor,
    observation_extractor: ObservationExtractor,
    reward_extractor: RewardExtractor,
    terminated_extractor: TerminatedExtractor,
    truncated_extractor: TruncatedExtractor,
    info_extractor: InfoExtractor,
    seed: int,
) -> None:
```
All of the above components (except the `Integrator` which is already implemented) are use-case dependent and need to be implemented by subclassing them.
> For a detailed tutorial of such implementation for a continous stirred tank reactor (CSTR) refer to [Creating New Environments with DEgym: A Comprehensive Tutorial](how_to_build_new_env.md).
>[!TIP]
> For a detailed tutorial of such implementation for a continous stirred tank reactor (CSTR) refer to [A Comprehensive Tutorial](how_to_build_new_env.md).
