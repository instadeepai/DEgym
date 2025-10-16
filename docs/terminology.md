# DEgym Terminology

This document defines key terms and concepts used throughout the DEgym framework.

## Table of Contents

- [Environment](#environment)
- [State Components](#state-components)
- [Action Components](#action-system)
- [Processing Pipeline](#processing-pipeline)
- [Extractors](#extractors)
- [Integration Components](#integration-components)
- [Environment Components](#environment-components)

## Environment

### **Environment**
The main interface between RL agents and reactor simulations. Inherits from Gymnasium's `Env` class and provides the standard RL interface (`step()`, `reset()`, etc.). The Environment coordinates all other components and maintains the internal simulation state.

**Usage in DEgym**: Acts as the central orchestrator that manages the complete simulation lifecycle, from initialization through episode execution.

### **Episode**
A complete simulation run from initialization to termination/truncation. Each episode represents one training or evaluation trajectory for the RL agent.

**Usage in DEgym**: Episodes are bounded by `reset()` and end when terminated or truncated conditions are met.

## State Components

### **State**
The complete internal representation of the environment at any given time. Contains all information needed to fully describe the reactor system and continue simulation from that point.

**Components**:
- `dae_state`: Variables that appear on the left-hand side of differential equations.
- `dae_params`: Parameters used in the differential equations.
- `non_dae_params`: Parameters not part of the differential equations but needed for simulation.

**Usage in DEgym**: The State is passed between components throughout the simulation pipeline and gets updated at each timestep.
#### What `State` is NOT!
*State* is a common term used in different disciplines where it points to somewhat different concepts. The state here is *not*:
- The state as in a point in phase space.
- The state as in DAEState (e.g., the variables that are differentiated).

#### Why do we organize the `State` this way?

We split the state values into these components so that we can easily retrieve and separate the state values required for implementing a reactor. This choice is motivated by the structure of the DAE and the way we implement the dynamics function.

### **DAEState**
Variables that appear on the left-hand side of differential equations (the variables being differentiated).

**Examples**: Concentrations (c_A, c_B), temperature (T), pressure (P).

**Usage in DEgym**: Updated by numerical integration and represents the evolving state of the system.

### **DAEParameters**
Parameters that appear in the differential equations but are not differentiated variables.

**Examples**: Flow rates (F), reactor volume (V), kinetic constants (k_0_a, k_0_b), activation energies (E_a), etc.

**Usage in DEgym**: Used by the integrator in differential equation calculations.

### **NonDAEParameters**
Parameters needed for simulation but not part of the differential equations.

**Examples**: Maximum heat input (q_max), episode length (max_timestep), current timestep, constraint limits.

**Usage in DEgym**: Used for simulation control, constraints, and episode management.

## **PhysicalParameters**
High-level parameters that characterize a specific reactor configuration. Used to generate both DAEParameters and NonDAEParameters.

>[!NOTE]
>`PhysicalParameter` is not the config that is passed to `make_cstr_environment`.

**Examples**: Reactor geometry, parameters of operating constraints, reaction kinetics, feed composition.

**Usage in DEgym**: Generated at episode start and remain constant within an episode, but can vary between episodes for domain randomization.

## Action Components

### **Raw Action**
The direct input passed to the environment's `step()` function by the RL agent. Can be scalars, numpy arrays, or other data types as defined by the action space.

**Usage in DEgym**: Raw actions are the starting point of the action processing pipeline.

### **Action**
An intermediate representation that gives semantic meaning to raw actions. Provides named fields and interpretable structure before conversion to physical control parameters.

**Purpose**:
- Makes control logic interpretable and debuggable.
- Bridges between RL agent outputs and physical system requirements.
- Enables clear mapping from agent decisions to control intentions.

**Usage in DEgym**: Actions are created by wrapping raw inputs and then converted to DAEActions.

### **DAEAction**
Control parameters as they appear directly in the differential equation system. These are domain-specific, physically meaningful control inputs that affect system dynamics.

**Purpose**:
- Encapsulates control parameters for differential equation integration.
- Ensures actions are in physically meaningful units and ranges.
- Enables direct use in system dynamics calculations.

**Usage in DEgym**: DAEActions are passed to the integrator for numerical solving.

### **ActionConverter**
Transforms semantic Actions into DAEActions and vice versa. Handles scaling, unit conversion, and complex control logic transformations.

**Usage in DEgym**: Called during action preprocessing to bridge semantic and physical representations.

### **ActionRegulator**
Enforces constraints on DAEActions to ensure they remain within safe and feasible bounds.

**Methods**:
- `is_legal()`: Checks if an action satisfies all constraints.
- `convert_to_legal_action()`: Corrects constraint-violating actions.

**Usage in DEgym**: Applied after action conversion to ensure safe operation.

### **ActionPreprocessor**
Orchestrates the complete action processing pipeline from raw inputs to validated DAEActions.

**Pipeline**:
1. Wrap raw input into semantic Action.
2. Convert Action to DAEAction using ActionConverter.
3. Apply constraints using ActionRegulator.
4. Return validated DAEAction for integration.

**Usage in DEgym**: Called by the Environment during each `step()` to process agent actions.

## Processing Pipeline

### **Preprocessor**
Coordinates preprocessing of both state and action before numerical integration.

**Usage in DEgym**: Called at the beginning of each `step()` to prepare inputs for the integrator.

### **StatePreprocessor**
Processes the state before numerical integration. Can apply transformations, corrections, or preparation steps.

**Usage in DEgym**: Ensures the state is in the correct format and range for numerical solving.

### **StatePostprocessor**
Processes the state after numerical integration to handle corrections, constraints, or transformations.

**Usage in DEgym**: Ensures the integrated state satisfies physical constraints and is ready for extraction.

## Extractors

Extractors convert internal state to the outputs required by the Gymnasium interface.

### **ObservationExtractor**
Converts environment state to observations that RL agents receive.

**Components**:
- `observation_space`: Defines the structure and bounds of observations.
- `extract_observation()`: Transforms state to agent-visible format.

**Usage in DEgym**: Called at the end of `step()` and `reset()` to provide agent observations.

### **RewardExtractor**
Computes reward signals based on state transitions (state, action, next_state).

**Purpose**: Implements the optimization objective for the RL agent.

**Usage in DEgym**: Called during `step()` to provide learning signals to the agent.

### **TerminatedExtractor**
Determines if an episode should terminate due to natural ending conditions of the MDP.

**Examples**: Achieving objectives, reaching unsafe conditions, violating constraints.

**Usage in DEgym**: Called during `step()` to check for episode termination.

### **TruncatedExtractor**
Determines if an episode should truncate due to artificial stopping conditions.

**Examples**: Time limits, maximum timesteps, computational constraints.

**Usage in DEgym**: Called during `step()` to check for episode truncation.

### **InfoExtractor**
Extracts additional diagnostic information useful for monitoring and analysis.

**Purpose**: Provides information for debugging, logging, and analysis but not used by the agent for learning.

**Usage in DEgym**: Called during `step()` and `reset()` to provide diagnostic data.

## Integration Components

### **Integrator**
Numerical solver that advances the DAE system over time. Handles the core mathematical simulation of reactor dynamics.

**Types**:
- `ScipyIntegrator`: Uses scipy's ODE solvers.
- `DiffeqpyIntegrator`: Uses DifferentialEquations.jl through Python interface.

**Usage in DEgym**: Called during `step()` to compute the next state given current state, parameters, and actions.

### **SystemDynamicsFn**
Implements the differential equations that describe the reactor dynamics.

**Purpose**: Defines the mathematical model f in: d/dt dae_state = f(dae_state, dae_params, dae_action).

**Usage in DEgym**: Used by the Integrator to evaluate derivatives during numerical solving.

### **IntegratorConfig**
Configuration parameters for the numerical integrator.

**Examples**: Time step size, solver method, tolerance settings, maximum iterations.

**Usage in DEgym**: Defines how the numerical integration should be performed.

## Environment Components

### **PhysicalParametersGenerator**
Generates PhysicalParameters for each episode, enabling domain randomization.

**Usage in DEgym**: Called during environment initialization and reset to create diverse training scenarios.

### **InitialStateGenerator**
Creates initial State instances at the start of episodes using PhysicalParameters.

**Process**:
1. Takes PhysicalParameters as input.
2. Creates initial DAEState (concentrations, temperature, etc.).
3. Generates DAEParameters for equations.
4. Sets up NonDAEParameters for simulation control.
5. Combines into complete State.

**Usage in DEgym**: Called during `reset()` to establish starting conditions for new episodes.

### **Observation**
Data structure that represents what the RL agent can perceive from the environment.

**Purpose**: Encapsulates agent-visible information with conversion to numpy arrays.

**Usage in DEgym**: Created by ObservationExtractor and returned to agents through `step()` and `reset()`.

---

## Key Relationships

1. **Raw Action → Action → DAEAction**: Action processing pipeline that transforms agent outputs to physical control parameters.

2. **PhysicalParameters → State**: Configuration-to-simulation mapping that establishes episode parameters.

3. **State → Integrator → Next State**: Core simulation loop that advances system dynamics.

4. **State → Extractors → RL Outputs**: Information extraction that converts internal state to agent-facing data.

5. **Environment**: Orchestrates all components and provides the standard RL interface.

This terminology forms the foundation for understanding how DEgym separates RL-specific logic (data flow, interfaces) from use-case-specific logic (what actions mean, how states are defined).
