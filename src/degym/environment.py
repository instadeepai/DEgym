# Copyright 2025 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import Optional, Tuple, final

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from degym.action import ActionPreprocessor, DAEAction, RawActionType
from degym.extractors import (
    InfoExtractor,
    Observation,
    ObservationExtractor,
    RewardExtractor,
    TerminatedExtractor,
    TruncatedExtractor,
)
from degym.integrators import Integrator, TimeSpan
from degym.physical_parameters import PhysicalParametersGenerator
from degym.state import (
    DAEParameters,
    DAEState,
    InitialStateGenerator,
    NonDAEParameters,
    State,
    StatePostprocessor,
    StatePreprocessor,
)
from degym.utils import NoOverrideMeta, no_override


class Environment(gym.Env, metaclass=NoOverrideMeta):
    """Environment class forms the interface between the agent and the environment."""

    @final
    @no_override
    def __init__(  # noqa: PLR0913
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
        """
        Initialize a DEgym environment for chemical and biological reactor simulations.

        Args:
            physical_parameters_generator: Generates physical parameters for the reactor
                system (e.g., flow rates, volumes, reaction constants). These parameters
                are sampled at environment reset and can vary between episodes to create
                diverse training scenarios.
            initial_state_generator: Generates initial states for new episodes, including
                initial concentrations, temperatures, and other system variables. Takes
                physical parameters as input to ensure consistent initialization.
            integrator: Numerical integrator (e.g., scipy or DiffEqPy-based) that solves
                the DAE system over time. Handles the core mathematical simulation of
                reactor dynamics between timesteps.
            action_preprocessor: Preprocesses RL actions before integration. Converts
                raw RL actions to DAE actions, applies constraints, and handles action
                validation and transformation.
            state_preprocessor: Preprocesses the environment state before integration.
                Applies any necessary state transformations, scaling, or corrections
                before passing to the numerical solver.
            state_postprocessor: Processes the state after numerical integration to
                handle any necessary corrections, constraints, or transformations
                before extracting observations and rewards.
            observation_extractor: Extracts observations from the environment state
                that will be provided to the RL agent. Defines the observation space
                and converts internal state representation to agent-visible format.
            reward_extractor: Computes reward signals based on state transitions
                (state, action, next_state). Implements the optimization objective
                for the RL agent (e.g., maximizing product concentration).
            terminated_extractor: Determines if an episode should terminate due to
                termination conditions (e.g., reaching unsafe operating conditions,
                achieving target objectives, or constraint violations).
            truncated_extractor: Determines if an episode should truncate due to
                time limits or other truncation conditions (e.g., maximum timesteps
                reached, computational limits exceeded).
            info_extractor: Extracts additional diagnostic information from state
                transitions that may be useful for monitoring, debugging, or analysis
                but is not part of the core RL loop.
            seed: Random seed for reproducible environment behavior. Controls random
                number generation for parameter sampling, initial state generation,
                and any stochastic processes within the environment.

        Note:
            All components except the integrator are use-case-specific and must be
            implemented by subclassing the corresponding abstract classes. The
            Environment class provides the RL-specific data flow and interfaces,
            while the injected components define the domain-specific behavior.

            The environment maintains internal state including current system state,
            simulation time, step counter, and physical parameters, which are
            automatically managed through the step() and reset() methods.
        """
        self._physical_parameters_generator = physical_parameters_generator
        self._integrator = integrator
        self._action_preprocessor = action_preprocessor
        self._state_preprocessor = state_preprocessor
        self._state_postprocessor = state_postprocessor
        self._observation_extractor = observation_extractor
        self._reward_extractor = reward_extractor
        self._terminated_extractor = terminated_extractor
        self._truncated_extractor = truncated_extractor
        self._info_extractor = info_extractor
        self._seed = seed
        self._initial_state_generator = initial_state_generator
        self._rng = np.random.default_rng(seed)

        # Generate physical parameters
        self._physical_parameters = self._physical_parameters_generator.generate(rng=self._rng)
        # Generate initial state
        self._state: State = self._initial_state_generator.generate(
            physical_parameters=self._physical_parameters
        )
        # Initialize time and step counter
        self._current_time: float = 0.0
        self._step_counter: int = 0

    @final
    @no_override
    def step(self, action: RawActionType) -> Tuple[NDArray[np.floating], float, bool, bool, dict]:
        """
        Given an input action, perform an update step in the environment
        to compute the next state. Outputs are computed by extractors.

        Args:
            action: RL action selected by the agent.
        Returns:
            observation: Observed version of the next environment state.
            reward: Reward returned for this (state, action) pair.
            terminated: Whether the episode terminated due to a termination condition.
            truncated: Whether the episode was truncated due to a time limit.
            info: Info dictionary returned by the environment after applying the action.
        """
        # Preprocess (state, action) before passing to integrator
        preprocessed_state = self._state_preprocessor.preprocess_state(self.state)
        dae_action = self._action_preprocessor.preprocess_action(action, self.state)
        # Calculate time span for this step
        time_span = self._calculate_time_span()

        # Use integrator to solve for next state
        next_state = self._compute_next_state(
            state=preprocessed_state, dae_action=dae_action, time_span=time_span
        )
        # Postprocess next environment state
        postprocessed_next_state = self._state_postprocessor.postprocess_state(next_state)

        # Extract outputs
        observation_array, reward, terminated, truncated, info = self._extract_step_outputs(
            state=self.state, action=dae_action, next_state=postprocessed_next_state
        )

        # Update internal state tracking: state, time, and step count
        self._state = postprocessed_next_state
        self._current_time = time_span.end_time
        self._step_counter += 1
        return observation_array, reward, terminated, truncated, info

    @final
    @no_override
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[NDArray[np.floating], dict]:
        """Reset the environment to the initial state and return observation and info."""
        # Generate new physical parameters, current state, and reset time
        self._physical_parameters = self._physical_parameters_generator.generate(rng=self._rng)
        self._state = self._initial_state_generator.generate(self._physical_parameters)
        self._current_time = 0.0
        self._step_counter = 0

        # Extract observation and info
        observation: Observation = self._observation_extractor.extract_observation(
            next_state=self._state
        )
        info = self._info_extractor.extract_info(state=None, action=None, next_state=self._state)
        return observation.to_np_array(), info

    @final
    @no_override
    def _extract_step_outputs(
        self, state: State, action: DAEAction, next_state: State
    ) -> Tuple[NDArray[np.floating], float, bool, bool, dict]:
        """
        Extract the step function's output from (s, a, s').

        Note: s is the state at the beginning of the step, s' is the state at the end of the step,
        i.e., the state after applying action a.

        Args:
            state: The state at the beginning of the step.
            action: The action taken in the step.
            next_state: The state at the end of the step.

        Returns:
            observation: Observation emitted from s'.
            reward: Reward returned for (s, a, s').
            terminated: Whether the episode terminated due to termination conditions.
            truncated: Whether the episode was truncated due to truncation conditions.
            info: Info dictionary returned by the environment.
        """
        observation = self._observation_extractor.extract_observation(next_state=next_state)
        reward = self._reward_extractor.extract_reward(
            state=state, action=action, next_state=next_state
        )
        terminated = self._terminated_extractor.extract_terminated(
            state=state, action=action, next_state=next_state
        )
        truncated = self._truncated_extractor.extract_truncated(
            state=state, action=action, next_state=next_state
        )
        info = self._info_extractor.extract_info(state=state, action=action, next_state=next_state)

        return observation.to_np_array(), reward, terminated, truncated, info

    @abstractmethod
    def _calculate_time_span(self) -> TimeSpan:
        """Calculate the time span for the step function."""

    @final
    @no_override
    def _return_next_dae_state(
        self, state: State, dae_action: DAEAction, time_span: TimeSpan
    ) -> DAEState:
        """
        Compute and return next DAE state by integrating the dynamics over the provided time span.

        Args:
            state: The state at the beginning of the step.
            dae_action: The action taken in the step.
            time_span: The time span of the step.

        Returns:
            next_dae_state: The next DAE state of the environment.
        """
        # Prepare state and action for use with integrator
        dae_state_array = state.dae_state.to_np_array()
        dae_parameters_array = state.dae_params.to_np_array()
        dae_action_array = dae_action.to_np_array()

        # Compute updated DAEState values using integrator
        next_dae_state_values = self._integrator.integrate(
            input_values=dae_state_array,
            parameters=dae_parameters_array,
            action=dae_action_array,
            time_span=time_span,
        )
        cls_ = state.dae_state.__class__
        return cls_.from_np_array(next_dae_state_values)

    @abstractmethod
    def _return_next_dae_params(self, state: State) -> DAEParameters:
        """Return the next DAE parameters of the state."""
        raise NotImplementedError

    @abstractmethod
    def _return_next_non_dae_params(self, state: State) -> NonDAEParameters:
        """Return the next non-DAE parameters of the state."""
        raise NotImplementedError

    @final
    @no_override
    def _compute_next_state(
        self, state: State, dae_action: DAEAction, time_span: TimeSpan
    ) -> State:
        """
        Compute and return next state by integrating the dynamics over the provided time span.

        The next state is computed by updating all parts of the state:
        * the dae_state
        * the dae_parameters
        * the non_dae_parameters

        Args:
            state: The state at the beginning of the step.
            dae_action: The action taken in the step.
            time_span: The time span of the step.

        Returns:
            next_state: The next state of the environment.
        """
        next_dae_state = self._return_next_dae_state(
            state=state, dae_action=dae_action, time_span=time_span
        )
        next_dae_params = self._return_next_dae_params(state=state)
        next_non_dae_params = self._return_next_non_dae_params(state=state)

        cls_ = state.__class__
        next_state = cls_(
            dae_state=next_dae_state,
            dae_params=next_dae_params,
            non_dae_params=next_non_dae_params,
        )
        return next_state

    @property
    @final
    @no_override
    def state(self) -> State:
        """Return the current state of the environment."""
        return self._state

    @property
    @final
    @no_override
    def current_time(self) -> float:
        """Return the current time of the environment."""
        return self._current_time

    @property
    @final
    @no_override
    def step_counter(self) -> int:
        """Return the current step counter of the environment."""
        return self._step_counter

    @property
    @final
    @no_override
    def observation_space(self) -> gym.spaces.Space:
        """Return the observation_space, as specified in the ObservationExtractor."""
        return self._observation_extractor.observation_space

    @property
    @final
    @no_override
    def action_space(self) -> gym.spaces.Space:
        """Return the action_space, as specified in the ActionPreprocessor."""
        return self._action_preprocessor.action_space
