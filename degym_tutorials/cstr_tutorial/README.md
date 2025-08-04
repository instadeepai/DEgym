# CSTR Tutorial: Creating Custom DEgym Environments

This tutorial demonstrates how to create a custom CSTR (Continuous Stirred-Tank Reactor) environment using the DEgym framework. This is a practical, hands-on implementation guide.

> [!TIP]
> **New to DEgym?** Start with the [DEgym Essentials](../../docs/degym_essentials.md) to understand the core architecture, then follow the [comprehensive tutorial](../../docs/how_to_build_new_env.md) for detailed step-by-step instructions.

## Quick Overview

This tutorial implements a CSTR with reversible reaction A ⇌ B, where:

<p align="center">
  <img src="../../docs/images/cstr_animation.gif" width="400">
</p>

>[!TIP]
> For a more  **detailed** instructiosn list :newspaper_roll: on how to implement a new environment using degym, refer to [comprehensive tutorial](../../docs/how_to_build_new_env.md)

In this example, we implemented a CSTR with reversible reaction A ⇌ B, where:
- **Goal**: Maximize product B concentration
- **Action**: Heat input (0 to Q_max)
- **Observation**: concentrations and temperature
- **Dynamics**: The chemical reactions (in form of mass and energy balance equations)

### Key Equations

Mass balances:
$$\frac{dc_A}{dt} = \frac{F}{V} (c_{A,0} - c_A) - k_A c_A + k_B c_B$$
$$\frac{dc_B}{dt} = \frac{F}{V} (-c_B) + k_A c_A - k_B c_B$$

Energy balance:
$$\frac{dT}{dt} = \frac{F \rho C_p (T_0 - T) + \dot{Q} - \Delta H (k_A c_A - k_B c_B)}{\rho C_p V}$$

## Implementation Summary
This tutorial follows the 9-step process outlined in the [main tutorial](../../docs/how_to_build_new_env.md):

- ✅ DAE formulation identification
- ✅ State classes (`DAEState`, `DAEParameters`, `NonDAEParameters`)
- ✅ Action classes (`Action`, `DAEAction`, converters, regulators)
- ✅ Physical parameters and generators
- ✅ System dynamics (SciPy and DiffEqPy)
- ✅ Extractors (observation, reward, termination, info)
- ✅ Environment class with time span and state computation
- ✅ Factory function for easy instantiation

This hands-on tutorial provides a complete, working implementation that you can use as a template for your own reactor environments.


## Running the Example

### Quick Start
```python
from make_env import make_cstr_environment
import numpy as np

# Minimal configuration
env_config = {
    "integrator": "scipy",
    "integrator_config": {
        "action_duration": 0.1,
        "method": "RK45",
        "rtol": 1e-6,
        "atol": 1e-8,
            },
            "random_seed": 0,
            "physical_parameters": {
                "fixed_values": {
                    "c_a_0": 0.3,
                    "c_p": 3.25,
                    "e_a": 41570,
                    "e_b": 45727,
                    "f": 0.0025,
                    "dh": 4157,
                    "k_0_a": 50_000,
                    "k_0_b": 100_000,
                    "r": 8.314,
                    "t_0": 300,
                    "v": 0.2,
                    "q_max": 5000,
                    "max_timestep": 100,
                },
                "sampled_values": {
            "p": {"distribution": "choice", "choices": [780, 790], "size": 1}
        },
    },
}

env = make_cstr_environment(env_config)
_, _ = env.reset()
done = False
while not done:
    action = np.random.randint(0, 5000)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    rgb_img = env.render(action)
    # save the image without background and with tight layout
    os.makedirs("images_for_vis", exist_ok=True)
    image_path = f"./images_for_vis/cstr_step_{env.step_counter}.png"
    plt.imsave(image_path, rgb_img)
```

## Next Steps

1. **Study the Code**: Examine each file to understand the implementation details
2. **Modify Parameters**: Try different reaction rates, temperatures, or constraints
3. **Custom Rewards**: Implement different reward functions
4. **Add Constraints**: Experiment with action and state limitations

## Documentation Links

- **[Installation Guide](../../docs/installation.md)**: Setup instructions
- **[DEgym Essentials](../../docs/degym_essentials.md)**: Core architecture and concepts
- **[Complete Tutorial](../../docs/how_to_build_new_env.md)**: Detailed implementation guide
- **[Main README](../../README.md)**: Project overview and examples
