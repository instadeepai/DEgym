from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.axes import Axes

from degym_tutorials.cstr_tutorial.physical_parameters import CSTRPhysicalParameters
from degym_tutorials.cstr_tutorial.state_concrete_classes import CSTRState


def draw_motor(ax: Axes) -> None:
    """
    Draw the motor and its connection to the shaft.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes object to draw on.

    Returns:
        None
    """
    motor_body = patches.Rectangle(
        (4.4, 8.5), 1.2, 0.3, edgecolor="black", facecolor="gray", linewidth=2
    )
    ax.add_patch(motor_body)

    # Connection from motor to the shaft
    motor_connection = patches.Rectangle((4.95, 8), 0.1, 0.5, edgecolor="black", facecolor="black")
    ax.add_patch(motor_connection)


def draw_pipes(ax: Axes) -> None:
    """
    Draw the inflow and outflow pipes for the reactor.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes object to draw on.

    Returns:
        None
    """
    # Inflow pipe and arrow entering the reactor
    inflow_pipe_outer = patches.FancyArrow(
        0.8, 7.8, 1.2, 0, width=0.2, head_width=0.3, head_length=0.4, color="gray"
    )
    ax.add_patch(inflow_pipe_outer)

    inflow_pipe_inner = patches.Rectangle((3, 8), -0.5, -0.4, edgecolor="gray", facecolor="gray")
    ax.add_patch(inflow_pipe_inner)

    # Outflow pipe and arrow outside the reactor
    outflow_pipe_outer = patches.FancyArrow(
        7.7, 3.2, 1.2, 0, width=0.2, head_width=0.3, head_length=0.4, color="gray"
    )
    ax.add_patch(outflow_pipe_outer)

    outflow_pipe_inner = patches.Rectangle((7, 3.4), 0.5, -0.4, edgecolor="gray", facecolor="gray")
    ax.add_patch(outflow_pipe_inner)


def draw_tank_walls_and_lid(ax: Axes) -> None:
    """
    Draw the outer and inner tank walls and the tank lid.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes object to draw on.

    Returns:
        None
    """
    # Outer tank wall
    outer_tank_body = patches.Rectangle(
        (2.8, 2.8), 4.4, 5.4, edgecolor="black", facecolor="gray", linewidth=2
    )
    ax.add_patch(outer_tank_body)

    # Inner tank wall
    inner_tank_body = patches.Rectangle(
        (3, 3), 4, 5, edgecolor="black", facecolor="white", zorder=1
    )
    ax.add_patch(inner_tank_body)

    # Tank lid (double line)
    lid_outer = patches.Arc(
        (5, 8.2), 4.4, 1.2, angle=0, theta1=0, theta2=180, edgecolor="black", linewidth=2
    )
    lid_inner = patches.Arc(
        (5, 8), 4, 1, angle=0, theta1=7, theta2=173, edgecolor="black", linewidth=1.5
    )
    ax.add_patch(lid_outer)
    ax.add_patch(lid_inner)


def draw_stirrer(ax: Axes) -> None:
    """
    Draw the stirrer shaft and propeller blades.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes object to draw on.

    Returns:
        None
    """
    # Thicker stirrer shaft
    stirrer_shaft = patches.Rectangle((4.95, 3.4), 0.1, 4.5, edgecolor="black", facecolor="black")
    ax.add_patch(stirrer_shaft)

    # Propeller blades (rectangles connected by small rods)
    propeller_blade1 = patches.Rectangle(
        (5, 3.3), 0.5, 0.1, angle=0, edgecolor="black", facecolor="black"
    )
    propeller_blade2 = patches.Rectangle(
        (4.5, 3.3), 0.5, 0.1, angle=0, edgecolor="black", facecolor="black"
    )

    ax.add_patch(propeller_blade1)
    ax.add_patch(propeller_blade2)


def draw_tank_structure(ax: Axes) -> None:
    """
    Draw the complete tank structure including walls, lid, stirrer, motor, and pipes.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes object to draw on.

    Returns:
        None
    """
    draw_tank_walls_and_lid(ax)
    draw_stirrer(ax)
    draw_motor(ax)
    draw_pipes(ax)


def annotate_flows_with_purity_bars(
    state: CSTRState, physical_parameters: CSTRPhysicalParameters, ax: Axes
) -> None:
    """
    Add flow annotations and purity bars showing inlet and outlet compositions.

    Args:
        state: State object containing concentration values (c_a, c_b).
        physical_parameters: Physical parameters object containing flow rate (F) and
        initial concentration (c_a_0).
        ax (matplotlib.axes.Axes): The matplotlib axes object to draw on.

    Returns:
        None
    """
    flow = physical_parameters.F
    c_a_0 = physical_parameters.c_a_0
    c_a = state.c_a
    c_b = state.c_b
    purity_bar_whole = patches.Rectangle((0.3, 7.6), 0.3, 1, facecolor="orange")
    ax.add_patch(purity_bar_whole)
    ax.text(
        1.4,
        6.8,
        r"$F$ = " + f"{flow}" + r"$\ \mathrm{mol/min}$",
        fontsize=10,
        color="black",
        ha="center",
    )
    ax.text(
        1.4,
        7.2,
        r"$C_{A,0} = $" + f"{c_a_0}" + r"$\ \mathrm{mol/L}$",
        fontsize=10,
        color="black",
        ha="center",
    )

    ax.text(
        7.8,
        4.1,
        r"$C_B = $" + f"{c_b:.2f}" + r"$\ \mathrm{mol/L}$",
        fontsize=10,
        color="black",
        ha="left",
    )
    purity = c_b / (c_a + c_b)
    ax.text(
        7.8,
        3.7,
        r"$\mathrm{purity} = $" + f"{int(purity * 100)} %",
        fontsize=10,
        color="black",
        ha="left",
    )
    purity_bar_whole = patches.Rectangle((9.7, 3), 0.3, 1, facecolor="orange")
    purity_bar_product = patches.Rectangle((9.7, 3), 0.3, purity, facecolor="blue")
    ax.add_patch(purity_bar_whole)
    ax.add_patch(purity_bar_product)


def annotate_reaction(ax: Axes) -> None:
    """
    Add reaction annotations including species labels and reaction equation.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes object to draw on.

    Returns:
        None
    """
    # Tank annotations
    ax.text(7.8, 8.1, r"$A$", fontsize=12, ha="center", color="black", fontweight="bold")
    ax.text(7.8, 7.6, r"$B$", fontsize=12, ha="center", color="black", fontweight="bold")
    ax.scatter([8.1], [7.7], color="blue", s=40)
    ax.scatter([8.1], [8.2], color="orange", s=40)

    # Reaction details
    ax.text(
        5, 9, r"$A \rightleftarrows B$", fontsize=16, ha="center", color="black", fontweight="bold"
    )


def draw_heater(ax: Axes) -> None:
    """
    Draw the heater element beneath the reactor.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes object to draw on.

    Returns:
        None
    """
    # Heater beneath the reactor
    shift = 0.3
    heater_body = patches.Rectangle(
        (3.8, 2 - shift), 2.4, 0.6, edgecolor="black", facecolor="red", linewidth=2
    )
    ax.add_patch(heater_body)
    ax.text(
        5,
        2.3 - shift,
        "Heater",
        fontsize=10,
        ha="center",
        va="center",
        color="black",
        fontweight="bold",
    )


def annotate_heater_with_action(ax: Axes, action: float) -> None:
    """
    Add heat input annotation showing the current action value.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes object to draw on.
        action (float): The heat input action value in KJ/min.

    Returns:
        None
    """
    ax.text(
        5,
        2.4,
        r"$\dot{Q} = \ $" + f"{int(action)}" + r"$\mathrm{KJ/min}$",
        fontsize=14,
        ha="center",
        color="black",
        fontweight="bold",
    )


def draw_circles(state: CSTRState, ax: Axes) -> None:
    """
    Draw circles inside the tank representing the concentration of species A and B.

    Args:
        state: State object containing concentration values (c_a, c_b).
        ax (matplotlib.axes.Axes): The matplotlib axes object to draw on.

    Returns:
        None
    """
    # Generate random positions for 100 circles inside the tank
    c_a = state.c_a
    c_b = state.c_b
    num_circles = 100
    blue_count = int(num_circles * c_a / (c_a + c_b))

    # Random positions for the circles
    x_positions = np.random.uniform(3.1, 6.9, size=num_circles)
    y_positions = np.random.uniform(3.2, 8.0, size=num_circles)

    # Plot blue circles
    ax.scatter(x_positions[:blue_count], y_positions[:blue_count], color="orange", s=40)

    # Plot orange circles
    ax.scatter(x_positions[blue_count:], y_positions[blue_count:], color="blue", s=40)


def draw_cstr_with_heater_and_circles(
    state: CSTRState, physical_parameters: CSTRPhysicalParameters, action: Optional[float] = None
) -> np.ndarray:
    """
    Create a complete visualization of the CSTR with heater and concentration representation.

    Args:
        state: State object containing concentration values (c_a, c_b).
        physical_parameters: Physical parameters object containing flow rate (F) and initial
        concentration (c_a_0).
        action (float, optional): The heat input action value in KJ/min.
        If provided, will be displayed on the heater.

    Returns:
        np.ndarray: A numpy array representation of the rendered image with shape with
          (height, width, channels).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the tank structure
    draw_tank_structure(ax)
    annotate_flows_with_purity_bars(state, physical_parameters, ax)
    annotate_reaction(ax)
    draw_heater(ax)
    if action is not None:
        annotate_heater_with_action(ax, action)

    draw_circles(state, ax)

    # General setting
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Convert figure to numpy array directly
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    buf = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # Convert RGBA to RGB by dropping alpha channel
    buf = buf[:, :, :3]
    plt.close(fig)

    return buf
