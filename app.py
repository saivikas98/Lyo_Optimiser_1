import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import fsolve

st.set_page_config(
    page_title="Primary Drying Simulator",
    page_icon="❄️",
    layout="wide",
)

st.title("Primary Drying Center vs Edge Vial Simulator")
st.caption(
    "Minute-based transient primary-drying model with exact VS Code legacy matching, "
    "transactional constraint mode, unit audit, plots, and downloadable results."
)


# ============================================================
# MODEL FUNCTIONS
# ============================================================
def product_resistance(ldry, a5, a6, a7):
    return a5 + (a6 * ldry) / (1.0 + a7 * ldry)


def kv_center(chamber_pressure):
    return (
        9.211
        + (0.066 * chamber_pressure * 1000.0)
        / (1.0 + 0.002 * chamber_pressure * 1000.0)
    ) * 0.001434


def kv_edge(chamber_pressure):
    return (
        (0.981 * chamber_pressure * 1000.0)
        / (1.0 + 0.02 * chamber_pressure * 1000.0)
    ) * 0.001434


def vapor_pressure_ice(temperature_c):
    temperature_k = temperature_c + 273.15
    return np.exp(-6144.96 / temperature_k + 24.01849)


def coupled_residual_transient(
    x,
    ldry,
    shelf_temperature,
    chamber_pressure,
    previous_product_temperature,
    dt,
    kv_function,
    parameters,
):
    interface_temperature, product_temperature = x
    ice_pressure = vapor_pressure_ice(interface_temperature)

    if ice_pressure <= chamber_pressure:
        return [1.0e6, 1.0e6]

    resistance = product_resistance(
        ldry,
        parameters["a5"],
        parameters["a6"],
        parameters["a7"],
    )

    sublimation_rate = max(
        0.0,
        (parameters["product_area"] / resistance)
        * (ice_pressure - chamber_pressure),
    )

    remaining_ice_height = max(
        parameters["initial_cake_height"] - ldry,
        1.0e-6,
    )

    thermal_mass = max(
        parameters["ice_density"]
        * parameters["product_area"]
        * ldry
        * parameters["solid_fraction"],
        1.0e-8,
    )

    kv_value = kv_function(chamber_pressure)

    heat_input = (
        kv_value
        * parameters["vial_area"]
        * (shelf_temperature - product_temperature)
    )

    heat_sublimation = (
        sublimation_rate
        * parameters["sublimation_enthalpy"]
    )

    heat_accumulation = (
        thermal_mass
        * parameters["heat_capacity"]
        * (product_temperature - previous_product_temperature)
        / dt
    )

    energy_residual = heat_input - (
        heat_sublimation + heat_accumulation
    )

    conduction = (
        parameters["ice_conductivity"]
        * parameters["vial_area"]
        * (product_temperature - interface_temperature)
        / remaining_ice_height
    )

    conduction_residual = heat_input - conduction

    return [energy_residual, conduction_residual]


def calculate_sublimation_rate(
    ldry,
    interface_temperature,
    chamber_pressure,
    parameters,
):
    resistance = product_resistance(
        ldry,
        parameters["a5"],
        parameters["a6"],
        parameters["a7"],
    )

    return max(
        0.0,
        (parameters["product_area"] / resistance)
        * (
            vapor_pressure_ice(interface_temperature)
            - chamber_pressure
        ),
    )


def solve_legacy_position(
    ldry,
    shelf_temperature,
    chamber_pressure,
    previous_product_temperature,
    interface_guess,
    dt,
    kv_function,
    parameters,
):
    """Match the original VS Code fsolve behavior.

    The original code uses the solution returned by fsolve without checking
    the convergence flag. This function intentionally preserves that behavior.
    """
    solution = fsolve(
        coupled_residual_transient,
        [interface_guess, previous_product_temperature],
        args=(
            ldry,
            shelf_temperature,
            chamber_pressure,
            previous_product_temperature,
            dt,
            kv_function,
            parameters,
        ),
    )

    interface_temperature, product_temperature = solution

    sublimation_rate = calculate_sublimation_rate(
        ldry,
        interface_temperature,
        chamber_pressure,
        parameters,
    )

    return (
        interface_temperature,
        product_temperature,
        sublimation_rate,
    )


def solve_transactional_position(
    ldry,
    shelf_temperature,
    chamber_pressure,
    previous_product_temperature,
    interface_guess,
    dt,
    kv_function,
    parameters,
):
    """Solve with an explicit convergence check for transactional mode."""
    solution, _, ier, message = fsolve(
        coupled_residual_transient,
        [interface_guess, previous_product_temperature],
        args=(
            ldry,
            shelf_temperature,
            chamber_pressure,
            previous_product_temperature,
            dt,
            kv_function,
            parameters,
        ),
        full_output=True,
        xtol=1.0e-8,
        maxfev=300,
    )

    if ier != 1 or not np.all(np.isfinite(solution)):
        raise RuntimeError(
            f"Nonlinear solver did not converge: {message}"
        )

    interface_temperature, product_temperature = solution

    sublimation_rate = calculate_sublimation_rate(
        ldry,
        interface_temperature,
        chamber_pressure,
        parameters,
    )

    return (
        interface_temperature,
        product_temperature,
        sublimation_rate,
    )


def make_record(
    time_minutes,
    shelf_temperature,
    center_interface,
    center_product,
    edge_interface,
    edge_product,
    center_dry_layer,
    edge_dry_layer,
    center_rate,
    edge_rate,
    parameters,
):
    initial_cake_height = parameters["initial_cake_height"]

    center_remaining = max(
        initial_cake_height - center_dry_layer,
        0.0,
    )
    edge_remaining = max(
        initial_cake_height - edge_dry_layer,
        0.0,
    )

    center_heat_input = (
        kv_center(parameters["pressure"])
        * parameters["vial_area"]
        * (shelf_temperature - center_product)
    )
    edge_heat_input = (
        kv_edge(parameters["pressure"])
        * parameters["vial_area"]
        * (shelf_temperature - edge_product)
    )

    center_conduction = (
        parameters["ice_conductivity"]
        * parameters["vial_area"]
        * (center_product - center_interface)
        / max(center_remaining, 1.0e-6)
    )
    edge_conduction = (
        parameters["ice_conductivity"]
        * parameters["vial_area"]
        * (edge_product - edge_interface)
        / max(edge_remaining, 1.0e-6)
    )

    return {
        "Time (min)": time_minutes,
        "Time (hr)": time_minutes / 60.0,
        "Shelf temperature (°C)": shelf_temperature,
        "Center interface temperature (°C)": center_interface,
        "Center product temperature (°C)": center_product,
        "Center dry layer (cm)": center_dry_layer,
        "Center remaining cake (cm)": center_remaining,
        "Center drying (%)": (
            100.0 * center_dry_layer / initial_cake_height
        ),
        "Center sublimation rate (g/min)": center_rate,
        "Center heat input (cal/min)": center_heat_input,
        "Center conduction (cal/min)": center_conduction,
        "Edge interface temperature (°C)": edge_interface,
        "Edge product temperature (°C)": edge_product,
        "Edge dry layer (cm)": edge_dry_layer,
        "Edge remaining cake (cm)": edge_remaining,
        "Edge drying (%)": (
            100.0 * edge_dry_layer / initial_cake_height
        ),
        "Edge sublimation rate (g/min)": edge_rate,
        "Edge heat input (cal/min)": edge_heat_input,
        "Edge conduction (cal/min)": edge_conduction,
    }


# ============================================================
# EXACT LEGACY VS CODE MODE
# ============================================================
def simulate_legacy(parameters):
    """Reproduce the supplied VS Code loop behavior line by line."""
    product_area = parameters["product_area"]
    vial_area = parameters["vial_area"]
    initial_cake_height = parameters["initial_cake_height"]
    dt = parameters["dt"]  # minutes
    chamber_pressure = parameters["pressure"]

    shelf_temperature = (
        6144.96
        / (24.01849 - math.log(chamber_pressure))
        - 273.15
    )

    time_minutes = 0.0
    center_dry_layer = 0.0
    edge_dry_layer = 0.0

    center_interface_guess = parameters[
        "initial_interface_temperature"
    ]
    edge_interface_guess = parameters[
        "initial_interface_temperature"
    ]

    center_interface_history = []
    center_product_history = []
    edge_interface_history = []
    edge_product_history = []

    records = []
    rejected_steps = 0
    loop_count = 0

    maximum_loops = (
        int(parameters["max_duration_hr"] * 60.0 / dt)
        + parameters["max_rejected_steps"]
        + 1000
    )

    while True:
        loop_count += 1

        if loop_count > maximum_loops:
            raise RuntimeError(
                "Maximum loop count reached before the original "
                "stop condition was satisfied."
            )

        # Exact stopping condition from the original script.
        if (
            center_dry_layer >= initial_cake_height
            and edge_dry_layer >= initial_cake_height
            and len(center_product_history) > 10
            and abs(
                center_product_history[-1]
                - shelf_temperature
            ) < 3.0
            and abs(
                edge_product_history[-1]
                - shelf_temperature
            ) < 2.0
        ):
            break

        # ------------------------- CENTER -------------------------
        if center_dry_layer < initial_cake_height:
            center_previous_product = (
                center_product_history[-1]
                if center_product_history
                else center_interface_guess
            )

            (
                center_interface,
                center_product,
                center_rate,
            ) = solve_legacy_position(
                center_dry_layer,
                shelf_temperature,
                chamber_pressure,
                center_previous_product,
                center_interface_guess,
                dt,
                kv_center,
                parameters,
            )

            center_increment = (
                center_rate
                * dt
                / (
                    parameters["ice_density"]
                    * product_area
                    * (1.0 - parameters["solid_fraction"])
                )
            )

            # Exact legacy behavior: update before Tcrit check.
            # Do not clip to the initial cake height.
            center_dry_layer += center_increment

        else:
            center_thermal_mass = (
                parameters["ice_density"]
                * product_area
                * initial_cake_height
                * parameters["solid_fraction"]
            )

            center_product = center_product_history[-1]
            center_product += (
                kv_center(chamber_pressure)
                * vial_area
                * (shelf_temperature - center_product)
                * dt
                / (
                    center_thermal_mass
                    * parameters["heat_capacity"]
                )
            )

            center_interface = center_interface_history[-1]
            center_rate = 0.0

        # -------------------------- EDGE --------------------------
        if edge_dry_layer < initial_cake_height:
            edge_previous_product = (
                edge_product_history[-1]
                if edge_product_history
                else edge_interface_guess
            )

            (
                edge_interface,
                edge_product,
                edge_rate,
            ) = solve_legacy_position(
                edge_dry_layer,
                shelf_temperature,
                chamber_pressure,
                edge_previous_product,
                edge_interface_guess,
                dt,
                kv_edge,
                parameters,
            )

            edge_increment = (
                edge_rate
                * dt
                / (
                    parameters["ice_density"]
                    * product_area
                    * (1.0 - parameters["solid_fraction"])
                )
            )

            # Exact legacy behavior: update before Tcrit check.
            # Do not clip to the initial cake height.
            edge_dry_layer += edge_increment

        else:
            edge_thermal_mass = (
                parameters["ice_density"]
                * product_area
                * initial_cake_height
                * parameters["solid_fraction"]
            )

            edge_product = edge_product_history[-1]
            edge_product += (
                kv_edge(chamber_pressure)
                * vial_area
                * (shelf_temperature - edge_product)
                * dt
                / (
                    edge_thermal_mass
                    * parameters["heat_capacity"]
                )
            )

            edge_interface = edge_interface_history[-1]
            edge_rate = 0.0

        # Exact original post-solve clipping.
        center_interface = min(
            center_interface,
            shelf_temperature - 1.0,
        )
        center_product = min(
            center_product,
            shelf_temperature - 1.0,
        )
        edge_interface = min(
            edge_interface,
            shelf_temperature - 1.0,
        )
        edge_product = min(
            edge_product,
            shelf_temperature - 1.0,
        )

        active_product_temperatures = []

        if center_dry_layer < initial_cake_height:
            active_product_temperatures.append(center_product)

        if edge_dry_layer < initial_cake_height:
            active_product_temperatures.append(edge_product)

        if (
            active_product_temperatures
            and max(active_product_temperatures)
            > parameters["critical_temperature"]
        ):
            shelf_temperature -= parameters["constraint_step"]
            rejected_steps += 1

            if (
                rejected_steps
                > parameters["max_rejected_steps"]
            ):
                raise RuntimeError(
                    "Too many Tcrit rejections in legacy mode."
                )

            # Exact legacy behavior:
            # - dry layer remains advanced
            # - time does not advance
            # - histories do not update
            # - interface guesses do not update
            # - shelf ramp does not occur
            continue

        shelf_temperature = min(
            shelf_temperature,
            parameters["maximum_shelf_temperature"],
        )

        # Accepted time step.
        time_minutes += dt

        center_interface_history.append(center_interface)
        center_product_history.append(center_product)
        edge_interface_history.append(edge_interface)
        edge_product_history.append(edge_product)

        records.append(
            make_record(
                time_minutes,
                shelf_temperature,
                center_interface,
                center_product,
                edge_interface,
                edge_product,
                center_dry_layer,
                edge_dry_layer,
                center_rate,
                edge_rate,
                parameters,
            )
        )

        # Exact original shelf control per accepted iteration.
        if shelf_temperature < -10.0:
            shelf_temperature += 0.2
        else:
            shelf_temperature += 0.1 / 3.0

        # Exact original: update guesses only after acceptance.
        center_interface_guess = center_interface
        edge_interface_guess = edge_interface

    if not records:
        raise RuntimeError("No legacy simulation points were accepted.")

    dataframe = pd.DataFrame(records)

    summary = {
        "completed": (
            center_dry_layer >= initial_cake_height
            and edge_dry_layer >= initial_cake_height
        ),
        "total_time_hr": dataframe["Time (hr)"].iloc[-1],
        "center_drying_percent": dataframe[
            "Center drying (%)"
        ].iloc[-1],
        "edge_drying_percent": dataframe[
            "Edge drying (%)"
        ].iloc[-1],
        "rejected_steps": rejected_steps,
        "solver_failures": 0,
        "mode": "Legacy VS Code match",
    }

    return dataframe, summary


# ============================================================
# TRANSACTIONAL CONSTRAINT MODE
# ============================================================
def simulate_transactional(parameters):
    product_area = parameters["product_area"]
    vial_area = parameters["vial_area"]
    initial_cake_height = parameters["initial_cake_height"]
    dt = parameters["dt"]
    chamber_pressure = parameters["pressure"]

    shelf_temperature = (
        6144.96
        / (24.01849 - math.log(chamber_pressure))
        - 273.15
    )

    time_minutes = 0.0
    center_dry_layer = 0.0
    edge_dry_layer = 0.0

    center_interface_guess = parameters[
        "initial_interface_temperature"
    ]
    edge_interface_guess = parameters[
        "initial_interface_temperature"
    ]

    center_interface_history = []
    center_product_history = []
    edge_interface_history = []
    edge_product_history = []

    records = []
    rejected_steps = 0
    solver_failures = 0

    maximum_steps = int(
        parameters["max_duration_hr"] * 60.0 / dt
    )

    for _ in range(maximum_steps):
        center_previous_product = (
            center_product_history[-1]
            if center_product_history
            else center_interface_guess
        )
        edge_previous_product = (
            edge_product_history[-1]
            if edge_product_history
            else edge_interface_guess
        )

        try:
            if center_dry_layer < initial_cake_height:
                (
                    center_interface,
                    center_product,
                    center_rate,
                ) = solve_transactional_position(
                    center_dry_layer,
                    shelf_temperature,
                    chamber_pressure,
                    center_previous_product,
                    center_interface_guess,
                    dt,
                    kv_center,
                    parameters,
                )

                center_increment = (
                    center_rate
                    * dt
                    / (
                        parameters["ice_density"]
                        * product_area
                        * (1.0 - parameters["solid_fraction"])
                    )
                )

                proposed_center_dry_layer = min(
                    initial_cake_height,
                    center_dry_layer + center_increment,
                )
            else:
                center_thermal_mass = (
                    parameters["ice_density"]
                    * product_area
                    * initial_cake_height
                    * parameters["solid_fraction"]
                )

                center_product = center_previous_product
                center_product += (
                    kv_center(chamber_pressure)
                    * vial_area
                    * (shelf_temperature - center_product)
                    * dt
                    / (
                        center_thermal_mass
                        * parameters["heat_capacity"]
                    )
                )

                center_interface = center_interface_history[-1]
                center_rate = 0.0
                proposed_center_dry_layer = center_dry_layer

            if edge_dry_layer < initial_cake_height:
                (
                    edge_interface,
                    edge_product,
                    edge_rate,
                ) = solve_transactional_position(
                    edge_dry_layer,
                    shelf_temperature,
                    chamber_pressure,
                    edge_previous_product,
                    edge_interface_guess,
                    dt,
                    kv_edge,
                    parameters,
                )

                edge_increment = (
                    edge_rate
                    * dt
                    / (
                        parameters["ice_density"]
                        * product_area
                        * (1.0 - parameters["solid_fraction"])
                    )
                )

                proposed_edge_dry_layer = min(
                    initial_cake_height,
                    edge_dry_layer + edge_increment,
                )
            else:
                edge_thermal_mass = (
                    parameters["ice_density"]
                    * product_area
                    * initial_cake_height
                    * parameters["solid_fraction"]
                )

                edge_product = edge_previous_product
                edge_product += (
                    kv_edge(chamber_pressure)
                    * vial_area
                    * (shelf_temperature - edge_product)
                    * dt
                    / (
                        edge_thermal_mass
                        * parameters["heat_capacity"]
                    )
                )

                edge_interface = edge_interface_history[-1]
                edge_rate = 0.0
                proposed_edge_dry_layer = edge_dry_layer

        except RuntimeError:
            solver_failures += 1
            shelf_temperature -= parameters["constraint_step"]

            if (
                solver_failures
                > parameters["max_solver_failures"]
            ):
                raise RuntimeError(
                    "The nonlinear solver repeatedly failed."
                )

            continue

        center_interface = min(
            center_interface,
            shelf_temperature - 1.0,
        )
        center_product = min(
            center_product,
            shelf_temperature - 1.0,
        )
        edge_interface = min(
            edge_interface,
            shelf_temperature - 1.0,
        )
        edge_product = min(
            edge_product,
            shelf_temperature - 1.0,
        )

        active_product_temperatures = []

        if proposed_center_dry_layer < initial_cake_height:
            active_product_temperatures.append(center_product)

        if proposed_edge_dry_layer < initial_cake_height:
            active_product_temperatures.append(edge_product)

        if (
            active_product_temperatures
            and max(active_product_temperatures)
            > parameters["critical_temperature"]
        ):
            shelf_temperature -= parameters["constraint_step"]
            rejected_steps += 1

            if (
                rejected_steps
                > parameters["max_rejected_steps"]
            ):
                raise RuntimeError(
                    "Too many Tcrit rejections."
                )

            continue

        center_dry_layer = proposed_center_dry_layer
        edge_dry_layer = proposed_edge_dry_layer

        shelf_temperature = min(
            shelf_temperature,
            parameters["maximum_shelf_temperature"],
        )

        time_minutes += dt

        center_interface_history.append(center_interface)
        center_product_history.append(center_product)
        edge_interface_history.append(edge_interface)
        edge_product_history.append(edge_product)

        records.append(
            make_record(
                time_minutes,
                shelf_temperature,
                center_interface,
                center_product,
                edge_interface,
                edge_product,
                center_dry_layer,
                edge_dry_layer,
                center_rate,
                edge_rate,
                parameters,
            )
        )

        if shelf_temperature < -10.0:
            shelf_temperature += (
                parameters["low_temperature_ramp"] * dt
            )
        else:
            shelf_temperature += (
                parameters["high_temperature_ramp"] * dt
            )

        center_interface_guess = center_interface
        edge_interface_guess = edge_interface

        drying_finished = (
            center_dry_layer >= initial_cake_height
            and edge_dry_layer >= initial_cake_height
        )

        thermal_equilibrium_reached = (
            len(center_product_history) > 10
            and abs(
                center_product_history[-1]
                - shelf_temperature
            ) < 3.0
            and abs(
                edge_product_history[-1]
                - shelf_temperature
            ) < 2.0
        )

        if drying_finished and thermal_equilibrium_reached:
            break

    if not records:
        raise RuntimeError(
            "No transactional simulation points were accepted."
        )

    dataframe = pd.DataFrame(records)

    summary = {
        "completed": (
            center_dry_layer >= initial_cake_height
            and edge_dry_layer >= initial_cake_height
        ),
        "total_time_hr": dataframe["Time (hr)"].iloc[-1],
        "center_drying_percent": dataframe[
            "Center drying (%)"
        ].iloc[-1],
        "edge_drying_percent": dataframe[
            "Edge drying (%)"
        ].iloc[-1],
        "rejected_steps": rejected_steps,
        "solver_failures": solver_failures,
        "mode": "Transactional constraint",
    }

    return dataframe, summary


def simulate(parameters):
    if parameters["simulation_mode"] == "Legacy VS Code match":
        return simulate_legacy(parameters)

    return simulate_transactional(parameters)


def make_excel(dataframe, parameters, summary):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(
            writer,
            sheet_name="Simulation",
            index=False,
        )

        pd.DataFrame(
            {
                "Parameter": list(parameters.keys()),
                "Value": list(parameters.values()),
            }
        ).to_excel(
            writer,
            sheet_name="Inputs",
            index=False,
        )

        pd.DataFrame(
            {
                "Metric": list(summary.keys()),
                "Value": list(summary.values()),
            }
        ).to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

    return output.getvalue()


# ============================================================
# SIDEBAR INPUTS
# ============================================================
with st.sidebar:
    st.header("Model inputs")

    simulation_mode = st.radio(
        "Simulation mode",
        [
            "Legacy VS Code match",
            "Transactional constraint",
        ],
        help=(
            "Legacy mode reproduces the original update order, "
            "including retained dry-layer advancement during a "
            "rejected Tcrit step."
        ),
    )

    critical_temperature = st.number_input(
        "Critical temperature (°C)",
        value=-10.0,
    )
    ice_density = st.number_input(
        "Ice density (g/cm³)",
        value=1.0,
        min_value=0.0001,
    )
    sublimation_enthalpy = st.number_input(
        "Sublimation enthalpy (cal/g)",
        value=680.0,
    )
    fill = st.number_input(
        "Fill volume (mL = cm³)",
        value=48.0,
        min_value=0.001,
    )
    outer_diameter = st.number_input(
        "Outer diameter (cm)",
        value=4.7,
        min_value=0.001,
    )
    wall_thickness = st.number_input(
        "Wall thickness (cm)",
        value=0.17,
        min_value=0.0,
    )
    pressure = st.number_input(
        "Chamber pressure (Torr)",
        value=0.15,
        min_value=0.0001,
        format="%.4f",
    )
    dt = st.number_input(
        "Time step (min)",
        value=1.0,
        min_value=0.01,
    )
    solid_fraction = st.number_input(
        "Solid fraction C",
        value=0.0625,
        min_value=0.0,
        max_value=0.99,
        format="%.4f",
    )
    heat_capacity = st.number_input(
        "Heat capacity Cp (cal/g/°C)",
        value=134.85,
    )

    with st.expander("Advanced parameters"):
        ice_conductivity = st.number_input(
            "Ice conductivity ki (cal/min/cm/°C)",
            value=0.3000,
            format="%.4f",
        )
        a5 = st.number_input(
            "Rp coefficient a5",
            value=44.59,
        )
        a6 = st.number_input(
            "Rp coefficient a6",
            value=1451.73,
        )
        a7 = st.number_input(
            "Rp coefficient a7",
            value=12.93,
        )
        initial_interface_temperature = st.number_input(
            "Initial interface guess (°C)",
            value=-20.0,
        )
        maximum_shelf_temperature = st.number_input(
            "Maximum shelf temperature (°C)",
            value=30.0,
        )
        constraint_step = st.number_input(
            "Tcrit correction step (°C)",
            value=0.2,
            min_value=0.001,
        )
        low_temperature_ramp = st.number_input(
            "Ramp below -10 °C (°C/min)",
            value=0.2,
        )
        high_temperature_ramp = st.number_input(
            "Ramp at/above -10 °C (°C/min)",
            value=0.1 / 3.0,
            format="%.5f",
        )
        max_duration_hr = st.number_input(
            "Maximum simulated duration (h)",
            value=120.0,
            min_value=0.1,
        )
        max_rejected_steps = st.number_input(
            "Maximum rejected steps",
            value=5000,
            min_value=1,
            step=100,
        )
        max_solver_failures = st.number_input(
            "Maximum solver failures",
            value=200,
            min_value=1,
            step=10,
        )

    run_simulation = st.button(
        "Run simulation",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# GEOMETRY AND PARAMETER DICTIONARY
# ============================================================
inner_diameter = outer_diameter - 2.0 * wall_thickness

if inner_diameter <= 0:
    st.error(
        "Outer diameter must be greater than twice the wall thickness."
    )
    st.stop()

product_area = np.pi * inner_diameter**2 / 4.0
vial_area = np.pi * outer_diameter**2 / 4.0
initial_cake_height = fill / product_area

parameters = {
    "simulation_mode": simulation_mode,
    "critical_temperature": critical_temperature,
    "ice_density": ice_density,
    "sublimation_enthalpy": sublimation_enthalpy,
    "fill": fill,
    "outer_diameter": outer_diameter,
    "wall_thickness": wall_thickness,
    "inner_diameter": inner_diameter,
    "product_area": product_area,
    "vial_area": vial_area,
    "initial_cake_height": initial_cake_height,
    "ice_conductivity": ice_conductivity,
    "pressure": pressure,
    "dt": dt,
    "solid_fraction": solid_fraction,
    "heat_capacity": heat_capacity,
    "a5": a5,
    "a6": a6,
    "a7": a7,
    "initial_interface_temperature": (
        initial_interface_temperature
    ),
    "maximum_shelf_temperature": (
        maximum_shelf_temperature
    ),
    "constraint_step": constraint_step,
    "low_temperature_ramp": low_temperature_ramp,
    "high_temperature_ramp": high_temperature_ramp,
    "max_duration_hr": max_duration_hr,
    "max_rejected_steps": int(max_rejected_steps),
    "max_solver_failures": int(max_solver_failures),
}

geometry_columns = st.columns(4)
geometry_columns[0].metric(
    "Inner diameter",
    f"{inner_diameter:.3f} cm",
)
geometry_columns[1].metric(
    "Product area",
    f"{product_area:.3f} cm²",
)
geometry_columns[2].metric(
    "Vial area",
    f"{vial_area:.3f} cm²",
)
geometry_columns[3].metric(
    "Initial cake height",
    f"{initial_cake_height:.3f} cm",
)

st.caption(
    f"Active mode: **{simulation_mode}** | "
    f"Internal time: minutes | Plot time: hours | "
    f"ki = {ice_conductivity:.4f} cal/(min·cm·°C)"
)


# ============================================================
# RUN AND STORE RESULTS
# ============================================================
if run_simulation:
    with st.spinner("Running transient simulation..."):
        try:
            simulation_data, simulation_summary = simulate(
                parameters
            )

            st.session_state["simulation_data"] = (
                simulation_data
            )
            st.session_state["simulation_summary"] = (
                simulation_summary
            )
            st.session_state["simulation_parameters"] = (
                parameters.copy()
            )
        except Exception as exc:
            st.exception(exc)

if "simulation_data" not in st.session_state:
    st.info(
        "Set the inputs in the sidebar and select **Run simulation**."
    )
    st.stop()


df = st.session_state["simulation_data"]
summary = st.session_state["simulation_summary"]
used_parameters = st.session_state["simulation_parameters"]


# ============================================================
# SUMMARY
# ============================================================
st.subheader("Simulation summary")

summary_columns = st.columns(6)
summary_columns[0].metric(
    "Mode",
    summary["mode"],
)
summary_columns[1].metric(
    "Cycle time",
    f"{summary['total_time_hr']:.2f} h",
)
summary_columns[2].metric(
    "Center drying",
    f"{summary['center_drying_percent']:.2f}%",
)
summary_columns[3].metric(
    "Edge drying",
    f"{summary['edge_drying_percent']:.2f}%",
)
summary_columns[4].metric(
    "Rejected steps",
    summary["rejected_steps"],
)
summary_columns[5].metric(
    "Solver failures",
    summary["solver_failures"],
)

if summary["completed"]:
    st.success(
        "Center and edge vial drying reached the model endpoint."
    )
else:
    st.warning(
        "Maximum duration was reached before both positions completed drying."
    )


# ============================================================
# PLOTS, AUDIT, AND DOWNLOADS
# ============================================================
(
    tab_dashboard,
    tab_temperature,
    tab_cake,
    tab_drying,
    tab_rate,
    tab_energy,
    tab_audit,
    tab_data,
) = st.tabs(
    [
        "Combined dashboard",
        "Temperature",
        "Cake height",
        "Drying %",
        "Sublimation",
        "Energy balance",
        "Unit audit",
        "Data",
    ]
)

plot_time = df["Time (hr)"]

with tab_dashboard:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(
        plot_time,
        df["Shelf temperature (°C)"],
        "--",
        label="Shelf",
    )
    axes[0, 0].plot(
        plot_time,
        df["Center product temperature (°C)"],
        label="Center product",
    )
    axes[0, 0].plot(
        plot_time,
        df["Edge product temperature (°C)"],
        "--",
        label="Edge product",
    )
    axes[0, 0].axhline(
        used_parameters["critical_temperature"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Tcrit",
    )
    axes[0, 0].set_title("Temperature")
    axes[0, 0].set_ylabel("Temperature (°C)")
    axes[0, 0].legend()

    axes[0, 1].plot(
        plot_time,
        df["Center remaining cake (cm)"],
        label="Center",
    )
    axes[0, 1].plot(
        plot_time,
        df["Edge remaining cake (cm)"],
        "--",
        label="Edge",
    )
    axes[0, 1].set_title("Remaining Frozen Cake Height")
    axes[0, 1].set_ylabel("Height (cm)")
    axes[0, 1].legend()

    axes[1, 0].plot(
        plot_time,
        df["Center drying (%)"],
        label="Center",
    )
    axes[1, 0].plot(
        plot_time,
        df["Edge drying (%)"],
        "--",
        label="Edge",
    )
    axes[1, 0].set_title("Drying Progression")
    axes[1, 0].set_ylabel("Drying (%)")
    axes[1, 0].legend()

    axes[1, 1].plot(
        plot_time,
        df["Center sublimation rate (g/min)"],
        label="Center",
    )
    axes[1, 1].plot(
        plot_time,
        df["Edge sublimation rate (g/min)"],
        "--",
        label="Edge",
    )
    axes[1, 1].set_title("Sublimation Rate")
    axes[1, 1].set_ylabel("Sublimation rate (g/min)")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Time (h)")
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_temperature:
    fig, axis = plt.subplots(figsize=(12, 6))

    axis.plot(
        plot_time,
        df["Shelf temperature (°C)"],
        "--",
        label="Shelf",
    )
    axis.plot(
        plot_time,
        df["Center product temperature (°C)"],
        label="Center product",
    )
    axis.plot(
        plot_time,
        df["Edge product temperature (°C)"],
        "--",
        label="Edge product",
    )
    axis.plot(
        plot_time,
        df["Center interface temperature (°C)"],
        label="Center interface",
    )
    axis.plot(
        plot_time,
        df["Edge interface temperature (°C)"],
        "--",
        label="Edge interface",
    )
    axis.axhline(
        used_parameters["critical_temperature"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Tcrit",
    )
    axis.set_xlabel("Time (h)")
    axis.set_ylabel("Temperature (°C)")
    axis.set_title("Temperature Profiles")
    axis.grid(True, alpha=0.3)
    axis.legend()

    st.pyplot(fig)
    plt.close(fig)

with tab_cake:
    fig, axis = plt.subplots(figsize=(12, 6))

    axis.plot(
        plot_time,
        df["Center remaining cake (cm)"],
        label="Center remaining cake",
    )
    axis.plot(
        plot_time,
        df["Edge remaining cake (cm)"],
        "--",
        label="Edge remaining cake",
    )
    axis.set_xlabel("Time (h)")
    axis.set_ylabel("Remaining frozen cake height (cm)")
    axis.set_title("Remaining Frozen Cake Height")
    axis.grid(True, alpha=0.3)
    axis.legend()

    st.pyplot(fig)
    plt.close(fig)

with tab_drying:
    fig, axis = plt.subplots(figsize=(12, 6))

    axis.plot(
        plot_time,
        df["Center drying (%)"],
        label="Center",
    )
    axis.plot(
        plot_time,
        df["Edge drying (%)"],
        "--",
        label="Edge",
    )
    axis.axhline(
        100.0,
        color="green",
        linestyle="--",
        label="Drying endpoint",
    )
    axis.set_xlabel("Time (h)")
    axis.set_ylabel("Drying (%)")
    axis.set_title("Drying Percentage")
    axis.grid(True, alpha=0.3)
    axis.legend()

    st.pyplot(fig)
    plt.close(fig)

with tab_rate:
    fig, axis = plt.subplots(figsize=(12, 6))

    axis.plot(
        plot_time,
        df["Center sublimation rate (g/min)"],
        label="Center",
    )
    axis.plot(
        plot_time,
        df["Edge sublimation rate (g/min)"],
        "--",
        label="Edge",
    )
    axis.set_xlabel("Time (h)")
    axis.set_ylabel("Sublimation rate (g/min)")
    axis.set_title("Sublimation Rate")
    axis.grid(True, alpha=0.3)
    axis.legend()

    st.pyplot(fig)
    plt.close(fig)

with tab_energy:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        plot_time,
        df["Center heat input (cal/min)"],
        label="Heat input",
    )
    axes[0].plot(
        plot_time,
        df["Center conduction (cal/min)"],
        "--",
        label="Conduction",
    )
    axes[0].set_title("Center Vial Energy Terms")
    axes[0].set_xlabel("Time (h)")
    axes[0].set_ylabel("Energy rate (cal/min)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        plot_time,
        df["Edge heat input (cal/min)"],
        label="Heat input",
    )
    axes[1].plot(
        plot_time,
        df["Edge conduction (cal/min)"],
        "--",
        label="Conduction",
    )
    axes[1].set_title("Edge Vial Energy Terms")
    axes[1].set_xlabel("Time (h)")
    axes[1].set_ylabel("Energy rate (cal/min)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_audit:
    st.subheader("Unit-consistency audit")
    st.caption(
        "Expected basis: minutes, centimetres, grams, calories, °C, and Torr."
    )

    audit_rows = [
        {
            "Quantity": "Time step, dt",
            "Current value": used_parameters["dt"],
            "Expected unit": "min",
            "Dimensional check": (
                "Used directly in accumulation and dry-layer equations"
            ),
        },
        {
            "Quantity": "Ice density, rho",
            "Current value": used_parameters["ice_density"],
            "Expected unit": "g/cm³",
            "Dimensional check": "rho × area × height = g",
        },
        {
            "Quantity": "Sublimation enthalpy",
            "Current value": used_parameters[
                "sublimation_enthalpy"
            ],
            "Expected unit": "cal/g",
            "Dimensional check": "m_dot × DeltaH = cal/min",
        },
        {
            "Quantity": "Ice conductivity, ki",
            "Current value": used_parameters[
                "ice_conductivity"
            ],
            "Expected unit": "cal/(min·cm·°C)",
            "Dimensional check": (
                "ki × area × DeltaT / length = cal/min"
            ),
        },
        {
            "Quantity": "Heat capacity, Cp",
            "Current value": used_parameters["heat_capacity"],
            "Expected unit": "cal/(g·°C)",
            "Dimensional check": (
                "M × Cp × DeltaT / dt = cal/min"
            ),
        },
        {
            "Quantity": "Chamber pressure",
            "Current value": used_parameters["pressure"],
            "Expected unit": "Torr",
            "Dimensional check": (
                "Must match vapor-pressure and Rp pressure basis"
            ),
        },
        {
            "Quantity": "Kv center",
            "Current value": kv_center(
                used_parameters["pressure"]
            ),
            "Expected unit": "cal/(min·cm²·°C)",
            "Dimensional check": (
                "Kv × area × DeltaT = cal/min"
            ),
        },
        {
            "Quantity": "Kv edge",
            "Current value": kv_edge(
                used_parameters["pressure"]
            ),
            "Expected unit": "cal/(min·cm²·°C)",
            "Dimensional check": (
                "Kv × area × DeltaT = cal/min"
            ),
        },
        {
            "Quantity": "Rp at 50% cake",
            "Current value": product_resistance(
                0.5 * used_parameters["initial_cake_height"],
                used_parameters["a5"],
                used_parameters["a6"],
                used_parameters["a7"],
            ),
            "Expected unit": "cm²·Torr·min/g",
            "Dimensional check": (
                "area × pressure difference / Rp = g/min"
            ),
        },
    ]

    st.dataframe(
        pd.DataFrame(audit_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Equation audit")

    equation_audit = pd.DataFrame(
        [
            {
                "Equation": "m_dot = Ap(Pice-Pc)/Rp",
                "Result unit": "g/min",
                "Status": "Consistent if Rp basis matches",
            },
            {
                "Equation": "heat_in = Kv Av (Ts-Tp)",
                "Result unit": "cal/min",
                "Status": "Consistent",
            },
            {
                "Equation": "heat_sub = m_dot DeltaH",
                "Result unit": "cal/min",
                "Status": "Consistent",
            },
            {
                "Equation": "heat_acc = M Cp DeltaT/dt",
                "Result unit": "cal/min",
                "Status": "Consistent when dt is minutes",
            },
            {
                "Equation": "cond = ki Av DeltaT/L",
                "Result unit": "cal/min",
                "Status": "Consistent",
            },
            {
                "Equation": "dL = m_dot dt/[rho Ap(1-C)]",
                "Result unit": "cm",
                "Status": "Consistent",
            },
        ]
    )

    st.dataframe(
        equation_audit,
        use_container_width=True,
        hide_index=True,
    )

    if used_parameters["heat_capacity"] > 10.0:
        st.warning(
            f"Cp = {used_parameters['heat_capacity']:.5g} "
            "cal/(g·°C) is unusually large for a mass-specific "
            "heat capacity. Verify whether 134.85 has a molar, "
            "SI, or another basis."
        )

    if abs(used_parameters["ice_conductivity"] - 0.3) > 0.01:
        st.warning(
            "The revised VS Code model uses ki = 0.3000 "
            "cal/(min·cm·°C)."
        )

    if used_parameters["simulation_mode"] == "Legacy VS Code match":
        st.warning(
            "Legacy mode intentionally retains dry-layer advancement "
            "during rejected Tcrit iterations and accepts the raw "
            "fsolve return value. This reproduces the original code "
            "but is not a transactional numerical controller."
        )

    st.markdown("#### Time basis")
    st.write(
        "The model uses minutes internally. The app stores both "
        "Time (min) and Time (hr). All plots use Time (hr), matching "
        "the original `time.append(t / 60)` logic."
    )

with tab_data:
    st.dataframe(
        df,
        use_container_width=True,
        height=500,
    )

    csv_data = df.to_csv(index=False).encode("utf-8")
    excel_data = make_excel(
        df,
        used_parameters,
        summary,
    )

    download_columns = st.columns(2)

    download_columns[0].download_button(
        "Download CSV",
        data=csv_data,
        file_name="primary_drying_simulation.csv",
        mime="text/csv",
        use_container_width=True,
    )

    download_columns[1].download_button(
        "Download Excel",
        data=excel_data,
        file_name="primary_drying_simulation.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True,
    )

st.divider()
st.caption(
    "Important: matching the legacy graph confirms behavioral "
    "consistency with the original script. It does not by itself "
    "validate the physical unit basis, especially Cp and Rp."
)
