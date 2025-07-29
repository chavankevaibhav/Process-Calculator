%%writefile app.py
import streamlit as st
from main import HeatExchanger, DistillationColumn, FluidMechanics, ChemicalReactor, FluidProperties
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Ensure matplotlib is imported for plotting

st.set_page_config(layout="wide", page_title="Chemical Engineering Toolkit")

st.title("Chemical Engineering Toolkit App")

# General Description
st.markdown("""
This application provides a collection of tools for common chemical engineering calculations and analysis.
Navigate through the sidebar to select a specific tool, input your parameters, and view the results and visualizations.
""")

st.sidebar.title("Navigation")
selected_tool = st.sidebar.radio(
    "Select a tool:",
    ("Heat Exchanger", "Distillation Column", "Fluid Mechanics", "Chemical Reactor")
)

def heat_exchanger_tool():
    st.header("Heat Exchanger Design and Analysis")
    st.markdown("""
    Use this tool to analyze the performance of counter-current or co-current heat exchangers.
    Input the fluid properties and the overall heat transfer coefficient to calculate the heat transfer rate,
    required area, Log Mean Temperature Difference (LMTD), and effectiveness.
    """)

    st.subheader("Hot Fluid Properties")
    with st.container(border=True):
        hot_t_in = st.number_input("Inlet Temperature (°C)", value=150.0, format="%.2f", key="hx_hot_tin",
                                   help="The temperature of the hot fluid entering the heat exchanger.")
        hot_t_out = st.number_input("Outlet Temperature (°C)", value=80.0, format="%.2f", key="hx_hot_tout",
                                    help="The temperature of the hot fluid leaving the heat exchanger.")
        hot_m_dot = st.number_input("Mass Flow Rate (kg/s)", value=2.0, min_value=0.0, format="%.2f", key="hx_hot_mdot",
                                    help="The mass flow rate of the hot fluid.")
        hot_cp = st.number_input("Specific Heat (J/kg·K)", value=4180.0, min_value=0.0, format="%.2f", key="hx_hot_cp",
                                  help="The specific heat capacity of the hot fluid.")

    st.subheader("Cold Fluid Properties")
    with st.container(border=True):
        cold_t_in = st.number_input("Inlet Temperature (°C)", value=20.0, format="%.2f", key="hx_cold_tin",
                                    help="The temperature of the cold fluid entering the heat exchanger.")
        cold_t_out = st.number_input("Outlet Temperature (°C)", value=60.0, format="%.2f", key="hx_cold_tout",
                                     help="The temperature of the cold fluid leaving the heat exchanger.")
        cold_m_dot = st.number_input("Mass Flow Rate (kg/s)", value=1.5, min_value=0.0, format="%.2f", key="hx_cold_mdot",
                                     help="The mass flow rate of the cold fluid.")
        cold_cp = st.number_input("Specific Heat (J/kg·K)", value=4180.0, min_value=0.0, format="%.2f", key="hx_cold_cp",
                                   help="The specific heat capacity of the cold fluid.")

    st.subheader("Heat Exchanger Parameters")
    with st.container(border=True):
        overall_u = st.number_input("Overall Heat Transfer Coefficient (W/m²·K)", value=500.0, min_value=0.0, format="%.2f", key="hx_u",
                                    help="The overall heat transfer coefficient of the exchanger, representing the overall thermal resistance.")
        exchanger_type = st.selectbox("Exchanger Type", ("counter-current", "co-current"), key="hx_type",
                                      help="Select the flow arrangement: counter-current or co-current.")

    if st.button("Calculate Heat Exchanger", key="hx_calculate"):
        hot_fluid_props = {'T_in': hot_t_in, 'T_out': hot_t_out, 'm_dot': hot_m_dot, 'cp': hot_cp}
        cold_fluid_props = {'T_in': cold_t_in, 'T_out': cold_t_out, 'm_dot': cold_m_dot, 'cp': cold_cp}

        try:
            hx = HeatExchanger(hot_fluid_props, cold_fluid_props, overall_u, exchanger_type)
            results = hx.calculate_heat_transfer()

            st.subheader("Results")
            st.markdown("""
            Here are the calculated performance metrics for the heat exchanger:
            """)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Heat Transfer Rate (Avg)", f"{results['Q_avg']/1000:.2f} kW", help="The average heat transferred between the hot and cold fluids.")
                st.metric("Required Area", f"{results['area']:.2f} m²", help="The surface area required for the heat exchanger based on the calculated LMTD and overall heat transfer coefficient.")
            with col2:
                st.metric("LMTD", f"{results['LMTD']:.2f} °C", help="The Log Mean Temperature Difference, a measure of the average temperature difference driving heat transfer.")
                st.metric("Effectiveness", f"{results['effectiveness']:.3f}", help="The effectiveness of the heat exchanger, which is the ratio of actual heat transfer to the maximum possible heat transfer.")

            # Include Temperature Profile Plot
            st.subheader("Temperature Profile")
            st.markdown("This plot shows the temperature change of the hot and cold fluids along the normalized length of the heat exchanger.")
            fig = None # Initialize fig to None
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                x_plot = np.array([0, 1])
                if hx.type == "counter-current":
                    hot_temps = [hx.hot['T_in'], hx.hot['T_out']]
                    cold_temps = [hx.cold['T_out'], hx.cold['T_in']]
                else: # co-current
                    hot_temps = [hx.hot['T_in'], hx.hot['T_out']]
                    cold_temps = [hx.cold['T_in'], hx.cold['T_out']]

                ax.plot(x_plot, hot_temps, 'r-o', linewidth=3, markersize=8, label='Hot Fluid')
                ax.plot(x_plot, cold_temps, 'b-o', linewidth=3, markersize=8, label='Cold Fluid')

                ax.set_xlabel('Normalized Length')
                ax.set_ylabel('Temperature (°C)')
                ax.set_title(f'Temperature Profile - {hx.type.title()} Heat Exchanger')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            except Exception as plot_e:
                st.warning(f"Could not generate plot: {plot_e}")
            finally:
                if fig is not None:
                    plt.close(fig) # Ensure the figure is closed


        except Exception as e:
            st.error(f"An error occurred: {e}")

def distillation_column_tool():
    st.header("Distillation Column Design and Analysis")
    st.markdown("""
    Analyze binary distillation columns using the McCabe-Thiele method.
    Calculate material balance, minimum reflux ratio (R_min), and minimum theoretical stages (N_min).
    A McCabe-Thiele diagram is also generated.
    """)

    st.subheader("Operating Parameters")
    with st.container(border=True):
        feed_flow = st.number_input("Feed Flow Rate (kmol/h)", value=100.0, min_value=0.0, format="%.2f", key="dist_feed_flow",
                                    help="The total molar flow rate of the feed stream.")
        feed_composition = st.slider("Feed Mole Fraction (z_F)", value=0.5, min_value=0.0, max_value=1.0, format="%.3f", key="dist_zF",
                                     help="The mole fraction of the light (more volatile) component in the feed.")
        distillate_composition = st.slider("Distillate Mole Fraction (x_D)", value=0.9, min_value=0.0, max_value=1.0, format="%.3f", key="dist_xD",
                                          help="The desired mole fraction of the light component in the distillate (top product).")
        bottoms_composition = st.slider("Bottoms Mole Fraction (x_B)", value=0.1, min_value=0.0, max_value=1.0, format="%.3f", key="dist_xB",
                                        help="The desired mole fraction of the light component in the bottoms (bottom product).")
        reflux_ratio = st.number_input("Reflux Ratio (R = L/D)", value=2.0, min_value=0.0, format="%.2f", key="dist_reflux",
                                       help="The ratio of liquid returned to the column (reflux, L) to the distillate removed (D).")
        relative_volatility = st.number_input("Relative Volatility (α)", value=2.5, min_value=0.1, format="%.2f", key="dist_alpha",
                                              help="A measure of the separability of the two components. Higher values indicate easier separation.")

    if st.button("Calculate Distillation", key="dist_calculate"):
        try:
            # Input validation for mole fractions
            if not (0 <= feed_composition <= 1 and 0 <= distillate_composition <= 1 and 0 <= bottoms_composition <= 1):
                 st.error("Mole fractions must be between 0 and 1.")
                 return
            if feed_composition <= bottoms_composition or feed_composition >= distillate_composition:
                 st.error("Feed composition must be between bottoms and distillate compositions.")
                 return
            if distillate_composition <= bottoms_composition:
                 st.error("Distillate composition must be greater than bottoms composition.")
                 return
            if reflux_ratio < 0:
                 st.error("Reflux ratio cannot be negative.")
                 return
            if relative_volatility <= 0:
                st.error("Relative volatility must be positive.")
                return


            column = DistillationColumn(
                feed_flow, feed_composition, distillate_composition,
                bottoms_composition, reflux_ratio, relative_volatility
            )

            D, B = column.material_balance()
            R_min = column.minimum_reflux_ratio()
            N_min = column.theoretical_stages()

            st.subheader("Results")
            st.markdown("""
            Here are the calculated material balance and minimum requirements:
            """)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Distillate Flow Rate", f"{D:.2f} kmol/h", help="The calculated molar flow rate of the distillate product.")
                st.metric("Minimum Reflux Ratio (R_min)", f"{R_min:.2f}", help="The theoretical minimum reflux ratio required for separation. Operating below this is impossible.")
            with col2:
                st.metric("Bottoms Flow Rate", f"{B:.2f} kmol/h", help="The calculated molar flow rate of the bottoms product.")
                st.metric("Minimum Theoretical Stages (N_min)", f"{N_min:.1f}", help="The theoretical minimum number of equilibrium stages required at total reflux (infinite reflux ratio).")


            # Include McCabe-Thiele Diagram
            st.subheader("McCabe-Thiele Diagram")
            st.markdown("""
            This diagram graphically represents the vapor-liquid equilibrium and operating lines.
            The equilibrium curve shows the vapor composition in equilibrium with the liquid composition.
            The operating lines represent the mass balance in the rectifying and stripping sections.
            The points represent the feed, distillate, and bottoms compositions.
            """)
            fig = None # Initialize fig to None
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                x = np.linspace(0, 1, 100)
                y_eq = column.equilibrium_curve(x)
                y_rect, y_strip = column.operating_lines(x)

                ax.plot(x, y_eq, 'g-', linewidth=2, label='Equilibrium')
                ax.plot(x, y_rect, 'r--', linewidth=2, label='Rectifying')
                ax.plot(x, y_strip, 'b--', linewidth=2, label='Stripping')
                ax.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='y = x')

                # Add points only if they are within the plot limits [0, 1]
                if 0 <= column.z_F <= 1: ax.plot(column.z_F, column.z_F, 'ko', markersize=8, label=f'Feed ({column.z_F:.2f})')
                if 0 <= column.x_D <= 1: ax.plot(column.x_D, column.x_D, 'ro', markersize=8, label=f'Distillate ({column.x_D:.2f})')
                if 0 <= column.x_B <= 1: ax.plot(column.x_B, column.x_B, 'bo', markersize=8, label=f'Bottoms ({column.x_B:.2f})')

                ax.set_xlabel('x (liquid mole fraction)')
                ax.set_ylabel('y (vapor mole fraction)')
                ax.set_title('McCabe-Thiele Diagram')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                st.pyplot(fig)
            except Exception as plot_e:
                st.warning(f"Could not generate plot: {plot_e}")
            finally:
                if fig is not None:
                    plt.close(fig) # Ensure the figure is closed


        except Exception as e:
            st.error(f"An error occurred: {e}")


def fluid_mechanics_tool():
    st.header("Fluid Mechanics - Pipe Flow Analysis")
    st.markdown("""
    Analyze fluid flow in a pipe. Calculate the Reynolds number, determine the flow regime (laminar, transition, turbulent),
    calculate the friction factor using a simplified Colebrook approximation, and estimate the pressure drop.
    """)

    st.subheader("Fluid Properties")
    with st.container(border=True):
        fluid_density = st.number_input("Fluid Density (kg/m³)", value=1000.0, min_value=0.0, format="%.2f", key="fm_density",
                                        help="The density of the fluid.")
        fluid_viscosity = st.number_input("Fluid Viscosity (Pa·s)", value=0.001, min_value=0.0, format="%.6f", key="fm_viscosity",
                                          help="The dynamic viscosity of the fluid.")

    st.subheader("Pipe Data")
    with st.container(border=True):
        pipe_diameter = st.number_input("Pipe Diameter (m)", value=0.1, min_value=0.001, format="%.3f", key="fm_diameter",
                                        help="The inner diameter of the pipe.")
        pipe_length = st.number_input("Pipe Length (m)", value=100.0, min_value=0.1, format="%.2f", key="fm_length",
                                      help="The length of the pipe section being analyzed.")
        fluid_velocity = st.number_input("Fluid Velocity (m/s)", value=2.0, min_value=0.0, format="%.2f", key="fm_velocity",
                                         help="The average velocity of the fluid in the pipe.")
        pipe_roughness = st.number_input("Pipe Roughness (m)", value=0.00005, min_value=0.0, format="%.6f", key="fm_roughness",
                                         help="The average height of the roughness elements on the inner pipe surface (absolute roughness).")

    if st.button("Analyze Pipe Flow", key="fm_calculate"):
        try:
            # Simple validation
            if fluid_density <= 0 or fluid_viscosity <= 0 or pipe_diameter <= 0 or pipe_length <= 0 or fluid_velocity < 0:
                 st.error("Input values must be positive (velocity can be zero for static fluid, but analysis needs flow).")
                 return

            fluid_props = FluidProperties(
                density=fluid_density,
                viscosity=fluid_viscosity,
                thermal_conductivity=0.6, # Placeholder, not used in current analysis
                specific_heat=4180.0 # Placeholder, not used in current analysis
            )
            pipe_data = {
                'diameter': pipe_diameter,
                'length': pipe_length,
                'velocity': fluid_velocity,
                'roughness': pipe_roughness
            }

            flow_analysis = FluidMechanics.pipe_flow_analysis(fluid_props, pipe_data)

            st.subheader("Results")
            st.markdown("""
            Here are the results of the pipe flow analysis:
            """)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Reynolds Number", f"{flow_analysis['Reynolds_number']:.0f}", help="A dimensionless number indicating the flow regime. Re < 2300 is laminar, Re > 4000 is turbulent.")
                st.metric("Friction Factor (f)", f"{flow_analysis['friction_factor']:.4f}", help="The Darcy friction factor, used to calculate pressure drop due to friction.")
            with col2:
                st.metric("Flow Regime", f"{flow_analysis['flow_regime']}", help="The type of flow based on the Reynolds number.")
                st.metric("Pressure Drop (ΔP)", f"{flow_analysis['pressure_drop']/1000:.2f} kPa", help="The pressure drop across the specified length of the pipe due to friction.")


        except Exception as e:
            st.error(f"An error occurred: {e}")


def chemical_reactor_tool():
    st.header("Chemical Reactor Design and Analysis")
    st.markdown("""
    Analyze the performance of Continuous Stirred-Tank Reactors (CSTR), Plug Flow Reactors (PFR), or Batch reactors
    based on simple reaction kinetics (power law). Calculate outlet concentration, conversion, and reaction rate.
    """)

    st.subheader("Reaction Parameters")
    with st.container(border=True):
        reactor_type = st.selectbox("Reactor Type", ("CSTR", "PFR", "Batch"), key="cr_type",
                                    help="Select the type of reactor to analyze.")
        rate_constant = st.number_input("Rate Constant (k)", value=0.1, min_value=0.0, format="%.4f", key="cr_k",
                                        help="The reaction rate constant. Units depend on the reaction order.")
        reaction_order = st.number_input("Reaction Order (n)", value=1.0, min_value=0.0, format="%.2f", key="cr_n",
                                         help="The order of the reaction with respect to the reactant.")
        initial_concentration = st.number_input("Initial/Inlet Concentration (C₀, mol/L)", value=1.0, min_value=0.01, format="%.2f", key="cr_c0",
                                                help="The initial concentration of the reactant (for Batch) or the inlet concentration (for CSTR/PFR).")

    st.subheader("Operating Time/Residence Time")
    with st.container(border=True):
        if reactor_type in ["CSTR", "PFR"]:
             time_or_tau_label = "Residence Time (τ, s)"
             default_time = 30.0
             help_text = "The average time the fluid spends inside the reactor."
        else: # Batch
             time_or_tau_label = "Reaction Time (t, s)"
             default_time = 30.0
             help_text = "The duration of the reaction in the batch reactor."

        time_or_tau = st.number_input(time_or_tau_label, value=default_time, min_value=0.1, format="%.2f", key="cr_time_tau",
                                      help=help_text)


    if st.button("Calculate Reactor Performance", key="cr_calculate"):
        try:
            # Simple validation
            if rate_constant < 0 or reaction_order < 0 or initial_concentration <= 0 or time_or_tau <= 0:
                 st.error("Input values must be non-negative (initial concentration and time/tau must be positive).")
                 return

            reactor = ChemicalReactor(
                reactor_type, rate_constant, reaction_order, initial_concentration
            )

            if reactor_type == 'CSTR':
                result = reactor.cstr_design(time_or_tau)
                st.subheader("CSTR Results")
            elif reactor_type == 'PFR':
                result = reactor.pfr_design(time_or_tau)
                st.subheader("PFR Results")
            else: # Batch
                result = reactor.batch_design(time_or_tau)
                st.subheader("Batch Reactor Results")

            st.markdown("""
            Here are the calculated performance metrics for the specified reactor and conditions:
            """)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Outlet Concentration", f"{result['C_out']:.4f} mol/L", help="The concentration of the reactant leaving the reactor (or at the end of the batch time).")
                st.metric("Reaction Rate", f"{result['rate']:.4f}", help="The rate of reaction under the final conditions in the reactor.")
            with col2:
                st.metric("Conversion", f"{result['conversion']:.4f}", help="The fraction of the initial/inlet reactant that has been consumed.")


            # Include Concentration Profile Plot
            st.subheader("Concentration Profile")
            st.markdown("""
            This plot shows how the reactant concentration changes with time (for Batch) or residence time/position (for CSTR/PFR).
            """)
            fig = None # Initialize fig to None
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                # Use a range for plotting if time_or_tau is the end point
                plot_time_or_tau = time_or_tau # Simulate up to the given time/tau
                t_vals, C_vals = reactor.concentration_profile(plot_time_or_tau, num_points=100)

                ax.plot(t_vals, C_vals, 'b-', linewidth=3)
                xlabel = 'Time (s)' if reactor_type == 'Batch' else 'Residence Time (s)'
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Concentration (mol/L)')
                ax.set_title(f'{reactor_type} Reactor Concentration Profile')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            except Exception as plot_e:
                st.warning(f"Could not generate plot: {plot_e}")
            finally:
                if fig is not None:
                    plt.close(fig) # Ensure the figure is closed


        except Exception as e:
            st.error(f"An error occurred: {e}")


# --- Main app logic based on selection ---
if selected_tool == "Heat Exchanger":
    heat_exchanger_tool()
elif selected_tool == "Distillation Column":
    distillation_column_tool()
elif selected_tool == "Fluid Mechanics":
    fluid_mechanics_tool()
elif selected_tool == "Chemical Reactor":
    chemical_reactor_tool()

# --- Footer ---
st.markdown("---")
st.markdown("Chemical Engineering Toolkit App | Created using Streamlit and a custom Python library.")
