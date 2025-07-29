import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math

# Page configuration
st.set_page_config(
    page_title="Chemical Engineering Toolkit",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .subheader {
        color: #ff7f0e;
        font-size: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .calculation-result {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">⚗️ Chemical Engineering Process Toolkit</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose Application",
    ["Heat Exchanger Design", "Distillation Column", "Reactor Design", "Fluid Properties", "Process Optimization"]
)

# Heat Exchanger Design Application
if app_mode == "Heat Exchanger Design":
    st.markdown('<h2 class="subheader">🔥 Heat Exchanger Design Calculator</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        
        # Hot fluid properties
        st.write("**Hot Fluid Properties:**")
        T_h_in = st.number_input("Hot fluid inlet temperature (°C)", value=150.0, min_value=0.0)
        T_h_out = st.number_input("Hot fluid outlet temperature (°C)", value=80.0, min_value=0.0)
        m_h = st.number_input("Hot fluid mass flow rate (kg/s)", value=2.0, min_value=0.1)
        cp_h = st.number_input("Hot fluid specific heat (kJ/kg·K)", value=4.18, min_value=0.1)
        
        # Cold fluid properties
        st.write("**Cold Fluid Properties:**")
        T_c_in = st.number_input("Cold fluid inlet temperature (°C)", value=20.0, min_value=0.0)
        T_c_out = st.number_input("Cold fluid outlet temperature (°C)", value=60.0, min_value=0.0)
        m_c = st.number_input("Cold fluid mass flow rate (kg/s)", value=1.5, min_value=0.1)
        cp_c = st.number_input("Cold fluid specific heat (kJ/kg·K)", value=4.18, min_value=0.1)
        
        # Heat exchanger properties
        st.write("**Heat Exchanger Properties:**")
        U = st.number_input("Overall heat transfer coefficient (W/m²·K)", value=500.0, min_value=1.0)
        exchanger_type = st.selectbox("Heat Exchanger Type", ["Counter-current", "Co-current"])
    
    with col2:
        st.subheader("Calculations")
        
        # Heat transfer calculations
        Q_h = m_h * cp_h * 1000 * (T_h_in - T_h_out)  # Convert kJ to W
        Q_c = m_c * cp_c * 1000 * (T_c_out - T_c_in)
        Q_avg = (Q_h + Q_c) / 2
        
        # Temperature differences
        if exchanger_type == "Counter-current":
            LMTD = ((T_h_in - T_c_out) - (T_h_out - T_c_in)) / np.log((T_h_in - T_c_out) / (T_h_out - T_c_in))
        else:  # Co-current
            LMTD = ((T_h_in - T_c_in) - (T_h_out - T_c_out)) / np.log((T_h_in - T_c_in) / (T_h_out - T_c_out))
        
        # Heat exchanger area
        A = Q_avg / (U * LMTD)
        
        # Effectiveness
        C_h = m_h * cp_h * 1000
        C_c = m_c * cp_c * 1000
        C_min = min(C_h, C_c)
        effectiveness = Q_avg / (C_min * (T_h_in - T_c_in))
        
        # Display results
        st.markdown(f"""
        <div class="calculation-result">
        <strong>Results:</strong><br>
        • Heat transfer rate (Hot): {Q_h/1000:.2f} kW<br>
        • Heat transfer rate (Cold): {Q_c/1000:.2f} kW<br>
        • Average heat transfer: {Q_avg/1000:.2f} kW<br>
        • LMTD: {LMTD:.2f} °C<br>
        • Required area: {A:.2f} m²<br>
        • Effectiveness: {effectiveness:.3f}
        </div>
        """, unsafe_allow_html=True)
        
        # Temperature profile plot
        fig = go.Figure()
        
        if exchanger_type == "Counter-current":
            x = [0, 1]
            hot_temps = [T_h_in, T_h_out]
            cold_temps = [T_c_out, T_c_in]
        else:
            x = [0, 1]
            hot_temps = [T_h_in, T_h_out]
            cold_temps = [T_c_in, T_c_out]
        
        fig.add_trace(go.Scatter(x=x, y=hot_temps, mode='lines+markers', 
                                name='Hot Fluid', line=dict(color='red', width=3)))
        fig.add_trace(go.Scatter(x=x, y=cold_temps, mode='lines+markers', 
                                name='Cold Fluid', line=dict(color='blue', width=3)))
        
        fig.update_layout(
            title="Temperature Profile",
            xaxis_title="Normalized Length",
            yaxis_title="Temperature (°C)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Distillation Column Application
elif app_mode == "Distillation Column":
    st.markdown('<h2 class="subheader">🏗️ Distillation Column Design</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        
        # Feed properties
        F = st.number_input("Feed flow rate (kmol/h)", value=100.0, min_value=1.0)
        z_F = st.slider("Feed composition (mole fraction)", 0.0, 1.0, 0.5, 0.01)
        x_D = st.slider("Distillate composition (mole fraction)", 0.0, 1.0, 0.9, 0.01)
        x_B = st.slider("Bottoms composition (mole fraction)", 0.0, 1.0, 0.1, 0.01)
        
        # Operating conditions
        R = st.number_input("Reflux ratio", value=2.0, min_value=0.1)
        alpha = st.number_input("Relative volatility", value=2.5, min_value=1.0)
        
        # Column properties
        P = st.number_input("Operating pressure (bar)", value=1.0, min_value=0.1)
        tray_efficiency = st.slider("Tray efficiency", 0.1, 1.0, 0.7, 0.05)
    
    with col2:
        st.subheader("McCabe-Thiele Analysis")
        
        # Material balance
        D = F * (z_F - x_B) / (x_D - x_B)
        B = F - D
        
        # Operating lines
        x_vals = np.linspace(0, 1, 100)
        
        # Equilibrium curve (using relative volatility)
        y_eq = (alpha * x_vals) / (1 + (alpha - 1) * x_vals)
        
        # Rectifying section operating line
        y_rect = (R / (R + 1)) * x_vals + x_D / (R + 1)
        
        # Stripping section operating line
        y_strip = ((B * x_B) / F - z_F) / ((B / F) - 1) * x_vals + z_F / ((B / F) - 1)
        
        # Create McCabe-Thiele diagram
        fig = go.Figure()
        
        # Equilibrium curve
        fig.add_trace(go.Scatter(x=x_vals, y=y_eq, mode='lines', 
                                name='Equilibrium', line=dict(color='green', width=2)))
        
        # Operating lines
        fig.add_trace(go.Scatter(x=x_vals, y=y_rect, mode='lines', 
                                name='Rectifying', line=dict(color='red', width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=x_vals, y=y_strip, mode='lines', 
                                name='Stripping', line=dict(color='blue', width=2, dash='dash')))
        
        # Diagonal line
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                name='y = x', line=dict(color='black', width=1, dash='dot')))
        
        fig.update_layout(
            title="McCabe-Thiele Diagram",
            xaxis_title="x (liquid mole fraction)",
            yaxis_title="y (vapor mole fraction)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate theoretical stages (simplified)
        N_theoretical = math.log((x_D / (1 - x_D)) * ((1 - x_B) / x_B)) / math.log(alpha)
        N_actual = N_theoretical / tray_efficiency
        
        st.markdown(f"""
        <div class="calculation-result">
        <strong>Results:</strong><br>
        • Distillate flow rate: {D:.2f} kmol/h<br>
        • Bottoms flow rate: {B:.2f} kmol/h<br>
        • Theoretical stages: {N_theoretical:.1f}<br>
        • Actual stages: {N_actual:.1f}<br>
        • Minimum reflux ratio: {alpha * z_F / (x_D * (alpha - 1)):.2f}
        </div>
        """, unsafe_allow_html=True)

# Reactor Design Application
elif app_mode == "Reactor Design":
    st.markdown('<h2 class="subheader">⚛️ Chemical Reactor Design</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Reaction Parameters")
        
        reactor_type = st.selectbox("Reactor Type", ["CSTR", "PFR", "Batch"])
        
        # Kinetic parameters
        k = st.number_input("Rate constant (1/s)", value=0.1, min_value=0.001, format="%.4f")
        n = st.number_input("Reaction order", value=1.0, min_value=0.1, max_value=3.0)
        C0 = st.number_input("Initial concentration (mol/L)", value=1.0, min_value=0.1)
        
        if reactor_type != "Batch":
            Q = st.number_input("Volumetric flow rate (L/s)", value=1.0, min_value=0.1)
            V = st.number_input("Reactor volume (L)", value=10.0, min_value=1.0)
            tau = V / Q  # Residence time
            st.write(f"**Residence time: {tau:.2f} s**")
        else:
            t_batch = st.number_input("Batch time (s)", value=60.0, min_value=1.0)
    
    with col2:
        st.subheader("Reactor Performance")
        
        if reactor_type == "CSTR":
            # CSTR design equation: tau = (C0 - C) / (k * C^n)
            # Solving for C: C = C0 / (1 + k * tau)^(1/n) for n=1
            if n == 1.0:
                C_out = C0 / (1 + k * tau)
            else:
                # Numerical solution for other orders
                from scipy.optimize import fsolve
                def cstr_eq(C):
                    return tau - (C0 - C) / (k * C**n)
                C_out = fsolve(cstr_eq, C0/2)[0]
            
            conversion = (C0 - C_out) / C0
            
        elif reactor_type == "PFR":
            # PFR design equation integration
            if n == 1.0:
                conversion = 1 - np.exp(-k * tau)
                C_out = C0 * (1 - conversion)
            else:
                # For other orders: integration of dC/dt = -k*C^n
                time_vals = np.linspace(0, tau, 100)
                conversion = 1 - (1 + (n-1) * k * C0**(n-1) * tau)**(-1/(n-1))
                C_out = C0 * (1 - conversion)
        
        else:  # Batch reactor
            if n == 1.0:
                conversion = 1 - np.exp(-k * t_batch)
                C_out = C0 * (1 - conversion)
            else:
                conversion = 1 - (1 + (n-1) * k * C0**(n-1) * t_batch)**(-1/(n-1))
                C_out = C0 * (1 - conversion)
        
        # Create concentration profile
        if reactor_type == "Batch":
            time_vals = np.linspace(0, t_batch, 100)
            if n == 1.0:
                C_vals = C0 * np.exp(-k * time_vals)
            else:
                C_vals = C0 * (1 + (n-1) * k * C0**(n-1) * time_vals)**(-1/(n-1))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_vals, y=C_vals, mode='lines', 
                                    name='Concentration', line=dict(color='blue', width=3)))
            fig.update_layout(
                title="Batch Reactor Concentration Profile",
                xaxis_title="Time (s)",
                yaxis_title="Concentration (mol/L)",
                height=400
            )
        
        else:  # CSTR or PFR
            # Show concentration vs residence time
            tau_vals = np.linspace(0, tau*2, 100)
            if reactor_type == "CSTR":
                if n == 1.0:
                    C_vals = C0 / (1 + k * tau_vals)
                else:
                    C_vals = []
                    for t in tau_vals:
                        def cstr_eq(C):
                            return t - (C0 - C) / (k * C**n) if C > 0 else float('inf')
                        try:
                            C_val = fsolve(cstr_eq, C0/2)[0]
                            C_vals.append(max(0, C_val))
                        except:
                            C_vals.append(0)
                    C_vals = np.array(C_vals)
            else:  # PFR
                if n == 1.0:
                    C_vals = C0 * np.exp(-k * tau_vals)
                else:
                    C_vals = C0 * (1 + (n-1) * k * C0**(n-1) * tau_vals)**(-1/(n-1))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tau_vals, y=C_vals, mode='lines', 
                                    name='Concentration', line=dict(color='blue', width=3)))
            fig.add_vline(x=tau, line_dash="dash", line_color="red", 
                         annotation_text=f"Operating Point (τ={tau:.1f}s)")
            fig.update_layout(
                title=f"{reactor_type} Concentration Profile",
                xaxis_title="Residence Time (s)",
                yaxis_title="Concentration (mol/L)",
                height=400
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        <div class="calculation-result">
        <strong>Results:</strong><br>
        • Final concentration: {C_out:.3f} mol/L<br>
        • Conversion: {conversion:.3f} ({conversion*100:.1f}%)<br>
        • Reaction rate: {k * C_out**n:.4f} mol/L·s
        </div>
        """, unsafe_allow_html=True)

# Fluid Properties Application
elif app_mode == "Fluid Properties":
    st.markdown('<h2 class="subheader">💧 Fluid Properties Calculator</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        
        fluid_type = st.selectbox("Fluid Type", ["Water", "Air", "Custom"])
        
        T = st.number_input("Temperature (°C)", value=25.0, min_value=-50.0, max_value=200.0)
        P = st.number_input("Pressure (bar)", value=1.0, min_value=0.1, max_value=100.0)
        
        if fluid_type == "Custom":
            MW = st.number_input("Molecular weight (g/mol)", value=18.0, min_value=1.0)
            critical_T = st.number_input("Critical temperature (°C)", value=374.0)
            critical_P = st.number_input("Critical pressure (bar)", value=221.0)
    
    with col2:
        st.subheader("Calculated Properties")
        
        # Temperature conversions
        T_K = T + 273.15
        T_R = T_K * 9/5  # Rankine
        
        if fluid_type == "Water":
            # Water properties (simplified correlations)
            # Density (kg/m³)
            rho = 1000 * (1 - 0.0002 * (T - 4))  # Simplified
            
            # Viscosity (Pa·s)
            mu = 0.001 * np.exp(1.3272 * (293.15 - T_K) / (T_K - 168.15))
            
            # Surface tension (N/m)
            sigma = 0.0728 * (1 - T_K/647.27)**1.256
            
            # Vapor pressure (bar)
            P_vap = np.exp(16.54 - 3985/T_K) / 100000  # Antoine equation
            
        elif fluid_type == "Air":
            # Air properties
            # Density (kg/m³) - ideal gas
            R = 287  # J/kg·K for air
            rho = P * 100000 / (R * T_K)
            
            # Viscosity (Pa·s) - Sutherland's law
            mu = 1.716e-5 * (T_K/273.15)**1.5 * (383.55/(T_K + 110.4))
            
            # Surface tension
            sigma = 0.0  # Gas has no surface tension
            
            # Vapor pressure
            P_vap = P  # For gas phase
        
        else:  # Custom fluid
            # Basic estimates for custom fluid
            R_gas = 8314 / MW  # J/kg·K
            rho = P * 100000 / (R_gas * T_K)
            mu = 1e-5  # Default estimate
            sigma = 0.02  # Default estimate
            P_vap = P * 0.1  # Default estimate
        
        # Reynolds number calculation helper
        st.write("**Reynolds Number Calculator:**")
        D = st.number_input("Pipe diameter (m)", value=0.1, min_value=0.001)
        v = st.number_input("Velocity (m/s)", value=1.0, min_value=0.1)
        Re = rho * v * D / mu
        
        # Flow regime
        if Re < 2300:
            flow_regime = "Laminar"
        elif Re > 4000:
            flow_regime = "Turbulent"
        else:
            flow_regime = "Transition"
        
        st.markdown(f"""
        <div class="calculation-result">
        <strong>Fluid Properties:</strong><br>
        • Density: {rho:.2f} kg/m³<br>
        • Viscosity: {mu*1000:.3f} mPa·s<br>
        • Surface tension: {sigma*1000:.2f} mN/m<br>
        • Vapor pressure: {P_vap:.4f} bar<br><br>
        <strong>Flow Analysis:</strong><br>
        • Reynolds number: {Re:.0f}<br>
        • Flow regime: {flow_regime}
        </div>
        """, unsafe_allow_html=True)
        
        # Property variation with temperature
        T_range = np.linspace(0, 100, 50)
        if fluid_type == "Water":
            rho_range = 1000 * (1 - 0.0002 * (T_range - 4))
            mu_range = 0.001 * np.exp(1.3272 * (293.15 - (T_range + 273.15)) / ((T_range + 273.15) - 168.15))
        else:
            rho_range = P * 100000 / (287 * (T_range + 273.15))
            mu_range = 1.716e-5 * ((T_range + 273.15)/273.15)**1.5 * (383.55/((T_range + 273.15) + 110.4))
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Density vs Temperature', 'Viscosity vs Temperature'))
        
        fig.add_trace(go.Scatter(x=T_range, y=rho_range, mode='lines', name='Density'),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=T_range, y=mu_range*1000, mode='lines', name='Viscosity'),
                     row=2, col=1)
        
        fig.update_xaxes(title_text="Temperature (°C)", row=2, col=1)
        fig.update_yaxes(title_text="Density (kg/m³)", row=1, col=1)
        fig.update_yaxes(title_text="Viscosity (mPa·s)", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Process Optimization Application
else:  # Process Optimization
    st.markdown('<h2 class="subheader">📊 Process Optimization</h2>', unsafe_allow_html=True)
    
    st.write("**Optimize operating conditions for maximum profit**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Process Parameters")
        
        # Economic parameters
        product_price = st.number_input("Product price ($/kg)", value=10.0, min_value=0.1)
        raw_material_cost = st.number_input("Raw material cost ($/kg)", value=2.0, min_value=0.1)
        utility_cost = st.number_input("Utility cost ($/kWh)", value=0.1, min_value=0.01)
        
        # Process variables
        temperature = st.slider("Operating temperature (°C)", 50, 200, 100)
        pressure = st.slider("Operating pressure (bar)", 1, 10, 5)
        residence_time = st.slider("Residence time (min)", 10, 120, 60)
        
        # Process model parameters
        st.write("**Process Model:**")
        st.write("Conversion = f(T, P, τ)")
        st.write("Energy consumption = g(T, P)")
    
    with col2:
        st.subheader("Optimization Results")
        
        # Simple process model
        # Conversion increases with temperature and pressure, decreases with high residence time
        conversion = 0.5 + 0.3 * (temperature - 50) / 150 + 0.2 * (pressure - 1) / 9 - 0.1 * (residence_time - 60) / 60
        conversion = max(0.1, min(0.95, conversion))  # Bounds
        
        # Energy consumption increases with temperature and pressure
        energy_consumption = 10 + 0.1 * (temperature - 50) + 2 * (pressure - 1)  # kW
        
        # Production rate (kg/h)
        production_rate = 100 * conversion
        
        # Economic calculation
        revenue = production_rate * product_price
        raw_material_costs = production_rate / conversion * raw_material_cost
        utility_costs = energy_consumption * utility_cost
        profit = revenue - raw_material_costs - utility_costs
        
        st.markdown(f"""
        <div class="calculation-result">
        <strong>Process Performance:</strong><br>
        • Conversion: {conversion:.1%}<br>
        • Production rate: {production_rate:.1f} kg/h<br>
        • Energy consumption: {energy_consumption:.1f} kW<br><br>
        <strong>Economics:</strong><br>
        • Revenue: ${revenue:.2f}/h<br>
        • Raw material cost: ${raw_material_costs:.2f}/h<br>
        • Utility cost: ${utility_costs:.2f}/h<br>
        • <strong>Net profit: ${profit:.2f}/h</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Optimization surface
        T_opt = np.linspace(50, 200, 20)
        P_opt = np.linspace(1, 10, 20)
        T_mesh, P_mesh = np.meshgrid(T_opt, P_opt)
        
        # Calculate profit surface
        conversion_surface = 0.5 + 0.3 * (T_mesh - 50) / 150 + 0.2 * (P_mesh - 1) / 9 - 0.1 * (residence_time - 60) / 60
        conversion_surface = np.maximum(0.1, np.minimum(0.95, conversion_surface))
        
        energy_surface = 10 + 0.1 * (T_mesh - 50) + 2 * (P_mesh - 1)
        production_surface = 100 * conversion_surface
        profit_surface = (production_surface * product_price - 
                         production_surface / conversion_surface * raw_material_cost - 
                         energy_surface * utility_cost)
        
        fig = go.Figure(data=go.Surface(z=profit_surface, x=T_mesh, y=P_mesh, 
                                       colorscale='Viridis'))
        
        fig.update_layout(
            title='Profit Optimization Surface',
            scene=dict(
                xaxis_title='Temperature (°C)',
                yaxis_title='Pressure (bar)',
                zaxis_title='Profit ($/h)'
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Find optimal conditions
        max_profit_idx = np.unravel_index(np.argmax(profit_surface), profit_surface.shape)
        optimal_T = T_mesh[max_profit_idx]
        optimal_P = P_mesh[max_profit_idx]
        max_profit = profit_surface[max_profit_idx]
        
        st.success(f"**Optimal conditions:** T = {optimal_T:.0f}°C, P = {optimal_P:.1f} bar, Profit = ${max_profit:.2f}/h")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Chemical Engineering Process Toolkit | Built with Streamlit</p>
    <p>For educational and professional use in chemical process engineering</p>
</div>
""", unsafe_allow_html=True)
