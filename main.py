"""
Chemical Engineering Toolkit
===========================

A comprehensive Python module for chemical engineering calculations and process design.
Contains classes and functions for heat transfer, mass transfer, reaction engineering,
fluid mechanics, and thermodynamics.

Author: Chemical Engineering Toolkit
Version: 1.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize
from scipy.integrate import odeint
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class FluidProperties:
    """Data class for fluid properties"""
    density: float  # kg/m³
    viscosity: float  # Pa·s
    thermal_conductivity: float  # W/m·K
    specific_heat: float  # J/kg·K
    surface_tension: Optional[float] = None  # N/m
    vapor_pressure: Optional[float] = None  # Pa

class HeatExchanger:
    """Heat exchanger design and analysis class"""
    
    def __init__(self, hot_fluid_props: dict, cold_fluid_props: dict, 
                 U: float, exchanger_type: str = "counter-current"):
        """
        Initialize heat exchanger
        
        Parameters:
        -----------
        hot_fluid_props : dict
            {'T_in': temp_in, 'T_out': temp_out, 'm_dot': mass_flow, 'cp': specific_heat}
        cold_fluid_props : dict
            {'T_in': temp_in, 'T_out': temp_out, 'm_dot': mass_flow, 'cp': specific_heat}
        U : float
            Overall heat transfer coefficient (W/m²·K)
        exchanger_type : str
            'counter-current' or 'co-current'
        """
        self.hot = hot_fluid_props
        self.cold = cold_fluid_props
        self.U = U
        self.type = exchanger_type
        
    def calculate_heat_transfer(self) -> dict:
        """Calculate heat transfer rates and required area"""
        # Heat transfer rates
        Q_hot = self.hot['m_dot'] * self.hot['cp'] * (self.hot['T_in'] - self.hot['T_out'])
        Q_cold = self.cold['m_dot'] * self.cold['cp'] * (self.cold['T_out'] - self.cold['T_in'])
        Q_avg = (Q_hot + Q_cold) / 2
        
        # Temperature differences
        if self.type == "counter-current":
            delta_T1 = self.hot['T_in'] - self.cold['T_out']
            delta_T2 = self.hot['T_out'] - self.cold['T_in']
        else:  # co-current
            delta_T1 = self.hot['T_in'] - self.cold['T_in']
            delta_T2 = self.hot['T_out'] - self.cold['T_out']
        
        # LMTD calculation
        if abs(delta_T1 - delta_T2) < 1e-6:
            LMTD = delta_T1
        else:
            LMTD = (delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)
        
        # Required area
        A = Q_avg / (self.U * LMTD)
        
        # Effectiveness
        C_hot = self.hot['m_dot'] * self.hot['cp']
        C_cold = self.cold['m_dot'] * self.cold['cp']
        C_min = min(C_hot, C_cold)
        effectiveness = Q_avg / (C_min * (self.hot['T_in'] - self.cold['T_in']))
        
        return {
            'Q_hot': Q_hot,
            'Q_cold': Q_cold,
            'Q_avg': Q_avg,
            'LMTD': LMTD,
            'area': A,
            'effectiveness': effectiveness
        }
    
    def plot_temperature_profile(self, save_path: Optional[str] = None):
        """Plot temperature profile along heat exchanger length"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.array([0, 1])
        if self.type == "counter-current":
            hot_temps = [self.hot['T_in'], self.hot['T_out']]
            cold_temps = [self.cold['T_out'], self.cold['T_in']]
        else:
            hot_temps = [self.hot['T_in'], self.hot['T_out']]
            cold_temps = [self.cold['T_in'], self.cold['T_out']]
        
        ax.plot(x, hot_temps, 'r-o', linewidth=3, markersize=8, label='Hot Fluid')
        ax.plot(x, cold_temps, 'b-o', linewidth=3, markersize=8, label='Cold Fluid')
        
        ax.set_xlabel('Normalized Length')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'Temperature Profile - {self.type.title()} Heat Exchanger')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class DistillationColumn:
    """Distillation column design and analysis class"""
    
    def __init__(self, feed_flow: float, feed_composition: float, 
                 distillate_composition: float, bottoms_composition: float,
                 reflux_ratio: float, relative_volatility: float):
        """
        Initialize distillation column
        
        Parameters:
        -----------
        feed_flow : float
            Feed flow rate (kmol/h)
        feed_composition : float
            Feed mole fraction of light component
        distillate_composition : float
            Distillate mole fraction of light component
        bottoms_composition : float
            Bottoms mole fraction of light component
        reflux_ratio : float
            Reflux ratio (L/D)
        relative_volatility : float
            Relative volatility (α)
        """
        self.F = feed_flow
        self.z_F = feed_composition
        self.x_D = distillate_composition
        self.x_B = bottoms_composition
        self.R = reflux_ratio
        self.alpha = relative_volatility
        
    def material_balance(self) -> Tuple[float, float]:
        """Calculate distillate and bottoms flow rates"""
        D = self.F * (self.z_F - self.x_B) / (self.x_D - self.x_B)
        B = self.F - D
        return D, B
    
    def minimum_reflux_ratio(self) -> float:
        """Calculate minimum reflux ratio"""
        y_feed_eq = (self.alpha * self.z_F) / (1 + (self.alpha - 1) * self.z_F)
        R_min = (self.x_D - y_feed_eq) / (y_feed_eq - self.z_F)
        return R_min
    
    def theoretical_stages(self) -> float:
        """Calculate theoretical number of stages using Fenske equation"""
        N_min = np.log((self.x_D / (1 - self.x_D)) * ((1 - self.x_B) / self.x_B)) / np.log(self.alpha)
        return N_min
    
    def operating_lines(self, x_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate rectifying and stripping operating lines"""
        D, B = self.material_balance()
        
        # Rectifying section
        y_rect = (self.R / (self.R + 1)) * x_range + self.x_D / (self.R + 1)
        
        # Stripping section
        L_bar = self.R * D + self.F  # Assuming saturated liquid feed
        y_strip = (L_bar / B) * x_range - ((L_bar / B) - 1) * self.x_B
        
        return y_rect, y_strip
    
    def equilibrium_curve(self, x_range: np.ndarray) -> np.ndarray:
        """Calculate vapor-liquid equilibrium curve"""
        return (self.alpha * x_range) / (1 + (self.alpha - 1) * x_range)
    
    def mccabe_thiele_plot(self, save_path: Optional[str] = None):
        """Generate McCabe-Thiele diagram"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x = np.linspace(0, 1, 100)
        
        # Equilibrium curve
        y_eq = self.equilibrium_curve(x)
        ax.plot(x, y_eq, 'g-', linewidth=2, label='Equilibrium')
        
        # Operating lines
        y_rect, y_strip = self.operating_lines(x)
        ax.plot(x, y_rect, 'r--', linewidth=2, label='Rectifying')
        ax.plot(x, y_strip, 'b--', linewidth=2, label='Stripping')
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='y = x')
        
        # Key points
        ax.plot(self.z_F, self.z_F, 'ko', markersize=8, label='Feed')
        ax.plot(self.x_D, self.x_D, 'ro', markersize=8, label='Distillate')
        ax.plot(self.x_B, self.x_B, 'bo', markersize=8, label='Bottoms')
        
        ax.set_xlabel('x (liquid mole fraction)')
        ax.set_ylabel('y (vapor mole fraction)')
        ax.set_title('McCabe-Thiele Diagram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class ChemicalReactor:
    """Chemical reactor design and analysis class"""
    
    def __init__(self, reactor_type: str, rate_constant: float, 
                 reaction_order: float, initial_concentration: float):
        """
        Initialize chemical reactor
        
        Parameters:
        -----------
        reactor_type : str
            'CSTR', 'PFR', or 'Batch'
        rate_constant : float
            Rate constant (units depend on reaction order)
        reaction_order : float
            Reaction order
        initial_concentration : float
            Initial concentration (mol/L)
        """
        self.type = reactor_type
        self.k = rate_constant
        self.n = reaction_order
        self.C0 = initial_concentration
        
    def cstr_design(self, residence_time: float) -> dict:
        """Design CSTR for given residence time"""
        if self.n == 1.0:
            C_out = self.C0 / (1 + self.k * residence_time)
        else:
            # Numerical solution for non-first order
            def cstr_equation(C):
                if C <= 0:
                    return float('inf')
                return residence_time - (self.C0 - C) / (self.k * C**self.n)
            
            try:
                C_out = fsolve(cstr_equation, self.C0/2)[0]
                C_out = max(0.001, C_out)  # Ensure positive
            except:
                C_out = 0.001
        
        conversion = (self.C0 - C_out) / self.C0
        rate = self.k * C_out**self.n
        
        return {'C_out': C_out, 'conversion': conversion, 'rate': rate}
    
    def pfr_design(self, residence_time: float) -> dict:
        """Design PFR for given residence time"""
        if self.n == 1.0:
            conversion = 1 - np.exp(-self.k * residence_time)
        else:
            # Numerical integration for non-first order
            conversion = 1 - (1 + (self.n - 1) * self.k * self.C0**(self.n-1) * residence_time)**(-1/(self.n-1))
        
        conversion = min(0.999, max(0.001, conversion))  # Bounds
        C_out = self.C0 * (1 - conversion)
        rate = self.k * C_out**self.n
        
        return {'C_out': C_out, 'conversion': conversion, 'rate': rate}
    
    def batch_design(self, time: float) -> dict:
        """Design batch reactor for given time"""
        if self.n == 1.0:
            conversion = 1 - np.exp(-self.k * time)
        else:
            conversion = 1 - (1 + (self.n - 1) * self.k * self.C0**(self.n-1) * time)**(-1/(self.n-1))
        
        conversion = min(0.999, max(0.001, conversion))
        C_out = self.C0 * (1 - conversion)
        rate = self.k * C_out**self.n
        
        return {'C_out': C_out, 'conversion': conversion, 'rate': rate}
    
    def concentration_profile(self, time_or_tau: float, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate concentration profile over time/residence time"""
        t_vals = np.linspace(0, time_or_tau, num_points)
        C_vals = np.zeros(num_points)
        
        for i, t in enumerate(t_vals):
            if self.type == 'CSTR':
                result = self.cstr_design(t)
            elif self.type == 'PFR':
                result = self.pfr_design(t)
            else:  # Batch
                result = self.batch_design(t)
            
            C_vals[i] = result['C_out']
        
        return t_vals, C_vals
    
    def plot_concentration_profile(self, time_or_tau: float, save_path: Optional[str] = None):
        """Plot concentration profile"""
        t_vals, C_vals = self.concentration_profile(time_or_tau)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_vals, C_vals, 'b-', linewidth=3)
        
        xlabel = 'Time (s)' if self.type == 'Batch' else 'Residence Time (s)'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Concentration (mol/L)')
        ax.set_title(f'{self.type} Reactor Concentration Profile')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class FluidMechanics:
    """Fluid mechanics calculations and pipe flow analysis"""
    
    @staticmethod
    def reynolds_number(density: float, velocity: float, diameter: float, viscosity: float) -> float:
        """Calculate Reynolds number"""
        return density * velocity * diameter / viscosity
    
    @staticmethod
    def friction_factor(Re: float, roughness: float = 0.0, diameter: float = 1.0) -> float:
        """Calculate friction factor using Colebrook equation"""
        if Re < 2300:
            # Laminar flow
            return 64 / Re
        else:
            # Turbulent flow - simplified approximation
            relative_roughness = roughness / diameter
            f = 0.25 / (np.log10(relative_roughness/3.7 + 5.74/Re**0.9))**2
            return f
    
    @staticmethod
    def pressure_drop(f: float, length: float, diameter: float, 
                     density: float, velocity: float) -> float:
        """Calculate pressure drop in pipe"""
        return f * (length / diameter) * (density * velocity**2) / 2
    
    @classmethod
    def pipe_flow_analysis(cls, fluid_props: FluidProperties, pipe_data: dict) -> dict:
        """Complete pipe flow analysis"""
        Re = cls.reynolds_number(fluid_props.density, pipe_data['velocity'], 
                                pipe_data['diameter'], fluid_props.viscosity)
        
        f = cls.friction_factor(Re, pipe_data.get('roughness', 0.0), pipe_data['diameter'])
        
        dp = cls.pressure_drop(f, pipe_data['length'], pipe_data['diameter'],
                              fluid_props.density, pipe_data['velocity'])
        
        # Flow regime
        if Re < 2300:
            flow_regime = "Laminar"
        elif Re > 4000:
            flow_regime = "Turbulent"
        else:
            flow_regime = "Transition"
        
        return {
            'Reynolds_number': Re,
            'friction_factor': f,
            'pressure_drop': dp,
            'flow_regime': flow_regime
        }

class Thermodynamics:
    """Thermodynamic property calculations and equations of state"""
    
    @staticmethod
    def antoine_equation(A: float, B: float, C: float, T: float) -> float:
        """Calculate vapor pressure using Antoine equation"""
        return 10**(A - B/(C + T))
    
    @staticmethod
    def ideal_gas_density(P: float, T: float, MW: float) -> float:
        """Calculate density using ideal gas law"""
        R = 8.314  # J/mol·K
        return (P * MW) / (R * T)
    
    @staticmethod
    def cp_correlation(T: float, A: float, B: float, C: float, D: float) -> float:
        """Heat capacity correlation: Cp = A + B*T + C*T² + D*T³"""
        return A + B*T + C*T**2 + D*T**3
    
    @classmethod
    def water_properties(cls, T: float, P: float = 101325) -> FluidProperties:
        """Calculate water properties at given temperature and pressure"""
        T_K = T + 273.15
        
        # Density (kg/m³) - simplified correlation
        rho = 1000 * (1 - 0.0002 * (T - 4))
        
        # Viscosity (Pa·s) - simplified correlation
        mu = 0.001 * np.exp(1.3272 * (293.15 - T_K) / (T_K - 168.15))
        
        # Thermal conductivity (W/m·K)
        k = 0.6 * (1 + 0.0015 * T)
        
        # Specific heat (J/kg·K)
        cp = 4180 * (1 + 0.0001 * T)
        
        # Surface tension (N/m)
        sigma = 0.0728 * (1 - T_K/647.27)**1.256
        
        # Vapor pressure (Pa) - Antoine equation
        P_vap = 133.322 * 10**(8.07131 - 1730.63/(T_K - 39.724))
        
        return FluidProperties(rho, mu, k, cp, sigma, P_vap)
    
    @classmethod
    def air_properties(cls, T: float, P: float = 101325) -> FluidProperties:
        """Calculate air properties at given temperature and pressure"""
        T_K = T + 273.15
        
        # Density (kg/m³) - ideal gas law
        R_specific = 287  # J/kg·K for air
        rho = P / (R_specific * T_K)
        
        # Viscosity (Pa·s) - Sutherland's law
        mu = 1.716e-5 * (T_K/273.15)**1.5 * (383.55/(T_K + 110.4))
        
        # Thermal conductivity (W/m·K)
        k = 0.0241 * (T_K/273.15)**0.9
        
        # Specific heat (J/kg·K)
        cp = 1005 + 0.017 * T
        
        return FluidProperties(rho, mu, k, cp, 0.0, P)

class ProcessOptimization:
    """Process optimization and economic analysis"""
    
    def __init__(self, process_model: callable, constraints: List[callable] = None):
        """
        Initialize process optimization
        
        Parameters:
        -----------
        process_model : callable
            Function that takes process variables and returns objective value
        constraints : List[callable]
            List of constraint functions
        """
        self.process_model = process_model
        self.constraints = constraints or []
    
    def optimize(self, initial_guess: List[float], bounds: List[Tuple[float, float]] = None,
                method: str = 'SLSQP') -> dict:
        """
        Optimize process conditions
        
        Parameters:
        -----------
        initial_guess : List[float]
            Initial guess for optimization variables
        bounds : List[Tuple[float, float]]
            Bounds for each variable
        method : str
            Optimization method
        
        Returns:
        --------
        dict : Optimization results
        """
        # Convert constraints to scipy format
        constraints = []
        for constraint in self.constraints:
            constraints.append({'type': 'ineq', 'fun': constraint})
        
        result = minimize(
            lambda x: -self.process_model(x),  # Maximize by minimizing negative
            initial_guess,
            method=method,
            bounds=bounds,
            constraints=constraints
        )
        
        return {
            'optimal_variables': result.x,
            'optimal_value': -result.fun,
            'success': result.success,
            'message': result.message,
            'iterations': result.nit
        }
    
    @staticmethod
    def economic_analysis(production_rate: float, product_price: float,
                         raw_material_cost: float, utility_cost: float,
                         fixed_costs: float = 0) -> dict:
        """Perform economic analysis"""
        revenue = production_rate * product_price
        variable_costs = raw_material_cost + utility_cost
        profit = revenue - variable_costs - fixed_costs
        profit_margin = profit / revenue if revenue > 0 else 0
        
        return {
            'revenue': revenue,
            'variable_costs': variable_costs,
            'fixed_costs': fixed_costs,
            'profit': profit,
            'profit_margin': profit_margin
        }

class ProcessSimulation:
    """Dynamic process simulation and control"""
    
    def __init__(self, model_equations: callable, initial_conditions: List[float]):
        """
        Initialize process simulation
        
        Parameters:
        -----------
        model_equations : callable
            Function defining differential equations dx/dt = f(x, t, u)
        initial_conditions : List[float]
            Initial values of state variables
        """
        self.model = model_equations
        self.x0 = initial_conditions
    
    def simulate(self, time_span: Tuple[float, float], 
                control_inputs: callable = None, num_points: int = 1000) -> dict:
        """
        Simulate process dynamics
        
        Parameters:
        -----------
        time_span : Tuple[float, float]
            Start and end time for simulation
        control_inputs : callable
            Function defining control inputs u(t)
        num_points : int
            Number of time points
        
        Returns:
        --------
        dict : Simulation results
        """
        t = np.linspace(time_span[0], time_span[1], num_points)
        
        if control_inputs is None:
            control_inputs = lambda t: [0]  # No control inputs
        
        def model_with_control(x, t):
            u = control_inputs(t)
            return self.model(x, t, u)
        
        solution = odeint(model_with_control, self.x0, t)
        
        return {
            'time': t,
            'states': solution,
            'final_state': solution[-1]
        }
    
    def plot_simulation(self, results: dict, state_names: List[str] = None, 
                       save_path: Optional[str] = None):
        """Plot simulation results"""
        fig, axes = plt.subplots(len(results['states'][0]), 1, figsize=(12, 8), sharex=True)
        
        if len(results['states'][0]) == 1:
            axes = [axes]
        
        state_names = state_names or [f'State {i+1}' for i in range(len(results['states'][0]))]
        
        for i, ax in enumerate(axes):
            ax.plot(results['time'], results['states'][:, i], 'b-', linewidth=2)
            ax.set_ylabel(state_names[i])
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time')
        plt.title('Process Simulation Results')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Utility functions
def create_process_report(results: dict, filename: str = "process_report.txt"):
    """Create a detailed process analysis report"""
    with open(filename, 'w') as f:
        f.write("CHEMICAL ENGINEERING PROCESS ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        for section, data in results.items():
            f.write(f"{section.upper()}\n")
            f.write("-" * len(section) + "\n")
            
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            else:
                f.write(f"{data}\n")
            
            f.write("\n")

def compare_reactors(conditions: dict, time_residence: float) -> pd.DataFrame:
    """Compare performance of different reactor types"""
    results = []
    
    for reactor_type in ['CSTR', 'PFR', 'Batch']:
        reactor = ChemicalReactor(
            reactor_type, 
            conditions['rate_constant'],
            conditions['reaction_order'],
            conditions['initial_concentration']
        )
        
        if reactor_type == 'CSTR':
            result = reactor.cstr_design(time_residence)
        elif reactor_type == 'PFR':
            result = reactor.pfr_design(time_residence)
        else:
            result = reactor.batch_design(time_residence)
        
        results.append({
            'Reactor Type': reactor_type,
            'Conversion': result['conversion'],
            'Final Concentration': result['C_out'],
            'Reaction Rate': result['rate']
        })
    
    return pd.DataFrame(results)

# Example usage and demonstration functions
def demo_heat_exchanger():
    """Demonstrate heat exchanger calculations"""
    print("Heat Exchanger Design Example")
    print("=" * 30)
    
    hot_fluid = {
        'T_in': 150,  # °C
        'T_out': 80,  # °C
        'm_dot': 2.0,  # kg/s
        'cp': 4180    # J/kg·K
    }
    
    cold_fluid = {
        'T_in': 20,   # °C
        'T_out': 60,  # °C
        'm_dot': 1.5, # kg/s
        'cp': 4180    # J/kg·K
    }
    
    hx = HeatExchanger(hot_fluid, cold_fluid, U=500, exchanger_type="counter-current")
    results = hx.calculate_heat_transfer()
    
    print(f"Heat transfer rate: {results['Q_avg']/1000:.2f} kW")
    print(f"Required area: {results['area']:.2f} m²")
    print(f"Effectiveness: {results['effectiveness']:.3f}")
    print(f"LMTD: {results['LMTD']:.2f} °C")
    
    return results

def demo_distillation():
    """Demonstrate distillation calculations"""
    print("\nDistillation Column Design Example")
    print("=" * 35)
    
    column = DistillationColumn(
        feed_flow=100,      # kmol/h
        feed_composition=0.5,
        distillate_composition=0.9,
        bottoms_composition=0.1,
        reflux_ratio=2.0,
        relative_volatility=2.5
    )
    
    D, B = column.material_balance()
    R_min = column.minimum_reflux_ratio()
    N_min = column.theoretical_stages()
    
    print(f"Distillate flow rate: {D:.2f} kmol/h")
    print(f"Bottoms flow rate: {B:.2f} kmol/h")
    print(f"Minimum reflux ratio: {R_min:.2f}")
    print(f"Minimum theoretical stages: {N_min:.1f}")
    
    return {'D': D, 'B': B, 'R_min': R_min, 'N_min': N_min}

def demo_reactor_comparison():
    """Demonstrate reactor comparison"""
    print("\nReactor Performance Comparison")
    print("=" * 30)
    
    conditions = {
        'rate_constant': 0.1,     # 1/s
        'reaction_order': 1.0,
        'initial_concentration': 1.0  # mol/L
    }
    
    comparison = compare_reactors(conditions, time_residence=30)  # 30 seconds
    print(comparison.to_string(index=False))
    
    return comparison

def demo_fluid_mechanics():
    """Demonstrate fluid mechanics calculations"""
    print("\nFluid Mechanics Analysis Example")
    print("=" * 32)
    
    # Water properties at 25°C
    water = Thermodynamics.water_properties(25)
    
    pipe_data = {
        'diameter': 0.1,      # m
        'length': 100,        # m
        'velocity': 2.0,      # m/s
        'roughness': 0.00005  # m
    }
    
    flow_analysis = FluidMechanics.pipe_flow_analysis(water, pipe_data)
    
    print(f"Reynolds number: {flow_analysis['Reynolds_number']:.0f}")
    print(f"Flow regime: {flow_analysis['flow_regime']}")
    print(f"Friction factor: {flow_analysis['friction_factor']:.4f}")
    print(f"Pressure drop: {flow_analysis['pressure_drop']/1000:.2f} kPa")
    
    return flow_analysis

def demo_process_optimization():
    """Demonstrate process optimization"""
    print("\nProcess Optimization Example")
    print("=" * 28)
    
    def profit_model(variables):
        """Simple profit model: variables = [temperature, pressure]"""
        T, P = variables
        
        # Conversion model (increases with T and P)
        conversion = 0.5 + 0.003 * (T - 100) + 0.05 * (P - 1)
        conversion = min(0.95, max(0.1, conversion))
        
        # Production rate
        production_rate = 100 * conversion  # kg/h
        
        # Costs
        energy_cost = 0.1 * (T - 50) + 2 * (P - 1)  # $/h
        raw_material_cost = production_rate / conversion * 2  # $/h
        
        # Revenue
        revenue = production_rate * 10  # $/h
        
        profit = revenue - energy_cost - raw_material_cost
        return profit
    
    def temperature_constraint(variables):
        """Temperature constraint: T <= 200°C"""
        return 200 - variables[0]
    
    def pressure_constraint(variables):
        """Pressure constraint: P <= 10 bar"""
        return 10 - variables[1]
    
    optimizer = ProcessOptimization(profit_model, [temperature_constraint, pressure_constraint])
    
    initial_guess = [120, 5]  # T=120°C, P=5 bar
    bounds = [(50, 200), (1, 10)]  # Temperature and pressure bounds
    
    result = optimizer.optimize(initial_guess, bounds)
    
    print(f"Optimal temperature: {result['optimal_variables'][0]:.1f}°C")
    print(f"Optimal pressure: {result['optimal_variables'][1]:.1f} bar")
    print(f"Maximum profit: ${result['optimal_value']:.2f}/h")
    print(f"Optimization successful: {result['success']}")
    
    return result

def demo_process_simulation():
    """Demonstrate dynamic process simulation"""
    print("\nProcess Simulation Example")
    print("=" * 26)
    
    def cstr_model(x, t, u):
        """CSTR dynamic model: dx/dt = (C_in - C)/tau - k*C"""
        C = x[0]  # Concentration
        C_in = u[0] if len(u) > 0 else 1.0  # Inlet concentration
        tau = 10  # Residence time (s)
        k = 0.1   # Rate constant (1/s)
        
        dCdt = (C_in - C) / tau - k * C
        return [dCdt]
    
    def step_input(t):
        """Step change in inlet concentration at t=20s"""
        return [1.5 if t >= 20 else 1.0]
    
    simulator = ProcessSimulation(cstr_model, [0.5])  # Initial concentration = 0.5
    results = simulator.simulate((0, 100), step_input, 1000)
    
    # Find steady-state values
    initial_ss = results['states'][190]  # Before step change
    final_ss = results['states'][-1]     # After step change
    
    print(f"Initial steady-state concentration: {initial_ss[0]:.3f} mol/L")
    print(f"Final steady-state concentration: {final_ss[0]:.3f} mol/L")
    print(f"Response time (95% of final value): ~{np.where(results['states'][:, 0] >= 0.95*final_ss[0])[0][0]/10:.1f}s")
    
    return results

class MassTransfer:
    """Mass transfer operations and equipment design"""
    
    @staticmethod
    def sherwood_correlation(Re: float, Sc: float, correlation: str = "packed_bed") -> float:
        """Calculate Sherwood number using various correlations"""
        if correlation == "packed_bed":
            # For packed beds: Sh = 2 + 0.6*Re^0.5*Sc^(1/3)
            return 2 + 0.6 * Re**0.5 * Sc**(1/3)
        elif correlation == "pipe_flow":
            # For pipe flow: Sh = 0.023*Re^0.8*Sc^0.4
            return 0.023 * Re**0.8 * Sc**0.4
        else:
            return 2.0  # Default minimum value
    
    @staticmethod
    def mass_transfer_coefficient(Sh: float, diffusivity: float, length: float) -> float:
        """Calculate mass transfer coefficient"""
        return Sh * diffusivity / length
    
    @classmethod
    def absorption_column_design(cls, gas_flow: float, liquid_flow: float,
                               inlet_concentration: float, outlet_concentration: float,
                               equilibrium_constant: float, column_height: float) -> dict:
        """Design absorption column using film theory"""
        
        # Operating line slope
        L_over_G = liquid_flow / gas_flow
        
        # Number of transfer units (NTU)
        y_in = inlet_concentration
        y_out = outlet_concentration
        
        # Absorption factor
        A = L_over_G / equilibrium_constant
        
        if A != 1.0:
            NTU = np.log((y_in - equilibrium_constant * 0) / (y_out - equilibrium_constant * 0)) / (A - 1)
        else:
            NTU = (y_in - y_out) / y_out
        
        # Height of transfer unit (HTU)
        HTU = column_height / NTU if NTU > 0 else column_height
        
        # Removal efficiency
        efficiency = (y_in - y_out) / y_in
        
        return {
            'NTU': NTU,
            'HTU': HTU,
            'absorption_factor': A,
            'removal_efficiency': efficiency,
            'liquid_concentration_out': (y_in - y_out) * gas_flow / liquid_flow
        }

class SeparationProcesses:
    """Advanced separation process calculations"""
    
    @staticmethod
    def adsorption_isotherm(C: float, q_max: float, K: float, isotherm_type: str = "langmuir") -> float:
        """Calculate adsorption equilibrium using various isotherms"""
        if isotherm_type == "langmuir":
            return q_max * K * C / (1 + K * C)
        elif isotherm_type == "freundlich":
            # q_max becomes 'a' and K becomes '1/n'
            return q_max * C**K
        elif isotherm_type == "linear":
            return K * C
        else:
            return 0.0
    
    @staticmethod
    def breakthrough_curve(t: np.ndarray, t_breakthrough: float, 
                          mass_transfer_zone: float) -> np.ndarray:
        """Calculate breakthrough curve for fixed bed adsorption"""
        # Simplified S-curve model
        tau = mass_transfer_zone / 2
        return 0.5 * (1 + np.tanh((t - t_breakthrough) / tau))
    
    @classmethod
    def membrane_separation(cls, feed_concentration: float, membrane_area: float,
                          permeability: float, thickness: float, 
                          pressure_difference: float) -> dict:
        """Calculate membrane separation performance"""
        
        # Permeate flux
        flux = permeability * pressure_difference / thickness
        
        # Permeate flow rate
        permeate_flow = flux * membrane_area
        
        # Rejection coefficient (simplified)
        rejection = 0.9  # Assumed 90% rejection
        
        # Permeate concentration
        permeate_concentration = feed_concentration * (1 - rejection)
        
        # Mass transfer rate
        mass_transfer_rate = permeate_flow * permeate_concentration
        
        return {
            'permeate_flux': flux,
            'permeate_flow': permeate_flow,
            'permeate_concentration': permeate_concentration,
            'rejection': rejection,
            'mass_transfer_rate': mass_transfer_rate
        }

class ProcessControl:
    """Process control and instrumentation calculations"""
    
    @staticmethod
    def pid_controller(error: float, integral_error: float, derivative_error: float,
                      Kp: float, Ki: float, Kd: float) -> float:
        """Calculate PID controller output"""
        return Kp * error + Ki * integral_error + Kd * derivative_error
    
    @staticmethod
    def transfer_function_response(numerator: List[float], denominator: List[float],
                                 time: np.ndarray, input_type: str = "step") -> np.ndarray:
        """Calculate transfer function response (simplified first-order)"""
        # For first-order system: G(s) = K/(τs + 1)
        if len(denominator) == 2 and len(numerator) == 1:
            K = numerator[0] / denominator[1]  # Gain
            tau = denominator[0] / denominator[1]  # Time constant
            
            if input_type == "step":
                return K * (1 - np.exp(-time / tau))
            elif input_type == "ramp":
                return K * (time - tau * (1 - np.exp(-time / tau)))
            else:
                return np.zeros_like(time)
        else:
            return np.zeros_like(time)
    
    @classmethod
    def control_loop_analysis(cls, setpoint: float, process_variable: float,
                            controller_params: dict, disturbance: float = 0) -> dict:
        """Analyze control loop performance"""
        
        error = setpoint - process_variable + disturbance
        
        # Calculate controller output (simplified)
        Kp, Ki, Kd = controller_params.get('Kp', 1), controller_params.get('Ki', 0), controller_params.get('Kd', 0)
        
        # Assume some integral and derivative terms for demonstration
        integral_error = error * 0.1  # Simplified
        derivative_error = error * 0.01  # Simplified
        
        controller_output = cls.pid_controller(error, integral_error, derivative_error, Kp, Ki, Kd)
        
        # Offset and overshoot calculations (simplified)
        steady_state_error = error / (1 + Kp)
        
        return {
            'error': error,
            'controller_output': controller_output,
            'steady_state_error': steady_state_error,
            'proportional_term': Kp * error,
            'integral_term': Ki * integral_error,
            'derivative_term': Kd * derivative_error
        }

class SafetyAnalysis:
    """Process safety and hazard analysis tools"""
    
    @staticmethod
    def explosion_limits(fuel_concentration: float, lower_limit: float, upper_limit: float) -> str:
        """Check if concentration is within explosion limits"""
        if fuel_concentration < lower_limit:
            return "Below LEL - Too lean to ignite"
        elif fuel_concentration > upper_limit:
            return "Above UEL - Too rich to ignite"
        else:
            return "Within explosion limits - HAZARDOUS!"
    
    @staticmethod
    def relief_valve_sizing(flow_rate: float, density: float, pressure_drop: float,
                          discharge_coefficient: float = 0.6) -> float:
        """Calculate relief valve orifice area"""
        # A = W / (Cd * sqrt(2 * rho * ΔP))
        return flow_rate / (discharge_coefficient * np.sqrt(2 * density * pressure_drop))
    
    @staticmethod
    def toxic_release_dispersion(release_rate: float, wind_speed: float,
                               atmospheric_stability: str = "neutral") -> dict:
        """Simplified toxic release dispersion calculation"""
        
        # Dispersion coefficients based on atmospheric stability
        stability_params = {
            "stable": {"a": 0.08, "b": 0.9},
            "neutral": {"a": 0.15, "b": 0.85},
            "unstable": {"a": 0.25, "b": 0.8}
        }
        
        params = stability_params.get(atmospheric_stability, stability_params["neutral"])
        
        # Simplified Gaussian plume model
        distances = np.array([100, 500, 1000, 2000])  # meters
        concentrations = []
        
        for x in distances:
            sigma_y = params["a"] * x**params["b"]
            sigma_z = 0.5 * sigma_y  # Simplified assumption
            
            # Ground-level centerline concentration
            concentration = release_rate / (np.pi * wind_speed * sigma_y * sigma_z)
            concentrations.append(concentration)
        
        return {
            'distances': distances,
            'concentrations': concentrations,
            'atmospheric_stability': atmospheric_stability
        }

# Advanced utility functions
def process_economics_analysis(capital_cost: float, operating_cost_annual: float,
                             revenue_annual: float, project_life: int,
                             discount_rate: float) -> dict:
    """Comprehensive economic analysis of chemical process"""
    
    # Net present value calculation
    cash_flows = [revenue_annual - operating_cost_annual] * project_life
    npv = -capital_cost + sum([cf / (1 + discount_rate)**i for i, cf in enumerate(cash_flows, 1)])
    
    # Internal rate of return (simplified approximation)
    irr_approx = (revenue_annual - operating_cost_annual) / capital_cost
    
    # Payback period
    annual_profit = revenue_annual - operating_cost_annual
    payback_period = capital_cost / annual_profit if annual_profit > 0 else float('inf')
    
    # Return on investment
    roi = annual_profit / capital_cost * 100
    
    return {
        'npv': npv,
        'irr_approximate': irr_approx,
        'payback_period': payback_period,
        'roi_percent': roi,
        'total_profit_over_life': annual_profit * project_life
    }

def create_process_flowsheet_data(units: List[dict]) -> pd.DataFrame:
    """Create process flowsheet data structure"""
    flowsheet_data = []
    
    for unit in units:
        unit_data = {
            'Unit_ID': unit.get('id', 'Unknown'),
            'Unit_Type': unit.get('type', 'Unknown'),
            'Inlet_Flow': unit.get('inlet_flow', 0),
            'Outlet_Flow': unit.get('outlet_flow', 0),
            'Temperature': unit.get('temperature', 25),
            'Pressure': unit.get('pressure', 1),
            'Duty': unit.get('duty', 0)
        }
        flowsheet_data.append(unit_data)
    
    return pd.DataFrame(flowsheet_data)

def sensitivity_analysis(base_case: dict, parameter_name: str, 
                        variation_range: Tuple[float, float], 
                        objective_function: callable, num_points: int = 20) -> dict:
    """Perform sensitivity analysis on process parameters"""
    
    base_value = base_case[parameter_name]
    min_val, max_val = variation_range
    
    # Create parameter range
    param_values = np.linspace(min_val, max_val, num_points)
    objective_values = []
    
    for param_val in param_values:
        # Update parameter in base case
        modified_case = base_case.copy()
        modified_case[parameter_name] = param_val
        
        # Calculate objective function
        obj_val = objective_function(modified_case)
        objective_values.append(obj_val)
    
    # Calculate sensitivity coefficient
    param_change_percent = (param_values - base_value) / base_value * 100
    obj_change_percent = (np.array(objective_values) - objective_values[num_points//2]) / objective_values[num_points//2] * 100
    
    # Linear sensitivity coefficient
    sensitivity_coeff = np.polyfit(param_change_percent, obj_change_percent, 1)[0]
    
    return {
        'parameter_values': param_values,
        'objective_values': objective_values,
        'parameter_change_percent': param_change_percent,
        'objective_change_percent': obj_change_percent,
        'sensitivity_coefficient': sensitivity_coeff
    }

# Extended demonstration functions
def demo_mass_transfer():
    """Demonstrate mass transfer calculations"""
    print("\nMass Transfer Example")
    print("=" * 21)
    
    # Absorption column design
    absorption_results = MassTransfer.absorption_column_design(
        gas_flow=100,           # m³/h
        liquid_flow=200,        # m³/h
        inlet_concentration=0.1, # mol/m³
        outlet_concentration=0.01, # mol/m³
        equilibrium_constant=0.5,
        column_height=10        # m
    )
    
    print(f"Number of Transfer Units: {absorption_results['NTU']:.2f}")
    print(f"Height of Transfer Unit: {absorption_results['HTU']:.2f} m")
    print(f"Removal Efficiency: {absorption_results['removal_efficiency']:.1%}")
    
    return absorption_results

def demo_safety_analysis():
    """Demonstrate safety analysis"""
    print("\nSafety Analysis Example")
    print("=" * 23)
    
    # Explosion limits check
    fuel_conc = 3.5  # % volume
    safety_status = SafetyAnalysis.explosion_limits(fuel_conc, 2.1, 9.5)
    print(f"Fuel concentration: {fuel_conc}%")
    print(f"Safety status: {safety_status}")
    
    # Relief valve sizing
    relief_area = SafetyAnalysis.relief_valve_sizing(
        flow_rate=10,        # kg/s
        density=1000,        # kg/m³
        pressure_drop=500000 # Pa
    )
    print(f"Required relief valve area: {relief_area*10000:.2f} cm²")
    
    return {'safety_status': safety_status, 'relief_area': relief_area}

def demo_economic_analysis():
    """Demonstrate economic analysis"""
    print("\nEconomic Analysis Example")
    print("=" * 25)
    
    economics = process_economics_analysis(
        capital_cost=1000000,      # $
        operating_cost_annual=200000, # $/year
        revenue_annual=500000,     # $/year
        project_life=10,           # years
        discount_rate=0.1          # 10%
    )
    
    print(f"Net Present Value: ${economics['npv']:,.0f}")
    print(f"Payback Period: {economics['payback_period']:.1f} years")
    print(f"Return on Investment: {economics['roi_percent']:.1f}%")
    
    return economics

if __name__ == "__main__":
    print("CHEMICAL ENGINEERING TOOLKIT DEMONSTRATION")
    print("=" * 45)
    
    # Run all demonstrations
    hx_results = demo_heat_exchanger()
    dist_results = demo_distillation()
    reactor_results = demo_reactor_comparison()
    fluid_results = demo_fluid_mechanics()
    optimization_results = demo_process_optimization()
    simulation_results = demo_process_simulation()
    mass_transfer_results = demo_mass_transfer()
    safety_results = demo_safety_analysis()
    economics_results = demo_economic_analysis()
    
    # Create comprehensive report
    all_results = {
        'Heat Exchanger Analysis': hx_results,
        'Distillation Column Design': dist_results,
        'Reactor Comparison': reactor_results.to_dict('records'),
        'Fluid Mechanics Analysis': fluid_results,
        'Process Optimization': optimization_results,
        'Mass Transfer Operations': mass_transfer_results,
        'Safety Analysis': safety_results,
        'Economic Analysis': economics_results
    }
    
    create_process_report(all_results)
    print(f"\nDetailed report saved as 'process_report.txt'")
    
    print(f"\nToolkit demonstration completed successfully!")
    print("All modules are ready for use in your chemical engineering projects.")
