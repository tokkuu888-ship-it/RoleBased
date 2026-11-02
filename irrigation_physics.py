"""
Advanced Furrow Irrigation Physics Module for TOKUMA 3-in-1 Research Platform
Provides Kostiakov equation implementation, advance/recession curves, and hydraulic analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import fsolve
from scipy.integrate import odeint


class FurrowIrrigationPhysics:
    """Advanced furrow irrigation physics modeling"""
    
    def __init__(self):
        # Soil type parameters for Kostiakov equation (k, a)
        self.soil_parameters = {
            "Clay": {"k": 0.15, "a": 0.6, "suction": 0.3, "conductivity": 0.5},
            "Sandy Loam": {"k": 2.5, "a": 0.4, "suction": 0.1, "conductivity": 3.0},
            "Silt Clay": {"k": 1.0, "a": 0.5, "suction": 0.2, "conductivity": 1.5},
            "Loam": {"k": 1.5, "a": 0.45, "suction": 0.15, "conductivity": 2.0},
            "Sandy": {"k": 5.0, "a": 0.35, "suction": 0.05, "conductivity": 5.0},
            "Silt": {"k": 3.0, "a": 0.4, "suction": 0.12, "conductivity": 3.5}
        }
        
        # Manning's roughness coefficients for different furrow conditions
        self.manning_n = {
            "Smooth concrete": 0.012,
            "Smooth earth": 0.020,
            "Natural earth": 0.025,
            "Vegetated": 0.035,
            "Rough earth": 0.030
        }
        
        # Typical furrow geometries
        self.furrow_shapes = {
            "Triangular": {"area_factor": 0.5, "wetted_perimeter_factor": 2.236},
            "Trapezoidal": {"area_factor": 0.7, "wetted_perimeter_factor": 2.5},
            "Parabolic": {"area_factor": 0.6, "wetted_perimeter_factor": 2.4}
        }

    def kostiakov_infiltration(self, t: np.ndarray, k: float, a: float, 
                              initial_cumulative: float = 0) -> np.ndarray:
        """
        Kostiakov infiltration equation: Z = k*t^a + Z0
        where Z is cumulative infiltration (mm), t is time (min)
        """
        return k * np.power(t, a) + initial_cumulative

    def kostiakov_infiltration_rate(self, t: np.ndarray, k: float, a: float) -> np.ndarray:
        """
        Instantaneous infiltration rate: dZ/dt = k*a*t^(a-1)
        """
        return k * a * np.power(t, a - 1)

    def modified_kostiakov(self, t: np.ndarray, k: float, a: float, 
                          fc: float, initial_cumulative: float = 0) -> np.ndarray:
        """
        Modified Kostiakov with steady-state infiltration: Z = k*t^a + fc*t + Z0
        where fc is the final steady-state infiltration rate (mm/min)
        """
        return k * np.power(t, a) + fc * t + initial_cumulative

    def philip_infiltration(self, t: np.ndarray, sorptivity: float, 
                           conductivity: float) -> np.ndarray:
        """
        Philip's infiltration equation: Z = S*t^0.5 + A*t
        where S is sorptivity and A is the steady infiltration rate
        """
        return sorptivity * np.sqrt(t) + conductivity * t

    def green_ampt_infiltration(self, t: np.ndarray, initial_moisture: float,
                               saturated_moisture: float, conductivity: float,
                               suction_head: float) -> np.ndarray:
        """
        Green-Ampt infiltration equation
        """
        delta_theta = saturated_moisture - initial_moisture
        infiltration = np.zeros_like(t)
        
        for i, time in enumerate(t):
            if time > 0:
                # Simplified Green-Ampt solution
                infiltration[i] = conductivity * time + delta_theta * suction_head * np.log(
                    1 + infiltration[i] / (delta_theta * suction_head)
                ) if i > 0 else 0
        
        return infiltration

    def furrow_hydraulics(self, flow_rate: float, slope: float, 
                         roughness: float, shape: str = "Triangular") -> Dict[str, float]:
        """
        Calculate furrow hydraulic parameters using Manning's equation
        """
        shape_params = self.furrow_shapes[shape]
        
        # For triangular furrow, assume top width to depth ratio of 2:1
        # Manning's equation: Q = (1/n) * A * R^(2/3) * S^(1/2)
        
        # Iterative solution for normal depth
        def normal_depth_equation(depth):
            area = shape_params["area_factor"] * depth * depth  # Simplified area calculation
            wetted_perimeter = shape_params["wetted_perimeter_factor"] * depth
            hydraulic_radius = area / wetted_perimeter
            
            q_calculated = (1 / roughness) * area * np.power(hydraulic_radius, 2/3) * np.sqrt(slope)
            return q_calculated - flow_rate
        
        # Solve for normal depth
        try:
            normal_depth = fsolve(normal_depth_equation, 0.1)[0]
            normal_depth = max(normal_depth, 0.01)  # Ensure positive depth
        except:
            normal_depth = 0.1  # Default value
        
        # Calculate hydraulic parameters
        area = shape_params["area_factor"] * normal_depth * normal_depth
        wetted_perimeter = shape_params["wetted_perimeter_factor"] * normal_depth
        hydraulic_radius = area / wetted_perimeter
        velocity = flow_rate / area if area > 0 else 0
        
        # Froude number
        froude = velocity / np.sqrt(9.81 * normal_depth) if normal_depth > 0 else 0
        
        return {
            "normal_depth_m": normal_depth,
            "area_m2": area,
            "wetted_perimeter_m": wetted_perimeter,
            "hydraulic_radius_m": hydraulic_radius,
            "velocity_m_s": velocity,
            "froude_number": froude
        }

    def advance_curve(self, distance: np.ndarray, flow_rate: float, 
                     infiltration_params: Dict, slope: float, 
                     roughness: float) -> np.ndarray:
        """
        Calculate advance curve (time for water to reach different distances)
        Using the volume balance approach
        """
        k = infiltration_params["k"]
        a = infiltration_params["a"]
        fc = infiltration_params.get("fc", 0.01)  # Final infiltration rate
        
        times = np.zeros_like(distance)
        
        for i, dist in enumerate(distance):
            if dist <= 0:
                times[i] = 0
                continue
            
            # Volume balance equation: Q*t = Volume in furrow + Volume infiltrated
            # Simplified solution using iterative approach
            def volume_balance(t):
                if t <= 0:
                    return -flow_rate
                
                # Volume in furrow (simplified)
                hydraulic_params = self.furrow_hydraulics(flow_rate, slope, roughness)
                furrow_volume = hydraulic_params["area_m2"] * dist
                
                # Volume infiltrated (integrated over distance and time)
                # Simplified: assume uniform infiltration along the furrow
                avg_infiltration = self.modified_kostiakov(
                    np.array([t]), k, a, fc
                )[0]
                infiltrated_volume = avg_infiltration * dist * 0.001  # Convert mm to m
                
                return flow_rate * t - furrow_volume - infiltrated_volume
            
            try:
                times[i] = fsolve(volume_balance, max(dist / (flow_rate * 10), 1))[0]
                times[i] = max(times[i], 0)  # Ensure non-negative
            except:
                times[i] = dist / (flow_rate * 5)  # Fallback calculation
        
        return times

    def recession_curve(self, distance: np.ndarray, advance_times: np.ndarray,
                        cutoff_time: float, infiltration_params: Dict) -> np.ndarray:
        """
        Calculate recession curve (time for water to recede from different distances)
        """
        k = infiltration_params["k"]
        a = infiltration_params["a"]
        fc = infiltration_params.get("fc", 0.01)
        
        recession_times = np.zeros_like(distance)
        
        for i, dist in enumerate(distance):
            advance_time = advance_times[i]
            if advance_time >= cutoff_time:
                recession_times[i] = 0  # Water never reached this point
                continue
            
            # Time water was present at this location
            opportunity_time = cutoff_time - advance_time
            
            # Recession occurs when infiltration opportunity is satisfied
            # Simplified approach
            total_infiltration = self.modified_kostiakov(
                np.array([opportunity_time]), k, a, fc
            )[0]
            
            # Recession time is when infiltration capacity is met
            # This is a simplified calculation
            recession_times[i] = cutoff_time + opportunity_time * 0.5
        
        return recession_times

    def application_efficiency(self, total_applied: float, 
                              required_depth: float, 
                              advance_times: np.ndarray,
                              recession_times: np.ndarray,
                              distance: np.ndarray) -> Dict[str, float]:
        """
        Calculate irrigation application efficiency metrics
        """
        # Distribution uniformity (DU)
        if len(recession_times) > 1:
            min_infiltration = np.min(recession_times - advance_times)
            avg_infiltration = np.mean(recession_times - advance_times)
            du = min_infiltration / avg_infiltration if avg_infiltration > 0 else 0
        else:
            du = 0
        
        # Application efficiency (AE)
        beneficial_use = required_depth * np.max(distance) * 0.001  # Convert to m³
        ae = beneficial_use / total_applied if total_applied > 0 else 0
        
        # Storage efficiency
        storage_efficiency = (total_applied - beneficial_use) / total_applied if total_applied > 0 else 0
        
        return {
            "distribution_uniformity": du,
            "application_efficiency": ae,
            "storage_efficiency": storage_efficiency,
            "total_applied_m3": total_applied,
            "beneficial_use_m3": beneficial_use
        }

    def plot_infiltration_curves(self, soil_type: str, time_range: Tuple[float, float]) -> go.Figure:
        """Plot infiltration curves for different models"""
        t = np.linspace(time_range[0], time_range[1], 100)
        params = self.soil_parameters[soil_type]
        
        fig = go.Figure()
        
        # Kostiakov
        kostiakov_cum = self.kostiakov_infiltration(t, params["k"], params["a"])
        kostiakov_rate = self.kostiakov_infiltration_rate(t, params["k"], params["a"])
        
        fig.add_trace(go.Scatter(
            x=t, y=kostiakov_cum,
            mode='lines', name=f'Kostiakov Cumulative (k={params["k"]}, a={params["a"]})',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=t, y=kostiakov_rate * 10,  # Scale for visibility
            mode='lines', name='Kostiakov Rate (×10)',
            line=dict(color='lightblue', dash='dash')
        ))
        
        # Modified Kostiakov
        fc = params["conductivity"] * 0.1  # Simplified final rate
        mod_kostiakov = self.modified_kostiakov(t, params["k"], params["a"], fc)
        
        fig.add_trace(go.Scatter(
            x=t, y=mod_kostiakov,
            mode='lines', name=f'Modified Kostiakov (fc={fc:.3f})',
            line=dict(color='red')
        ))
        
        # Philip
        philip = self.philip_infiltration(t, params["suction"], params["conductivity"])
        
        fig.add_trace(go.Scatter(
            x=t, y=philip,
            mode='lines', name=f'Philip (S={params["suction"]}, A={params["conductivity"]})',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title=f'Infiltration Models - {soil_type} Soil',
            xaxis_title='Time (minutes)',
            yaxis_title='Cumulative Infiltration (mm)',
            height=500
        )
        
        return fig

    def plot_advance_recession_curves(self, flow_rate: float, slope: float,
                                    soil_type: str, furrow_length: float) -> go.Figure:
        """Plot advance and recession curves"""
        distance = np.linspace(0, furrow_length, 50)
        infiltration_params = self.soil_parameters[soil_type]
        
        # Calculate advance curve
        advance_times = self.advance_curve(
            distance, flow_rate, infiltration_params, slope, 0.025
        )
        
        # Calculate recession curve (assuming 60 min cutoff)
        cutoff_time = 60
        recession_times = self.recession_curve(
            distance, advance_times, cutoff_time, infiltration_params
        )
        
        fig = go.Figure()
        
        # Advance curve
        fig.add_trace(go.Scatter(
            x=distance, y=advance_times,
            mode='lines+markers', name='Advance Time',
            line=dict(color='blue', width=3),
            marker=dict(size=4)
        ))
        
        # Recession curve
        valid_recession = recession_times > 0
        fig.add_trace(go.Scatter(
            x=distance[valid_recession], 
            y=recession_times[valid_recession],
            mode='lines+markers', name='Recession Time',
            line=dict(color='red', width=3),
            marker=dict(size=4)
        ))
        
        # Opportunity time
        opportunity_times = np.zeros_like(advance_times)
        opportunity_times[valid_recession] = recession_times[valid_recession] - advance_times[valid_recession]
        
        fig.add_trace(go.Scatter(
            x=distance, y=opportunity_times,
            mode='lines', name='Opportunity Time',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'Advance-Recession Curves - {soil_type} Soil',
            xaxis_title='Distance (m)',
            yaxis_title='Time (minutes)',
            height=500
        )
        
        return fig

    def render_irrigation_physics_interface(self) -> Dict[str, Any]:
        """Render the complete irrigation physics interface"""
        st.subheader("⚙️ Advanced Furrow Irrigation Physics")
        
        # Input parameters
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**Furrow Parameters**")
            furrow_length = st.number_input("Furrow Length (m)", value=150.0, min_value=10.0, max_value=500.0)
            furrow_slope_percent = st.number_input("Furrow Slope (%)", value=0.1, min_value=0.01, max_value=2.0)
            furrow_slope = furrow_slope_percent / 100.0
            flow_rate = st.number_input("Flow Rate (L/s)", value=2.0, min_value=0.1, max_value=10.0)
            
            st.write("**Soil Properties**")
            soil_type = st.selectbox("Soil Type", list(self.soil_parameters.keys()))
            furrow_condition = st.selectbox("Furrow Condition", list(self.manning_n.keys()))
            furrow_shape = st.selectbox("Furrow Shape", list(self.furrow_shapes.keys()))
            
            st.write("**Infiltration Model Parameters**")
            use_modified = st.checkbox("Use Modified Kostiakov", value=True)
            if use_modified:
                fc_rate = st.number_input("Final Infiltration Rate (mm/min)", 
                                        value=0.05, min_value=0.01, max_value=0.5, step=0.01)
            
            st.write("**Simulation Settings**")
            simulation_time = st.slider("Simulation Time (min)", 30, 240, 120)
            time_points = st.slider("Time Points", 20, 200, 100)
        
        with col2:
            # Get soil parameters
            soil_params = self.soil_parameters[soil_type]
            roughness = self.manning_n[furrow_condition]
            
            # Display soil parameters
            st.write(f"**{soil_type} Soil Parameters:**")
            col2a, col2b = st.columns(2)
            col2a.metric("k (Kostiakov)", f"{soil_params['k']:.3f}")
            col2b.metric("a (Kostiakov)", f"{soil_params['a']:.3f}")
            col2a.metric("Suction Head", f"{soil_params['suction']:.3f}")
            col2b.metric("Conductivity", f"{soil_params['conductivity']:.3f}")
            
            # Hydraulic calculations
            st.write("**Hydraulic Analysis:**")
            hydraulic_results = self.furrow_hydraulics(
                flow_rate / 1000,  # Convert L/s to m³/s
                furrow_slope,
                roughness,
                furrow_shape
            )
            
            col3a, col3b, col3c = st.columns(3)
            col3a.metric("Normal Depth", f"{hydraulic_results['normal_depth_m']:.3f} m")
            col3b.metric("Velocity", f"{hydraulic_results['velocity_m_s']:.3f} m/s")
            col3c.metric("Froude Number", f"{hydraulic_results['froude_number']:.3f}")
            
            # Infiltration curves plot
            st.write("**Infiltration Models Comparison:**")
            infiltration_fig = self.plot_infiltration_curves(
                soil_type, (1, simulation_time)
            )
            st.plotly_chart(infiltration_fig, use_container_width=True)
            
            # Advance-Recession curves
            st.write("**Advance-Recession Analysis:**")
            advance_recession_fig = self.plot_advance_recession_curves(
                flow_rate / 1000, furrow_slope, soil_type, furrow_length
            )
            st.plotly_chart(advance_recession_fig, use_container_width=True)
            
            # Efficiency calculations
            distance = np.linspace(0, furrow_length, 50)
            infiltration_params = soil_params.copy()
            if use_modified:
                infiltration_params["fc"] = fc_rate
            
            advance_times = self.advance_curve(
                distance, flow_rate / 1000, infiltration_params, furrow_slope, roughness
            )
            recession_times = self.recession_curve(
                distance, advance_times, simulation_time, infiltration_params
            )
            
            # Calculate total applied volume
            total_applied = flow_rate * simulation_time * 60 / 1000  # Convert to m³
            required_depth = 50  # mm, typical irrigation requirement
            
            efficiency_metrics = self.application_efficiency(
                total_applied, required_depth, advance_times, recession_times, distance
            )
            
            st.write("**Irrigation Efficiency Metrics:**")
            col4a, col4b, col4c = st.columns(3)
            col4a.metric("Distribution Uniformity", f"{efficiency_metrics['distribution_uniformity']:.2%}")
            col4b.metric("Application Efficiency", f"{efficiency_metrics['application_efficiency']:.2%}")
            col4c.metric("Storage Efficiency", f"{efficiency_metrics['storage_efficiency']:.2%}")
            
            # Detailed results table
            st.write("**Detailed Hydraulic Results:**")
            results_df = pd.DataFrame([
                ["Normal Depth (m)", f"{hydraulic_results['normal_depth_m']:.4f}"],
                ["Cross-sectional Area (m²)", f"{hydraulic_results['area_m2']:.4f}"],
                ["Wetted Perimeter (m)", f"{hydraulic_results['wetted_perimeter_m']:.4f}"],
                ["Hydraulic Radius (m)", f"{hydraulic_results['hydraulic_radius_m']:.4f}"],
                ["Flow Velocity (m/s)", f"{hydraulic_results['velocity_m_s']:.4f}"],
                ["Froude Number", f"{hydraulic_results['froude_number']:.4f}"],
                ["Total Applied Volume (m³)", f"{efficiency_metrics['total_applied_m3']:.2f}"],
                ["Beneficial Use (m³)", f"{efficiency_metrics['beneficial_use_m3']:.2f}"]
            ], columns=["Parameter", "Value"])
            
            st.dataframe(results_df, use_container_width=True)
        
        return {
            "soil_type": soil_type,
            "furrow_length": furrow_length,
            "flow_rate": flow_rate,
            "efficiency_metrics": efficiency_metrics,
            "hydraulic_results": hydraulic_results
        }


def get_irrigation_physics() -> FurrowIrrigationPhysics:
    """Get irrigation physics instance"""
    if 'irrigation_physics' not in st.session_state:
        st.session_state.irrigation_physics = FurrowIrrigationPhysics()
    return st.session_state.irrigation_physics
