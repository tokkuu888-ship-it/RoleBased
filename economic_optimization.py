"""
Economic Optimization Module for TOKUMA 3-in-1 Research Platform
Provides cost-benefit analysis, profit maximization, and economic nexus integration
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass


@dataclass
class EconomicParameters:
    """Economic parameters for agricultural optimization"""
    # Crop prices (USD per ton)
    wheat_price: float = 280.0
    maize_price: float = 200.0
    teff_price: float = 450.0
    
    # Input costs
    water_cost_per_m3: float = 0.05
    seed_cost_per_ha: float = 120.0
    fertilizer_cost_per_ha: float = 180.0
    labor_cost_per_ha: float = 200.0
    energy_cost_per_ha: float = 150.0
    
    # Irrigation system costs
    pump_cost_per_kw: float = 800.0
    pipe_cost_per_m: float = 25.0
    maintenance_cost_percent: float = 0.05
    
    # Economic parameters
    discount_rate: float = 0.08
    project_lifetime: int = 20
    inflation_rate: float = 0.03


class EconomicOptimizer:
    """Advanced economic optimization for agricultural systems"""
    
    def __init__(self):
        self.params = EconomicParameters()
        
        # Crop-specific parameters
        self.crop_params = {
            "Wheat": {
                "yield_potential": 5.0,  # t/ha
                "water_requirement": 450,  # mm/season
                "price": self.params.wheat_price,
                "seed_cost": self.params.seed_cost_per_ha * 1.2,
                "fertilizer_cost": self.params.fertilizer_cost_per_ha * 1.1
            },
            "Maize": {
                "yield_potential": 6.0,
                "water_requirement": 500,
                "price": self.params.maize_price,
                "seed_cost": self.params.seed_cost_per_ha * 0.8,
                "fertilizer_cost": self.params.fertilizer_cost_per_ha * 1.3
            },
            "Teff": {
                "yield_potential": 2.5,
                "water_requirement": 350,
                "price": self.params.teff_price,
                "seed_cost": self.params.seed_cost_per_ha * 1.5,
                "fertilizer_cost": self.params.fertilizer_cost_per_ha * 0.9
            }
        }
        
        # Irrigation system costs
        self.irrigation_costs = {
            "Traditional": {
                "capital_cost_per_ha": 500,
                "efficiency": 0.55,
                "energy_requirement": 0.3  # kWh/m³
            },
            "Improved": {
                "capital_cost_per_ha": 1500,
                "efficiency": 0.75,
                "energy_requirement": 0.2
            },
            "Drip": {
                "capital_cost_per_ha": 3000,
                "efficiency": 0.90,
                "energy_requirement": 0.1
            },
            "Sprinkler": {
                "capital_cost_per_ha": 2000,
                "efficiency": 0.80,
                "energy_requirement": 0.15
            }
        }

    def calculate_production_costs(self, crop_type: str, area_ha: float,
                                 irrigation_method: str, total_water_m3: float) -> Dict[str, float]:
        """Calculate total production costs"""
        crop_info = self.crop_params[crop_type]
        irrigation_info = self.irrigation_costs[irrigation_method]
        
        # Variable costs
        water_cost = total_water_m3 * self.params.water_cost_per_m3
        seed_cost = crop_info["seed_cost"] * area_ha
        fertilizer_cost = crop_info["fertilizer_cost"] * area_ha
        labor_cost = self.params.labor_cost_per_ha * area_ha
        
        # Energy cost for irrigation
        energy_kwh = total_water_m3 * irrigation_info["energy_requirement"]
        energy_cost = energy_kwh * 0.1  # $0.1 per kWh
        
        # Fixed costs (annualized)
        capital_cost = irrigation_info["capital_cost_per_ha"] * area_ha
        annualized_capital = self._annualize_cost(capital_cost)
        maintenance_cost = capital_cost * self.params.maintenance_cost_percent
        
        total_variable_cost = water_cost + seed_cost + fertilizer_cost + labor_cost + energy_cost
        total_fixed_cost = annualized_capital + maintenance_cost
        total_cost = total_variable_cost + total_fixed_cost
        
        return {
            "water_cost": water_cost,
            "seed_cost": seed_cost,
            "fertilizer_cost": fertilizer_cost,
            "labor_cost": labor_cost,
            "energy_cost": energy_cost,
            "annualized_capital": annualized_capital,
            "maintenance_cost": maintenance_cost,
            "total_variable_cost": total_variable_cost,
            "total_fixed_cost": total_fixed_cost,
            "total_cost": total_cost
        }

    def calculate_revenue(self, crop_type: str, yield_tons: float) -> float:
        """Calculate gross revenue from crop production"""
        crop_info = self.crop_params[crop_type]
        return yield_tons * crop_info["price"]

    def calculate_profitability_metrics(self, crop_type: str, area_ha: float,
                                     irrigation_method: str, yield_tons: float,
                                     total_water_m3: float) -> Dict[str, float]:
        """Calculate comprehensive profitability metrics"""
        costs = self.calculate_production_costs(crop_type, area_ha, irrigation_method, total_water_m3)
        revenue = self.calculate_revenue(crop_type, yield_tons)
        
        gross_profit = revenue - costs["total_variable_cost"]
        net_profit = revenue - costs["total_cost"]
        
        # Profitability ratios
        profit_margin = (net_profit / revenue) * 100 if revenue > 0 else 0
        return_on_investment = (net_profit / costs["total_fixed_cost"]) * 100 if costs["total_fixed_cost"] > 0 else 0
        
        # Water productivity (economic)
        economic_wp = revenue / total_water_m3 if total_water_m3 > 0 else 0
        
        # Land productivity
        land_productivity = revenue / area_ha if area_ha > 0 else 0
        
        return {
            "revenue": revenue,
            "gross_profit": gross_profit,
            "net_profit": net_profit,
            "profit_margin": profit_margin,
            "return_on_investment": return_on_investment,
            "economic_water_productivity": economic_wp,
            "land_productivity": land_productivity,
            "total_cost": costs["total_cost"],
            "variable_cost": costs["total_variable_cost"],
            "fixed_cost": costs["total_fixed_cost"]
        }

    def multi_objective_objective(self, x: np.ndarray, crop_type: str) -> np.ndarray:
        """
        Multi-objective function for NSGA-II optimization
        x = [area_ha, irrigation_level, irrigation_method_index]
        Returns [negative_profit, water_use, negative_land_productivity]
        """
        area_ha, irrigation_level, irr_method_idx = x
        irrigation_methods = list(self.irrigation_costs.keys())
        irrigation_method = irrigation_methods[int(irr_method_idx)]
        
        # Calculate yield based on irrigation level
        crop_info = self.crop_params[crop_type]
        potential_yield = crop_info["yield_potential"]
        water_req = crop_info["water_requirement"]
        
        # Yield response to water (simplified)
        actual_yield = potential_yield * min(1.0, irrigation_level)
        total_water = (water_req * area_ha * irrigation_level) / 1000  # Convert to m³
        
        # Calculate profitability
        metrics = self.calculate_profitability_metrics(
            crop_type, area_ha, irrigation_method, actual_yield, total_water
        )
        
        return np.array([
            -metrics["net_profit"],  # Minimize negative profit (maximize profit)
            total_water,             # Minimize water use
            -metrics["land_productivity"]  # Minimize negative land productivity
        ])

    def optimize_economic_objectives(self, crop_type: str, method: str = "nsga2") -> Dict[str, Any]:
        """Optimize economic objectives using different algorithms"""
        
        if method == "nsga2":
            return self._nsga2_optimization(crop_type)
        elif method == "differential_evolution":
            return self._differential_evolution_optimization(crop_type)
        else:
            return self._single_objective_optimization(crop_type)

    def _nsga2_optimization(self, crop_type: str) -> Dict[str, Any]:
        """NSGA-II multi-objective optimization"""
        try:
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.core.problem import ElementwiseProblem
            from pymoo.optimize import minimize
            from pymoo.termination import get_termination
            
            class EconomicProblem(ElementwiseProblem):
                def __init__(self, optimizer, crop):
                    self.optimizer = optimizer
                    self.crop = crop
                    super().__init__(
                        n_var=3,
                        n_obj=3,
                        n_ieq_constr=0,
                        xl=np.array([1.0, 0.3, 0.0]),
                        xu=np.array([100.0, 1.5, 3.0])
                    )
                
                def _evaluate(self, x, out, *args, **kwargs):
                    out["F"] = self.optimizer.multi_objective_objective(x, self.crop)
            
            problem = EconomicProblem(self, crop_type)
            algorithm = NSGA2(pop_size=50)
            termination = get_termination("n_gen", 100)
            
            res = minimize(problem, algorithm, termination, verbose=False)
            
            # Process results
            X = res.X
            F = res.F
            
            results = {
                "pareto_solutions": X,
                "pareto_objectives": F,
                "method": "NSGA-II"
            }
            
            # Convert to DataFrame for easier analysis
            solutions_df = pd.DataFrame(X, columns=['area_ha', 'irrigation_level', 'irrigation_method'])
            solutions_df['irrigation_method'] = solutions_df['irrigation_method'].astype(int).map(
                {0: 'Traditional', 1: 'Improved', 2: 'Drip', 3: 'Sprinkler'}
            )
            
            objectives_df = pd.DataFrame(F, columns=['negative_profit', 'water_use', 'negative_land_productivity'])
            objectives_df['profit'] = -objectives_df['negative_profit']
            objectives_df['land_productivity'] = -objectives_df['negative_land_productivity']
            
            results['solutions_df'] = pd.concat([solutions_df, objectives_df[['profit', 'water_use', 'land_productivity']]], axis=1)
            
            return results
            
        except ImportError:
            st.error("pymoo not installed. Using fallback optimization.")
            return self._single_objective_optimization(crop_type)

    def _differential_evolution_optimization(self, crop_type: str) -> Dict[str, Any]:
        """Differential evolution optimization for single objective"""
        def objective(x):
            area_ha, irrigation_level, irr_method_idx = x
            irrigation_methods = list(self.irrigation_costs.keys())
            irrigation_method = irrigation_methods[int(irr_method_idx)]
            
            crop_info = self.crop_params[crop_type]
            actual_yield = crop_info["yield_potential"] * min(1.0, irrigation_level)
            total_water = (crop_info["water_requirement"] * area_ha * irrigation_level) / 1000
            
            metrics = self.calculate_profitability_metrics(
                crop_type, area_ha, irrigation_method, actual_yield, total_water
            )
            
            return -metrics["net_profit"]  # Minimize negative profit
        
        bounds = [(1, 100), (0.3, 1.5), (0, 3)]
        result = differential_evolution(objective, bounds, maxiter=50, popsize=15)
        
        area_ha, irrigation_level, irr_method_idx = result.x
        irrigation_methods = list(self.irrigation_costs.keys())
        irrigation_method = irrigation_methods[int(irr_method_idx)]
        
        crop_info = self.crop_params[crop_type]
        actual_yield = crop_info["yield_potential"] * min(1.0, irrigation_level)
        total_water = (crop_info["water_requirement"] * area_ha * irrigation_level) / 1000
        
        metrics = self.calculate_profitability_metrics(
            crop_type, area_ha, irrigation_method, actual_yield, total_water
        )
        
        return {
            "optimal_solution": result.x,
            "optimal_objective": -result.fun,
            "metrics": metrics,
            "method": "Differential Evolution"
        }

    def _single_objective_optimization(self, crop_type: str) -> Dict[str, Any]:
        """Fallback single objective optimization"""
        def objective(x):
            area_ha, irrigation_level, irr_method_idx = x
            irrigation_methods = list(self.irrigation_costs.keys())
            irrigation_method = irrigation_methods[int(irr_method_idx)]
            
            crop_info = self.crop_params[crop_type]
            actual_yield = crop_info["yield_potential"] * min(1.0, irrigation_level)
            total_water = (crop_info["water_requirement"] * area_ha * irrigation_level) / 1000
            
            metrics = self.calculate_profitability_metrics(
                crop_type, area_ha, irrigation_method, actual_yield, total_water
            )
            
            return -metrics["net_profit"]
        
        # Simple grid search as fallback
        best_profit = -np.inf
        best_solution = None
        
        for area in [10, 25, 50, 75, 100]:
            for irr_level in [0.5, 0.75, 1.0, 1.25]:
                for method_idx in range(4):
                    profit = -objective([area, irr_level, method_idx])
                    if profit > best_profit:
                        best_profit = profit
                        best_solution = [area, irr_level, method_idx]
        
        return {
            "optimal_solution": best_solution,
            "optimal_objective": best_profit,
            "method": "Grid Search"
        }

    def _annualize_cost(self, capital_cost: float) -> float:
        """Annualize capital cost using annuity factor"""
        r = self.params.discount_rate
        n = self.params.project_lifetime
        annuity_factor = (r * (1 + r)**n) / ((1 + r)**n - 1)
        return capital_cost * annuity_factor

    def plot_pareto_front(self, pareto_objectives: np.ndarray) -> go.Figure:
        """Plot Pareto front for multi-objective optimization"""
        if pareto_objectives.shape[1] >= 2:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=pareto_objectives[:, 1],  # Water use
                y=-pareto_objectives[:, 0],  # Profit (negated back)
                mode='markers',
                marker=dict(
                    size=8,
                    color=-pareto_objectives[:, 2],  # Land productivity
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Land Productivity ($/ha)")
                ),
                text=[f"Profit: ${-p[0]:.0f}<br>Water: {p[1]:.0f} m³<br>Land Prod: ${-p[2]:.0f}/ha" 
                      for p in pareto_objectives],
                name='Pareto Solutions'
            ))
            
            fig.update_layout(
                title="Pareto Front: Profit vs Water Use",
                xaxis_title="Water Use (m³)",
                yaxis_title="Profit ($)",
                height=500
            )
            
            return fig
        else:
            return go.Figure()

    def plot_cost_breakdown(self, costs: Dict[str, float]) -> go.Figure:
        """Plot cost breakdown pie chart"""
        cost_items = {
            "Water Cost": costs["water_cost"],
            "Seed Cost": costs["seed_cost"],
            "Fertilizer Cost": costs["fertilizer_cost"],
            "Labor Cost": costs["labor_cost"],
            "Energy Cost": costs["energy_cost"],
            "Capital Cost": costs["annualized_capital"],
            "Maintenance Cost": costs["maintenance_cost"]
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(cost_items.keys()),
            values=list(cost_items.values()),
            hole=0.3
        )])
        
        fig.update_layout(
            title="Cost Breakdown",
            height=400
        )
        
        return fig

    def render_economic_interface(self) -> Dict[str, Any]:
        """Render the complete economic optimization interface"""
        st.subheader("💰 Economic Optimization Module")
        
        # Input parameters
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**Economic Parameters**")
            crop_type = st.selectbox("Crop Type", list(self.crop_params.keys()))
            
            st.write("**Input Costs (USD)**")
            water_cost = st.number_input("Water Cost ($/m³)", 0.01, 0.50, self.params.water_cost_per_m3, 0.01)
            seed_cost = st.number_input("Seed Cost ($/ha)", 50.0, 500.0, self.params.seed_cost_per_ha, 10.0)
            fertilizer_cost = st.number_input("Fertilizer Cost ($/ha)", 50.0, 500.0, self.params.fertilizer_cost_per_ha, 10.0)
            labor_cost = st.number_input("Labor Cost ($/ha)", 50.0, 500.0, self.params.labor_cost_per_ha, 10.0)
            
            st.write("**Crop Market Prices (USD/ton)**")
            crop_price = st.number_input(
                f"{crop_type} Price", 
                100.0, 1000.0, self.crop_params[crop_type]["price"], 10.0
            )
            
            st.write("**Optimization Settings**")
            optimization_method = st.selectbox(
                "Optimization Method",
                ["NSGA-II (Multi-objective)", "Differential Evolution", "Grid Search"]
            )
            
            # Update parameters based on user input
            self.params.water_cost_per_m3 = water_cost
            self.params.seed_cost_per_ha = seed_cost
            self.params.fertilizer_cost_per_ha = fertilizer_cost
            self.params.labor_cost_per_ha = labor_cost
            self.crop_params[crop_type]["price"] = crop_price
        
        with col2:
            st.write("**Profitability Analysis**")
            
            # Scenario analysis
            area_ha = st.slider("Area (ha)", 1, 200, 50)
            irrigation_level = st.slider("Irrigation Level", 0.3, 1.5, 1.0)
            irrigation_method = st.selectbox("Irrigation Method", list(self.irrigation_costs.keys()))
            
            # Calculate current scenario
            crop_info = self.crop_params[crop_type]
            actual_yield = crop_info["yield_potential"] * min(1.0, irrigation_level)
            total_water = (crop_info["water_requirement"] * area_ha * irrigation_level) / 1000
            
            costs = self.calculate_production_costs(crop_type, area_ha, irrigation_method, total_water)
            profitability = self.calculate_profitability_metrics(crop_type, area_ha, irrigation_method, actual_yield, total_water)
            
            # Display metrics
            col2a, col2b, col2c = st.columns(3)
            col2a.metric("Revenue", f"${profitability['revenue']:,.0f}")
            col2b.metric("Net Profit", f"${profitability['net_profit']:,.0f}")
            col2c.metric("Profit Margin", f"{profitability['profit_margin']:.1f}%")
            
            col2d, col2e, col2f = st.columns(3)
            col2d.metric("Economic WP", f"${profitability['economic_water_productivity']:.2f}/m³")
            col2e.metric("Land Productivity", f"${profitability['land_productivity']:.0f}/ha")
            col2f.metric("ROI", f"{profitability['return_on_investment']:.1f}%")
            
            # Cost breakdown
            st.write("**Cost Breakdown:**")
            cost_fig = self.plot_cost_breakdown(costs)
            st.plotly_chart(cost_fig, use_container_width=True)
            
            # Optimization
            st.write("**Economic Optimization:**")
            if st.button("Run Economic Optimization"):
                with st.spinner("Running economic optimization..."):
                    method_map = {
                        "NSGA-II (Multi-objective)": "nsga2",
                        "Differential Evolution": "differential_evolution",
                        "Grid Search": "single_objective"
                    }
                    
                    results = self.optimize_economic_objectives(
                        crop_type, method_map[optimization_method]
                    )
                    
                    st.session_state.economic_results = results
                    
                    if optimization_method == "NSGA-II (Multi-objective)" and "pareto_objectives" in results:
                        st.success("Multi-objective optimization completed!")
                        
                        # Plot Pareto front
                        pareto_fig = self.plot_pareto_front(results["pareto_objectives"])
                        st.plotly_chart(pareto_fig, use_container_width=True)
                        
                        # Display Pareto solutions table
                        if "solutions_df" in results:
                            st.write("**Pareto Solutions:**")
                            st.dataframe(results["solutions_df"], use_container_width=True)
                    
                    else:
                        st.success("Optimization completed!")
                        
                        if "optimal_solution" in results:
                            opt_area, opt_irr_level, opt_method_idx = results["optimal_solution"]
                            irrigation_methods = list(self.irrigation_costs.keys())
                            opt_method = irrigation_methods[int(opt_method_idx)]
                            
                            col_opt1, col_opt2, col_opt3 = st.columns(3)
                            col_opt1.metric("Optimal Area", f"{opt_area:.1f} ha")
                            col_opt2.metric("Optimal Irrigation", f"{opt_irr_level:.2f}")
                            col_opt3.metric("Optimal Method", opt_method)
                            
                            col_opt4, col_opt5 = st.columns(2)
                            col_opt4.metric("Max Profit", f"${results['optimal_objective']:,.0f}")
                            col_opt5.metric("Method", results["method"])
            
            # Sensitivity analysis
            st.write("**Sensitivity Analysis:**")
            sensitivity_param = st.selectbox(
                "Parameter for Sensitivity",
                ["water_cost_per_m3", "seed_cost_per_ha", "crop_price"]
            )
            
            if st.button("Run Sensitivity Analysis"):
                param_values = np.linspace(0.5, 2.0, 20)  # ±50% variation
                profits = []
                
                original_value = getattr(self.params, sensitivity_param, 
                                       self.crop_params[crop_type].get("price", 300))
                
                for multiplier in param_values:
                    if sensitivity_param in self.crop_params[crop_type]:
                        self.crop_params[crop_type]["price"] = original_value * multiplier
                    else:
                        setattr(self.params, sensitivity_param, original_value * multiplier)
                    
                    # Recalculate profitability
                    new_profitability = self.calculate_profitability_metrics(
                        crop_type, area_ha, irrigation_method, actual_yield, total_water
                    )
                    profits.append(new_profitability["net_profit"])
                
                # Restore original value
                if sensitivity_param in self.crop_params[crop_type]:
                    self.crop_params[crop_type]["price"] = original_value
                else:
                    setattr(self.params, sensitivity_param, original_value)
                
                # Plot sensitivity
                fig_sensitivity = go.Figure()
                fig_sensitivity.add_trace(go.Scatter(
                    x=param_values * 100,  # Convert to percentage
                    y=profits,
                    mode='lines+markers',
                    name='Net Profit'
                ))
                
                fig_sensitivity.update_layout(
                    title=f"Sensitivity Analysis: {sensitivity_param.replace('_', ' ').title()}",
                    xaxis_title="Parameter Change (%)",
                    yaxis_title="Net Profit ($)",
                    height=400
                )
                
                st.plotly_chart(fig_sensitivity, use_container_width=True)
        
        return {
            "crop_type": crop_type,
            "area_ha": area_ha,
            "irrigation_level": irrigation_level,
            "irrigation_method": irrigation_method,
            "profitability": profitability
        }


def get_economic_optimizer() -> EconomicOptimizer:
    """Get economic optimizer instance"""
    if 'economic_optimizer' not in st.session_state:
        st.session_state.economic_optimizer = EconomicOptimizer()
    return st.session_state.economic_optimizer
