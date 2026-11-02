"""
Climate Change Projection Downscaling Module for TOKUMA 3-in-1 Research Platform
Provides CMIP6 integration, statistical downscaling, and weather generation capabilities
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xarray as xr
import netCDF4 as nc


class ClimateDownscaling:
    """Advanced climate downscaling and weather generation"""
    
    def __init__(self):
        # CMIP6 scenarios and models
        self.cmip6_scenarios = {
            "SSP1-2.6": {"description": "Sustainability - Low emissions", "warming": 1.8},
            "SSP2-4.5": {"description": "Middle of the road - Medium emissions", "warming": 2.7},
            "SSP5-8.5": {"description": "Fossil-fueled development - High emissions", "warming": 4.4}
        }
        
        self.cmip6_models = [
            "GFDL-ESM4", "UKESM1-0-LL", "MPI-ESM1-2-HR", "CNRM-CM6-1",
            "CanESM5", "MIROC6", "IPSL-CM6A-LR", "CESM2"
        ]
        
        # Ethiopian climate zones with baseline parameters
        self.ethiopia_climate_zones = {
            "Highlands": {
                "lat_range": (8, 13), "lon_range": (36, 40),
                "elevation_range": (1500, 4500),
                "baseline_temp": 18.5, "baseline_rainfall": 1200,
                "temp_seasonality": 8.0, "rainfall_seasonality": 0.7
            },
            "Lowlands": {
                "lat_range": (4, 8), "lon_range": (38, 42),
                "elevation_range": (500, 1500),
                "baseline_temp": 27.5, "baseline_rainfall": 600,
                "temp_seasonality": 5.0, "rainfall_seasonality": 0.8
            },
            "Western": {
                "lat_range": (5, 10), "lon_range": (33, 36),
                "elevation_range": (500, 2000),
                "baseline_temp": 24.0, "baseline_rainfall": 1500,
                "temp_seasonality": 6.0, "rainfall_seasonality": 0.6
            },
            "Eastern": {
                "lat_range": (5, 12), "lon_range": (40, 48),
                "elevation_range": (200, 1500),
                "baseline_temp": 26.0, "baseline_rainfall": 400,
                "temp_seasonality": 7.0, "rainfall_seasonality": 0.9
            }
        }
        
        # Weather generation parameters
        self.weather_generator_params = {
            "temperature": {
                "annual_amplitude": 5.0,
                "daily_std": 3.0,
                "autocorrelation": 0.7
            },
            "rainfall": {
                "dry_spell_mean": 5.0,
                "wet_spell_mean": 3.0,
                "intensity_shape": 0.8,
                "intensity_scale": 10.0
            }
        }

    def generate_synthetic_cmip6_data(self, scenario: str, model: str, 
                                    years: Tuple[int, int], 
                                    variables: List[str]) -> pd.DataFrame:
        """Generate synthetic CMIP6 data for demonstration"""
        np.random.seed(42)
        
        start_year, end_year = years
        n_years = end_year - start_year + 1
        
        # Get scenario parameters
        scenario_info = self.cmip6_scenarios[scenario]
        warming_trend = scenario_info["warming"] / 100  # °C per year
        
        # Generate time series
        dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq='D')
        n_days = len(dates)
        
        data = pd.DataFrame({'date': dates})
        
        # Temperature generation
        if 'temperature' in variables:
            # Base temperature with seasonal cycle
            day_of_year = dates.dayofyear
            seasonal_temp = 20 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            
            # Add warming trend
            year_fraction = (dates.year - start_year) + (dates.dayofyear - 1) / 365
            warming = warming_trend * year_fraction
            
            # Add interannual variability
            interannual = np.sin(2 * np.pi * year_fraction / 11) * 0.5  # 11-year cycle
            
            # Add daily variability
            daily_noise = np.random.normal(0, 2, n_days)
            
            # Add autocorrelation
            for i in range(1, n_days):
                daily_noise[i] += 0.7 * daily_noise[i-1]
            
            data['temperature'] = seasonal_temp + warming + interannual + daily_noise
        
        # Rainfall generation
        if 'rainfall' in variables:
            # Generate occurrence (wet/dry days)
            prob_wet = 0.3 + 0.2 * np.sin(2 * np.pi * (dates.dayofyear - 100) / 365)
            wet_days = np.random.random(n_days) < prob_wet
            
            # Generate intensity for wet days
            intensity = np.zeros(n_days)
            intensity[wet_days] = np.random.gamma(
                shape=self.weather_generator_params["rainfall"]["intensity_shape"],
                scale=self.weather_generator_params["rainfall"]["intensity_scale"],
                size=np.sum(wet_days)
            )
            
            # Add climate change effect on rainfall
            rainfall_change = 1 - (warming_trend * year_fraction * 0.05)  # 5% change per °C
            data['rainfall'] = intensity * rainfall_change
        
        # Other variables
        if 'humidity' in variables:
            # Inverse relationship with temperature
            base_humidity = 70 - (data['temperature'] - 20) * 2
            data['humidity'] = np.clip(base_humidity + np.random.normal(0, 10, n_days), 20, 100)
        
        if 'solar_radiation' in variables:
            # Seasonal solar radiation
            day_angle = 2 * np.pi * (dates.dayofyear - 80) / 365
            base_radiation = 250 + 150 * np.cos(day_angle)
            cloud_effect = 1 - (data['rainfall'] > 0) * 0.7
            data['solar_radiation'] = base_radiation * cloud_effect + np.random.normal(0, 20, n_days)
        
        if 'wind_speed' in variables:
            data['wind_speed'] = np.random.gamma(2, 1.5, n_days) + 1
        
        return data

    def statistical_downscaling(self, large_scale_data: pd.DataFrame,
                             target_location: Tuple[float, float],
                             method: str = "delta") -> pd.DataFrame:
        """Perform statistical downscaling from large-scale to local scale"""
        
        # Get climate zone for target location
        lat, lon = target_location
        climate_zone = self._get_climate_zone(lat, lon)
        zone_params = self.ethiopia_climate_zones[climate_zone]
        
        downscaled = large_scale_data.copy()
        
        if method == "delta":
            # Delta method: apply changes to local baseline
            for col in ['temperature', 'rainfall', 'humidity', 'solar_radiation']:
                if col in downscaled.columns:
                    if col == 'temperature':
                        # Add delta to baseline temperature
                        baseline = zone_params['baseline_temp']
                        seasonal_cycle = zone_params['temp_seasonality'] * \
                                       np.sin(2 * np.pi * (downscaled['date'].dt.dayofyear - 80) / 365)
                        downscaled[col] = baseline + seasonal_cycle + \
                                        (downscaled[col] - downscaled[col].mean())
                    elif col == 'rainfall':
                        # Apply percentage change to baseline
                        baseline = zone_params['baseline_rainfall']
                        seasonal_factor = 1 + zone_params['rainfall_seasonality'] * \
                                        np.sin(2 * np.pi * (downscaled['date'].dt.dayofyear - 100) / 365)
                        change_factor = downscaled[col] / downscaled[col].mean()
                        downscaled[col] = baseline / 365 * seasonal_factor * change_factor
                    else:
                        # Apply relative changes
                        downscaled[col] = downscaled[col] * \
                                        (zone_params[f'baseline_{col}'] / downscaled[col].mean() 
                                         if f'baseline_{col}' in zone_params else 1)
        
        elif method == "quantile_mapping":
            # Quantile mapping approach
            for col in ['temperature', 'rainfall']:
                if col in downscaled.columns:
                    # Generate synthetic observations for calibration
                    n_obs = len(downscaled)
                    if col == 'temperature':
                        obs = np.random.normal(
                            zone_params['baseline_temp'], 
                            zone_params['temp_seasonality'], 
                            n_obs
                        )
                    else:
                        obs = np.random.exponential(
                            zone_params['baseline_rainfall'] / 365, 
                            n_obs
                        )
                    
                    # Simple quantile mapping (would need more sophisticated implementation)
                    obs_percentiles = np.percentile(obs, np.arange(1, 100))
                    model_percentiles = np.percentile(downscaled[col], np.arange(1, 100))
                    
                    # Apply mapping (simplified)
                    downscaled[col] = np.interp(downscaled[col], model_percentiles, obs_percentiles)
        
        return downscaled

    def weather_generator(self, climate_data: pd.DataFrame, 
                        generation_years: int,
                        preserve_statistics: bool = True) -> pd.DataFrame:
        """Generate synthetic daily weather from climate projections"""
        
        if preserve_statistics:
            # Calculate statistics from input data
            stats = {}
            for col in ['temperature', 'rainfall', 'humidity', 'solar_radiation', 'wind_speed']:
                if col in climate_data.columns:
                    if col == 'rainfall':
                        stats[col] = {
                            'wet_day_freq': (climate_data[col] > 0.1).mean(),
                            'wet_day_mean': climate_data[climate_data[col] > 0.1][col].mean(),
                            'wet_day_std': climate_data[climate_data[col] > 0.1][col].std(),
                            'dry_spell_length': self._calculate_spell_lengths(climate_data[col] <= 0.1).mean(),
                            'wet_spell_length': self._calculate_spell_lengths(climate_data[col] > 0.1).mean()
                        }
                    else:
                        stats[col] = {
                            'mean': climate_data[col].mean(),
                            'std': climate_data[col].std(),
                            'autocorr': self._calculate_autocorrelation(climate_data[col])
                        }
        
        # Generate new time series
        start_date = climate_data['date'].max() + pd.Timedelta(days=1)
        dates = pd.date_range(start_date, periods=generation_years * 365, freq='D')
        
        generated_data = pd.DataFrame({'date': dates})
        
        # Temperature generation
        if 'temperature' in climate_data.columns and preserve_statistics:
            temp_stats = stats['temperature']
            temp = np.random.normal(temp_stats['mean'], temp_stats['std'], len(dates))
            
            # Add autocorrelation
            for i in range(1, len(temp)):
                temp[i] += temp_stats['autocorr'] * temp[i-1]
            
            # Add seasonal cycle
            day_of_year = dates.dayofyear
            seasonal = 5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            generated_data['temperature'] = temp + seasonal
        
        # Rainfall generation (Markov chain approach)
        if 'rainfall' in climate_data.columns and preserve_statistics:
            rain_stats = stats['rainfall']
            
            # Markov chain for occurrence
            p_wet_wet = 0.6  # Probability of wet day given wet day
            p_wet_dry = rain_stats['wet_day_freq'] * (1 - p_wet_wet) / (1 - rain_stats['wet_day_freq'])
            
            rainfall = np.zeros(len(dates))
            is_wet = np.random.random(len(dates)) < rain_stats['wet_day_freq']
            
            for i in range(1, len(dates)):
                if is_wet[i-1]:
                    is_wet[i] = np.random.random() < p_wet_wet
                else:
                    is_wet[i] = np.random.random() < p_wet_dry
            
            # Generate intensity for wet days
            n_wet = np.sum(is_wet)
            if n_wet > 0:
                rainfall[is_wet] = np.random.gamma(
                    shape=2,  # Shape parameter
                    scale=rain_stats['wet_day_mean'] / 2,
                    size=n_wet
                )
            
            generated_data['rainfall'] = rainfall
        
        # Generate other variables
        for col in ['humidity', 'solar_radiation', 'wind_speed']:
            if col in climate_data.columns and preserve_statistics:
                col_stats = stats[col]
                generated_data[col] = np.random.normal(
                    col_stats['mean'], 
                    col_stats['std'], 
                    len(dates)
                )
                
                # Add autocorrelation
                if col_stats['autocorr'] > 0:
                    for i in range(1, len(generated_data)):
                        generated_data[col].iloc[i] += col_stats['autocorr'] * generated_data[col].iloc[i-1]
        
        return generated_data

    def _get_climate_zone(self, lat: float, lon: float) -> str:
        """Determine climate zone based on coordinates"""
        for zone, params in self.ethiopia_climate_zones.items():
            if (params['lat_range'][0] <= lat <= params['lat_range'][1] and
                params['lon_range'][0] <= lon <= params['lon_range'][1]):
                return zone
        return "Highlands"  # Default

    def _calculate_spell_lengths(self, condition: pd.Series) -> np.ndarray:
        """Calculate lengths of consecutive True values in boolean series"""
        spell_lengths = []
        current_length = 0
        
        for val in condition:
            if val:
                current_length += 1
            else:
                if current_length > 0:
                    spell_lengths.append(current_length)
                current_length = 0
        
        if current_length > 0:
            spell_lengths.append(current_length)
        
        return np.array(spell_lengths) if spell_lengths else np.array([1])

    def _calculate_autocorrelation(self, series: pd.Series, lag: int = 1) -> float:
        """Calculate autocorrelation at given lag"""
        return series.autocorr(lag=lag) if len(series) > lag else 0

    def plot_climate_projections(self, historical_data: pd.DataFrame,
                               projected_data: pd.DataFrame,
                               variable: str) -> go.Figure:
        """Plot historical vs projected climate data"""
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Annual Time Series', 'Monthly Climatology'),
            vertical_spacing=0.1
        )
        
        # Annual time series
        hist_annual = historical_data.groupby(historical_data['date'].dt.year)[variable].mean()
        proj_annual = projected_data.groupby(projected_data['date'].dt.year)[variable].mean()
        
        fig.add_trace(go.Scatter(
            x=hist_annual.index, y=hist_annual.values,
            mode='lines+markers', name='Historical',
            line=dict(color='blue')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=proj_annual.index, y=proj_annual.values,
            mode='lines+markers', name='Projected',
            line=dict(color='red', dash='dash')
        ), row=1, col=1)
        
        # Monthly climatology
        hist_monthly = historical_data.groupby(historical_data['date'].dt.month)[variable].mean()
        proj_monthly = projected_data.groupby(projected_data['date'].dt.month)[variable].mean()
        
        fig.add_trace(go.Scatter(
            x=hist_monthly.index, y=hist_monthly.values,
            mode='lines+markers', name='Historical Monthly',
            line=dict(color='blue')
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=proj_monthly.index, y=proj_monthly.values,
            mode='lines+markers', name='Projected Monthly',
            line=dict(color='red', dash='dash')
        ), row=2, col=1)
        
        fig.update_layout(
            title=f'Climate Projections: {variable.replace("_", " ").title()}',
            height=600
        )
        
        return fig

    def plot_downscaling_comparison(self, large_scale: pd.DataFrame,
                                 downscaled: pd.DataFrame,
                                 variable: str) -> go.Figure:
        """Plot comparison between large-scale and downscaled data"""
        
        fig = go.Figure()
        
        # Large-scale data
        fig.add_trace(go.Scatter(
            x=large_scale['date'], y=large_scale[variable],
            mode='lines', name='Large-scale',
            line=dict(color='blue', width=1),
            opacity=0.7
        ))
        
        # Downscaled data
        fig.add_trace(go.Scatter(
            x=downscaled['date'], y=downscaled[variable],
            mode='lines', name='Downscaled',
            line=dict(color='red', width=1)
        ))
        
        fig.update_layout(
            title=f'Downscaling Comparison: {variable.replace("_", " ").title()}',
            xaxis_title='Date',
            yaxis_title=variable.replace("_", " ").title(),
            height=400
        )
        
        return fig

    def render_climate_downscaling_interface(self) -> Dict[str, Any]:
        """Render the complete climate downscaling interface"""
        st.subheader("🌡️ Climate Change Projection Downscaling")
        
        # Configuration
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**CMIP6 Configuration**")
            scenario = st.selectbox("SSP Scenario", list(self.cmip6_scenarios.keys()))
            model = st.selectbox("GCM Model", self.cmip6_models)
            
            st.write("**Temporal Settings**")
            historical_start = st.number_input("Historical Start Year", 1980, 2000, 1990, step=1)
            historical_end = st.number_input("Historical End Year", 2010, 2020, 2020, step=1)
            projection_end = st.number_input("Projection End Year", 2030, 2100, 2050, step=1)
            
            st.write("**Downscaling Settings**")
            target_lat = st.number_input("Target Latitude", 3.0, 15.0, 8.5)
            target_lon = st.number_input("Target Longitude", 33.0, 48.0, 38.5)
            downscaling_method = st.selectbox("Downscaling Method", ["delta", "quantile_mapping"])
            
            st.write("**Weather Generation**")
            generate_weather = st.checkbox("Generate Synthetic Weather", value=True)
            generation_years = st.slider("Generation Years", 1, 30, 10) if generate_weather else 0
            
            variables = st.multiselect(
                "Variables to Process",
                ["temperature", "rainfall", "humidity", "solar_radiation", "wind_speed"],
                default=["temperature", "rainfall"]
            )
        
        with col2:
            if st.button("Run Climate Downscaling"):
                with st.spinner("Generating CMIP6 data and performing downscaling..."):
                    # Generate synthetic CMIP6 data
                    historical_data = self.generate_synthetic_cmip6_data(
                        scenario, model, (historical_start, historical_end), variables
                    )
                    
                    projected_data = self.generate_synthetic_cmip6_data(
                        scenario, model, (historical_end + 1, projection_end), variables
                    )
                    
                    # Perform downscaling
                    downscaled_data = self.statistical_downscaling(
                        projected_data, (target_lat, target_lon), downscaling_method
                    )
                    
                    # Store in session state
                    st.session_state.climate_historical = historical_data
                    st.session_state.climate_projected = projected_data
                    st.session_state.climate_downscaled = downscaled_data
                    
                    st.success("Climate downscaling completed successfully!")
            
            # Display results if available
            if 'climate_downscaled' in st.session_state:
                historical = st.session_state.climate_historical
                projected = st.session_state.climate_projected
                downscaled = st.session_state.climate_downscaled
                
                # Summary statistics
                st.write("**Summary Statistics:**")
                
                for var in variables:
                    if var in downscaled.columns:
                        col2a, col2b, col2c = st.columns(3)
                        
                        hist_mean = historical[var].mean()
                        proj_mean = downscaled[var].mean()
                        change = ((proj_mean - hist_mean) / hist_mean) * 100
                        
                        col2a.metric(f"Historical {var}", f"{hist_mean:.2f}")
                        col2b.metric(f"Projected {var}", f"{proj_mean:.2f}")
                        col2c.metric(f"Change %", f"{change:+.1f}%")
                
                # Plotting
                selected_var = st.selectbox("Select Variable for Visualization", variables)
                
                # Historical vs Projected
                st.write("**Historical vs Projected Climate:**")
                projection_fig = self.plot_climate_projections(
                    historical, downscaled, selected_var
                )
                st.plotly_chart(projection_fig, use_container_width=True)
                
                # Downscaling comparison
                st.write("**Downscaling Comparison:**")
                comparison_fig = self.plot_downscaling_comparison(
                    projected, downscaled, selected_var
                )
                st.plotly_chart(comparison_fig, use_container_width=True)
                
                # Weather generation
                if generate_weather and generation_years > 0:
                    st.write("**Synthetic Weather Generation:**")
                    
                    if st.button("Generate Synthetic Weather"):
                        with st.spinner("Generating synthetic weather data..."):
                            synthetic_weather = self.weather_generator(
                                downscaled, generation_years
                            )
                            
                            st.session_state.synthetic_weather = synthetic_weather
                            st.success(f"Generated {generation_years} years of synthetic weather!")
                    
                    if 'synthetic_weather' in st.session_state:
                        synthetic = st.session_state.synthetic_weather
                        
                        # Display synthetic weather statistics
                        st.write("**Synthetic Weather Statistics:**")
                        
                        for var in ['temperature', 'rainfall']:
                            if var in synthetic.columns:
                                col3a, col3b = st.columns(2)
                                col3a.metric(f"Mean {var}", f"{synthetic[var].mean():.2f}")
                                col3b.metric(f"Std {var}", f"{synthetic[var].std():.2f}")
                        
                        # Plot synthetic weather
                        fig_synthetic = go.Figure()
                        fig_synthetic.add_trace(go.Scatter(
                            x=synthetic['date'], y=synthetic[selected_var],
                            mode='lines', name='Synthetic Weather',
                            line=dict(color='green')
                        ))
                        fig_synthetic.update_layout(
                            title=f'Synthetic Weather: {selected_var.replace("_", " ").title()}',
                            xaxis_title='Date',
                            yaxis_title=selected_var.replace("_", " ").title()
                        )
                        st.plotly_chart(fig_synthetic, use_container_width=True)
                        
                        # Download synthetic weather
                        csv_data = synthetic.to_csv(index=False)
                        st.download_button(
                            label="Download Synthetic Weather (CSV)",
                            data=csv_data,
                            file_name=f"synthetic_weather_{scenario}_{model}.csv",
                            mime="text/csv"
                        )
                
                # Climate zone information
                climate_zone = self._get_climate_zone(target_lat, target_lon)
                zone_info = self.ethiopia_climate_zones[climate_zone]
                
                st.write("**Target Location Climate Zone:**")
                col4a, col4b = st.columns(2)
                col4a.metric("Climate Zone", climate_zone)
                col4b.metric("Baseline Temp", f"{zone_info['baseline_temp']:.1f}°C")
                col4a.metric("Baseline Rainfall", f"{zone_info['baseline_rainfall']:.0f} mm")
                col4b.metric("Temp Seasonality", f"{zone_info['temp_seasonality']:.1f}°C")
        
        return {
            "scenario": scenario,
            "model": model,
            "target_location": (target_lat, target_lon),
            "climate_zone": climate_zone if 'climate_zone' in locals() else None
        }


def get_climate_downscaling() -> ClimateDownscaling:
    """Get climate downscaling instance"""
    if 'climate_downscaling' not in st.session_state:
        st.session_state.climate_downscaling = ClimateDownscaling()
    return st.session_state.climate_downscaling
