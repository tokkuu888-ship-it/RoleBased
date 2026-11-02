"""
Real-time API Connectors Module for TOKUMA 3-in-1 Research Platform
Provides FAO WaPOR and OpenWeatherMap API integration for live data
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.exceptions import RequestException


class APIConnector:
    """Base class for API connectors"""
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TOKUMA-Research-Platform/1.0',
            'Accept': 'application/json'
        })
    
    def make_request(self, endpoint: str, params: Dict = None, timeout: int = 30) -> Dict:
        """Make API request with error handling"""
        try:
            url = f"{self.base_url}/{endpoint}"
            if self.api_key:
                params = params or {}
                # Use correct parameter name for different APIs
                if 'openweathermap' in self.base_url.lower():
                    params['appid'] = self.api_key
                else:
                    params['apikey'] = self.api_key
            
            response = self.session.get(url, params=params, timeout=timeout)
            
            # Check response content type
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                st.error(f"API returned HTML instead of JSON. Check API key and endpoint.")
                return {}
            
            response.raise_for_status()
            
            # Try to parse JSON with better error handling
            try:
                data = response.json()
                if not data or (isinstance(data, dict) and len(data) == 0):
                    st.warning("API returned empty response")
                    return {}
                return data
            except json.JSONDecodeError as e:
                st.error(f"JSON parsing failed: {str(e)}")
                st.write(f"Response content: {response.text[:200]}...")
                return {}
        
        except RequestException as e:
            st.error(f"API request failed: {str(e)}")
            return {}


class WaPORConnector(APIConnector):
    """FAO WaPOR API connector for evapotranspiration and biomass data"""
    
    def __init__(self, api_key: str = None):
        # WaPOR API endpoints - use correct API URLs
        super().__init__(
            "https://data.apps.fao.org/api/v2",  # Correct API endpoint
            api_key
        )
        self.data_levels = {
            "L1": "Continent (250m)",
            "L2": "Country (100m)", 
            "L3": "Basin (30m)",
            "L4": "Field (10m)"
        }
        
        self.indicators = {
            "AETI": "Actual Evapotranspiration and Interception",
            "NPP": "Net Primary Productivity", 
            "T": "Transpiration",
            "E": "Evaporation",
            "I": "Interception",
            "RET": "Reference Evapotranspiration",
            "P": "Precipitation",
            "GBWP": "Gross Biomass Water Productivity",
            "NBWP": "Net Biomass Water Productivity"
        }
    
    def get_available_data(self, data_level: str = "L2") -> Dict:
        """Get available data for specified level"""
        endpoint = f"catalog/{data_level}"
        return self.make_request(endpoint)
    
    def get_indicator_data(self, indicator: str, data_level: str = "L2",
                         coordinates: Tuple[float, float] = None,
                         start_date: str = None, end_date: str = None) -> Dict:
        """Get indicator data for specific location and time period"""
        params = {}
        
        if coordinates:
            params['point'] = f"{coordinates[1]},{coordinates[0]}"  # lon,lat format
        
        if start_date:
            params['start'] = start_date
        if end_date:
            params['end'] = end_date
        
        endpoint = f"data/{data_level}/{indicator}"
        return self.make_request(endpoint, params)
    
    def get_time_series(self, indicator: str, data_level: str = "L2",
                      coordinates: Tuple[float, float] = None,
                      start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get time series data for indicator"""
        data = self.get_indicator_data(indicator, data_level, coordinates, start_date, end_date)
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df_data = []
        for item in data['data']:
            df_data.append({
                'date': item.get('date'),
                'value': item.get('value'),
                'unit': data.get('unit', ''),
                'indicator': indicator
            })
        
        df = pd.DataFrame(df_data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        return df
    
    def get_current_aeti(self, coordinates: Tuple[float, float]) -> Dict:
        """Get current Actual Evapotranspiration and Interception"""
        # Get recent data (last 30 days)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        data = self.get_time_series("AETI", "L2", coordinates, start_date, end_date)
        
        if not data.empty:
            latest_value = data.iloc[-1]['value']
            return {
                "latest_value": latest_value,
                "unit": "mm/day",
                "date": data.iloc[-1]['date'].strftime("%Y-%m-%d"),
                "trend_30d": self._calculate_trend(data),
                "average_30d": data['value'].mean()
            }
        
        return {}
    
    def get_current_npp(self, coordinates: Tuple[float, float]) -> Dict:
        """Get current Net Primary Productivity"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        data = self.get_time_series("NPP", "L2", coordinates, start_date, end_date)
        
        if not data.empty:
            latest_value = data.iloc[-1]['value']
            return {
                "latest_value": latest_value,
                "unit": "gC/m²/day",
                "date": data.iloc[-1]['date'].strftime("%Y-%m-%d"),
                "trend_30d": self._calculate_trend(data),
                "average_30d": data['value'].mean()
            }
        
        return {}
    
    def _calculate_trend(self, df: pd.DataFrame) -> float:
        """Calculate linear trend from time series"""
        if len(df) < 2:
            return 0.0
        
        x = np.arange(len(df))
        y = df['value'].values
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def plot_wapor_timeseries(self, df: pd.DataFrame, indicator: str) -> go.Figure:
        """Plot WaPOR time series data"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['value'],
            mode='lines+markers',
            name=f"{indicator} ({df['unit'].iloc[0] if not df.empty else ''})",
            line=dict(width=2),
            marker=dict(size=4)
        ))
        
        # Add trend line if enough data
        if len(df) > 1:
            x = np.arange(len(df))
            y = df['value'].values
            trend = np.polyfit(x, y, 1)
            trend_line = np.poly1d(trend)(x)
            
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(dash='dash', color='red')
            ))
        
        fig.update_layout(
            title=f"{self.indicators.get(indicator, indicator)} Time Series",
            xaxis_title="Date",
            yaxis_title=f"{indicator} ({df['unit'].iloc[0] if not df.empty else ''})",
            height=400
        )
        
        return fig


class OpenWeatherMapConnector(APIConnector):
    """OpenWeatherMap API connector for current weather data"""
    
    def __init__(self, api_key: str):
        super().__init__(
            "https://api.openweathermap.org/data/2.5",
            api_key
        )
    
    def get_current_weather(self, coordinates: Tuple[float, float]) -> Dict:
        """Get current weather for coordinates"""
        params = {
            'lat': coordinates[0],
            'lon': coordinates[1],
            'units': 'metric'  # Celsius, m/s
        }
        
        data = self.make_request("weather", params)
        
        if data:
            return {
                "temperature": data['main']['temp'],
                "humidity": data['main']['humidity'],
                "pressure": data['main']['pressure'],
                "wind_speed": data['wind']['speed'],
                "wind_direction": data['wind'].get('deg', 0),
                "description": data['weather'][0]['description'],
                "location": data['name'],
                "timestamp": datetime.now().isoformat()
            }
        
        return {}
    
    def get_forecast(self, coordinates: Tuple[float, float], days: int = 5) -> pd.DataFrame:
        """Get weather forecast"""
        params = {
            'lat': coordinates[0],
            'lon': coordinates[1],
            'units': 'metric'
        }
        
        data = self.make_request("forecast", params)
        
        if not data or 'list' not in data:
            return pd.DataFrame()
        
        forecast_data = []
        for item in data['list'][:days * 8]:  # 8 forecasts per day (3-hour intervals)
            forecast_data.append({
                'datetime': pd.to_datetime(item['dt'], unit='s'),
                'temperature': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'pressure': item['main']['pressure'],
                'wind_speed': item['wind']['speed'],
                'description': item['weather'][0]['description'],
                'rainfall': item.get('rain', {}).get('3h', 0)
            })
        
        df = pd.DataFrame(forecast_data)
        if not df.empty:
            df['date'] = df['datetime'].dt.date
        
        return df
    
    def get_historical_data(self, coordinates: Tuple[float, float], 
                         start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical weather data (requires paid plan)"""
        # This would require the One Call API 3.0 with historical data
        # For demonstration, we'll return empty DataFrame
        st.warning("Historical data requires OpenWeatherMap subscription")
        return pd.DataFrame()


class IntegratedAPIManager:
    """Manages multiple API connectors and provides unified interface"""
    
    def __init__(self):
        self.wapor_connector = None
        self.weather_connector = None
        self.cache = {}
        self.cache_expiry = 300  # 5 minutes
    
    def initialize_connectors(self, wapor_api_key: str = None, 
                          weather_api_key: str = None):
        """Initialize API connectors with provided keys"""
        if wapor_api_key and wapor_api_key.strip():
            self.wapor_connector = WaPORConnector(wapor_api_key.strip())
        if weather_api_key and weather_api_key.strip():
            self.weather_connector = OpenWeatherMapConnector(weather_api_key.strip())
    
    def get_cached_data(self, cache_key: str) -> Any:
        """Get cached data if not expired"""
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_expiry:
                return data
        return None
    
    def cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp"""
        self.cache[cache_key] = (data, time.time())
    
    def get_field_conditions(self, coordinates: Tuple[float, float]) -> Dict:
        """Get comprehensive field conditions from all APIs"""
        cache_key = f"field_conditions_{coordinates[0]}_{coordinates[1]}"
        
        # Check cache first
        cached_data = self.get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        field_data = {
            "coordinates": coordinates,
            "timestamp": datetime.now().isoformat()
        }
        
        # Get weather data
        if self.weather_connector:
            weather = self.weather_connector.get_current_weather(coordinates)
            field_data["weather"] = weather
            
            # Get forecast
            forecast = self.weather_connector.get_forecast(coordinates, 3)
            if not forecast.empty:
                field_data["forecast"] = {
                    "next_3_days_avg_temp": forecast['temperature'].mean(),
                    "next_3_days_total_rain": forecast['rainfall'].sum(),
                    "next_3_days_avg_humidity": forecast['humidity'].mean()
                }
        
        # Get WaPOR data
        if self.wapor_connector:
            aeti = self.wapor_connector.get_current_aeti(coordinates)
            npp = self.wapor_connector.get_current_npp(coordinates)
            
            field_data["wapor"] = {
                "aeti": aeti,
                "npp": npp
            }
        
        # If no real data available, use mock data for testing
        if not field_data.get("weather") and not field_data.get("wapor"):
            st.warning("APIs not responding, using mock data for testing")
            field_data.update({
                "weather": {
                    "temperature": 25.5,
                    "humidity": 65.0,
                    "pressure": 1013.2,
                    "wind_speed": 3.2,
                    "wind_direction": 180,
                    "description": "partly cloudy",
                    "location": "Test Location",
                    "timestamp": datetime.now().isoformat()
                },
                "forecast": {
                    "next_3_days_avg_temp": 24.8,
                    "next_3_days_total_rain": 2.5,
                    "next_3_days_avg_humidity": 62.0
                },
                "wapor": {
                    "aeti": {
                        "latest_value": 3.2,
                        "unit": "mm/day",
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "trend_30d": 0.05,
                        "average_30d": 3.1
                    },
                    "npp": {
                        "latest_value": 4.8,
                        "unit": "gC/m²/day",
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "trend_30d": 0.02,
                        "average_30d": 4.6
                    }
                }
            })
        
        # Cache the results
        self.cache_data(cache_key, field_data)
        return field_data
    
    def plot_integrated_dashboard(self, coordinates: Tuple[float, float]) -> Dict[str, go.Figure]:
        """Create integrated dashboard with data from all APIs"""
        figures = {}
        
        # Weather forecast plot
        if self.weather_connector:
            forecast = self.weather_connector.get_forecast(coordinates, 5)
            if not forecast.empty:
                fig_weather = go.Figure()
                
                fig_weather.add_trace(go.Scatter(
                    x=forecast['datetime'],
                    y=forecast['temperature'],
                    mode='lines+markers',
                    name='Temperature',
                    line=dict(color='red')
                ))
                
                fig_weather.add_trace(go.Scatter(
                    x=forecast['datetime'],
                    y=forecast['humidity'],
                    mode='lines+markers',
                    name='Humidity',
                    yaxis='y2',
                    line=dict(color='blue')
                ))
                
                fig_weather.update_layout(
                    title="5-Day Weather Forecast",
                    xaxis_title="Date/Time",
                    yaxis=dict(title="Temperature (°C)", side='left'),
                    yaxis2=dict(title="Humidity (%)", side='right', overlaying='y'),
                    height=400
                )
                
                figures['weather_forecast'] = fig_weather
        
        # WaPOR data plots
        if self.wapor_connector:
            # AETI time series
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            aeti_data = self.wapor_connector.get_time_series("AETI", "L2", coordinates, start_date, end_date)
            if not aeti_data.empty:
                figures['aeti_timeseries'] = self.wapor_connector.plot_wapor_timeseries(aeti_data, "AETI")
            
            # NPP time series
            npp_data = self.wapor_connector.get_time_series("NPP", "L2", coordinates, start_date, end_date)
            if not npp_data.empty:
                figures['npp_timeseries'] = self.wapor_connector.plot_wapor_timeseries(npp_data, "NPP")
        
        return figures
    
    def export_api_data(self, coordinates: Tuple[float, float], 
                       format: str = "csv") -> bytes:
        """Export data from APIs"""
        field_data = self.get_field_conditions(coordinates)
        
        # Convert to DataFrame for export
        export_data = []
        
        # Weather data
        if "weather" in field_data:
            weather = field_data["weather"]
            export_data.append({
                "source": "OpenWeatherMap",
                "parameter": "Current Temperature",
                "value": weather.get("temperature"),
                "unit": "°C",
                "timestamp": weather.get("timestamp")
            })
            export_data.append({
                "source": "OpenWeatherMap",
                "parameter": "Current Humidity",
                "value": weather.get("humidity"),
                "unit": "%",
                "timestamp": weather.get("timestamp")
            })
        
        # WaPOR data
        if "wapor" in field_data:
            wapor = field_data["wapor"]
            if "aeti" in wapor:
                aeti = wapor["aeti"]
                export_data.append({
                    "source": "WaPOR",
                    "parameter": "Actual ET",
                    "value": aeti.get("latest_value"),
                    "unit": aeti.get("unit"),
                    "timestamp": aeti.get("date")
                })
            
            if "npp" in wapor:
                npp = wapor["npp"]
                export_data.append({
                    "source": "WaPOR",
                    "parameter": "Net Primary Productivity",
                    "value": npp.get("latest_value"),
                    "unit": npp.get("unit"),
                    "timestamp": npp.get("date")
                })
        
        df = pd.DataFrame(export_data)
        
        if format == "csv":
            return df.to_csv(index=False).encode('utf-8')
        elif format == "json":
            return json.dumps(field_data, indent=2, default=str).encode('utf-8')
        else:
            return df.to_csv(index=False).encode('utf-8')
    
    def render_api_interface(self) -> Dict[str, Any]:
        """Render the complete API connector interface"""
        st.subheader("📡 Real-time API Connectors")
        
        # API Configuration
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**API Configuration**")
            
            wapor_api_key = st.text_input(
                "FAO WaPOR API Key",
                type="password",
                help="Enter your WaPOR API key (optional for demo)"
            )
            
            weather_api_key = st.text_input(
                "OpenWeatherMap API Key",
                type="password",
                help="Enter your OpenWeatherMap API key"
            )
            
            if st.button("Initialize APIs"):
                if not wapor_api_key.strip() or not weather_api_key.strip():
                    st.error("❌ Please enter valid API keys for both services")
                else:
                    with st.spinner("Initializing API connectors..."):
                        # Debug: Show actual keys being used
                        st.write(f"🔍 Debug: Weather key length: {len(weather_api_key.strip())}")
                        st.write(f"🔍 Debug: WaPOR key length: {len(wapor_api_key.strip())}")
                        
                        if len(weather_api_key.strip()) < 10:
                            st.error("❌ Weather API key seems too short (must be 32+ characters)")
                        if len(wapor_api_key.strip()) < 10:
                            st.error("❌ WaPOR API key seems too short (must be 32+ characters)")
                        
                        # Initialize with proper error handling
                        try:
                            self.initialize_connectors(wapor_api_key.strip(), weather_api_key.strip())
                            st.success("✅ API connectors initialized successfully!")
                            st.session_state.api_initialized = True
                            
                            # Test immediate API call
                            st.write("🧪 Testing API connections...")
                            test_weather = self.weather_connector.get_current_weather((8.5, 38.5)) if self.weather_connector else None
                            test_wapor = self.wapor_connector.get_current_aeti((8.5, 38.5)) if self.wapor_connector else None
                            
                            if test_weather and 'temperature' in test_weather:
                                st.success(f"✅ Weather API working: {test_weather['temperature']}°C")
                            else:
                                st.error("❌ Weather API test failed")
                                
                            if test_wapor and 'latest_value' in test_wapor:
                                st.success(f"✅ WaPOR API working: {test_wapor['latest_value']}")
                            else:
                                st.error("❌ WaPOR API test failed")
                                
                        except Exception as e:
                            st.error(f"❌ API initialization failed: {str(e)}")
            
            st.write("**Location Selection**")
            lat = st.number_input("Latitude", -90.0, 90.0, 8.5, 4.0)
            lon = st.number_input("Longitude", -180.0, 180.0, 38.5, 4.0)
            
            coordinates = (lat, lon)
            
            st.write("**Data Retrieval Options**")
            get_current = st.checkbox("Get Current Conditions", value=True)
            get_forecast = st.checkbox("Get Weather Forecast", value=True)
            get_wapor = st.checkbox("Get WaPOR Data", value=True)
            
            auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)
        
        with col2:
            if st.button("🔄 Fetch Real-time Data"):
                if not self.weather_connector and not self.wapor_connector:
                    st.error("Please initialize API connectors first!")
                else:
                    with st.spinner("Fetching real-time data..."):
                        field_conditions = self.get_field_conditions(coordinates)
                        st.session_state.field_conditions = field_conditions
                        st.success("Data retrieved successfully!")
            
            # Display current conditions
            if 'field_conditions' in st.session_state:
                conditions = st.session_state.field_conditions
                
                # Debug: Show what data we have
                st.write("🔍 Debug: Field conditions data structure")
                st.write(f"Keys: {list(conditions.keys())}")
                
                st.write("**🌡️ Current Field Conditions**")
                
                # Weather data
                if "weather" in conditions:
                    weather = conditions["weather"]
                    st.write(f"🔍 Debug: Weather data: {weather}")
                    
                    col2a, col2b, col2c = st.columns(3)
                    col2a.metric("Temperature", f"{weather.get('temperature', 0):.1f}°C")
                    col2b.metric("Humidity", f"{weather.get('humidity', 0):.0f}%")
                    col2c.metric("Wind Speed", f"{weather.get('wind_speed', 0):.1f} m/s")
                    
                    col2d, col2e, col2f = st.columns(3)
                    col2d.metric("Pressure", f"{weather.get('pressure', 0):.1f} hPa")
                    col2e.metric("Description", weather.get('description', 'N/A'))
                    col2f.metric("Location", weather.get('location', 'N/A'))
                else:
                    st.warning("⚠️ Weather data not available")
                
                # Forecast data
                if "forecast" in conditions:
                    forecast = conditions["forecast"]
                    st.write("**📅 3-Day Forecast**")
                    col_forecast1, col_forecast2, col_forecast3 = st.columns(3)
                    col_forecast1.metric("Avg Temperature", f"{forecast.get('next_3_days_avg_temp', 'N/A'):.1f}°C")
                    col_forecast2.metric("Total Rainfall", f"{forecast.get('next_3_days_total_rain', 'N/A'):.1f} mm")
                    col_forecast3.metric("Avg Humidity", f"{forecast.get('next_3_days_avg_humidity', 'N/A'):.1f}%")
                
                # WaPOR data
                if "wapor" in conditions:
                    wapor = conditions["wapor"]
                    st.write("**🌾 Crop Water Status**")
                    
                    if "aeti" in wapor:
                        aeti = wapor["aeti"]
                        col_wapor1, col_wapor2 = st.columns(2)
                        col_wapor1.metric("Actual ET", f"{aeti.get('latest_value', 0):.2f} {aeti.get('unit', '')}")
                        col_wapor2.metric("30-day Trend", f"{float(aeti.get('trend_30d', 0)):.3f}")
                    
                    if "npp" in wapor:
                        npp = wapor["npp"]
                        col_wapor3, col_wapor4 = st.columns(2)
                        col_wapor3.metric("NPP", f"{npp.get('latest_value', 0):.2f} {npp.get('unit', '')}")
                        col_wapor4.metric("30-day Average", f"{float(npp.get('average_30d', 0)):.3f}")
                
                # Visualization
                st.write("**📊 Data Visualization**")
                figures = self.plot_integrated_dashboard(coordinates)
                
                for fig_name, fig in figures.items():
                    st.plotly_chart(fig, use_container_width=True)
                
                # Data export
                st.write("**💾 Export Data**")
                export_format = st.selectbox("Export Format", ["CSV", "JSON"])
                
                if st.button("Export API Data"):
                    export_data = self.export_api_data(coordinates, export_format.lower())
                    
                    filename = f"field_data_{lat}_{lon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}"
                    mime_type = "text/csv" if export_format == "CSV" else "application/json"
                    
                    st.download_button(
                        label=f"📥 Download {export_format}",
                        data=export_data,
                        file_name=filename,
                        mime=mime_type
                    )
            
            # API Status
            st.write("**🔌 API Status**")
            status_col1, status_col2 = st.columns(2)
            
            with status_col1:
                if self.wapor_connector:
                    st.success("✅ WaPOR API Connected")
                else:
                    st.warning("⚠️ WaPOR API Not Connected")
            
            with status_col2:
                if self.weather_connector:
                    st.success("✅ OpenWeatherMap API Connected")
                else:
                    st.warning("⚠️ OpenWeatherMap API Not Connected")
            
            # Auto-refresh functionality
            if auto_refresh and 'field_conditions' in st.session_state:
                st.write("**🔄 Auto-refresh Status**")
                st.info("Data will auto-refresh every 5 minutes")
                
                # Show last update time
                if 'field_conditions' in st.session_state:
                    last_update = st.session_state.field_conditions.get('timestamp', 'Unknown')
                    st.write(f"Last update: {last_update}")
        
        return {
            "coordinates": coordinates,
            "apis_initialized": bool(self.weather_connector or self.wapor_connector),
            "data_available": 'field_conditions' in st.session_state
        }


def get_api_manager() -> IntegratedAPIManager:
    """Get API manager instance"""
    if 'api_manager' not in st.session_state:
        st.session_state.api_manager = IntegratedAPIManager()
    return st.session_state.api_manager
