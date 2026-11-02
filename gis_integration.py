"""
Geospatial Integration Module for TOKUMA 3-in-1 Research Platform
Provides interactive mapping capabilities with Folium and shapefile support
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
import streamlit_folium
from shapely.geometry import Point, Polygon


class GISIntegration:
    """Enhanced GIS capabilities for the research platform"""
    
    def __init__(self):
        self.ethiopia_regions = {
            "Arsi": {"center": [7.75, 39.5], "zoom": 8},
            "Bale": {"center": [6.5, 40.0], "zoom": 8},
            "Harari": {"center": [9.3, 42.1], "zoom": 9},
            "Oromia": {"center": [8.5, 38.5], "zoom": 7},
            "Amhara": {"center": [11.5, 37.5], "zoom": 7},
            "SNNPR": {"center": [6.5, 36.5], "zoom": 7},
            "Tigray": {"center": [13.5, 38.5], "zoom": 8},
            "Somali": {"center": [6.5, 44.0], "zoom": 7},
            "Afar": {"center": [11.5, 41.0], "zoom": 7},
            "Benishangul-Gumuz": {"center": [10.5, 34.5], "zoom": 8},
            "Gambela": {"center": [8.0, 34.0], "zoom": 8},
            "Dire Dawa": {"center": [9.6, 41.9], "zoom": 10},
            "Addis Ababa": {"center": [9.0, 38.7], "zoom": 11}
        }
        
        self.soil_types = {
            "Clay": {"color": "#8B4513", "infiltration_rate": 0.5, "field_capacity": 0.35},
            "Sandy Loam": {"color": "#F4A460", "infiltration_rate": 2.5, "field_capacity": 0.20},
            "Silt Clay": {"color": "#CD853F", "infiltration_rate": 1.0, "field_capacity": 0.30},
            "Loam": {"color": "#DEB887", "infiltration_rate": 1.5, "field_capacity": 0.25},
            "Sandy": {"color": "#F5DEB3", "infiltration_rate": 5.0, "field_capacity": 0.15},
            "Silt": {"color": "#D2B48C", "infiltration_rate": 3.0, "field_capacity": 0.22}
        }
        
        self.climate_zones = {
            "Arid": {"color": "#FFD700", "annual_rainfall": 250, "temp_range": (20, 35)},
            "Semi-Arid": {"color": "#FFA500", "annual_rainfall": 500, "temp_range": (18, 32)},
            "Sub-Humid": {"color": "#90EE90", "annual_rainfall": 900, "temp_range": (15, 28)},
            "Humid": {"color": "#228B22", "annual_rainfall": 1400, "temp_range": (12, 25)}
        }

    def create_base_map(self, region: str = "Oromia") -> folium.Map:
        """Create base map centered on selected region"""
        if region not in self.ethiopia_regions:
            region = "Oromia"
        
        config = self.ethiopia_regions[region]
        m = folium.Map(
            location=config["center"],
            zoom_start=config["zoom"],
            tiles="OpenStreetMap"
        )
        
        # Add layer control
        folium.TileLayer('Stamen Terrain', attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap').add_to(m)
        folium.TileLayer('Stamen Toner', attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap').add_to(m)
        folium.TileLayer('CartoDB positron', attr='© CartoDB').add_to(m)
        folium.LayerControl().add_to(m)
        
        return m

    def add_sample_sites(self, map_obj: folium.Map, sites: List[Dict]) -> folium.Map:
        """Add sample field sites to the map"""
        for site in sites:
            popup_content = f"""
            <b>{site['name']}</b><br>
            Soil Type: {site['soil_type']}<br>
            Elevation: {site['elevation']}m<br>
            Last Irrigation: {site['last_irrigation']}mm
            """
            
            folium.Marker(
                location=[site['lat'], site['lon']],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=site['name'],
                icon=folium.Icon(color='green', icon='tint', prefix='fa')
            ).add_to(map_obj)
        
        return map_obj

    def process_shapefile(self, uploaded_file) -> Optional[gpd.GeoDataFrame]:
        """Process uploaded shapefile"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.shp') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Read shapefile
            gdf = gpd.read_file(tmp_path)
            
            # Ensure it's in WGS84
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)
            
            return gdf
        
        except Exception as e:
            st.error(f"Error processing shapefile: {str(e)}")
            return None

    def add_shapefile_layer(self, map_obj: folium.Map, gdf: gpd.GeoDataFrame, 
                           style: Dict = None) -> folium.Map:
        """Add shapefile as layer to map"""
        if style is None:
            style = {
                'fillColor': '#3186cc',
                'color': '#3186cc',
                'weight': 2,
                'fillOpacity': 0.3
            }
        
        # Convert to GeoJSON
        geojson_data = gdf.to_json()
        
        # Add to map
        folium.GeoJson(
            geojson_data,
            style_function=lambda x: style,
            tooltip=folium.GeoJsonTooltip(fields=list(gdf.columns))
        ).add_to(map_obj)
        
        return map_obj

    def extract_soil_climate_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """Extract soil and climate data based on location"""
        # Simplified soil assignment based on region
        if 7 <= lat <= 9 and 38 <= lon <= 42:  # Central Ethiopia
            soil_type = "Silt Clay"
            climate_zone = "Sub-Humid"
        elif lat < 7:  # Southern Ethiopia
            soil_type = "Sandy Loam"
            climate_zone = "Semi-Arid"
        elif lat > 10:  # Northern Ethiopia
            soil_type = "Clay"
            climate_zone = "Arid"
        else:  # Default
            soil_type = "Loam"
            climate_zone = "Sub-Humid"
        
        soil_data = self.soil_types[soil_type]
        climate_data = self.climate_zones[climate_zone]
        
        return {
            "soil_type": soil_type,
            "climate_zone": climate_zone,
            "infiltration_rate": soil_data["infiltration_rate"],
            "field_capacity": soil_data["field_capacity"],
            "annual_rainfall": climate_data["annual_rainfall"],
            "temp_range": climate_data["temp_range"],
            "soil_color": soil_data["color"],
            "climate_color": climate_data["color"]
        }

    def create_watershed_mask(self, gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Create watershed boundary mask from shapefile"""
        if gdf.empty:
            return {}
        
        # Calculate watershed statistics
        total_area = gdf.geometry.area.sum() / 1_000_000  # Convert to km²
        
        # Get centroid
        centroid = gdf.geometry.centroid.iloc[0]
        
        # Bounding box
        bounds = gdf.total_bounds.tolist()
        
        return {
            "total_area_km2": total_area,
            "centroid_lat": centroid.y,
            "centroid_lon": centroid.x,
            "bounds": bounds,
            "geometry_count": len(gdf)
        }

    def render_gis_interface(self) -> Tuple[Dict, Optional[folium.Map]]:
        """Render the complete GIS interface"""
        st.subheader("🗺️ Geospatial Integration (GIS)")
        
        # Region selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_region = st.selectbox(
                "Select Region",
                list(self.ethiopia_regions.keys()),
                index=list(self.ethiopia_regions.keys()).index("Oromia")
            )
            
            # Manual coordinate input
            st.write("**Or enter coordinates manually:**")
            lat_input = st.number_input("Latitude", value=8.5, min_value=-90.0, max_value=90.0)
            lon_input = st.number_input("Longitude", value=38.5, min_value=-180.0, max_value=180.0)
            
            # Extract soil and climate data
            if st.button("Extract Site Data"):
                site_data = self.extract_soil_climate_data(lat_input, lon_input)
                st.session_state.gis_site_data = site_data
                st.success("Site data extracted successfully!")
        
        with col2:
            # Create base map
            m = self.create_base_map(selected_region)
            
            # Add click handler for coordinate selection
            m.add_child(folium.LatLngPopup())
            
            # Add sample sites
            sample_sites = [
                {"name": "Kulumsa Research Station", "lat": 8.15, "lon": 39.92, 
                 "soil_type": "Silt Clay", "elevation": 2200, "last_irrigation": 45},
                {"name": "Wonji Sugar Estate", "lat": 8.48, "lon": 39.27,
                 "soil_type": "Sandy Loam", "elevation": 1500, "last_irrigation": 60},
                {"name": "Melkassa Agricultural Center", "lat": 8.43, "lon": 39.32,
                 "soil_type": "Loam", "elevation": 1550, "last_irrigation": 30}
            ]
            
            m = self.add_sample_sites(m, sample_sites)
            
            # Display map
            st_data = streamlit_folium.st_folium(m, width=700, height=500)
            
            # Store clicked coordinates
            if st_data.get('last_clicked'):
                clicked_lat = st_data['last_clicked']['lat']
                clicked_lon = st_data['last_clicked']['lng']
                st.session_state.clicked_coords = (clicked_lat, clicked_lon)
                st.write(f"**Selected Coordinates:** {clicked_lat:.4f}, {clicked_lon:.4f}")
        
        # Shapefile upload section
        st.write("---")
        st.write("**📁 Shapefile Upload for Watershed Delineation**")
        
        uploaded_shapefile = st.file_uploader(
            "Upload shapefile (.shp, .shx, .dbf, .prj)",
            type=['shp', 'shx', 'dbf', 'prj'],
            accept_multiple_files=True
        )
        
        if uploaded_shapefile:
            # Process the main .shp file
            shp_file = next((f for f in uploaded_shapefile if f.name.endswith('.shp')), None)
            if shp_file:
                gdf = self.process_shapefile(shp_file)
                if gdf is not None:
                    # Add shapefile to map
                    m_with_shapefile = self.create_base_map(selected_region)
                    m_with_shapefile = self.add_shapefile_layer(m_with_shapefile, gdf)
                    
                    st.write("Shapefile loaded successfully!")
                    st_data_shapefile = streamlit_folium.st_folium(
                        m_with_shapefile, width=700, height=400
                    )
                    
                    # Display watershed statistics
                    watershed_info = self.create_watershed_mask(gdf)
                    if watershed_info:
                        st.write("**Watershed Statistics:**")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Area", f"{watershed_info['total_area_km2']:.2f} km²")
                        col2.metric("Centroid", f"{watershed_info['centroid_lat']:.2f}°, {watershed_info['centroid_lon']:.2f}°")
                        col3.metric("Features", watershed_info['geometry_count'])
                    
                    st.session_state.watershed_gdf = gdf
        
        # Display extracted site data if available
        if 'gis_site_data' in st.session_state:
            st.write("---")
            st.write("**📍 Extracted Site Parameters:**")
            site_data = st.session_state.gis_site_data
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Soil Type", site_data['soil_type'])
                st.metric("Infiltration Rate", f"{site_data['infiltration_rate']} cm/hr")
                st.metric("Field Capacity", f"{site_data['field_capacity']:.2f}")
            
            with col2:
                st.metric("Climate Zone", site_data['climate_zone'])
                st.metric("Annual Rainfall", f"{site_data['annual_rainfall']} mm")
                st.metric("Temp Range", f"{site_data['temp_range'][0]}-{site_data['temp_range'][1]}°C")
        
        return st.session_state.get('gis_site_data', {}), m


def get_gis_integration() -> GISIntegration:
    """Get GIS integration instance"""
    if 'gis_integration' not in st.session_state:
        st.session_state.gis_integration = GISIntegration()
    return st.session_state.gis_integration
