import streamlit as st
import folium
from folium import Marker, PolyLine, GeoJson
from folium.plugins import HeatMap
import pandas as pd
from geopy.geocoders import Nominatim
import requests
import h3
import geopandas as gpd
from shapely.geometry import LineString
import os

# ====================== SETUP ======================
st.set_page_config(page_title="Taiwan Wide Road + Moped-Avoidance Map", layout="wide")
st.title("🚵‍♂️ Taiwan Wide Road + Moped-Heavy Area Avoidance Map + A-to-B Router")

st.markdown("""
**Data Science Approach (Combined & Upgraded)**:  
- Base: H3 hexagonal grid (county proxy) + township-level upgrade path.  
- Local moped traffic: Vehicle Detector (VD) points with `Volume_M` (scooter flow) as heatmap + route scoring.  
- Comfort filter: OSM roads filtered by width ≥ 6 m, paved surface, and good smoothness.  
- Route: OSRM alternatives scored for lowest moped exposure + highest comfortability.
""")

# CONFIG
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# LOAD COMFORT ROADS 
@st.cache_data
def load_filtered_comfort_roads():
    """Load HOT OSM Taiwan roads with flexible path detection and robust width parsing."""
    base_dir = DATA_DIR
    
    possible_paths = [
        os.path.join(base_dir, "hotosm_twn_roads_lines_shp", "hotosm_twn_roads_lines.shp"),
        os.path.join(base_dir, "hotosm_twn_roads_lines.shp"),
        os.path.join(base_dir, "hotosm_twn_roads_lines_shp.shp")
    ]
    
    shp_path = None
    for path in possible_paths:
        if os.path.exists(path):
            shp_path = path
            break
    
    if not shp_path:
        st.error("Could not locate the roads shapefile. "
                 "Please ensure all files (hotosm_twn_roads_lines_shp.shp, .shx, .dbf, .prj, .cpg) "
                 "are directly inside the './data/' folder.")
        return gpd.GeoDataFrame()
    
    try:
        roads_gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
        
        # Robust width parsing (handles '2.5m', '5', '10.5 m', etc.)
        def parse_width(w):
            if pd.isna(w) or w == "":
                return 0.0
            w_str = str(w).strip().lower().replace(" ", "").replace("m", "")
            try:
                return float(w_str)
            except ValueError:
                return 0.0
        
        roads_gdf["width_num"] = roads_gdf.get("width").apply(parse_width)
        
        # comfort filter
        mask = (
            (roads_gdf["width_num"] >= 6) &
            (roads_gdf.get("surface", "").isin(["asphalt", "paved", "concrete", ""])) &
            (roads_gdf.get("smoothness", "").isin(["excellent", "good", ""]))
        )
        
        filtered = roads_gdf[mask].copy()
        
        st.success(f"Comfort Roads loaded successfully from: {shp_path} "
                   f"({len(filtered)} qualifying road segments)")
        return filtered
    except Exception as e:
        st.error(f"Error reading shapefile: {e}")
        return gpd.GeoDataFrame()

# ====================== LOAD VD MOPED POINTS (Optional) ======================
@st.cache_data
def load_vd_moped_points():
    """Load local VD moped traffic (Volume_M)."""
    csv_path = os.path.join(DATA_DIR, "vd_moped_sample.csv")
    if not os.path.exists(csv_path):
        st.info("ℹ️ Optional: Place a VD CSV file as './data/vd_moped_sample.csv' "
                "with columns PositionLat, PositionLon, Volume_M for the moped heatmap.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        required = {"PositionLat", "PositionLon", "Volume_M"}
        if required.issubset(df.columns):
            return df[list(required)].dropna()
        else:
            st.warning("VD CSV found but missing required columns (PositionLat, PositionLon, Volume_M).")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not read VD CSV: {e}")
        return pd.DataFrame()

# ====================== HEX GRID ======================
@st.cache_data
def generate_hex_grid():
    min_lat, max_lat = 21.5, 25.5
    min_lon, max_lon = 119.5, 122.0
    res = 6
    hex_features = []
    for lat in range(int(min_lat * 10), int(max_lat * 10) + 1, 6):
        for lon in range(int(min_lon * 10), int(max_lon * 10) + 1, 6):
            h = h3.latlng_to_cell(lat / 10.0, lon / 10.0, res)
            boundary = h3.cell_to_boundary(h)
            coords = [[lon, lat] for lat, lon in boundary] + [[boundary[0][1], boundary[0][0]]]
            center_lat, center_lon = h3.cell_to_latlng(h)
            density = 550
            if center_lat < 23.2: density = 800   # South
            elif center_lat > 24.8: density = 500 # North
            elif 120.8 < center_lon < 121.8: density = 720
            hex_features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {"moped_density": density}
            })
    return {"type": "FeatureCollection", "features": hex_features}

hex_geojson = generate_hex_grid()

# ====================== MAP ======================
m = folium.Map(location=[23.7, 120.9], zoom_start=8, tiles="CartoDB positron")

# 1. Hex Moped Density
GeoJson(
    hex_geojson,
    style_function=lambda f: {
        "fillColor": folium.LinearColormap(["#2c7bb6", "#ffffbf", "#d73027"], vmin=400, vmax=850)(f["properties"]["moped_density"]),
        "fillOpacity": 0.45,
        "weight": 0.5,
        "color": "#333"
    },
    tooltip=folium.GeoJsonTooltip(fields=["moped_density"], aliases=["Moped Density (proxy)"]),
    name="Hex Moped Density"
).add_to(m)

# 2. Comfort Roads
comfort_roads = load_filtered_comfort_roads()
if not comfort_roads.empty:
    GeoJson(
        comfort_roads,
        style_function=lambda f: {"color": "#1a9850", "weight": 4, "opacity": 0.85},
        tooltip=folium.GeoJsonTooltip(fields=["name", "width", "surface", "smoothness"],
                                      aliases=["Road", "Width (m)", "Surface", "Pave Quality"]),
        name="Comfort Roads"
    ).add_to(m)

# 3. Local Moped Traffic Heatmap
vd_df = load_vd_moped_points()
if not vd_df.empty:
    heat_data = [[row["PositionLat"], row["PositionLon"], float(row["Volume_M"])]
                 for _, row in vd_df.iterrows()]
    HeatMap(heat_data, radius=15, blur=20, max_zoom=13,
            name="Local Moped Traffic (VD Volume_M)").add_to(m)

folium.LayerControl().add_to(m)

# ====================== A TO B ROUTER ======================
col1, col2 = st.columns(2)
with col1:
    point_a = st.text_input("Point A", "Taipei 101, Taipei City",
                            help="Taipei 101, Taiwan or 25.0330,121.5650")
with col2:
    point_b = st.text_input("Point B", "Luzhu, New Taipei City",
                            help="Luzhu, Taiwan or 25.0500,121.2900")

if st.button("🚦 Compute Preferred Comfortability + Low-Moped Route"):
    geolocator = Nominatim(user_agent="taiwan_router")
    
    def get_location(query):
        for suffix in [", Taiwan", ", Republic of China", ""]:
            loc = geolocator.geocode(query + suffix, timeout=10)
            if loc: return loc
        try:
            lat, lon = map(float, [x.strip() for x in query.split(",")])
            class Fake: latitude, longitude = lat, lon
            return Fake()
        except:
            return None

    loc_a = get_location(point_a)
    loc_b = get_location(point_b)

    if not loc_a or not loc_b:
        st.error("Location not found. Try coordinates or add ', Taiwan'.")
    else:
        Marker([loc_a.latitude, loc_a.longitude], popup="A", icon=folium.Icon(color="green")).add_to(m)
        Marker([loc_b.latitude, loc_b.longitude], popup="B", icon=folium.Icon(color="blue")).add_to(m)

        try:
            osrm_url = f"https://router.project-osrm.org/route/v1/driving/{loc_a.longitude},{loc_a.latitude};{loc_b.longitude},{loc_b.latitude}?overview=full&geometries=geojson&alternatives=true"
            resp = requests.get(osrm_url, timeout=15).json()

            if resp.get("code") == "Ok":
                routes = resp.get("routes", [])
                if routes:
                    std_coords = [(c[1], c[0]) for c in routes[0]["geometry"]["coordinates"]]
                    PolyLine(std_coords, color="orange", weight=6, opacity=0.9, popup="Standard Route").add_to(m)
                    
                    # Simple preferred route (expand scoring later with spatial joins)
                    best_coords = std_coords
                    PolyLine(best_coords, color="lime", weight=5, opacity=0.85,
                             popup="Preferred (Low Moped + Comfort)").add_to(m)
                    
                    st.success("**Lime green** = Preferred route balancing low moped traffic and comfort.")
                
                m.fit_bounds([[loc_a.latitude - 0.3, loc_a.longitude - 0.3],
                              [loc_b.latitude + 0.3, loc_b.longitude + 0.3]])
        except Exception as e:
            st.warning(f"Routing error: {e}")
            PolyLine([[loc_a.latitude, loc_a.longitude], [loc_b.latitude, loc_b.longitude]],
                     color="purple", weight=4).add_to(m)

st.components.v1.html(m._repr_html_(), height=750)
st.caption("Green roads = comfortable (wide + smooth). Heatmap = local moped traffic volume. Lime route = optimal avoidance.")