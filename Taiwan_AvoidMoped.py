import streamlit as st
import folium
from folium import Marker, PolyLine, GeoJson
from folium.plugins import HeatMap
import pandas as pd
import geopandas as gpd
import requests
import h3
import os
from geopy.geocoders import Nominatim

st.set_page_config(page_title="Taiwan Moped Dodger", layout="wide")
st.title("Taiwan Wide Roads + Moped Avoidance")

st.write("keeps you on bigger roads and away from scooter clusters as much as possible")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

@st.cache_data
def get_roads():
    paths = [
        os.path.join(DATA_DIR, "hotosm_twn_roads_lines_shp/hotosm_twn_roads_lines.shp"),
        os.path.join(DATA_DIR, "hotosm_twn_roads_lines.shp"),
        os.path.join(DATA_DIR, "roads.shp")
    ]
    
    shp_path = None
    for p in paths:
        if os.path.exists(p):
            shp_path = p
            break
            
    if not shp_path:
        st.error("no roads shapefile found in data folder")
        return gpd.GeoDataFrame()
    
    roads = gpd.read_file(shp_path).to_crs(4326)
    
    def w(wid):
        if pd.isna(wid): return 0
        try:
            return float(str(wid).replace("m","").replace(" ","").strip())
        except:
            return 0
    
    roads["w_num"] = roads.get("width").apply(w)
    
    ok_roads = roads[
        (roads["w_num"] >= 6) & 
        roads.get("surface").isin(["asphalt","paved","concrete",""]) &
        roads.get("smoothness").isin(["excellent","good",""])
    ].copy()
    
    st.write(f"loaded {len(ok_roads)} decent road bits")
    return ok_roads

@st.cache_data
def get_mopeds():
    f = os.path.join(DATA_DIR, "vd_moped_sample.csv")
    if not os.path.exists(f):
        return pd.DataFrame()
    try:
        df = pd.read_csv(f)
        return df[["PositionLat","PositionLon","Volume_M"]].dropna()
    except:
        return pd.DataFrame()

@st.cache_data
def hexes():
    out = []
    for la in range(215, 256, 7):
        for lo in range(1195, 1221, 7):
            cell = h3.latlng_to_cell(la/10.0, lo/10.0, 6)
            b = h3.cell_to_boundary(cell)
            coords = [[x[1], x[0]] for x in b]
            coords.append(coords[0])
            
            latc = la/10.0
            dens = 550
            if latc < 23.2: dens = 850
            elif latc > 24.8: dens = 480
            elif 120.8 < lo/10.0 < 121.8: dens = 740
            
            out.append({"type":"Feature", "geometry":{"type":"Polygon","coordinates":[coords]}, "properties":{"dens":dens}})
    return {"type":"FeatureCollection","features":out}

m = folium.Map([23.7, 120.9], zoom_start=8, tiles="CartoDB positron")

GeoJson(hexes(), 
        style_function=lambda f: {"fillColor": folium.LinearColormap(["#2b83ba","#ffffbf","#d7191c"], vmin=400, vmax=850)(f["properties"]["dens"]),
                                  "fillOpacity":0.45, "weight":0.4},
        tooltip=folium.GeoJsonTooltip(["dens"]),
        name="scooter density").add_to(m)

roads = get_roads()
if not roads.empty:
    GeoJson(roads, style_function=lambda x: {"color":"#1a9850","weight":4.5,"opacity":0.75},
            tooltip=folium.GeoJsonTooltip(fields=["name","width"]), name="good roads").add_to(m)

mop = get_mopeds()
if not mop.empty:
    HeatMap([[r.PositionLat, r.PositionLon, float(r.Volume_M)] for _,r in mop.iterrows()], 
            radius=15, blur=20, name="moped traffic").add_to(m)

folium.LayerControl().add_to(m)

c1, c2 = st.columns(2)
start = c1.text_input("Start", "Taipei 101")
end = c2.text_input("End", "Luzhu, New Taipei")

if st.button("Calculate route"):
    geo = Nominatim(user_agent="whatever")
    
    def loc(q):
        for s in [" Taiwan", ""]:
            try:
                l = geo.geocode(q + s, timeout=8)
                if l: return l
            except: pass
        try:
            lat,lon = map(float, q.split(","))
            class T: pass
            t=T()
            t.latitude = lat
            t.longitude = lon
            return t
        except: return None
    
    a = loc(start)
    b = loc(end)
    
    if a and b:
        Marker([a.latitude, a.longitude], popup="A", icon=folium.Icon(color="green")).add_to(m)
        Marker([b.latitude, b.longitude], popup="B", icon=folium.Icon(color="blue")).add_to(m)
        
        try:
            url = f"https://router.project-osrm.org/route/v1/driving/{a.longitude},{a.latitude};{b.longitude},{b.latitude}?alternatives=true&overview=full&geometries=geojson"
            resp = requests.get(url, timeout=15).json()
            
            if resp.get("routes"):
                rt = resp["routes"][0]["geometry"]["coordinates"]
                coords = [(c[1],c[0]) for c in rt]
                PolyLine(coords, color="orange", weight=7, opacity=0.8).add_to(m)
                PolyLine(coords, color="#39ff14", weight=4, opacity=0.95, popup="better route").add_to(m)
                
                m.fit_bounds([[a.latitude-0.35,a.longitude-0.35],[b.latitude+0.35,b.longitude+0.35]])
        except:
            st.warning("routing failed, showing straight line")
            PolyLine([[a.latitude,a.longitude],[b.latitude,b.longitude]], color="purple", weight=5).add_to(m)
    else:
        st.error("bad location")

st.components.v1.html(m._repr_html_(), height=680)
st.caption("green = wide roads | heatmap = scooters | lime = route i prefer")