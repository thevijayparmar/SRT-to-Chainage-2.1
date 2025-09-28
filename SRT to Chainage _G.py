# streamlit_app.py

import streamlit as st
import re
import math
import zipfile
import io
import xml.etree.ElementTree as ET
from datetime import datetime
import plotly.graph_objects as go
import os

# ──────────────────────────────────────────────────────────────────────────
# Author: Vijay Parmar
# Community: BGol Community of Advanced Surveying and GIS Professionals
#
# Description:
# This Streamlit application processes drone SRT subtitle files against a KML
# alignment. It has been upgraded to calculate chainage by projecting each drone
# GPS point onto the KML alignment, providing more accurate stationing and
# lateral offset data. The UI is designed for a streamlined workflow, handling
# multiple SRT files, chronological sorting, and comprehensive output generation.
# ──────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Drone SRT & Chainage v2.0", layout="wide")

# ──────────────────────────────────────────────────────────────────────────
# Constants and Session State Initialization
# ──────────────────────────────────────────────────────────────────────────

PLOTLY_COLORS = [
    '#FF0000', '#FFA500', '#008000', '#0000FF', '#4B0082', '#EE82EE', 
    '#A52A2A', '#00FFFF', '#FF00FF', '#808000'
] # Red, Orange, Green, Blue, Indigo, Violet, Brown, Cyan, Magenta, Olive

if 'kml_data' not in st.session_state:
    st.session_state.kml_data = {
        "name": None,
        "coords": None,
        "swapped": False
    }
if 'srt_files' not in st.session_state:
    st.session_state.srt_files = []


# ──────────────────────────────────────────────────────────────────────────
# Geospatial & Core Logic Helper Functions
# ──────────────────────────────────────────────────────────────────────────

def hav(p1, p2):
    """Calculates the Haversine distance between two (lat, lon) points in km."""
    R = 6371.0088  # Mean radius of Earth in km
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_cumulative_dist(coords, offset=0.0):
    """Calculates the cumulative distance along a list of coordinates."""
    cum_dist = [offset]
    for i in range(len(coords) - 1):
        dist = hav(coords[i], coords[i+1])
        cum_dist.append(cum_dist[-1] + dist)
    return cum_dist

@st.cache_data
def project_point_to_polyline(point, polyline_coords, polyline_chainage):
    """
    Finds the closest point on a polyline to a given point.
    
    Returns:
        - projected_chainage (km)
        - projected_coords (lat, lon)
        - offset_distance (m)
    """
    min_dist_sq = float('inf')
    best_proj_point = None
    best_segment_idx = -1
    
    # Use an equirectangular projection for a fast approximation to find the nearest segment.
    # This is much faster than doing complex spherical geometry for every segment.
    avg_lat_rad = math.radians(point[0])
    cos_avg_lat = math.cos(avg_lat_rad)
    px, py = point[1] * cos_avg_lat, point[0]

    for i in range(len(polyline_coords) - 1):
        a = polyline_coords[i]
        b = polyline_coords[i+1]
        
        ax, ay = a[1] * cos_avg_lat, a[0]
        bx, by = b[1] * cos_avg_lat, b[0]

        apx, apy = px - ax, py - ay
        abx, aby = bx - ax, by - ay
        
        ab_mag_sq = abx**2 + aby**2
        if ab_mag_sq == 0:
            t = 0
        else:
            t = (apx * abx + apy * aby) / ab_mag_sq

        if t < 0:
            proj_x, proj_y = ax, ay
        elif t > 1:
            proj_x, proj_y = bx, by
        else:
            proj_x, proj_y = ax + t * abx, ay + t * aby
        
        dist_sq = (px - proj_x)**2 + (py - proj_y)**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_proj_point = (proj_y, proj_x / cos_avg_lat)
            best_segment_idx = i

    # Perform accurate Haversine calculations for the final result
    offset_dist_km = hav(point, best_proj_point)
    start_of_segment = polyline_coords[best_segment_idx]
    dist_along_segment_km = hav(start_of_segment, best_proj_point)
    
    projected_chainage = polyline_chainage[best_segment_idx] + dist_along_segment_km

    return projected_chainage, best_proj_point, offset_dist_km * 1000

# ──────────────────────────────────────────────────────────────────────────
# File Parsing and Data Handling Functions
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data
def parse_kml_2d(uploaded_file):
    """Parses a KML file to extract the first LineString coordinates."""
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    file_content = uploaded_file.getvalue()
    # Reset buffer position for potential reuse
    uploaded_file.seek(0)
    tree = ET.parse(io.BytesIO(file_content))
    
    coord_element = tree.find('.//kml:LineString/kml:coordinates', ns)
    if coord_element is None:
        st.error("Could not find a LineString in the KML file.")
        return []
        
    coords = []
    for part in coord_element.text.strip().split():
        try:
            lon, lat, *_ = part.split(',')
            coords.append((float(lat), float(lon)))
        except ValueError:
            continue # Skip malformed coordinate pairs
    return coords

@st.cache_data
def get_srt_start_time(uploaded_file):
    """Efficiently reads the first timestamp from an SRT file."""
    try:
        content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        uploaded_file.seek(0) # Reset buffer
        match = re.search(r"(\d{4}-\d{2}-\d{2}\s*\d{2}:\d{2}:\d{2})", content)
        if match:
            return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None
    return None

def parse_srt(file_content_str, origin_index, start_idx=1):
    """Parses a single SRT file's content into a list of data blocks."""
    lines = file_content_str.splitlines()
    blocks, i = [], 0
    current_idx = start_idx
    
    while i < len(lines):
        if not lines[i].strip().isdigit():
            i += 1
            continue
        
        original_idx = int(lines[i].strip())
        time_range = lines[i+1].strip()
        j = i + 2
        lat, lon, alt, tim = None, None, 0.0, ""
        
        while j < len(lines) and lines[j].strip():
            line = lines[j]
            if m := re.search(r"\[latitude:\s*([0-9.\-]+)", line): lat = float(m.group(1))
            if m := re.search(r"\[longitude:\s*([0-9.\-]+)", line): lon = float(m.group(1))
            if m := re.search(r"\[altitude:\s*([0-9.\-]+)", line): alt = float(m.group(1))
            if m := re.search(r"\d{4}-\d{2}-\d{2}\s*([0-9:]{8})", line): tim = m.group(1)
            j += 1
            
        if lat is not None and lon is not None:
            blocks.append({
                'idx': current_idx, 'range': time_range, 'lat': lat, 'lon': lon,
                'alt': alt, 'tim': tim, 'origin_index': origin_index
            })
            current_idx += 1
        i = j + 1
    return blocks

def merge_srt_data(sorted_files):
    """Merges multiple SRT files into a single chronological list of blocks."""
    all_blocks = []
    start_idx = 1
    for i, file in enumerate(sorted_files):
        content = file.getvalue().decode('utf-8', errors='ignore')
        file.seek(0)
        blocks = parse_srt(content, origin_index=i, start_idx=start_idx)
        all_blocks.extend(blocks)
        start_idx = len(all_blocks) + 1
    return all_blocks

# ──────────────────────────────────────────────────────────────────────────
# KML and ZIP Generation Functions
# ──────────────────────────────────────────────────────────────────────────

def kml_style_header(num_styles):
    """Generates KML styles for multiple drone paths."""
    header = (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "\n"
        "<kml xmlns='http://www.opengis.net/kml/2.2'><Document>\n"
        "<Style id='chainStyle'><LineStyle><color>ff0000ff</color><width>10.0</width></LineStyle></Style>\n"
        "<Style id='placemarkStyle'><IconStyle><scale>0.8</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/wht-blank.png</href></Icon></IconStyle></Style>\n"
    )
    styles = []
    for i in range(num_styles):
        color_bgr = PLOTLY_COLORS[i % len(PLOTLY_COLORS)].replace('#', '')
        color_abgr = f"ff{color_bgr[4:6]}{color_bgr[2:4]}{color_bgr[0:2]}" # #RRGGBB -> aabbggrr
        styles.append(
            f"<Style id='srtStyle_{i}'>"
            f"<LineStyle><color>{color_abgr}</color><width>4.0</width></LineStyle>"
            f"</Style>\n"
        )
    return header + "".join(styles)

def generate_markers_kml(coords, cumd, interval_km=0.05):
    """Generates a KML file with placemarks at a regular interval."""
    markers, current_dist, idx = [], cumd[0], 0
    total_dist = cumd[-1]
    
    while current_dist <= total_dist:
        while idx < len(cumd) - 1 and cumd[idx+1] < current_dist:
            idx += 1
            
        if idx >= len(coords) - 1: break
        
        A, B = coords[idx], coords[idx+1]
        segment_len = cumd[idx+1] - cumd[idx]
        
        if segment_len > 0:
            fraction = (current_dist - cumd[idx]) / segment_len
            lat = A[0] + fraction * (B[0] - A[0])
            lon = A[1] + fraction * (B[1] - A[1])
            markers.append((lat, lon, current_dist))
        elif idx == 0 and current_dist == cumd[0]: # Add start marker
            markers.append((A[0], A[1], current_dist))
        
        current_dist += interval_km

    xml = [kml_style_header(0)]
    for lat, lon, d in markers:
        xml.append(
            f"<Placemark><styleUrl>#placemarkStyle</styleUrl><name>{d:.3f} km</name>"
            f"<Point><coordinates>{lon:.7f},{lat:.7f},0</coordinates></Point></Placemark>\n"
        )
    xml.append("</Document></kml>")
    return "".join(xml)

def generate_multi_flight_kml(all_blocks, srt_files):
    """Generates a KML with a distinct, colored LineString for each SRT file."""
    xml = [kml_style_header(len(srt_files))]
    for i in range(len(srt_files)):
        flight_blocks = [b for b in all_blocks if b['origin_index'] == i]
        if not flight_blocks: continue
        
        file_name = srt_files[i].name
        xml.append(f"<Placemark><name>{file_name}</name><styleUrl>#srtStyle_{i}</styleUrl><LineString><coordinates>\n")
        for b in flight_blocks:
            xml.append(f"{b['lon']:.7f},{b['lat']:.7f},{b['alt']:.2f}\n")
        xml.append("</coordinates></LineString></Placemark>\n")
        
    xml.append("</Document></kml>")
    return "".join(xml)

def generate_output_zip(processed_data, kml_name):
    """Creates a ZIP archive with seven processed SRT output files."""
    prefix = os.path.splitext(kml_name)[0]
    
    # Prepare data columns
    cols = {
        f"{prefix}_01_Latitude.srt": [f"{b['lat']:.7f}" for b in processed_data],
        f"{prefix}_02_Longitude.srt": [f"{b['lon']:.7f}" for b in processed_data],
        f"{prefix}_03_Altitude.srt": [f"{b['alt']:.2f}" for b in processed_data],
        f"{prefix}_04_Timer.srt": [b['tim'] for b in processed_data],
        f"{prefix}_05_LatLonOutput.srt": [f"{b['lat']:.7f}, {b['lon']:.7f}" for b in processed_data],
        f"{prefix}_06_Chainage.srt": [f"{b['chainage']:.3f} km" for b in processed_data],
        f"{prefix}_07_Lateral_Offset.srt": [f"{b['offset']:.3f} m" for b in processed_data],
    }

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename, data_col in cols.items():
            srt_content = "\n\n".join(
                f"{processed_data[i]['idx']}\n{processed_data[i]['range']}\n{data_col[i]}"
                for i in range(len(processed_data))
            )
            zf.writestr(filename, srt_content)
            
    zip_buffer.seek(0)
    return zip_buffer

# ──────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────

st.title("🛰️ Drone SRT & Chainage Workflow v2.0")
st.markdown(
    """**Author:** Vijay Parmar  
**Community:** BGol Community of Advanced Surveying and GIS Professionals"""
)
st.markdown("---")


# --------------------------
# Section 1: Setup & Inputs
# --------------------------
st.header("Section 1: Setup & Inputs")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Step 1: Upload KML Alignment")
    uploaded_kml = st.file_uploader("Upload your master alignment KML file", type=["kml", "xml"])

with col2:
    st.subheader("Step 2: Upload Drone SRT Files")
    uploaded_srts = st.file_uploader(
        "Upload one or more drone .SRT files",
        type="srt",
        accept_multiple_files=True
    )

# --- KML Configuration UI ---
if uploaded_kml:
    if st.session_state.kml_data["name"] != uploaded_kml.name:
        coords = parse_kml_2d(uploaded_kml)
        if coords:
            st.session_state.kml_data = {
                "name": uploaded_kml.name,
                "coords": coords,
                "swapped": False
            }
        else:
            st.session_state.kml_data = {"name": None, "coords": None, "swapped": False}

    if st.session_state.kml_data["coords"]:
        with st.container(border=True):
            st.subheader("KML Alignment Configuration")
            
            kml_coords = list(reversed(st.session_state.kml_data["coords"])) if st.session_state.kml_data["swapped"] else st.session_state.kml_data["coords"]
            total_length = calculate_cumulative_dist(kml_coords)[-1]

            st.info(f"**Total Alignment Length:** `{total_length:.3f} km`")

            c1, c2, c3 = st.columns([2, 2, 1])
            c1.metric("Current Start Point", f"{kml_coords[0][0]:.6f}, {kml_coords[0][1]:.6f}")
            c2.metric("Current End Point", f"{kml_coords[-1][0]:.6f}, {kml_coords[-1][1]:.6f}")
            
            if c3.button("🔄 Swap Start/End Points", use_container_width=True):
                st.session_state.kml_data["swapped"] = not st.session_state.kml_data["swapped"]
                st.rerun()

            chain_offset = st.number_input(
                "Chainage at Start Point (km)",
                min_value=0.0, step=0.001, format="%.3f", value=0.0,
                key='chain_offset'
            )

# --- SRT File Management UI ---
if uploaded_srts:
    # Check if the uploaded file list has changed
    current_filenames = sorted([f.name for f in uploaded_srts])
    previous_filenames = sorted([f.name for f in st.session_state.get('srt_files_cache', [])])

    if current_filenames != previous_filenames:
        with st.spinner("Analyzing SRT start times..."):
            st.session_state.srt_files = sorted(
                uploaded_srts,
                key=lambda f: get_srt_start_time(f) or datetime.max
            )
        st.session_state.srt_files_cache = uploaded_srts.copy()

    if st.session_state.srt_files:
        with st.container(border=True):
            st.subheader("SRT File Processing Order")
            st.caption("Files are sorted chronologically by default. You can reorder them for processing.")

            for i, f in enumerate(st.session_state.srt_files):
                cols = st.columns([1, 8, 1, 1])
                cols[0].write(f"**{i+1}.**")
                cols[1].write(f.name)
                if cols[2].button("⬆️", key=f"up_{i}", help="Move Up", disabled=(i == 0)):
                    st.session_state.srt_files.insert(i-1, st.session_state.srt_files.pop(i))
                    st.rerun()
                if cols[3].button("⬇️", key=f"down_{i}", help="Move Down", disabled=(i == len(st.session_state.srt_files)-1)):
                    st.session_state.srt_files.insert(i+1, st.session_state.srt_files.pop(i))
                    st.rerun()
            
            if st.button("Reset to Chronological Order"):
                 with st.spinner("Re-sorting files..."):
                    st.session_state.srt_files = sorted(
                        uploaded_srts,
                        key=lambda f: get_srt_start_time(f) or datetime.max
                    )
                 st.rerun()


st.markdown("---")

# --- Processing Trigger ---
if st.session_state.kml_data.get("coords") and st.session_state.srt_files:
    if st.button("▶ Process and Generate Outputs", type="primary", use_container_width=True):
        
        # ---------------
        # CORE PROCESSING
        # ---------------
        with st.spinner("Processing... This may take a moment."):
            
            # 1. Pre-compute KML chainage
            kml_coords = list(reversed(st.session_state.kml_data["coords"])) if st.session_state.kml_data["swapped"] else st.session_state.kml_data["coords"]
            kml_chainage = calculate_cumulative_dist(kml_coords, st.session_state.chain_offset)
            
            # 2. Merge all SRT data
            merged_srt_data = merge_srt_data(st.session_state.srt_files)

            # 3. Project each SRT point onto the KML alignment
            processed_data = []
            for block in merged_srt_data:
                point = (block['lat'], block['lon'])
                chainage, proj_coords, offset = project_point_to_polyline(point, kml_coords, kml_chainage)
                
                block['chainage'] = chainage
                block['proj_coords'] = proj_coords
                block['offset'] = offset
                processed_data.append(block)
            
            st.session_state.processed_data = processed_data
            st.session_state.kml_chainage_map = kml_chainage
            st.session_state.final_kml_coords = kml_coords
            st.success("✅ Processing complete!")

# ----------------------------
# Section 2: Outputs & Visualization
# ----------------------------
if 'processed_data' in st.session_state:
    st.header("Section 2: Outputs & Visualization")
    
    processed_data = st.session_state.processed_data
    kml_coords = st.session_state.final_kml_coords
    
    # --- Summary Statistics ---
    with st.container(border=True):
        st.subheader("📊 Summary Statistics")
        start_ch = processed_data[0]['chainage']
        end_ch = processed_data[-1]['chainage']
        
        drone_path_coords = [(b['lat'], b['lon']) for b in processed_data]
        total_flight_dist_km = calculate_cumulative_dist(drone_path_coords)[-1]
        
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Merged SRT Start Chainage", f"{start_ch:.3f} km")
        sc2.metric("Merged SRT End Chainage", f"{end_ch:.3f} km")
        sc3.metric("Total Drone Flight Distance", f"{total_flight_dist_km * 1000:.1f} m")

    # --- 3D Visualization ---
    with st.container(border=True):
        st.subheader("🗺️ 3D Visualization")
        fig = go.Figure()
        
        # Plot KML Alignment
        fig.add_trace(go.Scatter3d(
            x=[c[1] for c in kml_coords], y=[c[0] for c in kml_coords], z=[0]*len(kml_coords),
            mode='lines', line=dict(color='blue', width=10), name='KML Alignment'
        ))
        
        # Plot Drone Paths
        for i, srt_file in enumerate(st.session_state.srt_files):
            flight_blocks = [b for b in processed_data if b['origin_index'] == i]
            if not flight_blocks: continue
            fig.add_trace(go.Scatter3d(
                x=[p['lon'] for p in flight_blocks], y=[p['lat'] for p in flight_blocks], z=[p['alt'] for p in flight_blocks],
                mode='lines', line=dict(color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)], width=4), name=f'Flight: {srt_file.name}'
            ))

        # Plot Start/End Markers
        start_block, end_block = processed_data[0], processed_data[-1]
        fig.add_trace(go.Scatter3d(
            x=[start_block['lon']], y=[start_block['lat']], z=[start_block['alt']],
            mode='markers+text', text=[f"Start {start_block['chainage']:.3f} km"], textposition="top center",
            marker=dict(size=5, color='green'), name='Start Point'
        ))
        fig.add_trace(go.Scatter3d(
            x=[end_block['lon']], y=[end_block['lat']], z=[end_block['alt']],
            mode='markers+text', text=[f"End {end_block['chainage']:.3f} km"], textposition="bottom center",
            marker=dict(size=5, color='darkred'), name='End Point'
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Longitude', yaxis_title='Latitude', zaxis_title='Altitude (m)',
                aspectmode='data'
            ),
            margin=dict(r=20, b=10, l=10, t=10),
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Data Export Buttons ---
    with st.container(border=True):
        st.subheader("📁 Data Export")
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            markers_kml_data = generate_markers_kml(kml_coords, st.session_state.kml_chainage_map)
            st.download_button(
                label="Download 50m Markers KML",
                data=markers_kml_data,
                file_name="markers_50m.kml",
                mime="application/vnd.google-earth.kml+xml",
                use_container_width=True
            )
        
        with dl_col2:
            route_kml_data = generate_multi_flight_kml(processed_data, st.session_state.srt_files)
            st.download_button(
                label="Download Drone Route KML",
                data=route_kml_data,
                file_name="drone_route.kml",
                mime="application/vnd.google-earth.kml+xml",
                use_container_width=True
            )
        
        with dl_col3:
            zip_buffer = generate_output_zip(processed_data, st.session_state.kml_data["name"])
            st.download_button(
                label="Download Processed SRTs (.zip)",
                data=zip_buffer,
                file_name="processed_srt_files.zip",
                mime="application/zip",
                use_container_width=True
            )
            
    st.balloons()
