# streamlit_app.py

import streamlit as st
import re
import math
import zipfile
import io
import xml.et.ElementTree as ET
from datetime import datetime
import plotly.graph_objects as go
import os

# --- ADDED FOR HIGH-ACCURACY CALCULATIONS ---
from pyproj import Geod
import numpy as np
# ------------------------------------------

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
    '#A52A2A', '#00FFFF', '#FF00FF', '#808080'
] # Red, Orange, Green, Blue, Indigo, Violet, Brown, Cyan, Magenta, Grey

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

# --- MODIFIED FOR HIGH ACCURACY ---
def calculate_cumulative_dist(coords, offset=0.0):
    """
    Calculates the cumulative distance along a list of coordinates using the
    WGS84 ellipsoid for high accuracy.
    """
    if len(coords) < 2:
        return [offset]
    
    geod = Geod(ellps='WGS84')
    
    lats1 = [p[0] for p in coords[:-1]]
    lons1 = [p[1] for p in coords[:-1]]
    lats2 = [p[0] for p in coords[1:]]
    lons2 = [p[1] for p in coords[1:]]

    # Calculate distances for all segments at once (in meters)
    _, _, distances_m = geod.inv(lons1, lats1, lons2, lats2)
    
    # Create cumulative sum in km
    distances_km = np.array(distances_m) / 1000.0
    cum_dist = np.concatenate(([offset], offset + np.cumsum(distances_km)))
    
    return cum_dist.tolist()

# --- MODIFIED FOR HIGH ACCURACY ---
@st.cache_data
def project_point_to_polyline(point, polyline_coords, polyline_chainage):
    """
    Finds the closest point on a polyline to a given point using a fast
    approximation, then refines the result with high-accuracy geodesic math.
    
    Returns:
        - projected_chainage (km)
        - projected_coords (lat, lon)
        - offset_distance (m)
    """
    min_dist_sq = float('inf')
    best_proj_point = None
    best_segment_idx = -1
    
    # Use an equirectangular projection for a fast approximation to find the nearest segment.
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

    # --- ACCURACY UPGRADE: Perform accurate Geodesic calculations ---
    geod = Geod(ellps='WGS84')

    # 1. Calculate the accurate offset distance (point to its projection)
    _, _, offset_dist_m = geod.inv(
        point[1], point[0], 
        best_proj_point[1], best_proj_point[0]
    )

    # 2. Calculate the accurate distance from the start of the segment to the projected point
    start_of_segment = polyline_coords[best_segment_idx]
    _, _, dist_along_segment_m = geod.inv(
        start_of_segment[1], start_of_segment[0],
        best_proj_point[1], best_proj_point[0]
    )
    
    projected_chainage = polyline_chainage[best_segment_idx] + (dist_along_segment_m / 1000.0)

    return projected_chainage, best_proj_point, offset_dist_m

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
        # This regex now correctly handles milliseconds
        match = re.search(r"(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}[.,]\d{3})", content)
        if match:
            return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")
    except Exception:
        return None
    return None

# --- CORRECTED to handle full datetime objects ---
def parse_srt(file_content_str, origin_index, start_idx=1):
    """Parses a single SRT file's content, creating full datetime objects."""
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
        lat, lon, alt, datetime_obj = None, None, 0.0, None
        
        text_block = "\n".join(lines[j:])
        
        lat_match = re.search(r"\[latitude:\s*([0-9.\-]+)", text_block)
        lon_match = re.search(r"\[longitude:\s*([0-9.\-]+)", text_block)
        alt_match = re.search(r"\[altitude:\s*([0-9.\-]+)", text_block)
        time_match = re.search(r"(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}[.,]\d{3})", text_block)

        if lat_match: lat = float(lat_match.group(1))
        if lon_match: lon = float(lon_match.group(1))
        if alt_match: alt = float(alt_match.group(1))
        if time_match:
            try:
                datetime_obj = datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                datetime_obj = None

        if lat is not None and lon is not None and datetime_obj is not None:
            blocks.append({
                'idx': current_idx, 
                'lat': lat, 
                'lon': lon,
                'alt': alt, 
                'datetime_obj': datetime_obj, 
                'origin_index': origin_index
            })
            current_idx += 1
        
        # Move to the start of the next block
        next_block_start = file_content_str.find(f"\n\n{original_idx+2}\n", i)
        if next_block_start != -1:
            i = next_block_start
        else:
            break # End of file
            
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
    """Generates KML styles and the main document header with credits."""
    credit_text = (
        "<![CDATA["
        "# Author: Vijay Parmar<br>"
        "# Community: BGol Community of Advanced Surveying and GIS Professionals"
        "]]>"
    )
    header = (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<kml xmlns='http://www.opengis.net/kml/2.2'><Document>\n"
        f"<name>Chainage Outputs</name>\n"
        f"<description>{credit_text}</description>\n"
        "<Style id='chainStyle'><LineStyle><color>ff0000ff</color><width>4.0</width></LineStyle></Style>\n"
        "<Style id='placemarkStyle'><IconStyle><scale>0.8</scale><Icon><href>http://maps.google.com/mapfiles/ms/icons/red-dot.png</href></Icon></IconStyle></Style>\n"
    )
    styles = []
    for i in range(num_styles):
        color_bgr = PLOTLY_COLORS[i % len(PLOTLY_COLORS)].replace('#', '')
        color_abgr = f"ff{color_bgr[4:6]}{color_bgr[2:4]}{color_bgr[0:2]}"
        styles.append(
            f"<Style id='srtStyle_{i}'>"
            f"<LineStyle><color>{color_abgr}</color><width>4.0</width></LineStyle>"
            f"</Style>\n"
        )
    return header + "".join(styles)

def generate_combined_kml(kml_coords, kml_chainage_map, processed_data, srt_files):
    """Generates one KML file containing all data in organized folders."""
    num_styles = len(srt_files)
    xml = [kml_style_header(num_styles)]
    
    # --- Folder 1: User-provided Chainage KML ---
    xml.append("<Folder><name>Chainage KML</name>")
    xml.append("<Placemark><name>Original Alignment</name><styleUrl>#chainStyle</styleUrl><LineString><tessellate>1</tessellate><coordinates>")
    for lat, lon in kml_coords:
        xml.append(f"{lon:.7f},{lat:.7f},0")
    xml.append("</coordinates></LineString></Placemark>")
    xml.append("</Folder>")

    # --- Folder 2: 50m Markers ---
    xml.append("<Folder><name>markers_50m</name>")
    coords = kml_coords
    cumd = kml_chainage_map
    current_dist = cumd[0]
    total_dist = cumd[-1]
    interval_km = 0.050
    idx = 0
    
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
            xml.append(f"<Placemark><styleUrl>#placemarkStyle</styleUrl><name>{current_dist:.3f} km</name><Point><coordinates>{lon:.7f},{lat:.7f},0</coordinates></Point></Placemark>")
        elif idx == 0 and abs(current_dist - cumd[0]) < 1e-6:
             xml.append(f"<Placemark><styleUrl>#placemarkStyle</styleUrl><name>{current_dist:.3f} km</name><Point><coordinates>{coords[0][1]:.7f},{coords[0][0]:.7f},0</coordinates></Point></Placemark>")
        
        current_dist += interval_km
    xml.append("</Folder>")

    # --- Folder 3: Drone Routes (Projected over Chainage) ---
    xml.append("<Folder><name>drone_routes</name>")
    for i, srt_file in enumerate(srt_files):
        flight_blocks = [b for b in processed_data if b['origin_index'] == i]
        if not flight_blocks: continue
        
        file_name = srt_file.name
        start_block = flight_blocks[0]
        end_block = flight_blocks[-1]
        
        xml.append(f"<Folder><name>{file_name}</name>")
        xml.append(f"<Placemark><name>Route</name><styleUrl>#srtStyle_{i}</styleUrl><LineString><tessellate>1</tessellate><coordinates>")
        for b in flight_blocks:
            proj_lat, proj_lon = b['proj_coords']
            xml.append(f"{proj_lon:.7f},{proj_lat:.7f},{b['alt']:.2f}")
        xml.append("</coordinates></LineString></Placemark>")
        xml.append(f"<Placemark><name>Start: {start_block['chainage']:.3f} km</name><Point><coordinates>{start_block['proj_coords'][1]:.7f},{start_block['proj_coords'][0]:.7f},{start_block['alt']:.2f}</coordinates></Point></Placemark>")
        xml.append(f"<Placemark><name>End: {end_block['chainage']:.3f} km</name><Point><coordinates>{end_block['proj_coords'][1]:.7f},{end_block['proj_coords'][0]:.7f},{end_block['alt']:.2f}</coordinates></Point></Placemark>")
        xml.append("</Folder>")
    xml.append("</Folder>")
    
    # --- NEW FOLDER 4: Drone Full Path in 3D ---
    xml.append("<Folder><name>drone_full_path_3d</name>")
    for i, srt_file in enumerate(srt_files):
        flight_blocks = [b for b in processed_data if b['origin_index'] == i]
        if not flight_blocks: continue
        
        file_name = srt_file.name
        xml.append(f"<Placemark><name>{file_name}</name><styleUrl>#srtStyle_{i}</styleUrl><LineString>")
        xml.append("<extrude>1</extrude><tessellate>1</tessellate><altitudeMode>absolute</altitudeMode>")
        xml.append("<coordinates>")
        for b in flight_blocks:
            xml.append(f"{b['lon']:.7f},{b['lat']:.7f},{b['alt']:.2f}")
        xml.append("</coordinates></LineString></Placemark>")
    xml.append("</Folder>")
    
    xml.append("</Document></kml>")
    return "\n".join(xml)

# --- CORRECTED to generate progressive timestamps based on real datetime ---
def generate_master_zip(processed_data, kml_coords, kml_chainage_map, srt_files, kml_name):
    """Creates a single ZIP archive with all outputs and correct progressive timestamps."""
    zip_buffer = io.BytesIO()

    def _seconds_to_srt_time(seconds):
        """Helper to convert total seconds to HH:MM:SS,ms format."""
        if seconds < 0: seconds = 0
        millis = int(round((seconds * 1000) % 1000))
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. Create and add the single, combined KML
        combined_kml_data = generate_combined_kml(kml_coords, kml_chainage_map, processed_data, srt_files)
        zf.writestr("All_Outputs.kml", combined_kml_data)

        # 2. Prepare data columns for SRT files
        prefix = os.path.splitext(kml_name)[0]
        # We get the datetime_obj to create the Timer and Date files
        datetime_objs = [b['datetime_obj'] for b in processed_data]
        cols = {
            f"{prefix}_01_Latitude.srt": [f"{b['lat']:.7f}" for b in processed_data],
            f"{prefix}_02_Longitude.srt": [f"{b['lon']:.7f}" for b in processed_data],
            f"{prefix}_03_Altitude.srt": [f"{b['alt']:.2f}" for b in processed_data],
            f"{prefix}_04_Timer.srt": [dt.strftime('%H:%M:%S') for dt in datetime_objs],
            f"{prefix}_05_LatLonOutput.srt": [f"{b['lat']:.7f}, {b['lon']:.7f}" for b in processed_data],
            f"{prefix}_06_Chainage.srt": [f"{b['chainage']:.3f} km" for b in processed_data],
            f"{prefix}_07_Lateral_Offset.srt": [f"{b['offset']:.3f} m" for b in processed_data],
            f"{prefix}_08_Date.srt": [dt.strftime('%Y-%m-%d') for dt in datetime_objs],
        }

        # 3. Create progressive timestamps
        start_datetime = processed_data[0]['datetime_obj'] if processed_data else datetime.now()
        
        for filename, data_col in cols.items():
            srt_blocks = []
            for i, block_data in enumerate(processed_data):
                # THE FIX: Calculate time delta from the absolute start
                delta = block_data['datetime_obj'] - start_datetime
                start_time_sec = delta.total_seconds()
                # Use a small, consistent duration for each subtitle frame
                end_time_sec = start_time_sec + 0.033

                start_time_str = _seconds_to_srt_time(start_time_sec)
                end_time_str = _seconds_to_srt_time(end_time_sec)
                progressive_time_range = f"{start_time_str} --> {end_time_str}"
                
                srt_blocks.append(
                    f"{block_data['idx']}\n"
                    f"{progressive_time_range}\n"
                    f"{data_col[i]}"
                )
            srt_content = "\n\n".join(srt_blocks)
            zf.writestr(f"Processed_SRTs/{filename}", srt_content)
            
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
            
            kml_coords = list(reversed(st.session_state.kml_data["coords"])) if st.session_state.kml_data["swapped"] else st.session_state.kml_data["coords"]
            kml_chainage = calculate_cumulative_dist(kml_coords, st.session_state.chain_offset)
            
            merged_srt_data = merge_srt_data(st.session_state.srt_files)

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
        
        fig.add_trace(go.Scatter3d(
            x=[c[1] for c in kml_coords], y=[c[0] for c in kml_coords], z=[0]*len(kml_coords),
            mode='lines', line=dict(color='blue', width=10), name='KML Alignment'
        ))
        
        for i, srt_file in enumerate(st.session_state.srt_files):
            flight_blocks = [b for b in processed_data if b['origin_index'] == i]
            if not flight_blocks: continue
            fig.add_trace(go.Scatter3d(
                x=[p['lon'] for p in flight_blocks], y=[p['lat'] for p in flight_blocks], z=[p['alt'] for p in flight_blocks],
                mode='lines', line=dict(color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)], width=4), name=f'Flight: {srt_file.name}'
            ))

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

    # --- SINGLE DATA EXPORT BUTTON ---
    with st.container(border=True):
        st.subheader("📁 Data Export")
        st.write("Click the button below to download a single ZIP file containing all outputs: a structured KML and all processed SRT files.")

        zip_buffer = generate_master_zip(
            processed_data,
            kml_coords,
            st.session_state.kml_chainage_map,
            st.session_state.srt_files,
            st.session_state.kml_data["name"]
        )
        
        st.download_button(
            label="⬇️ Download All Outputs (.zip)",
            data=zip_buffer,
            file_name="All_Chainage_Outputs.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary"
        )
            
    st.balloons()

