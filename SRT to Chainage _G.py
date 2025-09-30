# streamlit_app.py

import streamlit as st
import re
import math
import zipfile
import io
from lxml import etree as ET
from datetime import datetime, timedelta
import plotly.graph_objects as go
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# --- ADDED for video creation ---
# Note: moviepy is a heavy library. If not available, the video section will be disabled.
try:
    from moviepy.editor import ImageSequenceClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

# --- ADDED FOR HIGH-ACCURACY CALCULATIONS ---
from pyproj import Geod
# ------------------------------------------

# ──────────────────────────────────────────────────────────────────────────
# Author: Vijay Parmar
# Community: BGol Community of Advanced Surveying and GIS Professionals
#
# Description:
# This Streamlit application processes drone SRT subtitle files against a KML
# alignment. It calculates chainage, allows data thinning, and generates both
# standard SRT/KML outputs and video clip overlays for editing software.
# ──────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Drone SRT & Chainage v3.0", layout="wide")

# ──────────────────────────────────────────────────────────────────────────
# Constants and Session State Initialization
# ──────────────────────────────────────────────────────────────────────────

PLOTLY_COLORS = [
    '#FF0000', '#FFA500', '#008000', '#0000FF', '#4B0082', '#EE82EE', 
    '#A52A2A', '#00FFFF', '#FF00FF', '#808080'
] # Red, Orange, Green, Blue, Indigo, Violet, Brown, Cyan, Magenta, Grey

if 'kml_data' not in st.session_state:
    st.session_state.kml_data = {"name": None, "coords": None, "swapped": False}
if 'srt_files' not in st.session_state:
    st.session_state.srt_files = []


# ──────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────

def format_timedelta(td: timedelta) -> str:
    """Formats a timedelta object into HH:MM:SS."""
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def calculate_cumulative_dist(coords, offset=0.0):
    """Calculates the cumulative distance along coordinates using WGS84 ellipsoid."""
    if len(coords) < 2: return [offset]
    geod = Geod(ellps='WGS84')
    lats1, lons1 = zip(*coords[:-1])
    lats2, lons2 = zip(*coords[1:])
    _, _, distances_m = geod.inv(list(lons1), list(lats1), list(lons2), list(lats2))
    distances_km = np.array(distances_m) / 1000.0
    cum_dist = np.concatenate(([offset], offset + np.cumsum(distances_km)))
    return cum_dist.tolist()

@st.cache_data
def project_point_to_polyline(point, polyline_coords, polyline_chainage):
    """Finds the closest point on a polyline to a given point with high accuracy."""
    min_dist_sq = float('inf')
    best_proj_point = None
    best_segment_idx = -1
    avg_lat_rad = math.radians(point[0])
    cos_avg_lat = math.cos(avg_lat_rad)
    px, py = point[1] * cos_avg_lat, point[0]

    for i in range(len(polyline_coords) - 1):
        a, b = polyline_coords[i], polyline_coords[i+1]
        ax, ay = a[1] * cos_avg_lat, a[0]
        bx, by = b[1] * cos_avg_lat, b[0]
        apx, apy = px - ax, py - ay
        abx, aby = bx - ax, by - ay
        ab_mag_sq = abx**2 + aby**2
        t = 0 if ab_mag_sq == 0 else (apx * abx + apy * aby) / ab_mag_sq
        t = max(0, min(1, t))
        proj_x, proj_y = ax + t * abx, ay + t * aby
        dist_sq = (px - proj_x)**2 + (py - proj_y)**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_proj_point = (proj_y, proj_x / cos_avg_lat)
            best_segment_idx = i

    geod = Geod(ellps='WGS84')
    _, _, offset_dist_m = geod.inv(point[1], point[0], best_proj_point[1], best_proj_point[0])
    start_of_segment = polyline_coords[best_segment_idx]
    _, _, dist_along_segment_m = geod.inv(start_of_segment[1], start_of_segment[0], best_proj_point[1], best_proj_point[0])
    projected_chainage = polyline_chainage[best_segment_idx] + (dist_along_segment_m / 1000.0)
    return projected_chainage, best_proj_point, offset_dist_m

# ──────────────────────────────────────────────────────────────────────────
# File Parsing Functions
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data
def parse_kml_2d(uploaded_file):
    """Parses a KML file to extract the first LineString coordinates."""
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    file_content = uploaded_file.getvalue()
    uploaded_file.seek(0)
    tree = ET.parse(io.BytesIO(file_content))
    coord_element = tree.find('.//kml:LineString/kml:coordinates', namespaces=ns)
    if coord_element is None:
        st.error("Could not find a LineString in the KML file.")
        return []
    coords = []
    if coord_element.text:
        for part in coord_element.text.strip().split():
            try:
                lon, lat, *_ = part.split(',')
                coords.append((float(lat), float(lon)))
            except ValueError:
                continue
    return coords

@st.cache_data
def get_srt_file_info(uploaded_file):
    """Efficiently reads an SRT file to get its start time and duration."""
    try:
        content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        uploaded_file.seek(0)
        timestamps = re.findall(r"(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}[.,]\d{3})", content)
        if not timestamps: return None, None
        datetimes = [datetime.strptime(ts.replace(',', '.'), "%Y-%m-%d %H:%M:%S.%f") for ts in timestamps]
        start_time = min(datetimes) if datetimes else None
        duration = max(datetimes) - start_time if len(datetimes) > 1 else timedelta(seconds=0)
        return start_time, duration
    except Exception:
        return None, None

def parse_srt(file_content_str, origin_index, start_idx=1):
    """Parses a single SRT file's content, creating full datetime objects."""
    blocks = []
    current_idx = start_idx
    srt_blocks = file_content_str.strip().split('\n\n')
    for block_text in srt_blocks:
        if not block_text.strip(): continue
        lat, lon, alt, datetime_obj = None, None, 0.0, None
        lat_match = re.search(r"\[latitude:\s*([0-9.\-]+)", block_text)
        lon_match = re.search(r"\[longitude:\s*([0-9.\-]+)", block_text)
        alt_match = re.search(r"\[altitude:\s*([0-9.\-]+)", block_text)
        time_match = re.search(r"(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}[.,]\d{3})", block_text)
        if lat_match: lat = float(lat_match.group(1))
        if lon_match: lon = float(lon_match.group(1))
        if alt_match: alt = float(alt_match.group(1))
        if time_match:
            try:
                time_str = time_match.group(1).replace(',', '.')
                datetime_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                datetime_obj = None
        if lat is not None and lon is not None and datetime_obj is not None:
            blocks.append({'idx': current_idx, 'lat': lat, 'lon': lon, 'alt': alt, 'datetime_obj': datetime_obj, 'origin_index': origin_index})
            current_idx += 1
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
# KML, ZIP, and Video Generation Functions
# ──────────────────────────────────────────────────────────────────────────

def kml_style_header(num_styles):
    """Generates KML styles and the main document header with credits."""
    credit_text = "<![CDATA[# Author: Vijay Parmar<br># Community: BGol Community of Advanced Surveying and GIS Professionals]]>"
    header = (f"<?xml version='1.0' encoding='UTF-8'?>\n<kml xmlns='http://www.opengis.net/kml/2.2'><Document>\n"
              f"<name>Chainage Outputs</name>\n<description>{credit_text}</description>\n"
              f"<Style id='chainStyle'><LineStyle><color>ff0000ff</color><width>4.0</width></LineStyle></Style>\n"
              f"<Style id='placemarkStyle'><IconStyle><scale>0.8</scale><Icon><href>http://maps.google.com/mapfiles/ms/icons/red-dot.png</href></Icon></IconStyle></Style>\n")
    styles = []
    for i in range(num_styles):
        color_bgr = PLOTLY_COLORS[i % len(PLOTLY_COLORS)].replace('#', '')
        color_abgr = f"ff{color_bgr[4:6]}{color_bgr[2:4]}{color_bgr[0:2]}"
        styles.append(f"<Style id='srtStyle_{i}'><LineStyle><color>{color_abgr}</color><width>4.0</width></LineStyle></Style>\n")
    return header + "".join(styles)

def generate_combined_kml(kml_coords, kml_chainage_map, processed_data, srt_files):
    """Generates one KML file containing all data in organized folders."""
    # This function remains the same as your working version
    num_styles = len(srt_files)
    xml = [kml_style_header(num_styles)]
    xml.append("<Folder><name>Chainage KML</name><Placemark><name>Original Alignment</name><styleUrl>#chainStyle</styleUrl><LineString><tessellate>1</tessellate><coordinates>")
    xml.append(" ".join([f"{lon:.7f},{lat:.7f},0" for lat, lon in kml_coords]))
    xml.append("</coordinates></LineString></Placemark></Folder>")
    xml.append("<Folder><name>markers_50m</name>")
    coords, cumd = kml_coords, kml_chainage_map
    current_dist, total_dist, interval_km, idx = cumd[0], cumd[-1], 0.050, 0
    while current_dist <= total_dist:
        while idx < len(cumd) - 1 and cumd[idx+1] < current_dist: idx += 1
        if idx >= len(coords) - 1: break
        A, B, segment_len = coords[idx], coords[idx+1], cumd[idx+1] - cumd[idx]
        if segment_len > 0:
            fraction = (current_dist - cumd[idx]) / segment_len
            lat, lon = A[0] + fraction * (B[0] - A[0]), A[1] + fraction * (B[1] - A[1])
            xml.append(f"<Placemark><styleUrl>#placemarkStyle</styleUrl><name>{current_dist:.3f} km</name><Point><coordinates>{lon:.7f},{lat:.7f},0</coordinates></Point></Placemark>")
        elif idx == 0 and abs(current_dist - cumd[0]) < 1e-6:
            xml.append(f"<Placemark><styleUrl>#placemarkStyle</styleUrl><name>{current_dist:.3f} km</name><Point><coordinates>{coords[0][1]:.7f},{coords[0][0]:.7f},0</coordinates></Point></Placemark>")
        current_dist += interval_km
    xml.append("</Folder>")
    xml.append("<Folder><name>drone_routes</name>")
    for i, srt_file in enumerate(srt_files):
        flight_blocks = [b for b in processed_data if b['origin_index'] == i]
        if not flight_blocks: continue
        file_name, start_block, end_block = srt_file.name, flight_blocks[0], flight_blocks[-1]
        xml.append(f"<Folder><name>{file_name}</name>")
        xml.append(f"<Placemark><name>Route</name><styleUrl>#srtStyle_{i}</styleUrl><LineString><tessellate>1</tessellate><coordinates>")
        xml.append(" ".join([f"{b['proj_coords'][1]:.7f},{b['proj_coords'][0]:.7f},{b['alt']:.2f}" for b in flight_blocks]))
        xml.append("</coordinates></LineString></Placemark>")
        xml.append(f"<Placemark><name>Start: {start_block['chainage']:.3f} km</name><Point><coordinates>{start_block['proj_coords'][1]:.7f},{start_block['proj_coords'][0]:.7f},{start_block['alt']:.2f}</coordinates></Point></Placemark>")
        xml.append(f"<Placemark><name>End: {end_block['chainage']:.3f} km</name><Point><coordinates>{end_block['proj_coords'][1]:.7f},{end_block['proj_coords'][0]:.7f},{end_block['alt']:.2f}</coordinates></Point></Placemark>")
        xml.append("</Folder>")
    xml.append("</Folder>")
    xml.append("<Folder><name>drone_full_path_3d</name>")
    for i, srt_file in enumerate(srt_files):
        flight_blocks = [b for b in processed_data if b['origin_index'] == i]
        if not flight_blocks: continue
        xml.append(f"<Placemark><name>{srt_file.name}</name><styleUrl>#srtStyle_{i}</styleUrl><LineString><extrude>1</extrude><tessellate>1</tessellate><altitudeMode>absolute</altitudeMode><coordinates>")
        xml.append(" ".join([f"{b['lon']:.7f},{b['lat']:.7f},{b['alt']:.2f}" for b in flight_blocks]))
        xml.append("</coordinates></LineString></Placemark>")
    xml.append("</Folder>")
    xml.append("</Document></kml>")
    return "\n".join(xml)

def generate_master_zip(processed_data, kml_coords, kml_chainage_map, srt_files, kml_name, thin_rate=None):
    """Creates a single ZIP archive with thinned SRTs and a combined KML."""
    # This function now also handles thinning
    zip_buffer = io.BytesIO()

    def _seconds_to_srt_time(seconds):
        if seconds < 0: seconds = 0
        millis = int(round((seconds * 1000) % 1000))
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    # Thin the data if a rate is specified
    if thin_rate and thin_rate > 0:
        interval = timedelta(seconds=1.0/thin_rate)
        thinned_data = []
        last_time = datetime.min
        for block in processed_data:
            if not thinned_data or (block['datetime_obj'] - last_time) >= interval:
                thinned_data.append(block)
                last_time = block['datetime_obj']
        data_to_process = thinned_data
    else:
        data_to_process = processed_data

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        combined_kml_data = generate_combined_kml(kml_coords, kml_chainage_map, data_to_process, srt_files)
        zf.writestr("All_Outputs.kml", combined_kml_data)

        prefix = os.path.splitext(kml_name)[0]
        datetime_objs = [b['datetime_obj'] for b in data_to_process]
        cols = {
            f"{prefix}_01_Latitude.srt": [f"{b['lat']:.7f}" for b in data_to_process],
            f"{prefix}_02_Longitude.srt": [f"{b['lon']:.7f}" for b in data_to_process],
            f"{prefix}_03_Altitude.srt": [f"{b['alt']:.2f}" for b in data_to_process],
            f"{prefix}_04_Timer.srt": [dt.strftime('%H:%M:%S') for dt in datetime_objs],
            f"{prefix}_05_LatLonOutput.srt": [f"{b['lat']:.7f}, {b['lon']:.7f}" for b in data_to_process],
            f"{prefix}_06_Chainage.srt": [f"{b['chainage']:.3f} km" for b in data_to_process],
            f"{prefix}_07_Lateral_Offset.srt": [f"{b['offset']:.3f} m" for b in data_to_process],
            f"{prefix}_08_Date.srt": [dt.strftime('%Y-%m-%d') for dt in datetime_objs],
        }

        start_datetime = data_to_process[0]['datetime_obj'] if data_to_process else datetime.now()
        
        for filename, data_col in cols.items():
            srt_blocks = []
            for i, block_data in enumerate(data_to_process):
                delta = block_data['datetime_obj'] - start_datetime
                start_time_sec = delta.total_seconds()
                # Use a small, consistent duration for each subtitle frame
                end_time_sec = start_time_sec + (1.0 / thin_rate if thin_rate else 0.033)

                start_time_str = _seconds_to_srt_time(start_time_sec)
                end_time_str = _seconds_to_srt_time(end_time_sec)
                progressive_time_range = f"{start_time_str} --> {end_time_str}"
                
                srt_blocks.append(f"{i + 1}\n{progressive_time_range}\n{data_col[i]}")
            srt_content = "\n\n".join(srt_blocks)
            zf.writestr(f"Processed_SRTs/{filename}", srt_content)
            
    zip_buffer.seek(0)
    return zip_buffer

# --- NEW: Video Generation Function ---
def generate_video_clips_zip(processed_data, kml_name, width, font_color, bg_color, fps, thin_rate):
    """Creates a ZIP file containing video clips for each SRT data type."""
    if not MOVIEPY_AVAILABLE:
        st.error("MoviePy library is not installed. Please install it (`pip install moviepy`) to generate video clips.")
        return None

    video_zip_buffer = io.BytesIO()

    # Thin the data for video generation as well
    if thin_rate and thin_rate > 0:
        interval = timedelta(seconds=1.0/thin_rate)
        thinned_data = []
        last_time = datetime.min
        for block in processed_data:
            if not thinned_data or (block['datetime_obj'] - last_time) >= interval:
                thinned_data.append(block)
                last_time = block['datetime_obj']
        data_to_process = thinned_data
    else:
        data_to_process = processed_data

    # Prepare data columns for video files
    prefix = os.path.splitext(kml_name)[0]
    datetime_objs = [b['datetime_obj'] for b in data_to_process]
    video_cols = {
        f"{prefix}_01_Latitude.mp4": [f"{b['lat']:.7f}" for b in data_to_process],
        f"{prefix}_02_Longitude.mp4": [f"{b['lon']:.7f}" for b in data_to_process],
        f"{prefix}_03_Altitude.mp4": [f"{b['alt']:.2f} m" for b in data_to_process],
        f"{prefix}_04_Timer.mp4": [dt.strftime('%H:%M:%S') for dt in datetime_objs],
        f"{prefix}_06_Chainage.mp4": [f"{b['chainage']:.3f} km" for b in data_to_process],
        f"{prefix}_07_Lateral_Offset.mp4": [f"{b['offset']:.3f} m" for b in data_to_process],
        f"{prefix}_08_Date.mp4": [dt.strftime('%Y-%m-%d') for dt in datetime_objs],
    }
    
    # Use a basic font that should be available on most systems
    try:
        font = ImageFont.truetype("arial.ttf", size=32)
    except IOError:
        font = ImageFont.load_default()

    progress_bar = st.progress(0)
    status_text = st.empty()
    num_videos = len(video_cols)

    with zipfile.ZipFile(video_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, (filename, data_col) in enumerate(video_cols.items()):
            status_text.info(f"Generating video for: `{filename}` ({i+1}/{num_videos})...")
            frames = []
            for text in data_col:
                # Dynamically determine image size
                try:
                    # Modern Pillow
                    left, top, right, bottom = font.getbbox(text)
                    text_width, text_height = right - left, bottom - top
                except AttributeError:
                    # Older Pillow
                    text_width, text_height = font.getsize(text)

                img_height = text_height + 20 # Add some padding
                img = Image.new('RGB', (width, img_height), color=bg_color)
                draw = ImageDraw.Draw(img)
                # Center the text
                draw.text(((width - text_width) / 2, 10), text, font=font, fill=font_color)
                frames.append(np.array(img))
            
            # Create video clip
            clip = ImageSequenceClip(frames, fps=fps)
            
            # Write to an in-memory file
            with io.BytesIO() as temp_buffer:
                clip.write_videofile(temp_buffer, codec="libx264", fps=fps, logger=None, preset='ultrafast')
                temp_buffer.seek(0)
                zf.writestr(filename, temp_buffer.read())
            
            progress_bar.progress((i + 1) / num_videos)

    status_text.success("All video clips generated successfully!")
    progress_bar.empty()
    video_zip_buffer.seek(0)
    return video_zip_buffer

# ──────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────

st.title("🛰️ Drone SRT & Chainage Workflow v3.0")
st.markdown(
    """**Author:** Vijay Parmar  
**Community:** BGol Community of Advanced Surveying and GIS Professionals"""
)
st.markdown("---")

# --------------------------
# Section 1: Setup & Inputs
# --------------------------
st.header("Phase 1: Setup & Inputs")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Step 1: Upload KML Alignment")
    uploaded_kml = st.file_uploader("Upload your master alignment KML file", type=["kml", "xml"])
with col2:
    st.subheader("Step 2: Upload Drone SRT Files")
    uploaded_srts = st.file_uploader("Upload one or more drone .SRT files", type="srt", accept_multiple_files=True)

if uploaded_kml:
    # KML processing logic remains the same
    # ... (code omitted for brevity but is identical to your working version)

# --- NEW: Moved processing options to their own phase ---
st.markdown("---")
st.header("Phase 2: Processing & Standard Export")

if not (uploaded_kml and uploaded_srts):
    st.info("Please upload a KML file and at least one SRT file to begin processing.")
else:
    with st.container(border=True):
        st.subheader("Processing Options")
        
        # --- NEW: Thinning Option ---
        use_thinning = st.toggle("Enable SRT Thinning (Recommended)", value=True)
        thin_rate = 0
        if use_thinning:
            thin_rate = st.number_input(
                "Data Blocks per Second", 
                min_value=1, 
                max_value=30, 
                value=5, 
                step=1, 
                help="Reduces the number of SRT entries for better performance in video editors. 5 is a good balance of accuracy and performance."
            )

        if st.button("▶ Process and Generate Outputs", type="primary", use_container_width=True):
            # --- NEW: Progress Bar ---
            st.subheader("⚙️ Processing...")
            progress_bar = st.progress(0, text="Initializing...")
            
            # (Core processing logic remains the same)
            # ... (code omitted for brevity but is identical to your working version)
            
            kml_coords = list(reversed(st.session_state.kml_data["coords"])) if st.session_state.kml_data["swapped"] else st.session_state.kml_data["coords"]
            kml_chainage = calculate_cumulative_dist(kml_coords, st.session_state.chain_offset)
            files_to_merge = [info['file'] for info in st.session_state.srt_files]
            merged_srt_data = merge_srt_data(files_to_merge)
            
            processed_data = []
            total_points = len(merged_srt_data)
            for i, block in enumerate(merged_srt_data):
                point = (block['lat'], block['lon'])
                chainage, proj_coords, offset = project_point_to_polyline(point, kml_coords, kml_chainage)
                block.update({'chainage': chainage, 'proj_coords': proj_coords, 'offset': offset})
                processed_data.append(block)
                # Update progress bar
                progress_bar.progress((i + 1) / total_points, text=f"Projecting point {i+1} of {total_points}...")

            st.session_state.processed_data = processed_data
            st.session_state.kml_chainage_map = kml_chainage
            st.session_state.final_kml_coords = kml_coords
            st.session_state.thin_rate = thin_rate if use_thinning else None
            
            progress_bar.empty()
            st.success("✅ Processing complete!")


# ----------------------------
# Section 3: Outputs & Visualization
# ----------------------------
if 'processed_data' in st.session_state:
    st.header("Phase 3: Visualization & Export")
    
    # (All visualization and standard export logic remains the same)
    # ... (code omitted for brevity but is identical to your working version)

# --- NEW: Phase 4 for Video Export ---
if 'processed_data' in st.session_state:
    st.markdown("---")
    st.header("Phase 4: Video Clip Export (Optional)")

    with st.expander("Click here to configure and generate video overlays"):
        if not MOVIEPY_AVAILABLE:
            st.error(
                "**Video Generation Disabled:** The `moviepy` library is not installed in this environment. "
                "Please run `pip install moviepy` in your terminal to enable this feature."
            )
        else:
            st.write(
                "This will generate a separate video clip for each data type (Latitude, Chainage, etc.) "
                "with the data burned in. These clips can be easily layered over your main video in an editor."
            )
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                vid_width = st.number_input("Max Video Width (px)", min_value=100, max_value=1920, value=300, step=10)
                font_color = st.color_picker("Font Color", value="#FFFFFF")
            with v_col2:
                vid_fps = 5 # As per your requirement
                st.text_input("Frame Rate (FPS)", value=f"{vid_fps} (Fixed)", disabled=True)
                bg_color = st.color_picker("Background Color", value="#008000") # Green

            if st.button("🎬 Generate and Download Video Clips", use_container_width=True):
                with st.spinner("Generating video clips... This may take a while."):
                    video_zip_buffer = generate_video_clips_zip(
                        st.session_state.processed_data,
                        st.session_state.kml_data["name"],
                        width=vid_width,
                        font_color=font_color,
                        bg_color=bg_color,
                        fps=vid_fps,
                        thin_rate=st.session_state.get('thin_rate')
                    )
                
                if video_zip_buffer:
                    st.download_button(
                        label="⬇️ Download All Video Clips (.zip)",
                        data=video_zip_buffer,
                        file_name="SRT_Video_Clips.zip",
                        mime="application/zip",
                        use_container_width=True,
                        type="primary"
                    )
