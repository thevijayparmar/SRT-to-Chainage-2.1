import streamlit as st
import re
import math
import zipfile
import io
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

# --- Dependency Check ---
# Check for plotting library (Plotly)
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Check for projection libraries (PyProj, Shapely, LXML)
try:
    from shapely.geometry import LineString, Point
    from pyproj import Proj, Transformer, CRS
    import lxml.etree
    PROJECTION_LIBS_AVAILABLE = True
except ImportError:
    PROJECTION_LIBS_AVAILABLE = False

# --- Constants ---
KML_NS = {'kml': 'http://www.opengis.net/kml/2.2'}
CREDIT_LINE = "Creator: Vijay Parmar — for BGol (Community of Surveyors and GIS Experts)"

# ──────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────
class SRTRecord(dict):
    """A dictionary-like object for a single SRT record."""
    pass

class ChainageResult(dict):
    """A dictionary-like object for a single chainage result record."""
    pass

# ──────────────────────────────────────────────────────────────────────────
# Main App UI
# ──────────────────────────────────────────────────────────────────────────
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Drone SRT & Chainage", layout="wide")
    st.title("Drone SRT & Chainage Tool")
    st.markdown(f"*{CREDIT_LINE}*")

    if not PROJECTION_LIBS_AVAILABLE:
        st.error(
            "**Critical libraries missing!** For meter-accurate results, please install `pyproj`, `shapely`, and `lxml`. "
            "Falling back to less accurate Haversine-based calculations. "
            "Install with: `pip install pyproj shapely lxml`"
        )

    # --- Session State Initialization ---
    if 'srt_files_info' not in st.session_state:
        st.session_state.srt_files_info = []
    if 'kml_data' not in st.session_state:
        st.session_state.kml_data = None
    if 'computation_results' not in st.session_state:
        st.session_state.computation_results = None
    if 'kml_swapped' not in st.session_state:
        st.session_state.kml_swapped = False


    # --- Step 1: File Upload ---
    st.header("Step 1: Upload Files")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_kml_file = st.file_uploader("Upload Chainage KML (LineString)", type=["kml", "xml"])
    with col2:
        uploaded_srt_files = st.file_uploader(
            "Upload Drone SRT Files", type="srt", accept_multiple_files=True
        )

    # --- Step 2: KML Configuration ---
    if uploaded_kml_file:
        process_kml_upload(uploaded_kml_file)

    # --- Step 3: SRT File Ordering ---
    if uploaded_srt_files:
        process_srt_uploads(uploaded_srt_files)

    # --- Step 4: Computation ---
    st.header("Step 4: Compute Chainage")
    if st.button("Compute Chainage (Merge & Project)", disabled=not (st.session_state.kml_data and st.session_state.srt_files_info)):
        with st.spinner("Processing... This may take a moment."):
            run_computation()

    # --- Step 5 & 6: Results Preview & Export ---
    if st.session_state.computation_results:
        display_results()
        display_export_options()


# ──────────────────────────────────────────────────────────────────────────
# UI Component Functions
# ──────────────────────────────────────────────────────────────────────────

def process_kml_upload(kml_file):
    """Parses the uploaded KML file and displays its properties."""
    st.header("Step 2: Configure KML Chain")
    try:
        kml_name = kml_file.name
        kml_bytes = kml_file.getvalue()
        linestrings = parse_kml_linestrings(kml_bytes, kml_name)

        if not linestrings:
            st.error("No LineString found in the KML file. Please upload a valid KML.")
            st.session_state.kml_data = None
            return

        if len(linestrings) > 1:
            st.warning(f"Found {len(linestrings)} LineStrings. Using the first one.")

        coords = linestrings[0]
        if len(coords) < 2:
            st.error("The KML LineString must have at least 2 vertices.")
            st.session_state.kml_data = None
            return

        # Store original coordinates
        if st.session_state.kml_data is None or st.session_state.kml_data.get('name') != kml_name:
             st.session_state.kml_data = {
                 'name': kml_name,
                 'original_coords': coords,
                 'swapped': st.session_state.kml_swapped
             }

        display_kml_info()

    except Exception as e:
        st.error(f"Error parsing KML file: {e}")
        st.session_state.kml_data = None

def display_kml_info():
    """Displays KML info and the swap toggle."""
    if not st.session_state.kml_data:
        return

    st.session_state.kml_swapped = st.toggle(
        "Swap chain start/end direction",
        value=st.session_state.kml_data.get('swapped', False),
        key='kml_swap_toggle'
    )
    st.session_state.kml_data['swapped'] = st.session_state.kml_swapped

    coords = list(reversed(st.session_state.kml_data['original_coords'])) if st.session_state.kml_swapped else st.session_state.kml_data['original_coords']
    st.session_state.kml_data['coords'] = coords

    length_km = calculate_linestring_length(coords)
    st.session_state.kml_data['length_km'] = length_km

    st.info(
        f"**KML Loaded:**\n"
        f"- **Vertices:** {len(coords)}\n"
        f"- **Total Length:** {length_km:.3f} km\n"
        f"- **Start (Lat, Lon):** {coords[0][0]:.6f}, {coords[0][1]:.6f}\n"
        f"- **End (Lat, Lon):** {coords[-1][0]:.6f}, {coords[-1][1]:.6f}"
    )


def process_srt_uploads(srt_files):
    """Parses uploaded SRT files, sorts them, and displays reordering UI."""
    st.header("Step 3: Order SRT Flights")
    
    current_files = {f.name for f in srt_files}
    previous_files = {f_info['name'] for f_info in st.session_state.srt_files_info}

    if current_files != previous_files:
        parsed_files = []
        for srt_file in srt_files:
            try:
                records, warnings = parse_srt(srt_file.getvalue(), srt_file.name)
                if records:
                    first_timestamp = min(rec['time'] for rec in records if rec.get('time'))
                    parsed_files.append({
                        "name": srt_file.name,
                        "records": records,
                        "first_timestamp": first_timestamp,
                        "warnings": warnings,
                    })
                else:
                    st.warning(f"Could not parse any valid records from '{srt_file.name}'.")
            except Exception as e:
                st.error(f"Error parsing SRT file '{srt_file.name}': {e}")
        
        # Auto-sort by timestamp
        st.session_state.srt_files_info = sorted(parsed_files, key=lambda x: x['first_timestamp'])
        
    display_srt_reorder_ui()


def display_srt_reorder_ui():
    """Renders the UI for reordering SRT files."""
    if not st.session_state.srt_files_info:
        return

    st.write("Drag and drop to reorder SRT files for merging. The top file is the start of the merged track.")

    for i, file_info in enumerate(st.session_state.srt_files_info):
        cols = st.columns([4, 1, 1])
        with cols[0]:
            st.text(f"{i+1}. {file_info['name']} (Starts at: {file_info['first_timestamp']})")
        with cols[1]:
            if i > 0:
                if st.button("⬆️", key=f"up_{i}", help="Move Up"):
                    st.session_state.srt_files_info.insert(i - 1, st.session_state.srt_files_info.pop(i))
                    st.rerun()
        with cols[2]:
            if i < len(st.session_state.srt_files_info) - 1:
                if st.button("⬇️", key=f"down_{i}", help="Move Down"):
                    st.session_state.srt_files_info.insert(i + 1, st.session_state.srt_files_info.pop(i))
                    st.rerun()

    if st.button("Reset to DateTime Order"):
        st.session_state.srt_files_info = sorted(st.session_state.srt_files_info, key=lambda x: x['first_timestamp'])
        st.rerun()


def run_computation():
    """Performs the main chainage computation."""
    kml_coords = st.session_state.kml_data['coords']
    srt_file_infos = st.session_state.srt_files_info

    # 1. Merge SRT files
    merged_records = []
    flight_segments = []
    start_index = 1
    warnings = []
    for file_info in srt_file_infos:
        records = file_info['records']
        for rec in records:
            rec['index'] = start_index
            start_index += 1
        merged_records.extend(records)
        flight_segments.append({
            'name': file_info['name'],
            'start_index': records[0]['index'],
            'end_index': records[-1]['index'],
            'records': records
        })
        warnings.extend(file_info['warnings'])

    # 2. Perform projection and chainage calculation
    if PROJECTION_LIBS_AVAILABLE:
        results_df, proj_transformer = compute_chainage_pyproj(merged_records, kml_coords)
    else:
        results_df = compute_chainage_fallback(merged_records, kml_coords)
        proj_transformer = None # No transformer in fallback

    st.session_state.computation_results = {
        'dataframe': results_df,
        'flight_segments': flight_segments,
        'kml_coords': kml_coords,
        'warnings': warnings,
        'proj_transformer': proj_transformer
    }
    st.success("Computation complete!")

def display_results():
    """Displays computation results, including a table and a map."""
    results = st.session_state.computation_results
    if not results:
        return

    st.header("Step 5: Results Preview")

    # Display warnings
    if results['warnings']:
        with st.expander("Show Warnings"):
            for warning in results['warnings']:
                st.warning(warning)
    
    # Table Preview
    st.subheader("Merged Track Preview")
    st.dataframe(results['dataframe'].head(20))

    # Map Preview
    st.subheader("3D Map Preview")
    if PLOTLY_AVAILABLE:
        fig = generate_3d_plot(
            results['dataframe'],
            results['kml_coords'],
            results['flight_segments']
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(
            "**Plotly library not found.** The 3D map preview cannot be displayed. "
            "To enable this feature, please install the required library by running: `pip install plotly`"
        )


def display_export_options():
    """Displays buttons for downloading all result files."""
    results = st.session_state.computation_results
    if not results:
        return

    st.header("Step 6: Export Results")

    # --- Create files in memory ---
    df = results['dataframe']
    flight_segments = results['flight_segments']
    
    # Merged files
    merged_csv_bytes = df.to_csv(index=False).encode('utf-8')
    merged_srt_bytes = records_to_srt(df.to_dict('records'), use_chainage=True).encode('utf-8')

    # Per-flight files
    per_flight_files = {}
    for segment in flight_segments:
        file_prefix = Path(segment['name']).stem
        segment_df = df[(df['index'] >= segment['start_index']) & (df['index'] <= segment['end_index'])]
        
        per_flight_files[f"{file_prefix}.csv"] = segment_df.to_csv(index=False).encode('utf-8')
        per_flight_files[f"{file_prefix}.srt"] = records_to_srt(segment_df.to_dict('records'), use_chainage=True).encode('utf-8')

    # KML file
    kml_bytes = generate_kml_output(
        results['kml_coords'],
        flight_segments,
        df
    ).encode('utf-8')

    # --- ZIP file ---
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("merged_track.csv", merged_csv_bytes)
        zf.writestr("merged_track.srt", merged_srt_bytes)
        zf.writestr("drone_flights.kml", kml_bytes)
        for name, data in per_flight_files.items():
            zf.writestr(f"per_flight/{name}", data)
    zip_buffer.seek(0)


    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button(
            label="⬇️ Download All (.zip)",
            data=zip_buffer,
            file_name="drone_chainage_results.zip",
            mime="application/zip",
        )
    with col2:
        st.download_button(
            label="Download Merged CSV",
            data=merged_csv_bytes,
            file_name="merged_track.csv",
            mime="text/csv",
        )
    with col3:
        st.download_button(
            label="Download KML",
            data=kml_bytes,
            file_name="drone_flights.kml",
            mime="application/vnd.google-earth.kml+xml",
        )


# ──────────────────────────────────────────────────────────────────────────
# Core Logic Functions
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data
def parse_kml_linestrings(kml_bytes: bytes, filename: str) -> List[List[Tuple[float, float]]]:
    """Parses a KML file and extracts all LineString coordinates."""
    root = ET.fromstring(kml_bytes)
    all_coords = []
    for coords_element in root.findall('.//kml:LineString/kml:coordinates', KML_NS):
        if coords_element.text:
            coords_list = []
            parts = coords_element.text.strip().split()
            for part in parts:
                try:
                    lon, lat, *_ = map(float, part.split(','))
                    coords_list.append((lat, lon))
                except ValueError:
                    continue # Skip malformed coordinate tuples
            if coords_list:
                all_coords.append(coords_list)
    return all_coords


@st.cache_data
def parse_srt(srt_bytes: bytes, filename: str) -> Tuple[List[SRTRecord], List[str]]:
    """Robustly parses an SRT file using multiple regex patterns."""
    content = srt_bytes.decode('utf-8', errors='ignore')
    blocks = content.split('\n\n')
    records = []
    warnings = []

    # Case-insensitive regex patterns
    PATTERNS = {
        'lat': [re.compile(r'\[latitude[:=]?\s*([\-0-9.]+)', re.IGNORECASE), re.compile(r'lat[:=]?\s*([\-0-9.]+)', re.IGNORECASE)],
        'lon': [re.compile(r'\[longitude[:=]?\s*([\-0-9.]+)', re.IGNORECASE), re.compile(r'lon[:=]?\s*([\-0-9.]+)', re.IGNORECASE)],
        'alt': [re.compile(r'\[altitude[:=]?\s*([\-0-9.]+)', re.IGNORECASE), re.compile(r'alt[:=]?\s*([\-0-9.]+)', re.IGNORECASE)],
        'time': [re.compile(r'(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)')],
        'gps_group': [re.compile(r'GPS:\s*([\-0-9.]+)\s*,\s*([\-0-9.]+)', re.IGNORECASE)]
    }

    def find_match(key, text):
        for pattern in PATTERNS[key]:
            match = pattern.search(text)
            if match:
                return match
        return None

    for block_text in blocks:
        if not block_text.strip():
            continue

        lines = block_text.strip().split('\n')
        if not lines or not lines[0].strip().isdigit():
            continue

        index = int(lines[0].strip())
        raw_block = "\n".join(lines)
        
        lat, lon, alt, time_obj = None, None, None, None

        # Try combined GPS pattern first
        gps_match = find_match('gps_group', raw_block)
        if gps_match:
            lat = float(gps_match.group(1))
            lon = float(gps_match.group(2))
        else:
            lat_match = find_match('lat', raw_block)
            lon_match = find_match('lon', raw_block)
            if lat_match: lat = float(lat_match.group(1))
            if lon_match: lon = float(lon_match.group(1))

        alt_match = find_match('alt', raw_block)
        if alt_match: alt = float(alt_match.group(1))
        
        time_match = find_match('time', raw_block)
        if time_match:
            try:
                time_str = time_match.group(1).replace(' ', 'T')
                time_obj = datetime.fromisoformat(time_str)
            except ValueError:
                pass

        if lat is not None and lon is not None and time_obj is not None:
            records.append(SRTRecord({
                'index': index,
                'time': time_obj,
                'lat': lat,
                'lon': lon,
                'alt': alt if alt is not None else 0.0,
                'raw_block': raw_block
            }))
        else:
            warnings.append(f"File '{filename}', Block #{index}: Skipped due to missing lat/lon/timestamp.")

    return records, warnings


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates Haversine distance between two points in kilometers."""
    R = 6371.0088  # Earth radius in kilometers
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_linestring_length(coords: List[Tuple[float, float]]) -> float:
    """Calculates the total length of a linestring (in km)."""
    total_length = 0.0
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i+1]
        total_length += haversine_distance(p1[0], p1[1], p2[0], p2[1])
    return total_length


def compute_chainage_pyproj(records: List[SRTRecord], kml_coords: List[Tuple[float, float]]) -> Tuple[pd.DataFrame, Transformer]:
    """Computes chainage using pyproj for accurate projection."""
    # 1. Set up projection
    kml_coords_arr = np.array(kml_coords)
    centroid = kml_coords_arr.mean(axis=0)
    
    aeqd_proj = CRS(f"+proj=aeqd +lat_0={centroid[0]} +lon_0={centroid[1]} +datum=WGS84")
    transformer = Transformer.from_crs("EPSG:4326", aeqd_proj, always_xy=True)

    # 2. Project KML chain and drone points to planar coordinates (meters)
    kml_lon, kml_lat = kml_coords_arr[:, 1], kml_coords_arr[:, 0]
    proj_kml_x, proj_kml_y = transformer.transform(kml_lon, kml_lat)
    proj_kml_points = np.column_stack([proj_kml_x, proj_kml_y])

    drone_lons = np.array([r['lon'] for r in records])
    drone_lats = np.array([r['lat'] for r in records])
    proj_drone_x, proj_drone_y = transformer.transform(drone_lons, drone_lats)
    proj_drone_points = np.column_stack([proj_drone_x, proj_drone_y])

    # 3. Precompute chain segment data
    segment_vectors = np.diff(proj_kml_points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))

    results = []
    # 4. For each drone point, find the closest projection on the chain
    for p_drone in proj_drone_points:
        min_dist = float('inf')
        best_chainage = 0.0
        
        # Calculate projection onto all segments and find the minimum distance
        p_drone_rep = np.tile(p_drone, (len(segment_vectors), 1))
        a_points = proj_kml_points[:-1]
        
        # Vectorized projection calculation
        dot_v_v = np.sum(segment_vectors * segment_vectors, axis=1)
        dot_p_a_v = np.sum((p_drone_rep - a_points) * segment_vectors, axis=1)
        
        # Avoid division by zero for zero-length segments
        t = np.divide(dot_p_a_v, dot_v_v, out=np.full_like(dot_v_v, 0), where=dot_v_v!=0)
        t_clamped = np.clip(t, 0, 1)

        projected_points = a_points + (segment_vectors * t_clamped[:, np.newaxis])
        distances = np.linalg.norm(p_drone_rep - projected_points, axis=1)
        
        # Find the segment with the minimum distance
        best_segment_idx = np.argmin(distances)
        
        min_dist = distances[best_segment_idx]
        best_t = t_clamped[best_segment_idx]
        
        chainage_m = cumulative_lengths[best_segment_idx] + best_t * segment_lengths[best_segment_idx]
        
        results.append({
            'chainage_km': chainage_m / 1000.0,
            'distance_to_chain_m': min_dist
        })
        
    df = pd.DataFrame(records)
    results_df = pd.DataFrame(results)
    final_df = pd.concat([df, results_df], axis=1)
    
    # Formatting
    final_df['chainage_km'] = final_df['chainage_km'].map('{:.3f}'.format)
    final_df['distance_to_chain_m'] = final_df['distance_to_chain_m'].map('{:.2f}'.format)
    final_df['timestamp'] = final_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return final_df[['index', 'timestamp', 'lat', 'lon', 'alt', 'chainage_km', 'distance_to_chain_m']], transformer


def compute_chainage_fallback(records: List[SRTRecord], kml_coords: List[Tuple[float, float]]) -> pd.DataFrame:
    """Fallback chainage computation using Haversine (less accurate)."""
    # Precompute cumulative distance along the chain
    cum_dist_km = [0.0]
    for i in range(len(kml_coords) - 1):
        p1 = kml_coords[i]
        p2 = kml_coords[i+1]
        cum_dist_km.append(cum_dist_km[-1] + haversine_distance(p1[0], p1[1], p2[0], p2[1]))

    results = []
    for rec in records:
        # Find the closest vertex on the chain (this is the approximation)
        drone_point = (rec['lat'], rec['lon'])
        distances_to_vertices = [haversine_distance(drone_point[0], drone_point[1], v[0], v[1]) for v in kml_coords]
        closest_vertex_idx = np.argmin(distances_to_vertices)
        
        chainage_km = cum_dist_km[closest_vertex_idx]
        distance_to_chain_m = distances_to_vertices[closest_vertex_idx] * 1000

        results.append({
            'chainage_km': f"{chainage_km:.3f}",
            'distance_to_chain_m': f"{distance_to_chain_m:.2f}"
        })

    df = pd.DataFrame(records)
    results_df = pd.DataFrame(results)
    final_df = pd.concat([df, results_df], axis=1)
    final_df['timestamp'] = final_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return final_df[['index', 'timestamp', 'lat', 'lon', 'alt', 'chainage_km', 'distance_to_chain_m']]


def generate_3d_plot(df: pd.DataFrame, kml_coords: List[Tuple[float, float]], flight_segments: List[Dict]) -> go.Figure:
    """Generates a 3D Plotly figure for visualization."""
    fig = go.Figure()

    # Plot Chain Polyline
    kml_arr = np.array(kml_coords)
    fig.add_trace(go.Scatter3d(
        x=kml_arr[:, 1], y=kml_arr[:, 0], z=np.zeros(len(kml_arr)),
        mode='lines',
        line=dict(color='blue', width=8),
        name='Chain Polyline'
    ))

    # Plot each flight segment with a different color
    colors = ['red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'olive', 'gray', 'lime']
    for i, segment in enumerate(flight_segments):
        segment_df = df[(df['index'] >= segment['start_index']) & (df['index'] <= segment['end_index'])]
        color = colors[i % len(colors)]
        
        # Flight path
        fig.add_trace(go.Scatter3d(
            x=segment_df['lon'], y=segment_df['lat'], z=segment_df['alt'],
            mode='lines',
            line=dict(color=color, width=4),
            name=f"Flight: {Path(segment['name']).stem}"
        ))

        # Start/End markers
        start_row = segment_df.iloc[0]
        end_row = segment_df.iloc[-1]
        fig.add_trace(go.Scatter3d(
            x=[start_row['lon'], end_row['lon']], y=[start_row['lat'], end_row['lat']], z=[start_row['alt'], end_row['alt']],
            mode='markers+text',
            marker=dict(size=5, color=color, symbol='diamond'),
            text=[f"START: {start_row['chainage_km']} km", f"END: {end_row['chainage_km']} km"],
            textposition="top center",
            name=f"Markers: {Path(segment['name']).stem}"
        ))

    fig.update_layout(
        title="3D View of Drone Flights and Chainage",
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Altitude (m)',
            aspectmode='data' # This makes the aspect ratio realistic
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600
    )
    return fig


def records_to_srt(records: List[Dict], use_chainage=False) -> str:
    """Converts a list of records back to SRT format."""
    srt_blocks = []
    for i, rec in enumerate(records):
        start_time = f"00:00:{i:02d},000"
        end_time = f"00:00:{i+1:02d},000"
        
        if use_chainage:
            content = f"Chainage: {rec['chainage_km']} km | Dist: {rec['distance_to_chain_m']} m\n" \
                      f"Lat: {rec['lat']:.6f}, Lon: {rec['lon']:.6f}, Alt: {rec['alt']:.2f}"
        else:
            content = f"Lat: {rec['lat']:.6f}, Lon: {rec['lon']:.6f}, Alt: {rec['alt']:.2f}"

        srt_blocks.append(f"{rec['index']}\n{start_time} --> {end_time}\n{content}")
    return "\n\n".join(srt_blocks)


def generate_kml_output(kml_coords, flight_segments, df) -> str:
    """Generates the final KML file with multiple colored flight paths."""
    
    # Color palette
    colors = ['ff0000ff', 'ff00ff00', 'ff00a5ff', 'ffff0000', 'ff00ffff', 'ffff00ff', 'ffffff00', 'ff458b00', 'ffe6e6fa', 'ff2e8b57', 'ffa020f0', 'ff00ced1'] # Red, Green, Orange, Blue, Cyan, Magenta, etc. (AABBGGRR format)

    # KML Document Start
    kml = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        f"<!-- {CREDIT_LINE} -->",
        f"<kml xmlns='{KML_NS['kml']}'>",
        "<Document>",
        f"<name>Drone Flight Chainage</name>",
        f"<description>{CREDIT_LINE}</description>"
    ]
    
    # Styles
    kml.append("<Style id='chainStyle'><LineStyle><color>ff00ffff</color><width>4</width></LineStyle></Style>") # Yellow for chain
    for i in range(len(flight_segments)):
        color = colors[i % len(colors)]
        kml.append(f"<Style id='flightStyle_{i}'><LineStyle><color>{color}</color><width>3</width></LineStyle></Style>")

    # Chain Placemark
    kml.append("<Folder><name>Chain Polyline</name>")
    kml.append("<Placemark><name>Chain</name><styleUrl>#chainStyle</styleUrl><LineString><coordinates>")
    kml.append(" ".join([f"{lon:.6f},{lat:.6f},0" for lat, lon in kml_coords]))
    kml.append("</coordinates></LineString></Placemark>")
    kml.append("</Folder>")

    # Flight Placemarks
    kml.append("<Folder><name>Drone Flights</name>")
    for i, segment in enumerate(flight_segments):
        segment_df = df[(df['index'] >= segment['start_index']) & (df['index'] <= segment['end_index'])]
        file_stem = Path(segment['name']).stem
        
        kml.append(f"<Folder><name>{file_stem}</name>")
        # LineString for the flight
        kml.append(f"<Placemark><name>{file_stem}</name><styleUrl>#flightStyle_{i}</styleUrl><LineString><coordinates>")
        kml.append(" ".join([f"{row['lon']:.6f},{row['lat']:.6f},{row['alt']:.2f}" for _, row in segment_df.iterrows()]))
        kml.append("</coordinates></LineString></Placemark>")

        # Start/End Placemarks
        start_row = segment_df.iloc[0]
        end_row = segment_df.iloc[-1]
        kml.append(f"<Placemark><name>START: {start_row['chainage_km']} km</name><Point><coordinates>{start_row['lon']:.6f},{start_row['lat']:.6f},{start_row['alt']:.2f}</coordinates></Point></Placemark>")
        kml.append(f"<Placemark><name>END: {end_row['chainage_km']} km</name><Point><coordinates>{end_row['lon']:.6f},{end_row['lat']:.6f},{end_row['alt']:.2f}</coordinates></Point></Placemark>")
        kml.append("</Folder>")

    kml.append("</Folder>")
    
    # KML Document End
    kml.append("</Document></kml>")
    return "\n".join(kml)


if __name__ == "__main__":
    main()

