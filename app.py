import streamlit as st
import pandas as pd
import numpy as np
import leafmap.foliumap as leafmap
import folium
from folium.plugins import MarkerCluster
import os
import math
import re

# ---------- HELPER FUNCTIONS ----------

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points (meters)"""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def bearing(lat1, lon1, lat2, lon2):
    """Returns bearing in degrees from (lat1, lon1) to (lat2, lon2)"""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    diff_lon = np.radians(lon2 - lon1)
    x = np.sin(diff_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(diff_lon)
    initial_bearing = np.degrees(np.arctan2(x, y))
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def angle_diff(a1, a2):
    """Returns smallest difference between two angles (degrees)"""
    d = abs(a1 - a2) % 360
    return min(d, 360-d)

def estimate_beamwidth(freq):
    if 5925 <= freq <= 6425:   # 6 GHz
        return 4
    elif 7125 <= freq <= 8500: # 7/8 GHz
        return 3.5
    elif 10700 <= freq <= 11700: # 11 GHz
        return 2.5
    elif 12700 <= freq <= 13250: # 13 GHz
        return 2.5
    elif 14500 <= freq <= 15350: # 15 GHz
        return 2.5
    elif 17700 <= freq <= 19700: # 18 GHz
        return 1.5
    elif 21200 <= freq <= 23600: # 23 GHz
        return 1.5
    else:
        return 5  # Default wider beamwidth if unrecognized

def polarization_discrimination(plzn1, plzn2):
    """
    Returns the cross-polar discrimination factor (XPD in dB) and a boolean
    indicating whether polarization reduces interference.
    Assumes MASTER_PLZN_CODE values:
        - 'H' or 'V' (horizontal/vertical)
        - If both are present and different, XPD is typically 25-35 dB
        - If same, no discrimination
    """
    if pd.isna(plzn1) or pd.isna(plzn2):
        return 0, False
    plzn1 = str(plzn1).upper()
    plzn2 = str(plzn2).upper()
    if plzn1 != plzn2 and plzn1 in ['H', 'V'] and plzn2 in ['H', 'V']:
        return 30, True  # typical XPD value (can adjust as needed)
    return 0, False

def analyze_conflicts(df, adjacent_thresh=14, spatial_thresh=5000, beam_overlap_margin=0.5):
    """
    Adds polarization check using MASTER_PLZN_CODE.
    If polarization is orthogonal, interference risk is reduced (XPD applied).
    """
    results = []
    n = len(df)
    for i in range(n):
        for j in range(i+1, n):
            s1 = df.iloc[i]
            s2 = df.iloc[j]
            freq1 = s1['FREQ']
            freq2 = s2['FREQ']
            bw1 = s1['BWIDTH']
            bw2 = s2['BWIDTH']
            kind = "Co-channel" if abs(freq1 - freq2) < 0.01 else "Adjacent channel" if abs(freq1 - freq2) < adjacent_thresh else None
            dist = haversine(s1['SID_LAT'], s1['SID_LONG'], s2['SID_LAT'], s2['SID_LONG'])
            spatial_overlap = dist < spatial_thresh
            beam_overlap = False
            overlap_detail = ""
            polarization_effect = ""
            xpd_db, polz_reduced = polarization_discrimination(
                s1.get('MASTER_PLZN_CODE', None), s2.get('MASTER_PLZN_CODE', None)
            )
            if kind and spatial_overlap:
                if 'AZIMUTH' in df.columns:
                    try:
                        azim1 = float(s1['AZIMUTH'])
                        azim2 = float(s2['AZIMUTH'])
                        beamwidth1 = float(s1['BEAMWIDTH']) if 'BEAMWIDTH' in df.columns and not pd.isna(s1.get('BEAMWIDTH', None)) else estimate_beamwidth(freq1)
                        beamwidth2 = float(s2['BEAMWIDTH']) if 'BEAMWIDTH' in df.columns and not pd.isna(s2.get('BEAMWIDTH', None)) else estimate_beamwidth(freq2)
                        bearing12 = bearing(s1['SID_LAT'], s1['SID_LONG'], s2['SID_LAT'], s2['SID_LONG'])
                        bearing21 = bearing(s2['SID_LAT'], s2['SID_LONG'], s1['SID_LAT'], s1['SID_LONG'])
                        overlap1 = angle_diff(bearing12, azim1) <= beam_overlap_margin * beamwidth1
                        overlap2 = angle_diff(bearing21, azim2) <= beam_overlap_margin * beamwidth2
                        beam_overlap = overlap1 and overlap2
                        overlap_detail = (
                            f"Az1={{azim1:.1f}},Bear12={{bearing12:.1f}},Δ1={{angle_diff(bearing12, azim1):.1f}},BW1={{beamwidth1:.1f}}; "
                            f"Az2={{azim2:.1f}},Bear21={{bearing21:.1f}},Δ2={{angle_diff(bearing21, azim2):.1f}},BW2={{beamwidth2:.1f}}"
                        )
                    except Exception as e:
                        overlap_detail = f"Error: {{e}}"
                else:
                    overlap_detail = "No AZIMUTH info"
                if polz_reduced:
                    polarization_effect = f"Interference reduced by polarization (XPD: {{xpd_db}} dB)"
                else:
                    polarization_effect = "No polarization discrimination"
                results.append({
                    'Station 1': s1['STN_NAME'],
                    'Station 2': s2['STN_NAME'],
                    'Freq 1 (MHz)': freq1,
                    'Freq 2 (MHz)': freq2,
                    'Bandwidth 1 (MHz)': bw1,
                    'Bandwidth 2 (MHz)': bw2,
                    'Type': kind,
                    'Distance (m)': round(dist,1),
                    'Beam Overlap': "Yes" if beam_overlap else "No",
                    'Overlap Detail': overlap_detail,
                    'Polarization 1': s1.get('MASTER_PLZN_CODE', ''),
                    'Polarization 2': s2.get('MASTER_PLZN_CODE', ''),
                    'Polarization Effect': polarization_effect,
                    'Link ID 1': s1['LINK_ID'],
                    'Link ID 2': s2['LINK_ID'],
                    'City 1': s1['CITY'],
                    'City 2': s2['CITY']
                })
    return pd.DataFrame(results)

def filter_dataframe_by_query(df, query):
    """
    Filter dataframe based on natural language query
    """
    if not query:
        return df
        
    # Convert query to lowercase for case-insensitive matching
    query = query.lower()
    
    # Define patterns for common query types
    radius_pattern = r'within\s+(\d+(?:\.\d+)?)\s*(?:km|kilometer|kilometers|klm)?\s+(?:radius\s+)?(?:of|from)\s+(?:site|station)?\s*([a-zA-Z0-9]+)'
    freq_range_pattern = r'(?:with\s+)?freq(?:uency)?\s+(?:between|from)?\s*(\d+(?:\.\d+)?)\s*(?:to|until|and|-)?\s*(\d+(?:\.\d+)?)'
    site_pattern = r'(?:site|station)\s+([a-zA-Z0-9]+)'
    city_pattern = r'(?:in|at|near)\s+(?:city|kota)?\s*([a-zA-Z]+(?:\s+[a-zA-Z]+)*)'
    
    filtered_df = df.copy()
    
    # Handle radius search
    radius_match = re.search(radius_pattern, query)
    if radius_match:
        radius_km = float(radius_match.group(1)) * 1000  # Convert to meters
        site_id = radius_match.group(2)
        
        # Find the reference site
        ref_site = df[df['STN_NAME'].str.contains(site_id, case=False) | 
                     df['LINK_ID'].str.contains(site_id, case=False)]
        
        if not ref_site.empty:
            ref_lat = ref_site.iloc[0]['SID_LAT']
            ref_long = ref_site.iloc[0]['SID_LONG']
            
            # Calculate distances for all stations
            distances = []
            for idx, row in filtered_df.iterrows():
                try:
                    dist = haversine(ref_lat, ref_long, row['SID_LAT'], row['SID_LONG'])
                    distances.append(dist)
                except:
                    distances.append(float('inf'))
            
            filtered_df['distance_to_ref'] = distances
            filtered_df = filtered_df[filtered_df['distance_to_ref'] <= radius_km]
            filtered_df = filtered_df.drop(columns=['distance_to_ref'])
    
    # Handle frequency range search
    freq_match = re.search(freq_range_pattern, query)
    if freq_match:
        min_freq = float(freq_match.group(1))
        max_freq = float(freq_match.group(2)) if freq_match.group(2) else min_freq + 100
        filtered_df = filtered_df[(filtered_df['FREQ'] >= min_freq) & (filtered_df['FREQ'] <= max_freq)]
    
    # Handle site search
    site_match = re.search(site_pattern, query)
    if site_match and not radius_match:  # Only if not already handled by radius search
        site_id = site_match.group(1)
        filtered_df = filtered_df[filtered_df['STN_NAME'].str.contains(site_id, case=False) | 
                                 filtered_df['LINK_ID'].str.contains(site_id, case=False)]
    
    # Handle city search
    city_match = re.search(city_pattern, query)
    if city_match:
        city_name = city_match.group(1)
        filtered_df = filtered_df[filtered_df['CITY'].str.contains(city_name, case=False)]
    
    return filtered_df

def create_map():
    df = st.session_state['license_df']

    # Add search query sidebar
    st.sidebar.subheader("Natural Language Filter")
    search_query = st.sidebar.text_input("Enter query (e.g., 'show stations within 2 km of site ABC345' or 'stations with freq 7000 until 7100')")
    
    if search_query:
        df = filter_dataframe_by_query(df, search_query)
        if df.empty:
            st.sidebar.warning("No stations match your query. Please try a different search.")
            return
        st.sidebar.success(f"Found {{len(df)}} stations matching your query.")

    # Add frequency filter sidebar
    st.sidebar.subheader("Frequency Filter")
    # Get unique frequency values
    freq_values = sorted(df['FREQ'].dropna().unique().tolist())
    
    # Add "Select All" option
    select_all = st.sidebar.checkbox("Select All Frequencies", value=True)
    
    # Create individual checkboxes for each frequency or use multiselect if many values
    if select_all:
        selected_freqs = freq_values
    else:
        if len(freq_values) > 10:
            selected_freqs = st.sidebar.multiselect(
                "Select Frequencies (MHz)", 
                options=freq_values,
                default=freq_values
            )
        else:
            selected_freqs = []
            for freq in freq_values:
                if st.sidebar.checkbox(f"{{freq}} MHz", value=False):
                    selected_freqs.append(freq)
    
    # Filter dataframe based on selected frequencies
    filtered_df = df[df['FREQ'].isin(selected_freqs)]
    
    # Continue with map creation using filtered_df instead of df
    center_lat = filtered_df['SID_LAT'].astype(float).mean()
    center_long = filtered_df['SID_LONG'].astype(float).mean()
    m = leafmap.Map(center=(center_lat, center_long), zoom=8)
    marker_cluster = MarkerCluster().add_to(m)
    stations_by_link = {}
    
    # Use filtered_df instead of df in the loop
    for idx, row in filtered_df.iterrows():
        try:
            lat = float(row['SID_LAT'])
            long = float(row['SID_LONG'])
            if np.isnan(lat) or np.isnan(long):
                continue
            popup_html = f"""
            <h4>{{row['STN_NAME']}}</h4>
            <b>Client:</b> {{row['CLNT_NAME']}}<br>
            <b>License Number:</b> {{row['CURR_LIC_NUM']}}<br>
            <b>Link ID:</b> {{row['LINK_ID']}}<br>
            <b>TX Frequency:</b> {{row['FREQ']}} MHz<br>
            <b>RX Frequency:</b> {{row['FREQ_PAIR']}} MHz<br>
            <b>Bandwidth:</b> {{row['BWIDTH']}} MHz<br>
            <b>Equipment Model:</b> {{row['EQ_MDL']}}<br>
            <b>City:</b> {{row['CITY']}}<br>
            <b>Polarization:</b> {{row.get('MASTER_PLZN_CODE', '')}}<br>
            """
            folium.Marker(
                location=[lat, long],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=row['STN_NAME'],
                icon=folium.Icon(color='blue', icon='antenna', prefix='fa')
            ).add_to(marker_cluster)
            link_id = row['LINK_ID']
            if link_id and link_id != '':
                if link_id not in stations_by_link:
                    stations_by_link[link_id] = []
                stations_by_link[link_id].append({
                    'lat': lat, 
                    'long': long, 
                    'name': row['STN_NAME'],
                    'far_end': row.get('STASIUN_LAWAN', '')
                })
        except Exception as e:
            st.error(f"Error adding marker for {{row['STN_NAME']}}: {{e}}")
    for link_id, stations in stations_by_link.items():
        if len(stations) == 2:
            station1 = stations[0]
            station2 = stations[1]
            link_popup = f"""
            <h4>Link ID: {{link_id}}</h4>
            <b>Station 1:</b> {{station1['name']}}<br>
            <b>Station 2:</b> {{station2['name']}}<br>
            """
            folium.PolyLine(
                locations=[[station1['lat'], station1['long']], [station2['lat'], station2['long']]],
                color='red', weight=2, opacity=0.7,
                popup=folium.Popup(link_popup, max_width=300)
            ).add_to(m)
    m.to_streamlit(height=600)

def create_relations():
    license_df = st.session_state['license_df']
    inspection_df = st.session_state['inspection_df']
    merged_df = pd.merge(
        license_df, 
        inspection_df, 
        on='LINK_ID', 
        how='left', 
        suffixes=('', '_inspection')
    )
    st.session_state['merged_df'] = merged_df
    link_groups = license_df.groupby('LINK_ID')
    for link_id, group in link_groups:
        if len(group) == 2:
            stations = group.index.tolist()
            for i in range(2):
                j = 1 - i
                if pd.isna(license_df.loc[stations[i], 'FREQ_PAIR']) or license_df.loc[stations[i], 'FREQ_PAIR'] == '':
                    license_df.loc[stations[i], 'FREQ_PAIR'] = license_df.loc[stations[j], 'FREQ']
    st.session_state['license_df'] = license_df

def load_sample_data():
    try:
        license_path = 'license.csv'
        if os.path.exists(license_path):
            license_df = pd.read_csv(license_path)
            if 'EPQ_MDL' in license_df.columns:
                license_df = license_df.rename(columns={'EPQ_MDL': 'EQ_MDL'})
            if 'BWIDTH' in license_df.columns:
                license_df['BWIDTH'] = license_df['BWIDTH'].astype(float) / 1000
            st.session_state['license_df'] = license_df
            st.success("Sample license data loaded!")
    except Exception as e:
        st.error(f"Error loading sample license data: {{e}}")
    try:
        inspection_path = 'inspectionreport.csv'
        if os.path.exists(inspection_path):
            inspection_df = pd.read_csv(inspection_path)
            if 'EPQ_MDL' in inspection_df.columns:
                inspection_df = inspection_df.rename(columns={'EPQ_MDL': 'EQ_MDL'})
            if 'BWIDTH' in inspection_df.columns:
                inspection_df['BWIDTH'] = inspection_df['BWIDTH'].astype(float) / 1000
            if 'BWIDTH_Actual' in inspection_df.columns:
                inspection_df['BWIDTH_Actual'] = inspection_df['BWIDTH_Actual'].astype(float) / 1000
            st.session_state['inspection_df'] = inspection_df
            st.success("Sample inspection data loaded!")
    except Exception as e:
        st.error(f"Error loading sample inspection data: {{e}}")

def extract_screenshot_info(df):
    st.subheader("Inspection Screenshots Data")
    st.write(df)

# ---------- MAIN STREAMLIT LAYOUT ----------

st.set_page_config(page_title="MW Link Inspection", layout="wide")

for key in [
    'license_df', 'inspection_df', 'screenshots_df', 'relations_created',
    'map_created', 'conflicts'
]:
    if key not in st.session_state:
        st.session_state[key] = None

st.title("MW Link Inspection Map")
st.markdown("""
This application visualizes radio station license data and inspection results on a map.
Upload your license database, inspection reports, and screenshots to get started.
Use the Analyze feature to check for potential frequency conflicts, including polarization effects.
""")

# ---------- UPLOAD FORM ----------
st.subheader("Upload Data Files")
file_types = {
    "License Database": "license_df",
    "Inspection Database": "inspection_df",
    "Screenshots Database": "screenshots_df"
}
upload_type = st.selectbox("Select the type of file to upload", list(file_types.keys()))
upload_file = st.file_uploader(f"Upload {{upload_type}} (CSV or XLSX)", type=["csv", "xlsx"])

if upload_file is not None:
    try:
        if upload_file.name.endswith('.csv'):
            df = pd.read_csv(upload_file)
        else:
            import openpyxl # make sure openpyxl is installed!
            df = pd.read_excel(upload_file)
        if 'EPQ_MDL' in df.columns:
            df = df.rename(columns={'EPQ_MDL': 'EQ_MDL'})
        if 'BWIDTH' in df.columns:
            df['BWIDTH'] = df['BWIDTH'].astype(float) / 1000
        st.session_state[file_types[upload_type]] = df
        st.success(f"{{upload_type}} uploaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {{e}}")

if (st.session_state['license_df'] is None or st.session_state['inspection_df'] is None) and st.button("Load Sample Data"):
    load_sample_data()

if st.session_state['license_df'] is not None and st.session_state['inspection_df'] is not None:
    if st.button("Create Relations"):
        try:
            create_relations()
            st.session_state['relations_created'] = True
            st.success("Relations created successfully!")
        except Exception as e:
            st.error(f"Error creating relations: {{e}}")

if st.session_state['license_df'] is not None:
    if st.button("Analyze Potential Interference"):
        st.session_state['conflicts'] = analyze_conflicts(st.session_state['license_df'])
        st.success("Analysis complete! See results below.")

st.subheader("Microwave Link Inspection Map")
if st.session_state['license_df'] is not None:
    create_map()
else:
    st.info("Upload a license database to view the map.")

if st.session_state['conflicts'] is not None:
    st.subheader("Potential Interference Analysis Results")
    if len(st.session_state['conflicts']) > 0:
        st.dataframe(st.session_state['conflicts'])
        st.info(f"{{len(st.session_state['conflicts'])}} potential conflicts detected.")
    else:
        st.success("No conflicts found with current criteria.")

if st.session_state['screenshots_df'] is not None:
    extract_screenshot_info(st.session_state['screenshots_df'])

if st.session_state['inspection_df'] is not None:
    st.subheader("Inspection Database Table Preview")
    st.dataframe(st.session_state['inspection_df'])

if st.session_state['license_df'] is not None:
    st.subheader("License Database Table Preview")
    st.dataframe(st.session_state['license_df'])

# Add help section for natural language queries
st.sidebar.subheader("Query Examples")
st.sidebar.markdown("""
### Examples of natural language queries:
- Show stations within 2 km of site ABC345
- Stations with frequency 7000 until 7100
- Stations in city Jakarta
- Show site XYZ789

### Example in Bahasa Indonesia:
- Tampilkan stasiun dalam radius 2 km dari site ABC345
- Stasiun dengan frekuensi 7000 sampai 7100
- Stasiun di kota Jakarta
""")

# ---------- FOOTER ----------
st.markdown("""
---
**References:**  
- ITU-R F.1245-3: Mathematical model of average radiation patterns for line-of-sight point-to-point radio-relay system antennas for use in interference assessment  
- ITU-R F.699-8: Reference radiation patterns of omnidirectional antennas for the fixed service operating in the frequency range 1-30 GHz  
- ITU-R F.1336-5: Reference radiation patterns for fixed wireless system antennas for use in coordination studies and interference assessment in the frequency range from 1 GHz to 86 GHz  
- ITU-R F.758-7: System parameters and considerations in the development of criteria for sharing or compatibility between digital fixed wireless systems in the frequency range 1-86 GHz  
""")