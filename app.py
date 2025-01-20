import streamlit as st
import pandas as pd
from folium import Map, Marker
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium


@st.cache_data
def load_and_clean_data() -> pd.DataFrame:
    """Read, clean, and cache the dataset."""
    data = pd.read_csv("data.csv", low_memory=False)

    # Data cleaning
    if 'Date Time' in data.columns:
        data['Date Time'] = pd.to_datetime(
            data['Date Time'], format='%m/%d/%Y %H:%M:%S', errors='coerce'
        )
        data['Year'] = data['Date Time'].dt.year

    if {'MP Major', 'MP Minor'}.issubset(data.columns):
        data['MP'] = data['MP Major'] + (data['MP Minor'] / 5280)

    if 'Subdivision' in data.columns:
        data['Subdivision'] = data['Subdivision'].str.replace(
            r'^(Moose Jaw-|Saskatoon-)', '', regex=True
        )

    desired_order = [
        'ID', 'MP', 'Subdivision', 'Linecode', 'Date Time', 'Type',
        'Value', 'Length', 'Severity', 'Sys', 'Repeat ID',
        'User Review Status', 'System Review Status', 'Vehicle', 'Speed',
        'PC', 'AC', 'TSC', 'Heading', 'Direction', 'MP Major',
        'MP Minor', 'Latitude', 'Longitude'
    ]
    existing_columns = [col for col in desired_order if col in data.columns]
    other_columns = [col for col in data.columns if col not in existing_columns]
    column_order = existing_columns + other_columns

    return data[column_order]


def filter_data(data: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply filters to the dataset."""
    filtered_data = data.copy()
    for column, selected_values in filters.items():
        if selected_values:
            filtered_data = filtered_data[filtered_data[column].isin(selected_values)]
    return filtered_data


def main():
    # Page layout
    st.set_page_config(layout="wide", page_title="CSV Filter Dashboard")

    # Title
    st.title("Dynamic CSV Filter Dashboard")

    # Load data (cached and cleaned)
    data = load_and_clean_data()

    # Sidebar filter section
    with st.sidebar.expander("Data Filter", expanded=True):
        exclude_filter_columns = [
            'ID', 'MP', 'MP Major', 'MP Minor', 'Date Time', 'Value',
            'PC', 'AC', 'TSC', 'Length', 'Latitude', 'Longitude', 'Heading'
        ]

        filters = {}
        for column in [c for c in data.columns if c not in exclude_filter_columns]:
            unique_values = data[column].dropna().unique()
            if len(unique_values) <= 200:
                filters[column] = st.multiselect(f"Filter by {column}", sorted(unique_values))
            else:
                st.warning(f"Column '{column}' has too many unique values for filtering.")
                filters[column] = []

        # Apply Filter Button
        apply_filters = st.button("Apply Filters")

    # Map filter options
    with st.sidebar.expander("Map Filters", expanded=True):
        map_view_mode = st.radio(
            "Select Map View",
            ("Scatter Plot (Markers)", "Heatmap"),
            index=0
        )

        basemap_options = {
            "Street (OpenStreetMap)": "OpenStreetMap",
            "Light (CartoDB positron)": "CartoDB positron",
            "Dark (CartoDB dark_matter)": "CartoDB dark_matter"
        }
        basemap_choice = st.selectbox(
            "Select Basemap Style",
            list(basemap_options.keys()),
            index=0
        )

    # Apply filters only on button click
    if apply_filters:
        filtered_data = filter_data(data, filters)
        st.session_state['filtered_data'] = filtered_data
    else:
        filtered_data = st.session_state.get('filtered_data', data)

    # Data Preview
    with st.expander("Filtered Data Preview", expanded=True):
        st.dataframe(filtered_data)
        st.write(f"{filtered_data.shape[0]} rows × {filtered_data.shape[1]} columns")

    # Map Visualization
    with st.expander("Map Visualization", expanded=True):
        if {'Latitude', 'Longitude'}.issubset(filtered_data.columns):
            map_data = filtered_data[['Latitude', 'Longitude']].dropna()

            if not map_data.empty:
                # Limit rows dynamically for performance
                if len(map_data) > 5000:
                    st.warning("Too many points to display! Showing the top 5,000 rows.")
                    map_data = map_data.head(5000)

                avg_lat = map_data['Latitude'].mean()
                avg_lon = map_data['Longitude'].mean()
                tile_style = basemap_options[basemap_choice]
                m = Map(location=[avg_lat, avg_lon], zoom_start=6, tiles=tile_style)

                # Scatter Plot vs Heatmap
                if map_view_mode == "Scatter Plot (Markers)":
                    marker_cluster = MarkerCluster(disableClusteringAtZoom=13).add_to(m)
                    for _, row in map_data.iterrows():
                        Marker(location=[row['Latitude'], row['Longitude']]).add_to(marker_cluster)
                else:  # Heatmap
                    heat_data = map_data[['Latitude', 'Longitude']].values.tolist()
                    HeatMap(heat_data).add_to(m)

                st_folium(m, width=700, height=500)
            else:
                st.warning("No valid data points to display.")
        else:
            st.warning("Dataset lacks 'Latitude'/'Longitude' columns.")


if __name__ == "__main__":
    main()