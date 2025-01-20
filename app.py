import streamlit as st
import pandas as pd
from folium import Map, Marker
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium


@st.cache_data
def load_csv() -> pd.DataFrame:
    """Read and cache the local data.csv file."""
    return pd.read_csv("data.csv", low_memory=False)


def main():
    # Wide page layout
    st.set_page_config(layout="wide")

    # Title
    st.title("Dynamic CSV Filter Dashboard")

    # Load data (cached)
    data = load_csv()

    # Data Cleaning / Transformation
    if 'Date Time' in data.columns:
        data['Date Time'] = pd.to_datetime(
            data['Date Time'],
            format='%m/%d/%Y %H:%M:%S',
            errors='coerce'
        )
        data['Year'] = data['Date Time'].dt.year

    if {'MP Major', 'MP Minor'}.issubset(data.columns):
        data['MP'] = data['MP Major'] + (data['MP Minor'] / 5280)

    if 'Subdivision' in data.columns:
        data['Subdivision'] = data['Subdivision'].str.replace(
            r'^(Moose Jaw-|Saskatoon-)', '',
            regex=True
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
    data = data[column_order]

    # Data Preview
    with st.expander("Data Preview", expanded=False):
        st.write("### Data Preview:")
        st.dataframe(data)
        st.write(f"{data.shape[0]} rows × {data.shape[1]} columns")

    # Filter Setup
    with st.sidebar.expander("Data Filter", expanded=False):
        exclude_filter_columns = [
            'ID', 'MP', 'MP Major', 'MP Minor', 'Date Time', 'Value',
            'PC', 'AC', 'TSC', 'Length', 'Latitude', 'Longitude', 'Heading'
        ]

        filter_selections = {}
        for column in [c for c in data.columns if c not in exclude_filter_columns]:
            unique_values = data[column].dropna().unique()
            if len(unique_values) <= 200:
                # Multi-select for fewer than or equal to 200 unique values
                filter_selections[column] = st.multiselect(
                    f"Filter by {column}",
                    sorted(unique_values),
                )
            else:
                # For large unique sets, you can handle sampling here as needed
                st.warning(f"{column} has more than 200 unique values. No multi-select displayed.")
                filter_selections[column] = []

    # Map Filters
    with st.sidebar.expander("Map Filters", expanded=False):
        map_view_mode = st.radio(
            "Select Map View",
            ("Standard Marker View", "Heatmap View"),
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

    # Filter the data dynamically (no button; re-run on every change)
    filtered_data = data.copy()
    for column, selected_values in filter_selections.items():
        if selected_values:
            filtered_data = filtered_data[filtered_data[column].isin(selected_values)]

    # Show Filtered Data
    with st.expander("Filtered Data", expanded=False):
        st.write("### Filtered Data:")
        st.dataframe(filtered_data)
        st.write(f"{filtered_data.shape[0]} rows × {filtered_data.shape[1]} columns")

    # Map Visualization (expanded by default)
    with st.expander("Interactive Map Visualization", expanded=True):
        if {'Latitude', 'Longitude'}.issubset(filtered_data.columns):
            map_data = filtered_data[['Latitude', 'Longitude']].dropna()

            if not map_data.empty:
                # Optionally limit the number of rows for performance
                if len(map_data) > 10000:
                    st.warning("Too many points to display! Showing the top 10,000 rows.")
                    map_data = map_data.head(10000)

                avg_lat = map_data['Latitude'].mean()
                avg_lon = map_data['Longitude'].mean()
                tile_style = basemap_options[basemap_choice]
                m = Map(location=[avg_lat, avg_lon], zoom_start=6, tiles=tile_style)

                if map_view_mode == "Standard Marker View":
                    marker_cluster = MarkerCluster(disableClusteringAtZoom=13).add_to(m)
                    for _, row in map_data.iterrows():
                        Marker(
                            location=[row['Latitude'], row['Longitude']]
                        ).add_to(marker_cluster)
                else:
                    heat_data = map_data.values.tolist()
                    HeatMap(heat_data).add_to(m)

                st_folium(m, width=700, height=600)
            else:
                st.warning("No valid data points to display on the map.")
        else:
            st.warning("Data lacks 'Latitude'/'Longitude' columns.")


if __name__ == "__main__":
    main()