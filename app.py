import streamlit as st  # Streamlit is used for creating interactive web applications
import pandas as pd  # Pandas is used for handling and analyzing data
import pydeck as pdk  # Pydeck is used for creating interactive maps

# Mapbox API Token (test without)
MAPBOX_API_TOKEN = "your_mapbox_api_token_here"

# Define available map styles to make the map visually customizable
MAP_STYLES = {
    "Satellite": "mapbox://styles/mapbox/satellite-v9",
    "Dark": "mapbox://styles/mapbox/dark-v10",
    "White": "mapbox://styles/mapbox/light-v10",
    "Outdoor": "mapbox://styles/mapbox/outdoors-v11",
    "Streets": "mapbox://styles/mapbox/streets-v11",
    "Light": "mapbox://styles/mapbox/light-v10",
    "Navigation Day": "mapbox://styles/mapbox/navigation-day-v1",
    "Navigation Night": "mapbox://styles/mapbox/navigation-night-v1",
}

# Define the preferred column order and columns to exclude for filtering
DESIRED_ORDER = ['ID', 'MP', 'Subdivision', 'Linecode', 'Date Time', 'Type', 'Value', 'Length', 'Severity',
                 'Sys', 'Repeat ID', 'User Review Status', 'System Review Status', 'Vehicle', 'Speed',
                 'PC', 'AC', 'TSC', 'Heading', 'Direction', 'MP Major', 'MP Minor', 'Latitude', 'Longitude']
EXCLUDE_FILTER_COLUMNS = ['ID', 'MP', 'MP Major', 'MP Minor', 'Date Time', 'Value', 'PC', 'AC', 'TSC', 'Length',
                          'Latitude', 'Longitude', 'Heading']

# Use caching to load and process the CSV file, so it doesn't reload unnecessarily
@st.cache_data
def load_and_process_csv(file):
    """
    Load and preprocess the CSV file.
    """
    data = pd.read_csv(file, low_memory=False)
    if 'Date Time' in data.columns:
        data['Date Time'] = pd.to_datetime(data['Date Time'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
        data['Year'] = data['Date Time'].dt.year
    if {'MP Major', 'MP Minor'}.issubset(data.columns):
        data['MP'] = data['MP Major'] + (data['MP Minor'] / 5280)
    if 'Subdivision' in data.columns:
        data['Subdivision'] = data['Subdivision'].str.replace(r'^(Moose Jaw-|Saskatoon-)', '', regex=True)
    existing_columns = [col for col in DESIRED_ORDER if col in data.columns]
    other_columns = [col for col in data.columns if col not in existing_columns]
    return data[existing_columns + other_columns]

def filter_data(data, exclude_columns):
    """
    Apply user-defined filters to the dataset.
    """
    filtered_data = data.copy()
    for column in [col for col in data.columns if col not in exclude_columns]:
        unique_values = data[column].dropna().unique()
        if len(unique_values) <= 200:
            selected_values = st.sidebar.multiselect(f"Filter by {column}", options=unique_values)
            if selected_values:
                filtered_data = filtered_data[filtered_data[column].isin(selected_values)]
    return filtered_data

def create_map(filtered_data, map_layer_type, map_style, manual_mp_input, recenter_map):
    fields_to_show = ['MP', 'Subdivision', 'Linecode', 'Year', 'Type', 'Sys', 'Value', 'Length', 'Severity', 'Speed']
    available_fields = [field for field in fields_to_show if field in filtered_data.columns]
    tooltip_dynamic_text = "<br>".join([f"<b>{field}:</b> {{{field}}}" for field in available_fields])
    TOOLTIP_TEXT = {"html": f"<div style='font-size: 12px;'>{tooltip_dynamic_text}</div>", "style": {"backgroundColor": "white", "border": "1px solid black"}}
    """
    Create and render a Pydeck map with IconLayer for filtered data.
    """
    if {'Latitude', 'Longitude', 'MP'}.issubset(filtered_data.columns):
        if recenter_map:
            filtered_data['Distance'] = (filtered_data['MP'] - manual_mp_input).abs()
            closest_point = filtered_data.loc[filtered_data['Distance'].idxmin()]
            view_state = pdk.ViewState(latitude=closest_point["Latitude"], longitude=closest_point["Longitude"], zoom=14, pitch=0)
        else:
            view_state = pdk.ViewState(
                latitude=filtered_data["Latitude"].mean(),
                longitude=filtered_data["Longitude"].mean(),
                zoom=12,
                pitch=0
            )

        ICON_URL = "https://upload.wikimedia.org/wikipedia/commons/6/64/Icone_Vermelho.svg"  # Example location marker icon
        filtered_data["icon_data"] = [{
            "url": ICON_URL,
            "width": 128,
            "height": 128,
            "anchorY": 128
        } for _ in range(len(filtered_data))]

        if map_layer_type == "Scatterplot":
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=filtered_data,
                get_position=["Longitude", "Latitude"],
                get_radius=100,
                radius_scale=1,
                radius_min_pixels=5,
                pickable=True,
                auto_highlight=True,
                get_fill_color=[255, 140, 0, 200]
            )
            deck_map = pdk.Deck(
                layers=[scatter_layer],
                initial_view_state=view_state,
                map_style=MAP_STYLES[map_style],
                api_keys={"mapbox": MAPBOX_API_TOKEN},
                tooltip=TOOLTIP_TEXT,
            )

        elif map_layer_type == "3D Visualization":
            column_layer = pdk.Layer(
                "ColumnLayer",
                data=filtered_data,
                get_position=["Longitude", "Latitude"],
                get_elevation="Value",
                elevation_scale=100,
                radius=200,
                get_fill_color=[0, 128, 255, 200],
                pickable=True,
                auto_highlight=True,
            )
            deck_map = pdk.Deck(
                layers=[column_layer],
                initial_view_state=view_state,
                map_style=MAP_STYLES[map_style],
                api_keys={"mapbox": MAPBOX_API_TOKEN},
                tooltip=TOOLTIP_TEXT,
            )

        st.pydeck_chart(deck_map)
    else:
        st.warning("No valid 'Latitude' or 'Longitude' columns found in the dataset.")

def main():
    st.set_page_config(page_title="Dynamic CSV Filter Dashboard", layout="wide")

    # Load the predefined CSV file
    data = load_and_process_csv("data.csv")
    st.write("### Data Preview")
    st.dataframe(data)
    st.write(f"Data contains {data.shape[0]} rows and {data.shape[1]} columns.")

    st.sidebar.title("Filters")
    manual_mp_input = st.sidebar.number_input("Center map to MP (milepost)", min_value=0.0, step=0.1)
    recenter_map = st.sidebar.button("Recenter Map")
    map_layer_type = st.sidebar.selectbox("Select Map Layer Type", ["Scatterplot", "Heatmap", "Hexagon", "3D Visualization"])
    map_style = st.sidebar.selectbox("Select Map Style", list(MAP_STYLES.keys()), index=0)
    filtered_data = filter_data(data, EXCLUDE_FILTER_COLUMNS)

    st.write("### Filtered Data")
    st.dataframe(filtered_data)
    st.write(f"Filtered Data contains {filtered_data.shape[0]} rows and {filtered_data.shape[1]} columns.")

    create_map(filtered_data, map_layer_type, map_style, manual_mp_input, recenter_map)

    csv = filtered_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered Data as CSV", data=csv, file_name="filtered_data.csv", mime="text/csv")


if __name__ == "__main__":
    main()