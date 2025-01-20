import streamlit as st
import pandas as pd
import pydeck as pdk

# Define available map styles
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

# Preferred column order and excluded columns
DESIRED_ORDER = [
    "ID", "MP", "Subdivision", "Linecode", "Date Time", "Type", "Value", "Length",
    "Severity", "Sys", "Repeat ID", "User Review Status", "System Review Status",
    "Vehicle", "Speed", "PC", "AC", "TSC", "Heading", "Direction", "MP Major",
    "MP Minor", "Latitude", "Longitude"
]
EXCLUDE_FILTER_COLUMNS = [
    "ID", "MP", "MP Major", "MP Minor", "Date Time", "Value", "PC",
    "AC", "TSC", "Length", "Latitude", "Longitude", "Heading"
]


@st.cache_data
def load_and_process_csv(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the CSV file.
    """
    data = pd.read_csv(file_path, low_memory=False)

    # Convert Date Time to datetime, add 'Year' column
    if 'Date Time' in data.columns:
        data['Date Time'] = pd.to_datetime(
            data['Date Time'],
            format='%m/%d/%Y %H:%M:%S',
            errors='coerce'
        )
        data['Year'] = data['Date Time'].dt.year

    # Calculate MP if MP Major and MP Minor exist
    if {'MP Major', 'MP Minor'}.issubset(data.columns):
        data['MP'] = data['MP Major'] + (data['MP Minor'] / 5280)

    # Clean up Subdivision if present
    if 'Subdivision' in data.columns:
        data['Subdivision'] = data['Subdivision'].str.replace(
            r'^(Moose Jaw-|Saskatoon-)',
            '',
            regex=True
        )

    # Reorder columns
    existing_columns = [col for col in DESIRED_ORDER if col in data.columns]
    other_columns = [col for col in data.columns if col not in existing_columns]

    return data[existing_columns + other_columns]


def collect_user_filters(data: pd.DataFrame, exclude_columns: list) -> dict:
    """
    Create sidebar widgets for each column (except excluded),
    collect selected filter values, and return them in a dict.
    """
    user_filters = {}
    for column in [col for col in data.columns if col not in exclude_columns]:
        unique_values = sorted(data[column].dropna().unique())
        # Only apply a sidebar filter if there are not too many unique values
        if len(unique_values) <= 200:
            selected_values = st.sidebar.multiselect(
                f"Filter by {column}",
                options=unique_values
            )
            user_filters[column] = selected_values
    return user_filters


@st.cache_data
def apply_filters(data: pd.DataFrame, user_filters: dict) -> pd.DataFrame:
    """
    Given the user-selected filters (from collect_user_filters),
    apply them to the DataFrame and return the filtered data.
    """
    filtered_data = data.copy()
    for column, selected_values in user_filters.items():
        if selected_values:  # Only filter if user selected any values
            filtered_data = filtered_data[filtered_data[column].isin(selected_values)]
    return filtered_data


def create_map(filtered_data: pd.DataFrame,
               map_layer_type: str,
               map_style: str,
               manual_mp_input: float,
               recenter_map: bool):
    """
    Create and render a Pydeck map with IconLayer (or other layers) for the filtered data.
    """
    fields_to_show = [
        'MP', 'Subdivision', 'Linecode', 'Year', 'Type', 'Sys',
        'Value', 'Length', 'Severity', 'Speed'
    ]
    available_fields = [field for field in fields_to_show if field in filtered_data.columns]
    tooltip_dynamic_text = "<br>".join([f"<b>{field}:</b> {{{field}}}" for field in available_fields])

    TOOLTIP_TEXT = {
        "html": f"<div style='font-size: 12px;'>{tooltip_dynamic_text}</div>",
        "style": {"backgroundColor": "white", "border": "1px solid black"}
    }

    if {'Latitude', 'Longitude', 'MP'}.issubset(filtered_data.columns) and len(filtered_data) > 0:
        if recenter_map:
            # Compute distance from user input MP to find a center
            filtered_data['Distance'] = (filtered_data['MP'] - manual_mp_input).abs()
            closest_point = filtered_data.loc[filtered_data['Distance'].idxmin()]
            view_state = pdk.ViewState(
                latitude=closest_point["Latitude"],
                longitude=closest_point["Longitude"],
                zoom=14,
                pitch=0
            )
        else:
            view_state = pdk.ViewState(
                latitude=filtered_data["Latitude"].mean(),
                longitude=filtered_data["Longitude"].mean(),
                zoom=12,
                pitch=0
            )

        # Example icon
        ICON_URL = "https://upload.wikimedia.org/wikipedia/commons/6/64/Icone_Vermelho.svg"
        filtered_data["icon_data"] = [{
            "url": ICON_URL,
            "width": 128,
            "height": 128,
            "anchorY": 128
        } for _ in range(len(filtered_data))]

        # Create layers based on user-selected layer type
        if map_layer_type == "Scatterplot":
            layer = pdk.Layer(
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
        elif map_layer_type == "3D Visualization":
            layer = pdk.Layer(
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
        elif map_layer_type == "Hexagon":
            layer = pdk.Layer(
                "HexagonLayer",
                data=filtered_data,
                get_position=["Longitude", "Latitude"],
                elevation_scale=50,
                radius=200,
                extruded=True,
                pickable=True,
            )
        elif map_layer_type == "Heatmap":
            layer = pdk.Layer(
                "HeatmapLayer",
                data=filtered_data,
                get_position=["Longitude", "Latitude"],
                radiusPixels=60,
            )
        else:
            # Fallback to Scatterplot if unrecognized
            layer = pdk.Layer(
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
            layers=[layer],
            initial_view_state=view_state,
            map_style=MAP_STYLES[map_style],
            tooltip=TOOLTIP_TEXT,
        )

        st.pydeck_chart(deck_map)
    else:
        st.warning("No valid 'Latitude' or 'Longitude' columns found in the dataset, or no data to map.")


def main():
    st.set_page_config(page_title="Dynamic CSV Filter Dashboard", layout="wide")

    # Load the predefined CSV file (cached)
    data = load_and_process_csv("data.csv")

    st.write("### Data Preview")
    st.dataframe(data)
    st.write(f"Data contains {data.shape[0]} rows and {data.shape[1]} columns.")

    st.sidebar.title("Filters")
    manual_mp_input = st.sidebar.number_input(
        "Center map to MP (milepost)",
        min_value=0.0,
        step=0.1
    )
    recenter_map = st.sidebar.button("Recenter Map")
    map_layer_type = st.sidebar.selectbox(
        "Select Map Layer Type",
        ["Scatterplot", "Heatmap", "Hexagon", "3D Visualization"]
    )
    map_style = st.sidebar.selectbox(
        "Select Map Style",
        list(MAP_STYLES.keys()),
        index=0
    )

    # Collect user filter selections (not cached because it involves UI)
    user_filters = collect_user_filters(data, EXCLUDE_FILTER_COLUMNS)

    # Apply filters in a cached function
    filtered_data = apply_filters(data, user_filters)

    st.write("### Filtered Data")
    st.dataframe(filtered_data)
    st.write(f"Filtered Data contains {filtered_data.shape[0]} rows and {filtered_data.shape[1]} columns.")

    # Create and display the map
    create_map(filtered_data, map_layer_type, map_style, manual_mp_input, recenter_map)

    # Download option
    csv = filtered_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()