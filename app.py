import os
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import pycountry

# Load CSV
csv_path = os.path.join(os.path.dirname(__file__), "hp_sites_with_continent.csv")
df = pd.read_csv(csv_path)

# City → Country map
city_to_country = {
    "Aguadilla": "Puerto Rico",
    "Almaty": "Kazakhstan",
    "Amsterdam": "Netherlands",
    "Ariana": "Tunisia",
    "Athens": "United States",
    "Kolkata": "India"
}
df['country'] = df['City'].map(city_to_country)

def get_iso_alpha(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None

df['iso_alpha'] = df['country'].apply(lambda x: get_iso_alpha(x) if pd.notna(x) else None)

# Risk classification
q1, q2, q3 = 42, 76, 140
df['Total_Risk_Score_Severity'] = df['Total_Risk_Score'].apply(
    lambda x: "Low" if x<q1 else "Medium" if x<q2 else "High" if x<q3 else "Very High"
)
color_map = {"Low":"#2ca02c","Medium":"#ff7f0e","High":"#d62728","Very High":"#9467bd"}

# Severity text
severity_cols = [
    'Accident_Severity','Active Shooter_Severity','Civil Unrest_Severity',
    'Criminal Activity_Severity','Health / Outbreak_Severity','Natural Disaster_Severity',
    'Others_Severity','Political Unrest_Severity','Terrorist Act_Severity',
    'Cyberattack_Severity','Power loss_Severity'
]
df['hover_text'] = df.apply(lambda r:
    f"{r['Closest HP Site']} ({r['City']})<br>Total Risk Score: {r['Total_Risk_Score']}<br>" +
    "<br>".join([f"{c}: {r[c]}" for c in severity_cols]), axis=1)

# Ensure Year column exists
if 'Year' not in df.columns:
    df['Year'] = 2023  # fallback if missing

# Dash app
app = dash.Dash(__name__)
server = app.server

# Animated scatter map
fig = px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    color="Total_Risk_Score_Severity",
    color_discrete_map=color_map,
    hover_name="City",
    hover_data={"Total_Risk_Score": True, "country": True, "Year": True},
    size="Total_Risk_Score",
    zoom=1,
    height=750,
    animation_frame="Year",
    mapbox_style="carto-darkmatter",
    title="HP Sites Global Risk Map — Animated by Year"
)

app.layout = html.Div([
    dcc.Graph(id='risk-map', figure=fig)
])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=True, host='0.0.0.0', port=port)
