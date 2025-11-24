
import dash
from dash import dcc, html
from dash.dependencies import Output, Input, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.interpolate import griddata


# Load CSV
csv_path = os.path.join(os.path.dirname(__file__), "predicted_crime_corrected.csv")
df = pd.read_csv(csv_path)

# ---------------- LOAD DATA ----------------
# df = pd.read_csv("predicted_crime_corrected.csv")
df['Predicted_Crimes'] = pd.to_numeric(df['Predicted_Crimes'], errors='coerce')
df = df.dropna(subset=['Predicted_Crimes', 'Crime_Type', 'Year', 'Month'])
df['Year'] = df['Year'].astype(int)
df['Month'] = df['Month'].astype(int)

# ---------------- 1️⃣ 3D Scatter ----------------
fig_3d = px.scatter_3d(
    df,
    x='Year',
    y='Month',
    z='Predicted_Crimes',
    color='Crime_Type',
    title='3D Visualization of Predicted Crimes',
    hover_name='Crime_Type'
)
fig_3d.update_layout(width=1300, height=760,legend=dict(
        x=0.8,          # horizontal position (0=left, 1=right)
        y=0.8,          # vertical position (0=bottom, 1=top)
        xanchor='left', # or 'center', 'right'
        yanchor='top',  # or 'middle', 'bottom'
        orientation='v', # 'v' vertical, 'h' horizontal
        bgcolor='rgba(255,255,255,0.5)', # optional background
        bordercolor='black',
        borderwidth=1
    ))


# ---------------- 2️⃣ Pie Chart for December ----------------
df_dec = df[df['Month'] == 12]
df_grouped = df_dec.groupby(['Year', 'Crime_Type'], as_index=False)['Predicted_Crimes'].sum()
years = sorted(df_grouped['Year'].unique())

fig_pie = px.pie(
    df_grouped[df_grouped['Year'] == 2025],
    values='Predicted_Crimes',
    names='Crime_Type',
    title='Crime Types in 2025',
    hole=0.3
)
fig_pie.update_traces(textposition='inside', textinfo='label+percent')

# Dropdown for years
buttons = []
for year in years:
    filtered_df = df_grouped[df_grouped['Year'] == year]
    buttons.append(dict(
        label=str(year),
        method='update',
        args=[{'values': [filtered_df['Predicted_Crimes']],
               'labels': [filtered_df['Crime_Type']],
               'title': f'Crime Types {year}'}]
    ))
#fig_pie.update_layout(updatemenus=[dict(type='dropdown', buttons=buttons, x=1.1, y=1.1)])
#fig_pie.update_layout(
#    updatemenus=[dict(type='dropdown', buttons=buttons, x=1.1, y=1.1)],
#    width=1200,   # set desired width in pixels
#    height=600   # set desired height in pixels
#)

fig_pie.update_layout(
    updatemenus=[dict(type='dropdown', buttons=buttons, x=0.9, y=1.0)],
    width=1200,
    height=600,
    legend=dict(
        x=0.85,        # horizontal position (0=left, 1=right)
        y=0.5,         # vertical position (0=bottom, 1=top)
        xanchor='left',  # anchors the legend box horizontally
        yanchor='middle', # anchors the legend box vertically
        bgcolor='rgba(255,255,255,0.7)',  # optional background
        bordercolor='black',
        borderwidth=1
    )
)



# ---------------- 3️⃣ Treemap ----------------
df_sorted = df.sort_values(['Year', 'Month'])
df_sorted['Year_str'] = df_sorted['Year'].astype(str)
df_sorted['Month_str'] = df_sorted['Month'].astype(str).str.zfill(2)
fig_treemap = px.treemap(
    df_sorted,
    path=['Year_str', 'Month_str', 'Crime_Type'],
    values='Predicted_Crimes',
    color='Predicted_Crimes',
    color_continuous_scale='RdBu',
    title='Predicted Crimes by Year, Month, and Crime Type'
)
fig_treemap.update_layout(margin=dict(t=50, l=0, r=0, b=0), coloraxis_colorbar=dict(title='Predicted Crimes'))

# ---------------- 45 3d Surface  ----------------


# Prepare grid
year_grid = np.linspace(df['Year'].min(), df['Year'].max(), 50)
month_grid = np.linspace(df['Month'].min(), df['Month'].max(), 40)
X, Y = np.meshgrid(year_grid, month_grid)
crime_types = df['Crime_Type'].unique()
colors = px.colors.qualitative.Bold

fig_surface = go.Figure()

for i, crime in enumerate(crime_types):
    df_c = df[df['Crime_Type'] == crime]
    Z = griddata(
        (df_c['Year'], df_c['Month']),
        df_c['Predicted_Crimes'],
        (X, Y),
        method='cubic'
    )
    color = colors[i % len(colors)]
    fig_surface.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        name=crime,
        legendgroup=crime,
        showlegend=True,
        colorscale=[[0, color],[1,color]],
        showscale=False,
        opacity=0.55,
        hovertemplate=(
            "<b>Crime Type:</b> " + crime + "<br>" +
            "<b>Year:</b> %{x}<br>" +
            "<b>Month:</b> %{y}<br>" +
            "<b>Predicted Crimes:</b> %{z}<extra></extra>"
        )
    ))

# Camera + rotation lock
fig_surface.update_scenes(
    camera=dict(eye=dict(x=1.8, y=-1.0, z=1.0)),  # lower start y
    dragmode='orbit',
    aspectratio=dict(x=1.4, y=1, z=0.7)
)

# Layout
#fig_surface.update_layout(title="Crime Prediction Surfaces by Crime Type",width=1200,\
 #   height=900,legend=dict(title="Crime Types",itemsizing='constant',\
 #       bgcolor="rgba(255,255,255,0.7)",x=1.05, y=1),scene=dict(xaxis_title='Year', yaxis_title='Month',zaxis_title='Predicted Crimes'))

#fig_surface.update_layout(
 #   title="Crime Prediction Surfaces by Crime Type",
  #  width=1200,
   # height=1100,
   # margin=dict(l=10, r=50, t=40, b=180),  # move chart lower (higher y position visually)
   # legend=dict(
   #     title="Crime Types",
   #     itemsizing='constant',
   #     bgcolor="rgba(255,255,255,0.7)",
   #     x=1.05,
   #     y=1
   # ),
   # scene=dict(
   #     xaxis_title='Year',
   #     yaxis_title='Month',
   #     zaxis_title='Predicted Crimes'
   # )
#)
fig_surface.update_layout(
    title="Crime Prediction Surfaces by Crime Type",
    width=1200,
    height=1100,
    margin=dict(l=10, r=50, t=40, b=180),  # chart lower
    legend=dict(
        title="Crime Types",
        itemsizing='constant',
        bgcolor="rgba(255,255,255,0.7)",
        x=1.05,
        y=0.5,           # lower y moves legend down
        yanchor='bottom'  # y refers to bottom of legend box
    ),
    scene=dict(
        xaxis_title='Year',
        yaxis_title='Month',
        zaxis_title='Predicted Crimes'
    )
)


# ---------------- 4️⃣ Animated December Bar Chart ----------------
df_dec_filtered = df[df['Year'].isin(range(2025, 2046)) & (df['Month']==12)]
fig_bar = px.bar(
    df_dec_filtered,
    y='Predicted_Crimes',
    x=df_dec_filtered.index,
    color='Crime_Type',
    animation_frame='Year',
    title='Predicted Crimes by Crime Type (2025-2045)',
    labels={'Predicted_Crimes':'Predicted Crimes'}
)

fig_bar.update_layout(
    width=1300,
    height=700,
    legend=dict(
        x=1.02,        # right of chart
        y=1,
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(255,255,255,0.5)',
        bordercolor='black',
        borderwidth=1,
        traceorder='normal',
        orientation='v',   # vertical layout
        itemsizing='constant',
        font=dict(size=12),
    )
)

fig_bar.update_xaxes(showticklabels=False, title_text='')

# ---------------- 5️⃣ London Map ----------------
df_london = df_dec_filtered.copy()
unique_years = sorted(df_london['Year'].unique())
LONDON = {"lat": 51.5074, "lon": -0.1278, "city": "LONDON"}

# ---------------- DASH APP ----------------
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("Crime Data Visualizations (2025-2045)", style={"textAlign":"center", "padding":"10px", "backgroundColor":"#EEE"}),
    
    dcc.Tabs(id="tabs", value='tab-3d', children=[
        dcc.Tab(label='London Map', value='tab-map', children=[
            dcc.Graph(id='crime-map'),
            html.Div([
                html.Button(
                    "Play",
                    id="playpause-btn",
                    n_clicks=0,
                    style={
                        "display": "inline-block",
                        "marginTop": "15px",
                        "padding": "14px 30px",
                        "fontSize": "20px",
                        "backgroundColor": "#0066FF",
                        "color": "white",
                        "borderRadius": "10px",
                        "cursor": "pointer",
                        "border": "none",
                        "marginLeft":"20px"
                    }
                )
            ]),
            dcc.Slider(
                id="year-slider",
                min=unique_years[0],
                max=unique_years[-1],
                value=unique_years[0],
                step=None,
                marks={str(y): str(y) for y in unique_years},
                tooltip={"placement":"bottom","always_visible":True}
            ),
            dcc.Interval(id="auto-interval", interval=1200, n_intervals=0, disabled=True),
            dcc.Interval(id="blink-interval", interval=500, n_intervals=0),
            dcc.Store(id='current-year', data=unique_years[0])  # Store selected year
        ]),
        dcc.Tab(label='3D Scatter Chart', value='tab-3d', children=[
            dcc.Graph(figure=fig_3d)
        ]),
        dcc.Tab(label='3D Crime Surfaces', value='tab-surface', children=[
            dcc.Graph(figure=fig_surface)
        ]),
        dcc.Tab(label='Pie Chart', value='tab-pie', children=[
            dcc.Graph(figure=fig_pie)
        ]),
        dcc.Tab(label='Treemap', value='tab-treemap', children=[
            dcc.Graph(figure=fig_treemap)
        ]),
        dcc.Tab(label='Bar', value='tab-bar', children=[
            dcc.Graph(figure=fig_bar)
        ])
        
    ])
])

# ---------------- CALLBACKS FOR MAP ----------------
@app.callback(
    Output("auto-interval", "disabled"),
    Input("playpause-btn", "n_clicks"),
    State("auto-interval", "disabled")
)
def toggle_play(n_clicks, disabled):
    if n_clicks>0:
        return not disabled
    return disabled

@app.callback(
    Output("playpause-btn", "children"),
    Input("auto-interval", "disabled")
)
def update_play_button(disabled):
    return "Play" if disabled else "Pause"

@app.callback(
    Output("year-slider", "value"),
    Output("current-year", "data"),
    Input("auto-interval", "n_intervals"),
    State("year-slider", "value"),
    State("year-slider", "min"),
    State("year-slider", "max")
)
def advance_year(n, current, min_y, max_y):
    next_y = current + 1
    if next_y > max_y:
        next_y = min_y
    return next_y, next_y

@app.callback(
    Output("crime-map", "figure"),
    Input("year-slider", "value"),
    Input("blink-interval", "n_intervals")
)
def update_map(year, blink_n):
    opacity = 1 if blink_n % 2 == 0 else 0.15
    year_df = df_london[df_london['Year']==year]
    hover = f"<b>Year:</b> {year}<br>" + "<br>".join(
        [f"<b>{r['Crime_Type']}:</b> {int(r['Predicted_Crimes'])}" for _, r in year_df.iterrows()]
    )
    fig = go.Figure(go.Scattermapbox(
        lat=[LONDON["lat"]],
        lon=[LONDON["lon"]],
        mode="markers+text",
        marker=dict(size=25, color="red", opacity=opacity),
        text=[LONDON["city"]],
        textposition="top center",
        textfont=dict(color="red", size=16),
        hovertext=hover,
        hoverinfo="text"
    ))
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox=dict(center=dict(lat=LONDON["lat"], lon=LONDON["lon"]), zoom=10),
        margin=dict(l=0,r=0,t=0,b=0)
    )
    return fig

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8029)
