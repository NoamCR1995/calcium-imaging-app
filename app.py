import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import plotly.graph_objs as go
import base64
import io
import pandas as pd
import numpy as np

app = dash.Dash(__name__)
app.title = 'Calcium Imaging Analyzer'


def compute_dff(trace, baseline_method='percentile', percentile=5):
    if baseline_method == 'percentile':
        F0 = np.percentile(trace, percentile)
    elif baseline_method == 'min':
        F0 = np.min(trace)
    elif baseline_method == 'mean5':
        F0 = np.mean(np.sort(trace)[:int(len(trace)*0.05)])
    else:
        F0 = np.mean(trace)
    return (trace - F0) / F0


app.layout = html.Div([
    html.H2("Calcium Imaging Analysis Web App"),

    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),

    html.Div([
        html.Label('Baseline method:'),
        dcc.Dropdown(
            id='baseline-method',
            options=[
                {'label': 'Percentile (5%)', 'value': 'percentile'},
                {'label': 'Min Value', 'value': 'min'},
                {'label': 'Mean of Bottom 5%', 'value': 'mean5'},
                {'label': 'Mean', 'value': 'mean'}
            ],
            value='percentile'
        )
    ], style={'width': '40%', 'display': 'inline-block'}),

    dcc.Graph(id='raw-traces'),
    dcc.Graph(id='dff-traces'),

    dcc.Store(id='stored-data')
])


@app.callback(
    Output('stored-data', 'data'),
    Output('raw-traces', 'figure'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def parse_contents(contents, filename):
    if contents is None:
        return dash.no_update, go.Figure()

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(y=df[col], mode='lines', name=col))
    fig.update_layout(title='Raw Traces', xaxis_title='Frame', yaxis_title='Fluorescence')

    return df.to_json(date_format='iso', orient='split'), fig


@app.callback(
    Output('dff-traces', 'figure'),
    Input('stored-data', 'data'),
    Input('baseline-method', 'value')
)
def update_dff_plot(data, baseline_method):
    if data is None:
        return go.Figure()

    df = pd.read_json(data, orient='split')
    dff_df = pd.DataFrame()
    for col in df.columns:
        dff_df[col] = compute_dff(df[col].values, baseline_method)

    fig = go.Figure()
    for col in dff_df.columns:
        fig.add_trace(go.Scatter(y=dff_df[col], mode='lines', name=col))
    fig.update_layout(title='dF/F Traces', xaxis_title='Frame', yaxis_title='dF/F')

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
