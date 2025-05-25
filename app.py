import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objs as go
import base64
import io
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

app = dash.Dash(__name__)
app.title = 'Calcium Imaging Analyzer'

def compute_dff(trace, baseline_method='percentile', percentile=5, frame_range=None):
    if baseline_method == 'percentile':
        F0 = np.percentile(trace, percentile)
    elif baseline_method == 'min':
        F0 = np.min(trace)
    elif baseline_method == 'mean5':
        F0 = np.mean(np.sort(trace)[:int(len(trace)*0.05)])
    elif baseline_method == 'mean_range' and frame_range is not None:
        start, end = frame_range
        F0 = np.mean(trace[start:end])
    else:
        F0 = np.mean(trace)
    return (trace - F0) / F0

def low_pass_filter(trace, sampling_rate=10, cutoff_freq=1, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, trace)

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
                {'label': 'Mean', 'value': 'mean'},
                {'label': 'Mean of selected frame range', 'value': 'mean_range'}
            ],
            value='percentile'
        ),
        html.Div([
            html.Label('First frame:'),
            dcc.Input(id='frame-start', type='number', value=0, min=0, step=1),
            html.Label('Last frame:'),
            dcc.Input(id='frame-end', type='number', value=10, min=1, step=1),
        ], id='frame-range-input', style={'marginTop': '10px', 'display': 'none'})
    ], style={'width': '60%', 'display': 'inline-block'}),

    html.Div([
        html.Label('Apply low-pass filter:'),
        dcc.Checklist(
            id='apply-filter',
            options=[{'label': ' Enable', 'value': 'on'}],
            value=[]
        ),
        html.Div([
            html.Label('Sampling rate (Hz):'),
            dcc.Input(id='sampling-rate', type='number', value=5.3, step=0.1),
            html.Label('Cutoff frequency (Hz):'),
            dcc.Input(id='cutoff-freq', type='number', value=1.0, step=0.1),
            html.Label('Filter order:'),
            dcc.Input(id='filter-order', type='number', value=4, step=1)
        ], id='filter-settings', style={'marginTop': '10px'})
    ], style={'marginTop': '20px'}),

    dcc.Graph(id='raw-traces'),
    dcc.Graph(id='dff-traces'),
    dcc.Graph(id='filtered-traces'),

    dcc.Store(id='stored-data')
])

@app.callback(
    Output('frame-range-input', 'style'),
    Input('baseline-method', 'value')
)
def toggle_frame_range_inputs(method):
    if method == 'mean_range':
        return {'marginTop': '10px', 'display': 'block'}
    return {'display': 'none'}

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
    Output('filtered-traces', 'figure'),
    Input('stored-data', 'data'),
    Input('baseline-method', 'value'),
    Input('frame-start', 'value'),
    Input('frame-end', 'value'),
    Input('apply-filter', 'value'),
    Input('sampling-rate', 'value'),
    Input('cutoff-freq', 'value'),
    Input('filter-order', 'value')
)
def update_dff_plot(data, baseline_method, frame_start, frame_end, apply_filter, sampling_rate, cutoff_freq, filter_order):
    if data is None:
        return go.Figure(), go.Figure()

    df = pd.read_json(data, orient='split')
    dff_df = pd.DataFrame()
    filtered_df = pd.DataFrame()

    frame_range = (frame_start, frame_end) if baseline_method == 'mean_range' else None

    for col in df.columns:
        dff_trace = compute_dff(df[col].values, baseline_method, frame_range=frame_range)
        dff_df[col] = dff_trace
        if 'on' in apply_filter:
            filtered_df[col] = low_pass_filter(dff_trace, sampling_rate, cutoff_freq, filter_order)

    dff_fig = go.Figure()
    for col in dff_df.columns:
        dff_fig.add_trace(go.Scatter(y=dff_df[col], mode='lines', name=col))
    dff_fig.update_layout(title='dF/F Traces', xaxis_title='Frame', yaxis_title='dF/F')

    filtered_fig = go.Figure()
    if 'on' in apply_filter:
        for col in filtered_df.columns:
            filtered_fig.add_trace(go.Scatter(y=filtered_df[col], mode='lines', name=col))
        filtered_fig.update_layout(title='Filtered dF/F Traces', xaxis_title='Frame', yaxis_title='Filtered dF/F')

    return dff_fig, filtered_fig

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=10000)