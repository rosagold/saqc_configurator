# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import inspect

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_table
import plotly.express as px
from dash_helper import text_value_input, DivRow

import pandas as pd
import base64
import datetime
import io
import saqc.funcs
import saqc.lib.types as saqc_types

AGG_METHODS = ['mean', 'min', 'max', 'sum']  # first is default

SAQC_FUNCS = {
    "None": None,  # default
    'flagRange': saqc.funcs.flagRange,
    'flagMAD': saqc.funcs.flagMAD,
    'flagDummy': saqc.funcs.flagDummy,
}

TYPE_MAPPING = {
    str: 'text',
    int: 'number',
    float: 'number',
    saqc_types.ColumnName: 'text',
    saqc_types.TimestampColumnName: 'text',
    saqc_types.FreqString: 'text',
    saqc_types.IntegerWindow: 'number',
    saqc_types.PositiveFloat: 'number',
    saqc_types.PositiveInt: 'number',
}

df_name = "https://raw.githubusercontent.com/plotly/datasets/master/solar.csv"
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv',
                 index_col=None)
df.T.index.name = ''
df = df.T.reset_index().T
df = df.reset_index(drop=True)
df.index.name = ''
df = df.reset_index()

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

input_section = html.Div([
    html.H4('Input section'),
    html.Br(),

    DivRow(["Data", dcc.Upload(dbc.Button("Upload File"), id='upload-data')]),
    html.Div([]),

    text_value_input("Header row", type='n', value=0, id=f"header-row"),
    text_value_input("Index column", type="number", min=0, value=0, id="index-col"),
    text_value_input("Data column", type="number", min=0, value=1, id="data-col"),
    html.Br(),

    # Dataframe Preview
    dbc.Table.from_dataframe(df, bordered=True, hover=True),
    DivRow(["Config (optional)", dcc.Upload(dbc.Button("Upload File"),
                                            id='upload-config')]),
    html.Div(id='config-preview'),
])

preview_section = html.Div([
    html.H4('Preview section'),
    html.Br(),
    DivRow([
        'preview aggregation method :',
        dbc.Select(
            options=[dict(label=m, value=m) for m in AGG_METHODS],
            value=AGG_METHODS[0],
            id='agg-select'
        )
    ]),

    html.Br(),
    DivRow([
        'function: ',
        dbc.Select(
            options=[dict(label=f, value=f) for f in SAQC_FUNCS.keys()],
            placeholder="Select a function",
            id='function-select'
        )
    ]),

    html.Br(),
    dbc.Card([dbc.Form("Parameter to the selected Function")], id='parameters')
])


@app.callback(
    Output('parameters', 'children'),
    Input('function-select', 'value'),
)
def update_parameters(funcname):
    if funcname is None:
        funcname = 'None'
    func = SAQC_FUNCS[funcname]
    if func is None:
        return dbc.Form(['No parameters to set'])

    parameters = inspect.signature(func).parameters
    pnames = [p for p in parameters][3:]

    forms = []
    for name in pnames:
        p = parameters[name]

        type_ = TYPE_MAPPING.get(p.annotation, 'text')
        default = p.default
        if default == inspect._empty:
            default = None

        form = text_value_input(text=name,
                                id=f"param-input-{name}",
                                type=type_,
                                value=default,
                                raw=True)
        forms.append(form)

    return dbc.Form(forms)



app.layout = dbc.Container(
    [
        html.H1("SaQC Configurator"),
        dbc.Card([input_section]),
        html.Br(),
        dbc.Card([preview_section]),
    ],
)

if __name__ == '__main__':
    app.run_server(debug=True)
