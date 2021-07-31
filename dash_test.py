# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import inspect
import json

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import dash_table
import plotly.express as px

from dash_helper import text_value_input, DivRow

import pandas as pd
import numpy as np
import base64
import datetime
import io
import saqc.funcs

AGG_METHODS = ['mean', 'min', 'max', 'sum']  # first is default


def test(
        a, b, c, arg, kw1: int = 9, kw2: bool = False, kw3=None,
        kw4=np.nan, kw5=-np.inf,
        kw6: float = np.nan, kw7: float = -np.inf

):
    pass


SAQC_FUNCS = {
    "None": None,  # default
    'flagRange': saqc.funcs.flagRange,
    'flagMAD': saqc.funcs.flagMAD,
    'flagDummy': saqc.funcs.flagDummy,
    'test': test,
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
    DivRow([
        'preview aggregation method :',
        dbc.Select(
            options=[dict(label=m, value=m) for m in AGG_METHODS],
            value=AGG_METHODS[0],
            id='agg-select'
        )
    ]),

    DivRow([
        'function: ',
        dbc.Select(
            options=[dict(label=f, value=f) for f in SAQC_FUNCS.keys()],
            placeholder="Select a function",
            id='function-select'
        )
    ]),

    dbc.Card([
        dbc.CardHeader(f'Parameter for selected function'),
        dbc.CardBody([dbc.Form("")], id='parameters')
    ]),

    html.Br(),

    # live preview and submit Button/Checkbox
    # dbc.FormGroup(
    #     [
    #         dbc.Checkbox(id="live-preview-toggle", className="form-check-input"),
    #         dbc.Label(
    #             "Live Preview",
    #             html_for="live-preview-toggle",
    #             className="form-check-label",
    #         ),
    #     ],
    #     check=True,
    # ),
    dbc.Button("Submit", id='submit', block=True),

    dbc.Card("Result", body=True, id='show'),
])


# @app.callback(
#     Output('submit', 'disabled'),
#     Input('live-preview-toggle', 'checked')
# )
# def cb_live_preview_toggle(checked):
#     """
#     Disable/Enable the `Submit` button, depending on the
#     `Live-Preview` checkbox.
#     """
    # return bool(checked)
    # return False


def parse_param(name, value, funcname):
    func = SAQC_FUNCS[funcname]
    if func is None:
        return 'nothing parsed'
    param = inspect.signature(func).parameters[name]
    type_ = param.annotation
    if type_ is inspect._empty:
        type_ = str

    # parse
    try:
        v = type_(value)
    except ValueError:
        v = value

    return f"{v}({type(v)})"


@app.callback(
    Output('show', 'children'),
    Input('submit', 'n_clicks'),
    State({'group': 'param', 'id': ALL}, 'value'),
    State('function-select', 'value'),
    # State('live-preview-toggle', 'checked'),  # ignored
)
def cb_update_graph(submit, params, funcname):
    ctx = dash.callback_context

    id = val = None
    if ctx.triggered:
        id = ctx.triggered[0]['prop_id'].split('.')[0]
        val = ctx.triggered[0]['value']

    if id is None:
        return ['Nothing happened yet']

    # if id == 'submit':
    #     s = parse_param(id, val, funcname)
    #     return [f'']

    ctx_msg = json.dumps({
        'states': ctx.states,
        'triggered': ctx.triggered,
        'inputs': ctx.inputs
    }, indent=2)

    return html.Div([f"{id} is now {val}", html.Br(), html.Pre(ctx_msg)])


@app.callback(
    Output('parameters', 'children'),
    Input('function-select', 'value'),
)
def cb_update_parameters(funcname):
    if funcname is None:
        funcname = 'None'
    func = SAQC_FUNCS[funcname]
    if func is None:
        return dbc.Form(['No parameters to set'])

    parameters = inspect.signature(func).parameters
    ignore = ['data', 'flags', 'field', 'kwargs']
    pnames = [p for p in parameters if p not in ignore]

    forms = []
    for name in pnames:
        p = parameters[name]

        if p.default == inspect._empty:
            default = None
        else:
            default = str(p.default)

        # using a dict as ``id`` makes pattern matching callbacks possible
        id = {"group": "param", "id": name}
        form = text_value_input(text=name, id=id, type='text', value=default, raw=True)
        forms.append(form)

    return dbc.Form(forms)


app.layout = dbc.Container(
    [
        html.H1("SaQC Configurator"),

        # dbc.Card([
        #     dbc.CardHeader('Input section'),
        #     dbc.CardBody([input_section]),
        # ]),

        html.Br(),

        dbc.Card([
            dbc.CardHeader('Preview section'),
            dbc.CardBody([preview_section]),
        ]),
    ],
)

if __name__ == '__main__':
    app.run_server(debug=True)
