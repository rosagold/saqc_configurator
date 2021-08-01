# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import inspect
import json
import typing

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc
import dash_table
import plotly.express as px

from dash_helper import text_value_input, DivRow, parse_param, type_repr

import pandas as pd
import numpy as np
import base64
import datetime
import io
import saqc.funcs
from saqc.lib.types import FreqString, PositiveInt

AGG_METHODS = ['mean', 'min', 'max', 'sum']  # first is default


def test(
        data, field, flags,
        kw1: int = 9,
        kw2: bool = False,
        kw3: int = None,
        kw31=None,
        kw4=np.nan,
        kw6=-np.inf,
        kw7: PositiveInt = 9,
        freq: FreqString = '9d',
        union: typing.Union[int, float] = 0,
        tup: (int, float) = 0,
        li: typing.Literal['a', 'b', 'c'] = 'a',
        op1: int = None,
        op2: typing.Optional[int] = None,
        op3: typing.Optional[int] = 7,
        **kwargs
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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
            id='function-select',
        )
    ]),

    dbc.Card([
        dbc.CardHeader(f'Parameter for selected function'),
        dbc.CardBody([dbc.Form("")], id='parameters')
    ]),

    html.Br(),

    dbc.Button("Submit", id='submit', block=True),
    html.Div([], id='error-container'),
    dbc.Card("Result", body=True, id='show'),
])




@app.callback(
    Output({'group': 'param', 'id': MATCH}, 'invalid'),
    Input({'group': 'param', 'id': MATCH}, 'value'),
)
def cb_validate_param_input(value):
    return value is None or value == ''


@app.callback(
    Output('show', 'children'),
    Output('error-container', 'children'),
    Input('submit', 'n_clicks'),
    State({'group': 'param', 'id': ALL}, 'value'),
    State('function-select', 'value'),
)
def cb_update_graph(submit_n, params_unused, funcname):
    alerts = []
    if submit_n is None:
        return ['Nothing happened yet'], alerts

    ctx = dash.callback_context

    out = []
    kws_to_func = {}
    submit = True
    # a state entry for a parameter look like this:
    # '{"group": "param", "id": name}.value': '255.0'
    for key, value in ctx.states.items():
        key = key.split('.')[0]
        if not key.startswith('{'):
            continue

        param_name = json.loads(key)['id']
        target_type = inspect.signature(SAQC_FUNCS[funcname]).parameters[
            param_name].annotation

        # process and parse value
        try:
            if value is None or value == "":
                raise ValueError(f"Missing value for parameter '{param_name}'")
            parsed = parse_param(value, target_type)
        except ValueError as e:
            alerts.append(dbc.Alert(str(e), color="danger"))
            submit = False
            continue

        kws_to_func[param_name] = parsed

    if submit:
        txt, color = 'Success\n', 'success'
    else:
        txt, color = 'Failed\n', 'danger'
    for k, v in kws_to_func.items():
        txt += f"'{k}' is '{v}' of type '{type(v).__name__}'\n"
    out.append(dbc.Alert(html.Pre(txt), color=color))

    return html.Div(out), alerts


@app.callback(
    Output('parameters', 'children'),
    Output('submit', 'disabled'),
    Input('function-select', 'value'),
)
def cb_fill_parameters(funcname):
    if funcname is None:
        funcname = 'None'
    func = SAQC_FUNCS[funcname]
    if func is None:
        return dbc.Form(['No parameters to set']), True

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

        type_ = p.annotation
        hint = type_repr(type_)
        if type_ is inspect._empty:
            type_, hint = None, ''

        # using a dict as ``id`` makes pattern matching callbacks possible
        id = {"group": "param", "id": name}
        form = text_value_input(
            text=f"{name}: {hint}", id=id, type='text', value=default, raw=True
        )
        forms.append(form)

    return dbc.Form(forms), False


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
