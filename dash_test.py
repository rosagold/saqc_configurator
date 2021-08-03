# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import inspect
import json
import typing
import typeguard
import pandas as pd
import numpy as np
import base64
import datetime
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc
import dash_table
import plotly.express as px

import docstring_parser as docparse
from helper import text_value_input, DivRow, type_repr, literal_eval_extended
from const import AGG_METHODS, SAQC_FUNCS, TYPE_MAPPING, IGNORED_PARAMS

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
        dbc.CardHeader([], id='parameter-header'),  # filled by callback
        dbc.CardBody([], id='parameters'),  # filled by callback
        dbc.CardFooter([
            dbc.Button("Submit", id='submit', block=True),
            html.Div([], id='error-container'),  # filled by callback
        ]),
    ]),

    html.Br(),

    dbc.Card("Result", body=True, id='show'),
])


@app.callback(
    Output('parameter-header', 'children'),
    Input('function-select', 'value'),
)
def cb_param_header(funcname):
    if funcname is None:
        return []
    return html.H4(funcname)


@app.callback(
    Output({'group': 'param', 'id': MATCH}, 'invalid'),
    Output({'group': 'param-err', 'id': MATCH}, 'children'),
    Input({'group': 'param', 'id': MATCH}, 'value'),
    State({'group': 'param', 'id': MATCH}, 'id'),
    State('function-select', 'value'),
)
def cb_validate_param_input(value, id, funcname):
    param_name = id['id']
    failed, msg = False, ""

    if value is None:
        return True, []

    # Empty value after user already was in the input form
    if value == "":
        failed, msg = 'danger', f"Missing value."

    else:
        # prepare type check
        param = inspect.signature(SAQC_FUNCS[funcname]).parameters[param_name]
        a = param.annotation
        a = TYPE_MAPPING.get(a, a)
        # sometimes the written typehints in saqc aren't explicit about None
        if param.default is None:
            a = typing.Union[a, None]

        # parse and check
        try:
            parsed = literal_eval_extended(value)
            try:
                typeguard.check_type(param_name, parsed, a)
            except TypeError as e:
                failed, msg = 'warning', f"Type check failed: {e}"
        except (TypeError, SyntaxError):
            failed, msg = 'danger', f"Invalid Syntax."
        except ValueError:
            failed, msg = 'danger', f"Invalid value."

    if failed == 'danger':
        children = dbc.Alert([html.B('Error: '), msg], color=failed)
    elif failed == 'warning':
        children = dbc.Alert([html.B('Warning: '), msg], color=failed)
        failed = False
    else:
        children = []

    return bool(failed), children


@app.callback(
    Output('error-container', 'children'),
    Output('show', 'children'),
    Input('submit', 'n_clicks'),
    State({'group': 'param', 'id': ALL}, 'id'),
    State({'group': 'param', 'id': ALL}, 'value'),
)
def cb_update_graph(submit_n, param_ids, param_values):
    if submit_n is None:
        return [], []

    kws_to_func = {}
    submit = True

    # parse values, all checks are already done, in the input-form-callback
    for i, id_dict in enumerate(param_ids):
        param_name = id_dict['id']
        value = param_values[i]
        try:
            kws_to_func[param_name] = literal_eval_extended(value)
        except (SyntaxError, ValueError):
            submit = False

    if submit:
        txt = 'Great Success\n=============\n'
        for k, v in kws_to_func.items():
            txt += f"{k}={repr(v)} ({type(v).__name__})\n"
        alert, out = [], html.Pre(txt)
    else:
        alert = dbc.Alert("Missing fields or errors above.", color='danger'),
        out = html.Pre('Failed')

    return alert, out


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

    children = []
    param_forms = []

    # docstring
    docstr = docparse.parse(func.__doc__)
    for o in [docstr.short_description, docstr.long_description, *docstr.meta]:
        if o is None or isinstance(o, (
        docparse.DocstringParam, docparse.DocstringReturns)):
            continue
        if not isinstance(o, str):
            o = f"#### {o.args[0].capitalize()}\n{o.description}"
        children.append(dcc.Markdown(o))

    # dynamically add param input fields
    params_docstr = {p.arg_name: p.description for p in docstr.params}
    params = inspect.signature(func).parameters
    pnames = [p for p in params if p not in IGNORED_PARAMS]
    for name in pnames:
        p = params[name]

        if p.default == inspect.Parameter.empty:
            default = None
        else:
            default = repr(p.default)

        type_ = p.annotation
        hint = type_repr(type_)
        if type_ is inspect.Parameter.empty:
            type_, hint = None, ''

        # using a dict as ``id`` makes pattern matching callbacks possible
        id = {"group": "param", "id": name}

        docu = dcc.Markdown(params_docstr.get(name, []))

        form = dbc.FormGroup(
            [
                dbc.Label(html.B(name), html_for=id, width=2),
                dbc.Col(
                    [
                        dbc.Input(type='text', value=default, id=id),
                        dbc.FormText(html.Pre(hint))
                    ], width=10
                ),
                dbc.Col(docu, width=12),
                dbc.Col([], width=12, id={"group": "param-err", "id": name}),
            ],
            row=True
        )
        param_forms.append(html.Hr())
        param_forms.append(form)

    children.append(dbc.Form(param_forms))

    return children, False


app.layout = dbc.Container(
    [
        html.H1("SaQC Configurator"),

        dbc.Card([
            dbc.CardHeader('Input section'),
            dbc.CardBody([input_section]),
        ]),

        html.Br(),

        dbc.Card([
            dbc.CardHeader('Preview section'),
            dbc.CardBody([preview_section]),
        ]),
    ],
)

if __name__ == '__main__':
    app.run_server(debug=True)
