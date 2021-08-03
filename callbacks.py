# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import inspect
import json
import typing

import numpy as np
import typeguard
import pandas as pd
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
from app import app


# ======================================================================
# Input section
# ======================================================================

@app.callback(
    Output('upload-data', 'contents'),
    Input('random-data', 'n_clicks'),
)
def cb_random_data(clicked):
    if clicked:
        return 'random'
    return None


@app.callback(
    Output('df-preview', 'children'),
    Input('upload-data', 'contents'),
    State('header-row', 'value'),
    State('index-column', 'value'),
    State('data-column', 'value'),
)
def cb_df_preview(data, hrow, icol, dcol):
    if data is None:
        return []

    if data == 'random':
        r = np.random.rand(990, 10) * 10
        df = pd.DataFrame(data=r, columns=list('abcdefghij'))
        df = df.round(2)
        name = 'random_data'
    else:
        name = ''
        df = pd.DataFrame()

    return [
        html.B(name),
        dash_table.DataTable(
            data=df.to_dict(orient='records'),
            columns=[{'name': str(i), 'id': str(i)} for i in df.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold'
            },
        ),
    ]
    return dbc.Table.from_dataframe(
        df,
        bordered=True,
        hover=True,
    )


# ======================================================================
# Function section
# ======================================================================

@app.callback(
    Output('params-header', 'children'),
    Input('function-select', 'value'),
)
def cb_params_header(funcname):
    if funcname is None:
        return []
    return html.H4(funcname)


@app.callback(
    Output('params-body', 'children'),
    Output('submit', 'disabled'),
    Input('function-select', 'value'),
)
def cb_params_body(funcname):
    """
    Fill param fields according to selected function
    adds:
        - docstring section
        - param:
            - name
            - docstring
            - type
            - input-field
            - alert
    """
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

                # filled by cb_param_validation()
                dbc.Col([], width=12, id={"group": "param-validation", "id": name}),
            ],
            row=True
        )
        param_forms.append(html.Hr())
        param_forms.append(form)

    children.append(dbc.Form(param_forms))

    return children, False


@app.callback(
    Output({'group': 'param', 'id': MATCH}, 'invalid'),
    Output({'group': 'param-validation', 'id': MATCH}, 'children'),
    Input({'group': 'param', 'id': MATCH}, 'value'),
    State({'group': 'param', 'id': MATCH}, 'id'),
    State('function-select', 'value'),
)
def cb_param_validation(value, id, funcname):
    """
    validated param input and show an alert if validation fails
    """
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
    Output('submit-error', 'children'),
    Output('result', 'children'),
    Input('submit', 'n_clicks'),
    State({'group': 'param', 'id': ALL}, 'id'),
    State({'group': 'param', 'id': ALL}, 'value'),
)
def cb_submit(submit_n, param_ids, param_values):
    """
    parse all inputs.
    if successful calculate result TODO: text and plotted flags
    on parsing errors show an alert.
    """
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
        alert = dbc.Alert("Errors or missing fields above.", color='danger'),
        out = html.Pre('Failed')

    return alert, out
