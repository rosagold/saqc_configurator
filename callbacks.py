# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import base64
import inspect
import json
import typing

import numpy as np
import typeguard
import pandas as pd

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash_table
import plotly.express as px


import docstring_parser as docparse

import saqc
from helper import (
    parse_data, type_repr, parse_keywords, cache_get, cache_set,
    saqc_func_repr, param_parse, param_typecheck
)
from const import AGG_METHODS, SAQC_FUNCS, TYPE_MAPPING, IGNORED_PARAMS, PARSER_MAPPING
from app import app


# ======================================================================
# Input section
# ======================================================================

@app.callback(
    Output('upload-data', 'filename'),
    Input('random-data', 'n_clicks'),
    State('session-id', 'data'),
)
def cb_click_to_random(n, session_id):
    if n is None:
        raise PreventUpdate
    rows = 990
    start = np.random.randint(9466848, 16094592) * 10 ** 11
    r = np.random.rand(rows, 10) * 10
    i = pd.date_range(start=start, periods=rows, freq='10min')
    df = pd.DataFrame(index=i, data=r, columns=list('abcdefghij'))
    df = df.round(2)
    cache_set(session_id, 'df', df)
    return 'random_data'


@app.callback(
    Output('df-present', 'data'),
    Output('parser-kwargs', 'invalid'),
    Output('upload-alert', 'children'),
    Input('upload-data', 'filename'),
    Input('upload-data', 'contents'),
    Input('datafile-type', 'value'),
    Input('parser-kwargs', 'value'),
    State('session-id', 'data'),
)
def cb_parse_data(filename, content, filetype, parser_kws, session_id):
    if filename is None:
        raise PreventUpdate

    if filename == 'random_data':
        # df_present, kws_invalid, alerts
        return True, False, []

    try:
        parser_kws = parse_keywords(parser_kws)
    except SyntaxError:
        msg = "Keyword parsing error: Syntax: 'key1=..., key3=..., key3=...'"
        # df_present, kws_invalid, alerts
        return False, True, dbc.Alert(msg, color='danger')

    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = parse_data(filename, filetype, decoded, parser_kws)
        if len(df.columns) == 0:
            raise ValueError("DataFrame must not be empty.")
    except Exception as e:
        msg = f"Data parsing error: {repr(e)}"
        # df_present, kws_invalid, alerts
        return False, False, dbc.Alert(msg, color='danger')

    cache_set(session_id, 'df', df)
    # df_present, kws_invalid, alerts
    return True, False, []


@app.callback(
    Output('df-preview', 'children'),
    Input('df-present', 'data'),
    State('upload-data', 'filename'),
    State('session-id', 'data'),
)
def cb_df_preview(df_present, filename, session_id):
    # todo: same as this : plot
    if not df_present or filename is None:
        raise PreventUpdate

    df = cache_get(session_id, 'df')
    df = df.reset_index()

    columns = []
    for c in df.columns:
        dtype = df[c].dtype
        if pd.api.types.is_datetime64_dtype(dtype):
            t = 'datetime'
        elif pd.api.types.is_numeric_dtype(dtype):
            t = 'numeric'
        else:
            t = 'any'
        d = dict(name=str(c), id=str(c), type=t)
        columns.append(d)

    table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=columns,
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold'
        },
    )
    # table = dbc.Table.from_dataframe(
    #     df, striped=False, bordered=True, hover=True, responsive=True
    # )
    return [html.B(filename), table]


# ======================================================================
# Function section and param validation
# ======================================================================

@app.callback(
    Output('func-selected', 'data'),
    Output('params-header', 'children'),
    Input('function-select', 'value'),
    State('session-id', 'data'),
)
def cb_params_header(funcname, session_id):
    if funcname is None:
        raise PreventUpdate
    func = SAQC_FUNCS[funcname]
    cache_set(session_id, 'func', func)
    return True, html.H4(funcname)


@app.callback(
    Output('params-body', 'children'),
    Input('func-selected', 'data'),
    State('default-field', 'data'),
    State('session-id', 'data')
)
def cb_params_body(func_selected, default_field, session_id):
    """
    Fill param fields according to selected function
    adds:
        - docstring section
        - param with [name, docstring, type, input-field, alert]
    """
    if not func_selected:
        raise PreventUpdate

    func = cache_get(session_id, 'func')
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

        docu = dcc.Markdown(params_docstr.get(name, []))

        if name == 'field':
            default = default_field
            if default is None:
                df = cache_get(session_id, 'df', pd.DataFrame(columns=[None]))
                default = df.columns[0]  # we don't allow empty DataFrames
                if default is not None:
                    default = repr(default)
            docu = "A column name holding the data."

        # using a dict as ``id`` makes pattern matching callbacks possible
        id = {"group": "param", "id": name}
        form = dbc.FormGroup(
            [
                dbc.Label(html.B(name), html_for=id, width=2),
                dbc.Col(
                    [
                        dbc.Input(type='text', value=default, debounce=True, id=id),
                        dbc.FormText(hint, color='secondary')
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

    return children


@app.callback(
    Output({'group': 'param', 'id': MATCH}, 'invalid'),
    Output({'group': 'param-validation', 'id': MATCH}, 'children'),
    Input({'group': 'param', 'id': MATCH}, 'value'),
    State({'group': 'param', 'id': MATCH}, 'id'),
    State('session-id', 'data')
)
def cb_param_validation(value, param_id, session_id):
    """
    validated param input and show an alert if validation fails
    """
    param_name = param_id['id']
    failed, msg = False, ""

    if value is None:
        return True, []

    # Empty value after user already was in the input form
    if value == "":
        failed, msg = 'danger', f"Missing value."

    else:
        # prepare type check
        func = cache_get(session_id, 'func')
        df = cache_get(session_id, 'df', None)
        param = inspect.signature(func).parameters[param_name]
        a = param.annotation
        # sometimes the written typehints in saqc aren't explicit about None
        if param.default is None:
            a = typing.Union[a, None]

        try:
            parsed = param_parse(value)
            param_typecheck(param_name, parsed, a, df)
        except (TypeError, ValueError) as e:
            failed, msg = e.args

    if failed == 'danger':
        children = dbc.Alert([html.B('Error: '), msg], color=failed)
    elif failed == 'warning':
        children = dbc.Alert([html.B('Warning: '), msg], color=failed)
        failed = False
    else:
        children = []

    return bool(failed), children


@app.callback(
    Output('params-parsed', 'data'),
    Output('default-field', 'data'),
    Input({'group': 'param', 'id': ALL}, 'invalid'),
    State({'group': 'param', 'id': ALL}, 'value'),
    State({'group': 'param', 'id': ALL}, 'id'),
    State('default-field', 'data'),
    State('func-selected', 'data'),
    State('session-id', 'data'),
)
def cb_parsing_done_and_default_field(
        invalids, values, param_ids, default_field, func_seleced, session_id
):
    if not func_seleced:
        raise PreventUpdate

    # set default value for field
    default = default_field
    for i, param_id in enumerate(param_ids):
        name = param_id['id']
        value = values[i]
        valid = not invalids[i]
        if name == 'field':
            if valid:
                default = value
            break

    if any(invalids):
        return False, default

    # store params in cache
    params = dict()
    for i, param_id in enumerate(param_ids):
        name = param_id['id']
        value = values[i]
        # We have to parse the values here again, but that isn't to expensive i
        # guess. We cannot store the values in the validation callback above,
        # because of the pattern-matching with MATCH, which make the callback
        # possibly run in parallel. That could generate a race-condition on the cache
        # in the read-update-write workflow, because of it non-atomic nature.
        params[name] = param_parse(value)
    cache_set(session_id, 'params', params)

    return True, default

# ======================================================================
# Config
# ======================================================================


@app.callback(
    Output('add-to-config', 'disabled'),
    Input('params-parsed', 'data'),
)
def cb_enable_add_to_config(parsed):
    """ enable add-button if all param-inputs are valid. """
    return not parsed


@app.callback(
    Output('config-preview', 'value'),
    Input('add-to-config', 'n_clicks'),
    Input('clear-config', 'n_clicks'),
    Input('upload-config', 'contents'),
    State('config-preview', 'value'),
    State('session-id', 'data'),
)
def cb_config_preview(add, clear, content, config, session_id):
    if config is None:
        config = ''
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == 'add-to-config':
        func = cache_get(session_id, 'func')
        params = cache_get(session_id, 'params')
        line = "qc." + saqc_func_repr(func, params)
        return config + line + '\n'

    if trigger == 'clear-config':
        return ''

    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    return decoded.decode('utf-8')


# ======================================================================
# Preview / Plot
# ======================================================================

@app.callback(
    Output('preview', 'disabled'),
    Input('params-parsed', 'data'),
    Input('df-present', 'data'),
)
def cb_enable_preview(parsed, df_present):
    """ enable preview-button if we have data and all param-inputs are valid. """
    return not parsed or not df_present


@app.callback(
    Output('preview-alert', 'children'),
    Output('result', 'children'),
    Output('plot', 'children'),
    Input('preview', 'n_clicks'),
    State('session-id', 'data'),
)
def cb_preview(clicked, session_id):
    """
    parse all inputs.
    if successful calculate result TODO: text and plotted flags
    on parsing errors show an alert.
    """
    if not clicked:
        raise PreventUpdate

    df = cache_get(session_id, 'df')
    func = cache_get(session_id, 'func')
    params = cache_get(session_id, 'params')
    field = params['field']

    frpr = saqc_func_repr(func, params)
    txt = f"call `qc.{frpr}`\n"

    txt += '\ndetailed params:\n-----------------\n'
    for k, v in params.items():
        txt += f"{k}={repr(v)} ({type(v).__name__})\n"

    alert, plot = [], []
    try:
        qc = saqc.SaQC(df)
        func = getattr(qc, f"{func._module}.{func.__name__}")

        # plot data
        fig = px.line(df, x=df.index, y=field)
        result = func(**params)

        # plot flags
        flags = result.flags[field]
        flagged = flags > saqc.UNFLAGGED
        x = flags[flagged].index
        y = df.loc[flagged, field]
        fig.add_scatter(x=x, y=y, mode="markers",)

        plot = dcc.Graph(figure=fig)
    except Exception as e:
        alert = dbc.Alert(repr(e), color='danger')
        txt = 'Errors during ' + txt
    else:
        txt = 'Great Success\n=============\n' + txt

    return alert, html.Pre(txt), plot
