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
import litereval
import plotly.express as px

import docstring_parser as docparse

import saqc
from helper import parse_data, type_repr, literal_eval_extended, parse_keywords
from const import AGG_METHODS, SAQC_FUNCS, TYPE_MAPPING, IGNORED_PARAMS, PARSER_MAPPING
from app import app, cache
import flask_caching

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
    Output('upload-error', 'children'),
    Output('parser-kwargs', 'invalid'),
    Output('df-exist', 'data'),
    # Datafile
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
        return [], False, True

    try:
        parser_kws = parse_keywords(parser_kws)
    except SyntaxError:
        msg = "Keyword parsing error: Syntax: 'key1=..., key3=..., key3=...'"
        return dbc.Alert(msg, color='danger'), True, False

    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = parse_data(filename, filetype, decoded, parser_kws)
    except Exception as e:
        msg = f"Data parsing error: {repr(e)}"
        return dbc.Alert(msg, color='danger'), False, False

    cache_set(session_id, 'df', df)
    return [], False, True


@app.callback(
    Output('df-preview', 'children'),
    Input('df-exist', 'data'),
    State('upload-data', 'filename'),
    State('session-id', 'data'),
)
def cb_df_preview(df_exist, filename, session_id):
    if not df_exist or filename is None:
        raise PreventUpdate

    df = cache_get(session_id, 'df')
    if df is None:
        raise AssertionError("DataFrame is not present in cache")

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
        print(d)
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
    Output('params-header', 'children'),
    Input('function-select', 'value'),
    Input('session-id', 'data')
)
def cb_params_header(funcname, session_id):
    if funcname is None:
        raise PreventUpdate
    func = SAQC_FUNCS[funcname]
    cache_set(session_id, 'func', func)
    return html.H4(funcname)


@app.callback(
    Output('params-body', 'children'),
    Input('function-select', 'value'),
)
def cb_params_body(funcname):
    """
    Fill param fields according to selected function
    adds:
        - docstring section
        - param with [name, docstring, type, input-field, alert]
    """
    if funcname is None:
        raise PreventUpdate

    func = SAQC_FUNCS[funcname]
    if func is None:
        return dbc.Form(['No parameters to set'])

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
def cb_param_validation(value, id, session_id):
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
        func = cache_get(session_id, 'func')
        param = inspect.signature(func).parameters[param_name]
        a = param.annotation
        a = TYPE_MAPPING.get(a, a)
        # sometimes the written typehints in saqc aren't explicit about None
        if param.default is None:
            a = typing.Union[a, None]

        # parse and check
        parsed = NotImplemented
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

        # store result in cache
        if parsed is not NotImplemented:
            params = cache_get(session_id, 'params', dict())
            params[param_name] = parsed
            cache_set(session_id, 'params', params)

    if failed == 'danger':
        children = dbc.Alert([html.B('Error: '), msg], color=failed)
    elif failed == 'warning':
        children = dbc.Alert([html.B('Warning: '), msg], color=failed)
        failed = False
    else:
        children = []

    return bool(failed), children


# ======================================================================
# Config
# ======================================================================


@app.callback(
    Output('add-to-config', 'disabled'),
    Input({'group': 'param', 'id': ALL}, 'invalid'),
    State('function-select', 'value'),
)
def cb_enable_add_to_config(invalids, funcname):
    """ enable add-button if all param-inputs are valid. """
    return any(invalids) or funcname is None


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
        func_repr = f"{func._module}.{func.__name__}"
        params = cache_get(session_id, 'params')
        paramstr = ', '.join([f"{k}={repr(v)}" for k, v in params.items()])
        line = f"qc.{func_repr}({paramstr})"
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
    Input({'group': 'param', 'id': ALL}, 'invalid'),
    Input('df-exist', 'data'),
    State('function-select', 'value'),
)
def cb_enable_preview(invalids, df_exist, fname):
    """ enable preview-button if we have data and all param-inputs are valid. """
    return any(invalids) or not df_exist or fname is None


@app.callback(
    Output('submit-error', 'children'),
    Output('result', 'children'),
    Input('preview', 'n_clicks'),
    State('session-id', 'data'),
)
def cb_submit(clicked, session_id):
    """
    parse all inputs.
    if successful calculate result TODO: text and plotted flags
    on parsing errors show an alert.
    """
    if not clicked:
        raise PreventUpdate

    if cache_get(session_id, 'df', None) is None:
        alert = dbc.Alert("No data. Please upload or generate data.", color='warning')
        out = html.Pre('Failed')
        return alert, out

    func = cache_get(session_id, 'func')
    params = cache_get(session_id, 'params')

    func_repr = f"{func._module}.{func.__name__}"
    paramstr = ', '.join([f"{k}={repr(v)}" for k, v in params.items()])
    txt = f"call `qc.{func_repr}({paramstr})`\n"

    txt += '\ndetailed params:\n-----------------\n'
    for k, v in params.items():
        txt += f"{k}={repr(v)} ({type(v).__name__})\n"

    df = cache_get(session_id, 'df')
    if df is None:
        raise AssertionError("DataFrame is not present in cache")

    try:
        qc = saqc.SaQC(df)
        func = getattr(qc, func_repr)
        field = 'a'
        result = func(field=field, **params)
    except Exception as e:
        alert = dbc.Alert(repr(e), color='danger')
        txt = 'Errors during ' + txt
    else:
        alert = []
        txt = 'Great Success\n=============\n' + txt

    return alert, html.Pre(txt)
