# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import base64
import inspect
import json
import typing

import numpy as np
from numpy.random import randint, rand
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
from const import AGG_METHODS, SAQC_FUNCS, TYPE_MAPPING, IGNORED_PARAMS, PARSER_MAPPING, \
    AGG_THRESHOLD, MAX_DF_ROWS
from app import app


# ======================================================================
# Input section
# ======================================================================

@app.callback(
    Output('upload-data', 'filename'),
    Input('random-data', 'n_clicks'),
    State('session-id', 'data'),
    State('random-type', 'value')
)
def cb_click_to_random(n, session_id, random_type):
    """
    Generate random data.

    After generation we fake a click to upload-button by
    setting a new filename, this will trigger the `cb_upload_data`
    callback.
    """
    random_type = random_type or []
    if n is None:
        raise PreventUpdate

    rows, cols = 990, 10
    points = rows * cols
    r = np.random.rand(points) * 10

    if 'outliers' in random_type:
        nr = np.random.randint(points * 0.01, points * 0.1)
        idx = np.random.choice(range(points), size=nr, replace=False)
        r[idx] = (r[idx] + 2) * 10

    if 'gaps' in random_type:
        nr = np.random.randint(points * 0.01, points * 0.1)
        idx = np.random.choice(range(points), size=nr, replace=False)
        r[idx] = np.nan

    start = np.random.randint(9466848, 16094592) * 10 ** 11
    i = pd.date_range(start=start, periods=rows, freq='10min')
    df = pd.DataFrame(index=i, data=r.reshape(rows, cols), columns=list('abcdefghij'))

    if 'plateaus' in random_type:
        for j in range(len(df.columns)):
            nr = np.random.randint(1, 11)
            idx = np.random.choice(df.index, size=nr, replace=False)
            for i in idx:
                size = np.random.randint(5, 20)
                i = df.index.get_loc(i)
                size = len(df.iloc[i:i + size])  # shrink if `i+size > len(df.index)`
                df.iloc[i:i + size, j] = randint(2, 11) * 10 + np.random.rand(size)

    df = df.round(2)
    flags = pd.DataFrame(False, index=df.index, columns=df.columns, dtype=bool)
    cache_set(session_id, 'df', df)
    cache_set(session_id, 'flags', flags)
    return 'random_data'


@app.callback(
    Output('new-upload', 'data'),
    Output('parser-kwargs', 'invalid'),
    Output('upload-alert', 'children'),
    Input('upload-data', 'filename'),
    Input('upload-data', 'contents'),
    Input('datafile-type', 'value'),
    Input('parser-kwargs', 'value'),
    State('session-id', 'data'),
)
def cb_upload_data(filename, content, filetype, parser_kws, session_id):
    if filename is None:
        raise PreventUpdate

    if filename == 'random_data':
        # new-upload, kws_invalid, alerts
        return True, False, []

    try:
        parser_kws = parse_keywords(parser_kws)
    except SyntaxError:
        msg = "Keyword parsing error: Syntax: 'key1=..., key3=..., key3=...'"
        # new-upload, kws_invalid, alerts
        return False, True, dbc.Alert(msg, color='danger')

    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = parse_data(filename, filetype, decoded, parser_kws)
        if len(df.columns) == 0:
            raise ValueError("DataFrame must not be empty.")
    except Exception as e:
        msg = f"Data parsing error: {repr(e)}"
        # new-upload, kws_invalid, alerts
        return False, False, dbc.Alert(msg, color='danger')

    alert = []
    if len(df.index) > MAX_DF_ROWS:
        df = df.iloc[:MAX_DF_ROWS, :]
        msg = f'Maximum data set size exceeded. ' \
              f'The data is truncated to {MAX_DF_ROWS} rows per column'
        alert = dbc.Alert([html.B('Warning: '), msg], color='warning')
    print(alert)

    flags = pd.DataFrame(False, index=df.index, columns=df.columns, dtype=bool)
    cache_set(session_id, 'df', df)
    cache_set(session_id, 'flags', flags)
    return True, False, alert  # new-upload, kws_invalid, alerts


@app.callback(
    Output('new-data', 'data'),
    Input('new-upload', 'data'),
)
def cb_new_data(upload):
    return upload


# ======================================================================
# Data Table
# ======================================================================

@app.callback(
    Output('data-table', 'children'),
    Input('new-data', 'data'),
    State('upload-data', 'filename'),
    State('session-id', 'data'),
)
def cb_data_table(new_data, filename, session_id):
    if not new_data or filename is None:
        raise PreventUpdate

    df = cache_get(session_id, 'df')
    df = df.reset_index()  # otherwise the index will not show up

    columns = []
    for c in df.columns:
        dtype, t = df[c].dtype, 'any'
        if pd.api.types.is_datetime64_dtype(dtype):
            t = 'datetime'
        elif pd.api.types.is_numeric_dtype(dtype):
            t = 'numeric'
        columns.append(dict(name=str(c), id=str(c), type=t))

    table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=columns,
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
    )
    return [html.B(filename), table]


# ======================================================================
# Plot
# ======================================================================

@app.callback(
    Output('plot-column', 'options'),
    Output('plot-column', 'value'),
    Input('new-data', 'data'),
    Input('preview', 'n_clicks'),  # trigger a recheck of `preselect` on `Preview` click
    State('func-field', 'data'),
    State('default-field', 'data'),
    State('session-id', 'data'),
)
def cb_fill_plot_column_select(_0, _1, func_field, default_field, session_id):
    df = cache_get(session_id, 'df', None)
    if df is None:
        return None, None

    if func_field is not None and func_field in df.columns:
        preselect = func_field
    elif default_field is not None and default_field in df.columns:
        preselect = default_field
    else:
        preselect = df.columns[0]

    options = [dict(label=c, value=c) for c in df.columns]
    value = preselect
    return options, value

    # fig = px.line(df, x=df.index, y=preselect)
    # graph = dcc.Graph(figure=fig, id='graph')
    # return [html.Br(), col_chooser, graph]


@app.callback(
    Output('graph', 'figure'),
    Input('plot-column', 'value'),
    State('session-id', 'data'),
)
def cb_plot_var(plotcol, session_id):
    if not plotcol:
        return dict(data=[], layout={}, frames=[])
    df: pd.DataFrame = cache_get(session_id, 'df', None)
    df = df[[plotcol]]

    # aggregation for big data
    # # df = df.reset_index()
    # window = len(df.index) // AGG_THRESHOLD
    # if window > 2:
    #     if df.index.inferred_type.startswith('datetime'):
    #         window = (df.index.max() - df.index.min()) / AGG_THRESHOLD
    #     df = df.resample(window).agg('mean')

    indexname = df.index.name or 'index'
    x, y = df.index, df[plotcol]
    fig = px.line(x=x, y=y, labels=dict(x=indexname, y=plotcol))
    return fig


# ======================================================================
# Function section and param parsing
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


def _get_docstring_description(docstr):
    dc = docparse
    children = []
    for o in [docstr.short_description, docstr.long_description, *docstr.meta]:
        if o is None or isinstance(o, (dc.DocstringParam, dc.DocstringReturns)):
            continue
        if not isinstance(o, str):
            o = f"#### {o.args[0].capitalize()}\n{o.description}"
        children.append(dcc.Markdown(o))
    return children


@app.callback(
    Output('default-field', 'data'),
    Input('new-data', 'data'),
    Input('func-field', 'data'),  # the `field` param of the selected function
    State('default-field', 'data'),
    State('session-id', 'data')
)
def cb_maybe_update_default_field(_0, func_field, default_field, session_id):
    """
    - update the default if the user entered a new valid value
    - keep the old value otherwise
    - initially suggest a df-column
    """

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # once the user set a valid field we keep it forever
    if func_field is not None:
        return func_field

    # now func_field is None !

    df = cache_get(session_id, 'df', None)
    if df is None or df.columns.empty:
        columns = [None]
    else:
        columns = df.columns

    # check if our possibly suggested default is still valid
    if default_field in columns:
        return default_field

    return str(columns[0])  # suggest a default


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
    docstr_obj = docparse.parse(func.__doc__)
    children += _get_docstring_description(docstr_obj)
    param_descriptions = {p.arg_name: p.description for p in docstr_obj.params}

    # dynamically add param input fields
    for name, param in inspect.signature(func).parameters.items():
        if name in IGNORED_PARAMS:
            continue

        default = None
        if param.default != inspect.Parameter.empty:
            default = repr(param.default)

        hint = type_repr(param.annotation)
        if param.annotation is inspect.Parameter.empty:
            hint = ''

        description = dcc.Markdown(param_descriptions.get(name, []))

        # special param `field`
        if name == 'field':
            default = default_field
            if default is not None:
                default = repr(default)
            description = "A column name holding the data."

        # using a dict as `id` makes pattern matching callbacks possible
        id = {"group": "param", "id": name}
        id_valid = {"group": "param-validation", "id": name}

        # html creation
        name = html.B(name)
        inp = dbc.Input(type='text', value=default, debounce=True, id=id)
        hint = dbc.FormText(hint, color='secondary')
        form = dbc.FormGroup([
            dbc.Label(name, html_for=id, width=2),
            dbc.Col([inp, hint], width=10),
            dbc.Col(description, width=12),
            dbc.Col([], width=12, id=id_valid),  # filled by cb_param_validation()
        ], row=True)
        param_forms.append(html.Hr())
        param_forms.append(form)

    children.append(dbc.Form(param_forms))

    return children


@app.callback(
    Output({'group': 'param', 'id': MATCH}, 'invalid'),
    Output({'group': 'param-validation', 'id': MATCH}, 'children'),
    Input('new-data', 'data'),  # trigger a re-check on new data
    Input({'group': 'param', 'id': MATCH}, 'value'),
    State({'group': 'param', 'id': MATCH}, 'id'),
    State('session-id', 'data')
)
def cb_param_validation(new_data, value, param_id, session_id):
    """
    validated param input and show an alert if validation fails
    """
    if value is None:
        return True, []

    # Empty value after user already was in the input form
    if value == "":
        return True, dbc.Alert([html.B('Error: '), 'Missing value'], color='danger')

    # existence of func is ensured by earlier callbacks
    func = cache_get(session_id, 'func')
    df = cache_get(session_id, 'df', None)

    param_name = param_id['id']
    param = inspect.signature(func).parameters[param_name]

    # sometimes the written typehints in saqc aren't explicit about None
    annotation = param.annotation
    if param.default is None:
        annotation = typing.Union[annotation, None]

    try:
        parsed = param_parse(value)
        param_typecheck(param_name, parsed, annotation, df)
    except (TypeError, ValueError) as e:
        failed, msg = e.args
    else:
        failed, msg = False, ""

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
    Output('func-field', 'data'),
    Input({'group': 'param', 'id': ALL}, 'invalid'),
    State({'group': 'param', 'id': ALL}, 'value'),
    State({'group': 'param', 'id': ALL}, 'id'),
    State('func-selected', 'data'),
    State('session-id', 'data'),
)
def cb_parsing_done(
        invalids, values, param_ids, func_selected, session_id
):
    if not func_selected:
        raise PreventUpdate

    # if the user enters a value for `field`, which is always present (!)
    # we update the default field, unless the value is not valid, then we
    # keep its former default
    i = param_ids.index({'group': 'param', 'id': 'field'})
    if invalids[i]:
        func_field = None
    # unfortunately valid means ugly quotes
    else:
        func_field = values[i][1:-1]

    if any(invalids):
        return False, func_field

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

    return True, func_field


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
    Input('add-to-config', 'n_clicks'),  # trigger by `add`
    Input('clear-config', 'n_clicks'),  # trigger by `clear`
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
# Processing
# ======================================================================

@app.callback(
    Output('preview', 'disabled'),
    Input('params-parsed', 'data'),
    Input('new-data', 'data'),
)
def cb_enable_preview(parsed, new_data):
    """ enable preview-button if we have data and all param-inputs are valid. """
    return not parsed or not new_data


@app.callback(
    Output('preview-alert', 'children'),
    Output('result', 'children'),
    # Output('plot', 'children'),
    Output('ignore', 'data'),
    Input('preview', 'n_clicks'),
    State('session-id', 'data'),
)
def cb_process(clicked, session_id):
    """
    parse all inputs.
    if successful calculate result
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
        fig.add_scatter(x=x, y=y, mode="markers", )

        plot = dcc.Graph(figure=fig)
    except Exception as e:
        alert = dbc.Alert(repr(e), color='danger')
        txt = 'Errors during ' + txt
    else:
        txt = 'Great Success\n=============\n' + txt

    return alert, html.Pre(txt), plot
