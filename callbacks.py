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
import dios
from saqc import UNFLAGGED

from helper import (
    parse_data, type_repr, parse_keywords, cache_get, cache_set,
    saqc_func_repr, param_parse, param_typecheck, aggregate
)
from const import (
    AGG_METHODS, SAQC_FUNCS, TYPE_MAPPING, IGNORED_PARAMS, PARSER_MAPPING,
    AGG_THRESHOLD, MAX_DF_ROWS)
from app import app

send_signal = True
send_no_signal = dash.no_update


def _on_change(new, old):
    if new == old:
        return dash.no_update
    return new


def _get_trigger():
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    return ctx.triggered[0]['prop_id'].split('.')[0]


# ======================================================================
# Input section
# ======================================================================

@app.callback(
    Output('new-random-signal', 'data'),
    Output('upload-data', 'filename'),  # reset data upload
    Input('random-data', 'n_clicks'),  # trigger only
    State('session-id', 'data'),
    State('random-type', 'value'),
    prevent_initial_call=True,
)
def cb_click_to_random(_, session_id, random_type):
    """
    Generate random data.

    After generation we fake a click to upload-button by
    setting a new filename, this will trigger the `cb_upload_data`
    callback.
    """
    random_type = random_type or []
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
    flags = pd.DataFrame(UNFLAGGED, index=df.index, columns=df.columns, dtype=float)
    cache_set(session_id, 'df', df)
    cache_set(session_id, 'flags', flags)
    return send_signal, None


@app.callback(
    Output('new-upload-signal', 'data'),
    Output('parser-kwargs', 'invalid'),
    Output('upload-alert', 'children'),
    Input('upload-data', 'filename'),
    Input('upload-data', 'contents'),
    Input('datafile-type', 'value'),
    Input('parser-kwargs', 'value'),
    State('session-id', 'data'),
    prevent_initial_call=True,
)
def cb_upload_data(filename, content, filetype, parser_kws, session_id):
    kws_valid, kws_invalid = False, True
    no_alert = []

    # parse keywords
    try:
        parser_kws = parse_keywords(parser_kws)
    except SyntaxError:
        msg = "Keyword parsing error: Syntax: 'key1=..., key3=..., key3=...'"
        return send_no_signal, kws_invalid, dbc.Alert(msg, color='danger')

    if filename is None or content is None:
        return send_no_signal, kws_valid, no_alert

    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = parse_data(filename, filetype, decoded, parser_kws)
        if len(df.columns) == 0:
            raise ValueError("DataFrame must not be empty.")
    except Exception as e:
        msg = f"Data parsing error: {repr(e)}"
        return send_no_signal, kws_valid, dbc.Alert(msg, color='danger')

    alert = []
    if len(df.index) > MAX_DF_ROWS:
        df = df.iloc[:MAX_DF_ROWS, :]
        msg = f'Maximum data set size exceeded. ' \
              f'The data is truncated to {MAX_DF_ROWS} rows per column'
        alert = dbc.Alert([html.B('Warning: '), msg], color='warning')

    flags = pd.DataFrame(UNFLAGGED, index=df.index, columns=df.columns, dtype=float)
    cache_set(session_id, 'df', df)
    cache_set(session_id, 'flags', flags)
    return send_signal, kws_valid, alert


@app.callback(
    Output('data-src-and-signal', 'data'),
    Input('new-random-signal', 'data'),  # just trigger
    Input('new-upload-signal', 'data'),  # just trigger
    Input('data-update-signal', 'data'),  # just trigger
    prevent_initial_call=True,
)
def cb_new_data(*_):
    trigger = _get_trigger()
    if trigger == 'new-random-signal':
        return 'random'
    if trigger == 'new-upload-signal':
        return 'upload'
    if trigger == 'data-update-signal':
        return 'update'
    raise RuntimeError('unknown trigger in `cb_new_data`')


# ======================================================================
# Data Table
# ======================================================================

@app.callback(
    Output('data-table', 'children'),
    Input('data-src-and-signal', 'data'),
    State('upload-data', 'filename'),
    State('session-id', 'data'),
    prevent_initial_call=True,
)
def cb_data_table(data_src, filename, session_id):
    if not data_src:
        raise PreventUpdate

    if data_src == 'random':
        filename = 'random_data'

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
    Input('data-src-and-signal', 'data'),
    State('session-id', 'data'),
    prevent_initial_call=True
)
def cb_new_plotcol_selector(data_src, session_id):
    """ make a new selector if we have new data """
    if data_src is None:
        raise PreventUpdate
    df = cache_get(session_id, 'df', None)
    return [dict(label=c, value=c) for c in df.columns]


@app.callback(
    Output('plot-column', 'value'),
    Input('preview', 'n_clicks'),  # trigger only
    Input('data-src-and-signal', 'data'),  # trigger only
    State('func-field', 'data'),
    State('plot-column', 'value'),
    State('session-id', 'data'),
    prevent_initial_call=True
)
def cb_plotcol_selector_value(_0, data_src, func_field, curr_val, session_id):
    if data_src is None:
        raise PreventUpdate
    df = cache_get(session_id, 'df', None)
    trigger = _get_trigger()
    if trigger is None or df is None:
        raise PreventUpdate
    if trigger == 'preview' and func_field in df.columns:
        return func_field
    if curr_val is None or curr_val not in df.columns:
        return df.columns[0]
    raise PreventUpdate


@app.callback(
    Output('graph', 'figure'),
    Input('plot-column', 'value'),
    Input('data-src-and-signal', 'data'),  # trigger only
    State('session-id', 'data'),
    prevent_initial_call=True
)
def cb_plot_var(plotcol, data_src, session_id):
    if plotcol is None or data_src is None:
        raise PreventUpdate
    # now data must exist

    df = cache_get(session_id, 'df')
    fl = cache_get(session_id, 'flags')

    if plotcol not in df.columns:
        raise PreventUpdate

    data = df[plotcol]
    flags = fl[plotcol]
    indexname = df.index.name or 'index'

    # plot data
    fig = px.line(x=data.index, y=data, labels=dict(x=indexname, y=plotcol))
    # plot flags
    flagged = flags > saqc.UNFLAGGED
    y = data.loc[flagged]
    fig.add_scatter(x=y.index, y=y, mode="markers")
    return fig


# ======================================================================
# Function section and param parsing
# ======================================================================

@app.callback(
    Output('func-selected-signal', 'data'),
    Output('params-header', 'children'),
    Input('function-select', 'value'),
    State('session-id', 'data'),
    prevent_initial_call=True
)
def cb_params_header(funcname, session_id):
    if funcname is None:
        raise PreventUpdate
    func = SAQC_FUNCS[funcname]
    cache_set(session_id, 'func', func)
    return send_signal, html.H4(funcname)


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
    Input('data-src-and-signal', 'data'),  # just a trigger
    Input('func-field', 'data'),  # the `field` param of the selected function
    State('default-field', 'data'),
    State('session-id', 'data'),
    prevent_initial_call=True
)
def cb_maybe_update_default_field(_0, func_field, curr_value, session_id):
    """
    - we update the default-field if the user enter a new valid value
    - otherwise we keep its value
    - initially suggest a df-column
    """

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # once the user set a valid field we update the default
    if func_field is not None:
        if func_field == curr_value:
            raise PreventUpdate
        return func_field

    # now `func_field` is None, this means
    # its not present at all or not valid.

    df = cache_get(session_id, 'df', None)
    if df is None or df.columns.empty:
        columns = [None]
    else:
        columns = df.columns

    # is the current value still valid ?
    if curr_value in columns:
        raise PreventUpdate

    return str(columns[0])  # suggest a default


@app.callback(
    Output('params-body', 'children'),
    Input('func-selected-signal', 'data'),
    State('default-field', 'data'),
    State('session-id', 'data'),
    prevent_initial_call=True
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
        id_valid = {"group": "param-alert", "id": name}

        # html creation
        name = html.B(name)
        inp = dbc.Input(type='text', value=default, debounce=False, id=id)
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
    Output({'group': 'param-alert', 'id': MATCH}, 'children'),
    Input('data-src-and-signal', 'data'),  # trigger a re-check on new data
    Input({'group': 'param', 'id': MATCH}, 'value'),
    State({'group': 'param', 'id': MATCH}, 'id'),
    State('session-id', 'data'),

    # initial here means: when the element inserted in the layout,
    # which happens dynamically in `cb_params_body` and we do want
    # to check the (default) values
    prevent_initial_call=False,
)
def cb_param_validation(_0, value, param_id, session_id):
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
    Output('params-parsed-and-signal', 'data'),
    Output('func-field', 'data'),
    Input({'group': 'param', 'id': ALL}, 'value'),
    Input({'group': 'param', 'id': ALL}, 'invalid'),
    State({'group': 'param', 'id': ALL}, 'id'),
    State('func-selected-signal', 'data'),
    State('params-parsed-and-signal', 'data'),
    State('session-id', 'data'),
)
def cb_parsing_done(
        values, invalids, param_ids, func_selected, curr_value, session_id
):
    if not func_selected or None in invalids:
        return _on_change(False, curr_value), None

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
        return _on_change(False, curr_value), func_field

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

    return _on_change(True, curr_value), func_field


# ======================================================================
# Processing
# ======================================================================

@app.callback(
    Output('preview', 'disabled'),
    Input('params-parsed-and-signal', 'data'),
    Input('data-src-and-signal', 'data'),
)
def cb_enable_preview(parsed, data_exist):
    """ enable preview-button if we have data and all param-inputs are valid. """
    return not (parsed and data_exist)


@app.callback(
    Output("data-update-signal", 'data'),
    Output('preview-alert', 'children'),
    Output('result', 'children'),
    Input('preview', 'n_clicks'),
    State('session-id', 'data'),
    prevent_initial_call=True,
)
def cb_process(clicked, session_id):
    """
    parse all inputs.
    if successful calculate result
    on parsing errors show an alert.
    """
    df = cache_get(session_id, 'df')
    flags = cache_get(session_id, 'flags')
    func = cache_get(session_id, 'func')
    params = cache_get(session_id, 'params')
    field = params['field']

    frpr = f"`qc.{saqc_func_repr(func, params)}`"

    alert, plot = [], []
    try:
        qc = saqc.SaQC(df)
        func = getattr(qc, f"{func._module}.{func.__name__}")
        result = func(**params)
        flag_col = result.flags[field]
        flags[field] = flag_col

    except Exception as e:
        alert = dbc.Alert(repr(e), color='danger')
        txt = f'Errors during call of {frpr}\n'
    else:
        flagged = flag_col > UNFLAGGED
        n, N = len(flagged[flagged]), len(flagged)
        txt = f'Great Success\n' \
              f'=============\n' \
              f'call of {frpr}\n' \
              f'flagged: {n}/{N} data points ({round(n / N * 100, 2)}%)\n'

    # detailed params
    txt += '\nparsed parameter:' \
           '\n-----------------\n'
    for k, v in params.items():
        txt += f"{k}={repr(v)} ({type(v).__name__})\n"

    cache_set(session_id, 'flags', flags)
    return send_signal, alert, html.Pre(txt)


# ======================================================================
# Config
# ======================================================================


@app.callback(
    Output('add-to-config', 'disabled'),
    Input('params-parsed-and-signal', 'data'),
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
    trigger = _get_trigger()
    if trigger is None:
        raise PreventUpdate

    if config is None:
        config = ''

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
