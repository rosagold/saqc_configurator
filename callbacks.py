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
from app import app


def df_dumps(df):
    if not df.index.name:
        df.index.name = 'index'
    df = df.reset_index()
    return df.to_json(orient='split', double_precision=15),


def df_loads(s):
    return pd.read_json(s[0], orient='split', precise_float=True)


# ======================================================================
# Input section
# ======================================================================

@app.callback(
    Output('upload-data', 'contents'),
    Output('upload-data', 'filename'),
    Input('random-data', 'n_clicks'),
)
def random_to_content(n):
    if n is None:
        raise PreventUpdate
    rows = 990
    start = np.random.randint(9466848, 16094592) * 10 ** 11
    r = np.random.rand(rows, 10) * 10
    i = pd.date_range(start=start, periods=rows, freq='10min')
    df = pd.DataFrame(index=i, data=r, columns=list('abcdefghij'))
    df = df.round(2)

    return df_dumps(df), 'random_data'


@app.callback(
    Output('upload-error', 'children'),
    Output('parser-kwargs', 'invalid'),
    Output('df', 'data'),
    # Datafile
    Input('upload-data', 'contents'),
    Input('datafile-type', 'value'),
    Input('parser-kwargs', 'value'),
    State('upload-data', 'filename'),
)
def cb_parse_data(content, filetype, parser_kws, filename):
    if filename is None:
        raise PreventUpdate

    if filename == 'random_data':
        return [], False, content

    try:
        parser_kws = parse_keywords(parser_kws)
    except SyntaxError:
        msg = "Keyword parsing error: Syntax: 'key1=..., key3=..., key3=...'"
        return dbc.Alert(msg, color='danger'), True, None

    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = parse_data(filename, filetype, decoded, parser_kws)
    except Exception as e:
        msg = f"Data parsing error: {repr(e)}"
        return dbc.Alert(msg, color='danger'), False, None

    return [], False, df_dumps(df)


@app.callback(
    Output('df-preview', 'children'),
    Input('df', 'data'),
    State('upload-data', 'filename'),
)
def cb_df_preview(df_jstr, filename):
    if df_jstr is None:
        raise PreventUpdate
    df = df_loads(df_jstr)
    preview = [
        html.B(filename),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': str(c), 'id': str(c)} for c in df.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold'
            },
        ),
    ]
    return preview


# ======================================================================
# Function section and param validation
# ======================================================================

@app.callback(
    Output('params-header', 'children'),
    Input('function-select', 'value'),
)
def cb_params_header(funcname):
    if funcname is None:
        raise PreventUpdate
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
    Output('func_repr', 'data'),
    Input('add-to-config', 'n_clicks'),
    State({'group': 'param', 'id': ALL}, 'id'),
    State({'group': 'param', 'id': ALL}, 'value'),
    State('function-select', 'value'),
)
def cb_add_to_config(add, param_ids, param_values, funcname):
    if add is None:
        raise PreventUpdate
    fkws = dict()
    for i, id_dict in enumerate(param_ids):
        param_name = id_dict['id']
        value = param_values[i]
        fkws[param_name] = value
    return get_func_repr(funcname, fkws, method=str)


@app.callback(
    Output('config-preview', 'value'),
    Input('func_repr', 'data'),
    Input('clear-config', 'n_clicks'),
    Input('upload-config', 'contents'),
    State('upload-config', 'filename'),
    State('config-preview', 'value'),
    State('function-select', 'value'),
)
def cb_config_preview(func_repr, clear, content, filename, config, funcname):
    if config is None:
        config = ''
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == 'func_repr':
        return config + func_repr

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
    Input('df', 'data'),
    State('function-select', 'value'),
)
def cb_enable_preview(invalids, data, fname):
    """ enable preview-button if we have data and all param-inputs are valid. """
    return any(invalids) or data is None or fname is None


@app.callback(
    Output('submit-error', 'children'),
    Output('result', 'children'),
    Input('preview', 'n_clicks'),
    State({'group': 'param', 'id': ALL}, 'id'),
    State({'group': 'param', 'id': ALL}, 'value'),
    State('function-select', 'value'),
    State('df', 'data'),
)
def cb_submit(submit, param_ids, param_values, funcname, df_records):
    """
    parse all inputs.
    if successful calculate result TODO: text and plotted flags
    on parsing errors show an alert.
    """
    if not submit:
        return [], []

    kws_to_func = {}
    submit = True

    if df_records is None:
        alert = dbc.Alert("No data selected", color='warning'),
        out = html.Pre('Failed')
        return alert, out

    # parse values, all checks are already done, in the input-form-callback
    for i, id_dict in enumerate(param_ids):
        param_name = id_dict['id']
        value = param_values[i]
        try:
            kws_to_func[param_name] = literal_eval_extended(value)
        except (SyntaxError, ValueError):
            submit = False

    if not submit:
        alert = dbc.Alert("Errors or missing fields above.", color='danger'),
        out = html.Pre('Failed')
        return alert, out

    txt = 'Great Success\n=============\n'
    for k, v in kws_to_func.items():
        txt += f"{k}={repr(v)} ({type(v).__name__})\n"

    df = pd.DataFrame.from_records(df_records)
    qc = saqc.SaQC(df)
    func = SAQC_FUNCS[funcname]
    saqcobj_funcname = f"{func._module}.{func.__name__}"
    func = getattr(qc, saqcobj_funcname)
    field = 'a'
    result = func(field=field, **kws_to_func)
    txt = get_func_repr(funcname, kws_to_func)
    print(result, type(result))
    return [], html.Pre(txt)


def get_func_repr(fname, fkwargs: dict, ostr='qc', method=repr):
    func = SAQC_FUNCS[fname]
    saqcobj_funcname = f"{func._module}.{func.__name__}"
    rpr = f"{saqcobj_funcname}("
    if ostr:
        rpr = f"{ostr}.{rpr}"
    for k, v in fkwargs.items():
        rpr += f"{k}={method(v)},"
    if fkwargs:  # remove trailing comma
        rpr = rpr[:-1]
    rpr += ')\n'
    return rpr
