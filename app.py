#!/usr/bin/env python
import uuid

import dash
import os
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from flask_caching import Cache
from layout import layout


def serve_layout():
    session_id = str(uuid.uuid4())
    layout.session_id = session_id

    outlay = html.Div([
        dcc.Store(id='session-id', data=session_id),  # used to cache data
        dcc.Store(id="new-data", data=False),  # new data/flags in cache
        dcc.Store(id="new-upload", data=False),  # indicates data was uploaded
        dcc.Store(id="data-update", data=False),  # function result changed data/flags
        dcc.Store(id="func-selected", data=False),  # saqc-function in cache
        dcc.Store(id='plot-column-preselect', data=None),  # first to show column
        dcc.Store(id='func-field', data=None),  # data user put in input `field`
        dcc.Store(id="params-parsed", data=False),  # indicate if all params are parsed successfully
        dcc.Store(id='default-field', data=None),  # default value or suggestion for `field`
        dcc.Store(id='ignore'),  # (debug) placeholder
        layout
    ])
    return outlay


app = dash.Dash(
    name=__name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    url_base_pathname=os.getenv("APP_URL", None),
    suppress_callback_exceptions=True,
)
app.layout = serve_layout

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': '.server_cache'
})
