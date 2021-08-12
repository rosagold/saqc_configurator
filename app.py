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
    return html.Div([
        layout,

        # used to cache data per user
        dcc.Store(id='session-id', data=session_id),

        # signals, even if their value keeps the same, they will trigger callback,
        # which use them as `Input(..., 'data')`
        dcc.Store(id="new-random-signal", data=False),  # new data was uploaded
        dcc.Store(id="new-upload-signal", data=False),  # new data was uploaded
        dcc.Store(id="data-update-signal", data=False),  # function result changed data/flags
        dcc.Store(id="func-selected-signal", data=False),  # saqc-function in cache

        # signals which also hold data
        dcc.Store(id="data-src-and-signal", data=None),  # new data/flags in cache (None/random/update/upload)
        dcc.Store(id="params-parsed-and-signal", data=False),  # all params parsed successfully (true/false)

        # default value in select-column-to-plot
        dcc.Store(id='plot-column-preselect', data=None),
        # the valid (!) str value the user put in the input `field` otherwise None
        dcc.Store(id='func-field', data=None),
        # default value or suggestion for `field` on creation
        dcc.Store(id='default-field', data=None),
        # (debug) placeholder
        dcc.Store(id='ignore'),
    ])


app = dash.Dash(
    name=__name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    url_base_pathname=os.getenv("APP_URL", None),
    suppress_callback_exceptions=False,
)
app.layout = serve_layout

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': '.server_cache',
    'CACHE_DEFAULT_TIMEOUT': 60*60*12+60,  # in seconds (12h1min)
    'CACHE_THRESHOLD': 1000*12,  # max number of parallel users per 12 hours
})
