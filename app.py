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
        # used to cache data per user
        dcc.Store(id='session-id', data=session_id),

        # signals, even if their value keeps the same, they will trigger callback,
        # which use them as `Input(..., 'data')`
        dcc.Store(id="new-data", data=False),  # new data/flags in cache
        dcc.Store(id="new-upload", data=False),  # new data was uploaded
        dcc.Store(id="data-update", data=False),  # function result changed data/flags
        dcc.Store(id="func-selected", data=False),  # saqc-function in cache
        dcc.Store(id="params-parsed", data=False),  # all params parsed successfully

        # default value in select-column-to-plot
        dcc.Store(id='plot-column-preselect', data=None),
        # the valid (!) str value the user put in the input `field` otherwise None
        dcc.Store(id='func-field', data=None),
        # default value or suggestion for `field` on creation
        dcc.Store(id='default-field', data=None),
        # (debug) placeholder
        dcc.Store(id='ignore'),
        layout
    ])
    return outlay


app = dash.Dash(
    name=__name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    url_base_pathname=os.getenv("APP_URL", None),
    suppress_callback_exceptions=False,
)
app.layout = serve_layout

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': '.server_cache'
})
