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
        dcc.Store(id='session-id', data=session_id),
        dcc.Store(id="df-present", data=False),
        dcc.Store(id="func-selected", data=False),
        dcc.Store(id="params-parsed", data=False),
        dcc.Store(id='default-field', data=None),
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
