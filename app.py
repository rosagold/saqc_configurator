#!/usr/bin/env python
import uuid

import dash
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


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# app.config.suppress_callback_exceptions = True
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': '.server_cache'
})
app.layout = serve_layout
server = app.server
