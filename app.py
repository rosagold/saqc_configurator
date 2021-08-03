#!/usr/bin/env python


import dash
import dash_bootstrap_components as dbc
from layout import layout

app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    # suppress_callback_exceptions=True,
)

app.layout = layout
