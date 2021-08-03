# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc
import dash_table
import plotly.express as px

from helper import text_value_input, DivRow
from const import AGG_METHODS, SAQC_FUNCS


input_section = html.Div(
    [
        # Data upload, params and preview
        html.Div([
            dbc.Form(dbc.FormGroup(
                [
                    dbc.Label('Data', width=2),
                    dbc.Col(dcc.Upload(dbc.Button("Upload File"), id="upload-data"), width=2),
                    dbc.Col(' or ', width=1, align='center'),
                    dbc.Col(dbc.Button("Random Data", id="random-data"), width=2),
                ],
                row=True,
            )),
            text_value_input("Header row", value=0, id=f"header-row"),
            text_value_input("Index column", value=0, id="index-column"),
            text_value_input("Data column", value=1, id="data-column"),
            html.Br(),

            html.Div([], id="df-preview"),  # filled by cb_df_preview()
        ]),

        # config upload and preview
        html.Div([
            DivRow(["Config (optional)", dcc.Upload(dbc.Button("Upload File"),
                                                    id="upload-config")]),
            html.Div(id="config-preview"),  # filled by cb_config_preview
        ]),
    ]
)

function_section = html.Div(
    [
        DivRow(
            [
                "preview aggregation method :",
                dbc.Select(
                    options=[dict(label=m, value=m) for m in AGG_METHODS],
                    value=AGG_METHODS[0],
                    id="agg-select",
                ),
            ]
        ),
        DivRow(
            [
                "function: ",
                dbc.Select(
                    options=[dict(label=f, value=f) for f in SAQC_FUNCS.keys()],
                    placeholder="Select a function",
                    id="function-select",
                ),
            ]
        ),
        dbc.Card(
            [
                dbc.CardHeader([], id="params-header"),  # filled by cb_params_header()
                dbc.CardBody([], id="params-body"),  # filled by cb_params_body()
                dbc.CardFooter(
                    [
                        dbc.Button("Submit", id="submit", block=True),
                        html.Div([], id="submit-error"),  # filled by cb_submit()
                    ]
                ),
            ]
        ),
        html.Br(),
        dbc.Card("Result", body=True, id="result"),  # filled by cb_submit()
    ]
)

layout = dbc.Container(
    [
        html.H1("SaQC Configurator"),
        dbc.Card(
            [
                dbc.CardHeader("Input section"),
                dbc.CardBody([input_section]),
            ]
        ),
        html.Br(),
        dbc.Card(
            [
                dbc.CardHeader("Function section"),
                dbc.CardBody([function_section]),
            ]
        ),
    ],
)
