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

from helper import FormGroupInput, DivRow
from const import AGG_METHODS, SAQC_FUNCS, PARSER_MAPPING, RANDOM_TYPES

random_data_section = dbc.Form(
    [
        # Data        [Generate]  [x] outlier [x] plateaus [x] gaps/NaN
        dbc.FormGroup(
            [
                dbc.Label("Data", width=2),
                dbc.Col(
                    dbc.Button("Generate", id="random-data"),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Checklist(
                        options=RANDOM_TYPES,
                        inline=True,
                        id="random-type",
                    ),
                    align="center",
                    width="auto",
                ),
            ],
            row=True,
        ),
    ]
)

data_input_section = dbc.Form(
    [
        # Data        [Upload File]  [mean|v]
        # Header      [_________]
        # Index       [_________]
        # DataCol     [_________]
        # Parsing kw  [_________]
        dbc.FormGroup(
            [
                dbc.Label("Data", width=2),
                dbc.Col(
                    dcc.Upload(dbc.Button("Upload File"), id="upload-data"),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Select(
                        options=[dict(label=m, value=m) for m in PARSER_MAPPING.keys()],
                        value=list(PARSER_MAPPING.keys())[0],
                        id="datafile-type",
                    ),
                    width="auto",
                ),
            ],
            row=True,
            inline=True,
        ),
        # FormGroupInput("Header row", value=0, id=f"header-row"),
        # FormGroupInput("Index column", value=0, id="index-column"),
        # FormGroupInput("Data column", value=1, id="data-column"),
        dbc.FormGroup(
            [
                dbc.Label("Parser kwargs", html_for="parser-kwargs", width=2),
                dbc.Col(
                    [
                        dbc.Input(
                            type="text",
                            value="header=0, index_col=0, parse_dates=True",
                            id="parser-kwargs",
                        ),
                        dbc.FormText(
                            dcc.Markdown(
                                "kwargs passed to `pandas.read_csv` or `pandas.read_exel`",
                            ),
                            color="secondary",
                        ),
                    ],
                    width=10,
                ),
            ],
            row=True,
        ),
        html.Div([], id="upload-error"),  # TODO
    ]
)

data_preview = html.Div([], id="df-preview")  # filled by cb_df_preview()

config_input_section = dbc.Form(
    [
        # Config  [Upload File]
        dbc.FormGroup(
            [
                dbc.Label("Config (optional)", width=2),
                dbc.Col(
                    dcc.Upload(dbc.Button("Upload File"), id="upload-config"),
                    width="auto",
                ),
                dbc.Col(dbc.Button("Clear", id="clear-config"), width="auto"),
            ],
            row=True,
            inline=True,
        ),
    ]
)

# filled by cb_config_preview()
config_preview = html.Div(
    dbc.Textarea(
        debounce=True,
        rows=10,
        wrap=True,
        bs_size="sm",
        readOnly=True,
        className="mb-3",
        id="config-preview",
    )
)

input_section = html.Div(
    [
        random_data_section,
        html.Hr(),
        data_input_section,
        data_preview,
    ]
)

config_section = html.Div(
    [
        config_input_section,
        config_preview,
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
        html.Br(),
        dbc.Card([]),
        html.Br(),
        dbc.Card(
            [
                dbc.CardHeader([], id="params-header"),  # filled by cb_params_header()
                dbc.CardBody([], id="params-body"),  # filled by cb_params_body()
                dbc.CardFooter(
                    [
                        dbc.FormGroup(
                            [
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Preview",
                                            block=True,
                                            disabled=True,
                                            id="preview",
                                        ),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Add to config",
                                            block=True,
                                            disabled=True,
                                            id="add-to-config",
                                        ),
                                    ],
                                    width=6,
                                ),
                            ],
                            row=True,
                            inline=True,
                        ),
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
        dcc.Store(id="df"),
        dcc.Store(id="func_repr"),
        html.H1("SaQC Configurator"),
        dbc.Card(
            [
                dbc.CardHeader("1. generate or upload data"),
                dbc.CardBody([input_section]),
            ]
        ),
        html.Br(),
        dbc.Card(
            [
                dbc.CardHeader("2. optionally upload an existing config"),
                dbc.CardBody([config_section]),
            ]
        ),
        html.Br(),
        dbc.Card(
            [
                dbc.CardHeader("3. choose a function"),
                dbc.CardBody([function_section]),
            ]
        ),
    ],
)
