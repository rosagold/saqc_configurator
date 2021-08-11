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
from const import AGG_METHODS, SAQC_FUNCS, PARSER_MAPPING, RANDOM_TYPES, MAX_FILE_SIZE, \
    MEGA, PARSER_KW_DEFAULTS

data_section = html.Div(
    [

        # ######################################################################
        # Random Data
        # ######################################################################
        dbc.Form(
            [
                dbc.FormGroup(
                    [
                        dbc.Label("Data", width=2),
                        dbc.Col(
                            # trigger -> cb_click_to_random
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
        ),

        html.Hr(),

        # ######################################################################
        # upload data
        # ######################################################################
        dbc.Form(
            [
                dbc.FormGroup(
                    [
                        dbc.Label("Data", width=2),
                        dbc.Col(
                            [
                                dcc.Upload(
                                    dbc.Button("Upload File"),
                                    max_size=MAX_FILE_SIZE,
                                    id="upload-data",
                                ),
                                dbc.FormText(
                                    f"Maximum upload file size: {MAX_FILE_SIZE//MEGA}M",
                                    color="secondary",
                                ),
                            ],
                            width="auto",
                        ),
                        dbc.Col(
                            [
                                dbc.Select(
                                    options=[dict(label=m, value=m) for m in
                                             PARSER_MAPPING.keys()],
                                    value=list(PARSER_MAPPING.keys())[0],
                                    id="datafile-type",
                                ),

                            ],
                            width="auto",
                        ),
                    ],
                    row=True,
                    inline=True,
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Parser kwargs", html_for="parser-kwargs", width=2),
                        dbc.Col(
                            [
                                dbc.Input(
                                    type="text",
                                    value=PARSER_KW_DEFAULTS,
                                    id="parser-kwargs",
                                ),
                                dbc.FormText(
                                    dcc.Markdown(
                                        "kwargs passed to `pandas.read_csv` or "
                                        "`pandas.read_exel`",
                                    ),
                                    color="secondary",
                                ),
                            ],
                            width=10,
                        ),
                    ],
                    row=True,
                ),
                html.Div([], id="upload-alert")
            ]
        ),
    ]
)

config_section = html.Div(
    [

        dbc.Form(
            [
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
        ,

        # filled by cb_config_preview()
        html.Div(
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
    ]
)

function_section = html.Div(
    [
        # DivRow(
        #     [
        #         "preview aggregation method :",
        #         dbc.Select(
        #             options=[dict(label=m, value=m) for m in AGG_METHODS],
        #             value=AGG_METHODS[0],
        #             id="agg-select",
        #         ),
        #     ]
        # ),
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
                        html.Div([], id="preview-alert"),  # filled by cb_process()
                    ]
                ),
            ]
        ),
    ]
)

data_table = html.Div(["No data yet"], id='data-table')
plot_container = html.Div([
    dbc.FormGroup(
        [
            dbc.Label("Column to plot", width='auto', html_for="plot-column"),
            dbc.Col(
                dbc.Select(
                    id="plot-column",
                    placeholder='No data to plot'
                ),
                width="auto",
            )
        ], row=True, inline=True
    ),
    dcc.Graph(id='graph')
])

title = html.H1("SaQC Configuration App")
footer = html.Div([html.Br()] * 5)

data_card = dbc.Card(
    [
        dbc.CardHeader("Input"),
        dbc.CardBody([data_section]),
    ]
)

preview_card = dbc.Card(
    [
        dbc.CardHeader("Data / Plot"),
        dbc.CardBody(
            dbc.Tabs(
                [
                    dbc.Tab([html.Br(), plot_container], label="Plot"),  # tab-0
                    dbc.Tab([html.Br(), data_table], label="Data"),  # tab-1
                ],
                active_tab='tab-0',
            )
        ),
    ]
)

function_card = dbc.Card(
    [
        dbc.CardHeader("Function"),
        dbc.CardBody([function_section]),
    ]
)

config_card = dbc.Card(
    [
        dbc.CardHeader("Config"),
        dbc.CardBody([config_section]),
    ]
)

result_card = dbc.Card("Result", body=True, id="result")  # filled by cb_process()

# ######################################################################
# Final layouts
# ######################################################################


# all in columns
layout0 = dbc.Container([
    title,
    data_card,
    preview_card,
    function_card,
    result_card,
    config_card,
    footer,
])

# row(col,col)
# row(col,col)
# ------.--------
# data  | config
#       | config
# ------|--------
# func  | preview
# func  |
# func  |
# ------^--------
layout1 = dbc.Container(
    [
        html.Br(),
        title,
        html.Br(),
        dbc.Row([
            dbc.Col([data_card], width=5),
            dbc.Col([config_card], width=7),
        ]),

        html.Br(),

        dbc.Row([
            dbc.Col([function_card], width=5),
            dbc.Col([preview_card, result_card], width=7),
        ]),

        footer,
    ],
    fluid=True,
)

# row(col,col)
# row(col,col)
# ------.--------
# data  | config
# func  | config
# func  | preview
# func  |
# func  |
# ------^--------
layout2 = dbc.Container(
    [
        html.Br(),
        title,
        html.Br(),
        dbc.Row([
            dbc.Col([data_card, html.Br(), function_card, html.Br(), result_card], width=5),
            dbc.Col([config_card, html.Br(), preview_card], width=7),
        ]),
        footer,
    ],
    fluid=True,
)

layout = layout2
