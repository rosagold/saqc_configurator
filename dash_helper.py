#!/usr/bin/env python

import dash_bootstrap_components as dbc
import saqc.lib.types as saqc_types

TYPE_MAPPING = {
    str: 'text',
    int: 'number',
    float: 'number',
    bool: 'text',
    saqc_types.ColumnName: 'text',
    saqc_types.TimestampColumnName: 'text',
    saqc_types.FreqString: 'text',
    saqc_types.IntegerWindow: 'number',
    saqc_types.PositiveFloat: 'number',
    saqc_types.PositiveInt: 'number',
}


def text_value_input(text, id, type="text", widths=(2, 10), row=True, raw=False, **kws):
    if type in ['t', 'txt', 'text']:
        type = 'text'
    if type in ['n', 'nr', 'number']:
        type = 'number'

    form = dbc.FormGroup(
        [
            dbc.Label(text, html_for=id, width=widths[0]),
            dbc.Col(dbc.Input(type=type, id=id, **kws), width=widths[1]),
        ],
        row=row,
    )
    if raw:
        return form
    return dbc.Form(form)


def DivRow(children, widths=(2, 10)):
    if len(children) != 2:
        raise ValueError(f"length of children ({len(children)}) > 2")

    return dbc.Form(dbc.FormGroup(
        [
            dbc.Label(children[0], width=widths[0]),
            dbc.Col(children[1], width=widths[1]),
        ],
        row=True,
    ))
