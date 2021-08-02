#!/usr/bin/env python
import typing

import dash_bootstrap_components as dbc
import saqc.lib.types as saqc_types
import inspect


# ===========================================================================
# Parsing
# ===========================================================================

TYPE_MAPPING = {
    inspect._empty: typing.Any,
    saqc_types.ColumnName: str,
    saqc_types.TimestampColumnName: str,
    saqc_types.FreqString: str,
    saqc_types.IntegerWindow: int,
    saqc_types.PositiveInt: int,
    saqc_types.PositiveFloat: float,
}

def literal_eval_extended(node_or_string: str):
    """
    Safely evaluate an expression node or a string containing a Python
    expression.  The string or node provided may only consist of the following
    Python literal structures: strings, bytes, numbers, tuples, lists, dicts,
    sets, booleans, and None.

    Taken from ast package, thanks.
    + Added 'nan' and 'inf' support
    """
    from ast import (parse, Expression, Constant, Name, UnaryOp, UAdd, USub, Tuple,
                     List, Dict, Set, BinOp, Add, Sub)
    if isinstance(node_or_string, str):
        node_or_string = parse(node_or_string, mode='eval')
    if isinstance(node_or_string, Expression):
        node_or_string = node_or_string.body

    def _convert_num(node):
        if isinstance(node, Constant):
            if type(node.value) in (int, float, complex):
                return node.value
        elif isinstance(node, Name) and node.id in ['nan', 'inf']:
            return float(node.id)
        raise ValueError('malformed node or string: ' + repr(node))

    def _convert_signed_num(node):
        if isinstance(node, UnaryOp) and isinstance(node.op, (UAdd, USub)):
            operand = _convert_num(node.operand)
            if isinstance(node.op, UAdd):
                return + operand
            else:
                return - operand
        return _convert_num(node)

    def _convert(node):
        if isinstance(node, Constant):
            return node.value
        elif isinstance(node, Tuple):
            return tuple(map(_convert, node.elts))
        elif isinstance(node, List):
            return list(map(_convert, node.elts))
        elif isinstance(node, Set):
            return set(map(_convert, node.elts))
        elif isinstance(node, Dict):
            return dict(zip(map(_convert, node.keys),
                            map(_convert, node.values)))
        elif isinstance(node, BinOp) and isinstance(node.op, (Add, Sub)):
            left = _convert_signed_num(node.left)
            right = _convert_num(node.right)
            if isinstance(left, (int, float)) and isinstance(right, complex):
                if isinstance(node.op, Add):
                    return left + right
                else:
                    return left - right
        return _convert_signed_num(node)

    return _convert(node_or_string)


def type_repr(t, pretty=True):
    # HINT: a.o. Union[.] has no __name__ attribute
    s = getattr(t, '__name__', repr(t))
    if pretty:
        s = (s.replace('typing.', '')
             .replace('NoneType', 'None')
             .replace('Union', '')
             )
        # rm `[` and `]`
        if typing.get_origin(t) is typing.Union:
            s = s[1:-1]
    return s


# ===========================================================================
# html creation helper
# ===========================================================================


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
