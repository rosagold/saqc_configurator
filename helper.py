#!/usr/bin/env python
import typing
import litereval
import pandas as pd
import io
import dash_bootstrap_components as dbc
import saqc.lib.types
from const import TYPE_MAPPING
import typeguard


# ======================================================================
# Cache
# ======================================================================

def cache_set(session_id, key, value):
    from app import cache  # late import to avoid circles
    ca = cache.get(session_id) or dict()
    ca[key] = value
    cache.set(session_id, ca)


def cache_get(session_id, key, *args):
    from app import cache  # late import to avoid circles
    ca = cache.get(session_id) or dict()
    if len(args) == 0:
        return ca[key]  # raise KeyError
    if len(args) == 1:
        return ca.get(key, args[0])
    raise TypeError(f"cache_get() expected at most 3 arguments, got {2 + len(args)}")


# ===========================================================================
# Parsing
# ===========================================================================

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


def saqc_func_repr(func, kws):
    f = f"{func._module}.{func.__name__}"
    p = ', '.join([f"{k}={repr(v)}" for k, v in kws.items()])
    return f"{f}({p})"


def parse_keywords(kwstr):
    try:
        parsed = litereval.litereval('{' + kwstr + '}')
    except (ValueError, TypeError, SyntaxError):
        raise SyntaxError
    if not isinstance(parsed, dict):
        raise SyntaxError
    for key in parsed.keys():
        if not isinstance(key, str):
            raise SyntaxError
    return parsed


def parse_data(filename: str, file_type, content, parse_kws=None):
    if parse_kws is None:
        parse_kws = {}
    if file_type is None:
        file_type = 'csv'
        if filename.endswith('.csv'):
            file_type = 'csv'
        elif filename.endswith('.xls'):
            file_type = 'xls'
    if file_type == 'csv':
        return pd.read_csv(io.StringIO(content.decode('utf-8')), **parse_kws)
    elif file_type == 'xls':
        return pd.read_excel(io.BytesIO(content), **parse_kws)
    else:
        raise ValueError('Unknown filetype')


# ===========================================================================
# html creation helper
# ===========================================================================

def FormGroupInput(text, id, type="text", widths=(2, 10), row=True, raw=False, **kws):
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


def param_parse(value: str):
    """
    Raises
    ======
    ValueError: if parsing fails
    """
    try:
        return literal_eval_extended(value)
    except (TypeError, SyntaxError, IndexError):
        raise ValueError('danger', f"Invalid syntax")
    except ValueError:
        raise ValueError('danger', f"Invalid value")


def param_typecheck(name, value, expected_type, df=None):
    """
    Raises
    ======
    TypeError: if simple type-check fails
    ValueError: if advanced type-check fails
    """
    simple = TYPE_MAPPING.get(expected_type, expected_type)
    try:
        typeguard.check_type(name, value, simple)
    except TypeError as e:
        raise TypeError('warning', f"TypeCheckError: {e}")

    # extras
    if expected_type == saqc.lib.types.ColumnName:
        if df is not None and value not in df.columns:
            raise ValueError(
                'warning',
                f"ValueError: column '{value}' is not present in data."
            )

    elif (
            expected_type == saqc.lib.types.PositiveFloat
            or expected_type == saqc.lib.types.PositiveInt
    ):
        if value < 0:
            raise ValueError('warning', f"ValueError: value is negative")

    elif expected_type == saqc.lib.types.FreqString:
        try:
            pd.Timedelta(value)  # not accurate enough
        except (KeyError, ValueError) as e:
            raise ValueError('warning', f"{repr(e)}")

    return value






