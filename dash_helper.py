#!/usr/bin/env python
import typing

import dash_bootstrap_components as dbc
import saqc.lib.types as saqc_types
import inspect

TYPE_MAPPING = {
    saqc_types.ColumnName: str,
    saqc_types.TimestampColumnName: str,
    saqc_types.FreqString: str,
    saqc_types.IntegerWindow: int,
    saqc_types.PositiveInt: int,
    saqc_types.PositiveFloat: float,
}


def parse_param(value, target_type):
    """
    handle type hints:
        - str, int, float, bool, None
        - saqc.types (see TYPE_MAPPING above)
        - Union[.], Optional[.], Literal[.]
    todo:
        - list
        - Tuple[.], List[.]
        - Callable[.]
    """

    # Special
    if value == 'None':
        return None

    # forced string
    if (value.startswith("\"") and value.endswith("\"")
            or value.startswith("\"") and value.endswith("\"")):
        return value[1:-1]

    # evaluate target type
    if typing.get_origin(target_type) is typing.Union:
        target_type = typing.get_args(target_type)

    # recurse for Union and tuple typehints
    if isinstance(target_type, tuple):
        for t in target_type:
            if t == type(None):  # noqa
                continue
            try:
                return parse_param(value, t)
            except (ValueError, TypeError):
                continue
        raise ValueError(
            f"Could not cast '{value}' any type of '{type_repr(target_type)}'."
        )

    target_type = TYPE_MAPPING.get(target_type, target_type)

    if typing.get_origin(target_type) is typing.Literal:
        target_type = str

    # Parsing
    if target_type is bool and value == 'False':
        return False
    if target_type is not inspect._empty:  # we have a casting type
        try:
            return target_type(value)
        except (ValueError, TypeError):
            raise ValueError(
                f"Could not cast '{value}' to needed type '{type_repr(target_type)}'."
            )

    # Guessing by user submitted value
    if value in ['True', 'False']:
        return eval(value)
    try:
        return int(value)
    except (ValueError, TypeError):
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
    return value  # fall back to plain string


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


def type_repr(t, pretty=True):
    # HINT: a.o. Union[.] has no __name__ attribute
    s = getattr(t, '__name__', repr(t))
    if pretty:
        if isinstance(t, tuple) and len(t):
            s = '('
            for sub in t:
                s += type_repr(sub, pretty=True) + ','
            if len(t) > 1:
                s = s[:-1]
            return s + ')'

        s = (s.replace('typing.', '')
             .replace('NoneType', 'None')
             .replace('Union', '')
             )
        # rm `[` and `]`
        if typing.get_origin(t) is typing.Union:
            s = s[1:-1]
    return s


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
