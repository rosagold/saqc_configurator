#!/usr/bin/env python

import typing
import inspect

import pandas as pd

import saqc.funcs
import saqc
import numpy as np
from saqc.lib.types import (
    PositiveInt, PositiveFloat, FreqString, ColumnName, TimestampColumnName,
    IntegerWindow,
)

IGNORED_PARAMS = ['data', 'flags', 'field', 'kwargs']
AGG_METHODS = ['mean', 'min', 'max', 'sum']  # first is default

RANDOM_TYPES = [
    {"label": "with outlier", "value": 1},
    {"label": "with plateaus", "value": 2},
    {"label": "with gaps/NaNs", "value": 3},
]

TYPE_MAPPING = {
    inspect.Parameter.empty: typing.Any,
    ColumnName: str,
    TimestampColumnName: str,
    FreqString: str,
    IntegerWindow: int,
    PositiveInt: int,
    PositiveFloat: float,
}

PARSER_MAPPING = {
    'csv': pd.read_csv,
    'xls': pd.read_excel,
}


def test(
        data, field, flags,
        extralong_neverending_yes_its_long_kw=None,
        kw1: int = 9,
        kw2: bool = False,
        kw3: int = None,
        kw31=None,
        kw4=np.nan,
        kw6=-np.inf,
        kw7: PositiveInt = 9,
        freq: FreqString = '9d',
        union: typing.Union[int, float] = 0,
        tup: (int, float) = 0,
        li: typing.Literal['a', 'b', 'c'] = 'a',
        lilong: typing.Literal['a',
                               'some foo and stuff that makes the line long',
                               'some foo and stuff that makes the line long',
                               'some foo and stuff that makes the line long',
                               'oh no another very long literal','b', 'c'] = 'a',
        op1: int = None,
        op2: typing.Optional[int] = None,
        op3: typing.Optional[int] = 7,
        **kwargs
):
    pass


SAQC_FUNCS = {
    # "None": None,  # default
    'flagRange': saqc.funcs.flagRange,
    'flagMAD': saqc.funcs.flagMAD,
    'flagDummy': saqc.funcs.flagDummy,
    'flagCrossStatistic': saqc.funcs.flagCrossStatistic,
    'test': test,
}
