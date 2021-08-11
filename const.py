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


AGG_THRESHOLD = 100
MAX_DF_ROWS = 10000
MEGA = 10**6
MAX_FILE_SIZE = 10 * MEGA  # in byte
PARSER_KW_DEFAULTS = "header=0, index_col=0, parse_dates=True, nrows=10000"

IGNORED_PARAMS = ['data', 'flags', 'kwargs']
AGG_METHODS = ['mean', 'min', 'max', 'sum']  # first is default

RANDOM_TYPES = [
    {"label": "with outlier", "value": 'outliers'},
    {"label": "with plateaus", "value": 'plateaus'},
    {"label": "with gaps/NaNs", "value": 'gaps'},
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
    'flagRange': saqc.funcs.flagRange,
    'flagMAD': saqc.funcs.flagMAD,
    'flagDummy': saqc.funcs.flagDummy,
    'flagCrossStatistic': saqc.funcs.flagCrossStatistic,
    'test': test,
}
