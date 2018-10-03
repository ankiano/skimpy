#!/usr/bin/env python3.6
# coding=utf-8

# data typyzation
from typing import Callable, Any, Dict, List, Optional
from dataclasses import dataclass

# data analysis and wrangling
import pandas as pd


class Preprocessing:
    """
    Class for
    1) accumulating data transformation steps during the variables analysis
    2) applying transformations for a training dataset before feeding
    into the model
    3) applying transformations for a testing dataset before predicting
    based on existing model

    """

    @dataclass
    class Step:
        input_var: str
        output_var: str = None
        transformations: Optional[List[Callable[[Any], Any]]] = None
        info: str = None

    def __init__(self, items: Dict[str, Step]=None):
        self._items = items or {}

    def __repr__(self):
        return 'Preprocessing({!r})'.format(self._items)

    def add(self, step: Step):
        self._items[step.output_var] = step

    def transform(self, X: pd.DataFrame):
        __X = pd.DataFrame()  # dataframe for all transformed output variables
        for s in self._items.values():
            # if single source variable
            if (set(s.input_var).issubset(X.columns)
                    # or list of variables
                    or s.input_var in X.columns):
                _X = X[s.input_var]
                #  apply transformation steps
                if s.transformations:
                    for t in s.transformations:
                        _X = t(_X)
            else:
                print('variable: {} not found in dataset'.format(s.input_var))
            # add one more output variable to whole result dataframe
            __X[s.output_var or s.input_var] = _X
        return __X

    def info(self, keys: List, sort_by: str = None) -> pd.DataFrame:
        result = pd.DataFrame()
        for s in self._items.values():
            row = {}
            row['feature'] = s.output_var or s.input_var
            if s.info is not None:
                for k in keys:
                    row[k] = s.info.get(k)
            result = result.append(row, ignore_index=True)
        if sort_by:
            result.sort_values(sort_by, ascending=False, inplace=True)
        return result


__version__ = '0.0.2'
