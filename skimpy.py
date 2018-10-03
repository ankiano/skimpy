#!/usr/bin/env python3.6
# coding=utf-8

# data typyzation
from typing import Callable, Any, Dict, List, Optional
from dataclasses import dataclass

# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt


class Preprocessing:
    """
    Object for
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


class WoE:
    """
    Object for Weight of Evidence calculation and transfomation.

    The WoE transformation has (at least) three positive effects:
    1) It can transform an independent variable so that it establishes
    monotonic relationship to the dependent variable.
    The WoE transformation actually orders the categories on a "logistic"
    scale which is natural for logistic regression
    2) For variables with too many (sparsely populated) discrete values,
    these can be grouped into categories (densely populated) and the WoE
    can be used to express information for the whole category
    3) The (univariate) effect of each category on dependent variable can
    be simply compared across categories and across variables because WoE
    is standardized value

    It also has (at least) three drawbacks:
    1) Loss of information (variation) due to binning to few categories
    2) It is a "univariate" measure so it does not take into account
    correlation between independent variables
    3) It is easy to manipulate (overfit) the effect of variables according
    to how categories are created
    """

    def __init__(self):
        self.dict = {}
        self.iv = None

    @staticmethod
    def get_iv_description(value):
        """
        Describe the information value predictive power in a human language
        """

        return {
            value is None: '',
            value < 0.02: 'useless',
            0.02 <= value < 0.1: 'week',
            0.1 <= value < 0.3: 'medium',
            0.3 <= value < 0.5: 'strong',
            0.5 <= value: 'excellent'
        }[True]

    def calculate(self, H, A, woe_sorted=False):
        """
        Calculate WOE (Weight Of Evidence) and Information Value of binary
        classification dataset.

        This analysis is in the verification of the presence and strength
        of the relationship between one dependent and independent variables,
        which allows you to determine which variables are the most
        accurate predictors of the model.

        The Information Value (IV) measures predictive power of the
        characteristic, which is used for feature selection.
        It is also khown as Kullback–Leibler divergence or relative entropy.

        Returns dataframe with information value, woe, etc dictionaries of
        each category of feature.

        :param H:  feature category e.g. feature "sex" has 2 categories -
        "male" and "female"
        :param A: А - target event e.g. churn, buy, fraud. which should be
        binary. Ā - opposite to target event.
        """

        def safe_ln(x):
            try:
                result = np.log(x)
            except ZeroDivisionError:
                result = 0
            return result

        # prepare distribution of target classed by feature categories
        t = pd.crosstab(H, A)
        t.rename(columns={0: 'Ā', 1: 'А'}, inplace=True)
        t.index = t.index.astype('str')  # ?
        # calculate probabilities, odds, woe and iv
        t['P(Hi)'] = (t['А']+t['Ā']) / (t['А'].sum()+t['Ā'].sum())
        t['P(Hi|Ā)'] = t['Ā']/t['Ā'].sum()
        t['P(Hi|A)'] = t['А']/t['А'].sum()
        t['posterior-odds'] = t['P(Hi|A)'].div(t['P(Hi|Ā)'])
        t['weight-of-evidence'] = t['posterior-odds'].\
            apply(lambda x: safe_ln(x)).replace([np.inf, -np.inf], 0).fillna(0)
        t['information-value'] = (t['P(Hi|A)']-t['P(Hi|Ā)']) *\
            t['weight-of-evidence']
        # sort table by woe
        if woe_sorted:
            t.sort_values('weight-of-evidence', inplace=True)
        # calculate totals
        t.loc['total'] = t.sum()
        t['conclusion'] = t['information-value'].fillna(0).\
            apply(lambda x: self.get_iv_description(x))
        # clear some not meanfull totals
        t.loc['total', 'posterior-odds'] = ''
        t.loc['total', 'weight-of-evidence'] = ''
        # prepare results
        woe_report = t
        woe_dict = t['weight-of-evidence'].iloc[:-1].to_dict()
        self.dict = woe_dict
        iv_info = t[['information-value', 'conclusion']].\
            loc['total'].to_dict()
        self.iv = iv_info
        return woe_report

    @staticmethod
    def plot(woe_report):
        """
        Creates charts with woe results
        """
        # get data without totals
        data = woe_report[woe_report.index != 'total']
        # setup panel
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        plt.subplots_adjust(wspace=0.3)
        # first chart
        data['P(Hi|A)'].plot(ax=axs[0], linewidth=3, alpha=0.7)
        data['P(Hi|Ā)'].plot(ax=axs[0], linewidth=3, alpha=0.7)
        axs[0].set_title('Probability distribution')
        axs[0].set_xlabel(data.index.name)
        axs[0].set_ylabel('probability')
        axs[0].legend(['P(Hi|A)', 'P(Hi|Ā)'])
        # secound chart
        data['weight-of-evidence'].plot(ax=axs[1], linewidth=3, alpha=0.7)
        axs[1].set_title('WoE')
        axs[1].set_xlabel(data.index.name)
        axs[1].set_ylabel('WoE')
        # third chart
        data['information-value'].plot(ax=axs[2], linewidth=3, alpha=0.7)
        axs[2].set_title('Information value')
        axs[2].set_ylabel('IV')


__version__ = '0.0.3'
