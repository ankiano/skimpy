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
import seaborn as sns

# model results and validation
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

# model benchmark
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


import itertools


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
    Object for Weight of Evidence calculation and transromation.

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
            0.02 <= value < 0.1: 'weak',
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
        t.index = t.index.astype('str')
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
        self.woe_report = t
        woe_dict = t['weight-of-evidence'].iloc[:-1].to_dict()
        self.dict = woe_dict
        iv_info = t[['information-value', 'conclusion']].\
            loc['total'].to_dict()
        self.iv = iv_info
        return t

    def plot(self):
        """
        Creates charts with woe results
        """
        # get data without totals
        data = self.woe_report[self.woe_report.index != 'total']
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
        # second chart
        data['weight-of-evidence'].plot(ax=axs[1], linewidth=3, alpha=0.7)
        axs[1].set_title('WoE')
        axs[1].set_xlabel(data.index.name)
        axs[1].set_ylabel('WoE')
        # third chart
        data['information-value'].plot(ax=axs[2], linewidth=3, alpha=0.7)
        axs[2].set_title('Information value')
        axs[2].set_ylabel('IV')


# Feature exploration

def plot_density(data: pd.DataFrame, target: str, feature: str):
    """
    Density estimation chart for one feature
    """

    plt.figure(figsize=(16, 4))

    sns.kdeplot(
        data[feature][data[target] == 1],
        shade=True, label='{}=1'.format(target), linewidth=3)
    sns.kdeplot(
        data[feature][data[target] == 0],
        shade=True, label='{}=0'.format(target), linewidth=3)

    min_v = data[feature].min()
    max_v = data[feature].max()
    plt.xlim(min_v, max_v)

    plt.title('Distribution of {} by {} value'.format(
        feature.upper(), target.upper()))
    plt.xlabel('{}'.format(feature))
    plt.ylabel('Density')


# Model metrics

def predict_with_threshold(y_pred_proba, threshold):
    """
    Calculates predicted classes (0,1) by continuous probabilities with
    custom threshold
    """

    y_pred = [1 if x >= threshold else 0 for x in y_pred_proba]
    return pd.Series(data=y_pred, name='y_pred')


def accuracy_score(y_true, y_pred_proba, threshold=0.5):
    """
    Calculates accuracy score by true and predicted values with custom
    threshold
    """

    y_pred = predict_with_threshold(y_pred_proba, threshold)

    acc = pd.concat([y_true.rename('y_true'), y_pred], axis=1)
    tp = acc.apply(
        lambda row: 1 if (row.y_true == row.y_pred) else 0,
        axis=1
    ).sum()
    return tp/acc['y_pred'].count()


def plot_precision_recall_curve(y_true, y_pred_proba, threshold=0.5):
    """
    Creates precision and recall chart for selected threshold (cut-off)
    level for the model.
    """

    y_pred = predict_with_threshold(y_pred_proba, threshold)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    precisions, recalls, thresholds = \
        precision_recall_curve(y_true, y_pred_proba)

    plt.plot(
        thresholds, precisions[:-1], "b",
        label="Precision={:.3f}".format(precision), linewidth=3, alpha=0.7)
    plt.plot(
        thresholds, recalls[:-1], "g",
        label="Recall={:.3f}".format(recall), linewidth=3, alpha=0.7)
    plt.plot(  # threshold line
        [threshold, threshold], [0, 1], 'r--',
        label='threshold={}'.format(threshold), linewidth=3, alpha=0.3,)

    plt.title("Precision and Recall curve")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc="lower left")

    f1 = 2 * (precision * recall) / (precision + recall)
    plt.text(0.6, 0.05, "F-1={:.3f}".format(f1),
             bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))


def plot_roc_curve(y_true, y_pred_proba, threshold=0.5):
    """
    Creates Roc curve and AUC value
    """

    y_pred = predict_with_threshold(y_pred_proba, threshold)
    roc_auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    plt.plot(  # roc auc line
        fpr, tpr,
        label='AUC={:.3f}'.format(roc_auc),
        linewidth=3, alpha=0.7)
    plt.plot(  # base line
        [0, 1], [0, 1], 'r--',
        label='baseline=0.5',
        linewidth=3, alpha=0.3)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")


def plot_probability_distribution(
        y_true, y_pred_proba, threshold, class_labels=[0, 1]):
    """
    Creates probability distribution chart by classes
    """

    _y = pd.concat([y_true, y_pred_proba], axis=1)

    sns.kdeplot(
        _y[_y.iloc[:, 0] == 1].iloc[:, 1],
        shade=True, label=class_labels[1], linewidth=3, alpha=0.7)
    sns.kdeplot(
        _y[_y.iloc[:, 0] == 0].iloc[:, 1],
        shade=True, label=class_labels[0], linewidth=3, alpha=0.7)

    plt.plot(  # threshold line
        [threshold, threshold],
        [plt.ylim()[0], plt.ylim()[1]],
        'r--', linewidth=3,
        alpha=0.3, label='threshold={}'.format(threshold))

    plt.xlim(0, 1)
    plt.title("Class probability distribution")
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.legend(loc='upper center')
    score = accuracy_score(y_true, y_pred_proba, threshold)
    plt.text(
        0.05, 0.2,
        "Score={:.3f}".format(score),
        bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))


def plot_confusion_matrix(
        y_true, y_pred_proba, threshold=0.5,
        class_labels=[0, 1], normalize=True):
    """
    Create the confusion matrix.
    Normalization can be turn-off by setting `normalize=False`.
    """

    y_pred = predict_with_threshold(y_pred_proba, threshold)
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels)
    plt.yticks(tick_marks, class_labels, rotation=90)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_model_summary(
        y_true, y_pred_proba, threshold=0.5,
        class_labels=[0, 1], normalize=True):
    """
    Creates:
        1) precision and recall chart
        2) ROC chart
        3) probability distribution by clases chart
        4) confusion matrix
    in one panel
    """

    plt.figure(figsize=(16, 4))
    # first chart
    plt.subplot(1, 4, 1)
    plot_precision_recall_curve(y_true, y_pred_proba, threshold)
    # secound chart
    plt.subplot(1, 4, 2)
    plot_roc_curve(y_true, y_pred_proba, threshold)
    # third chart
    plt.subplot(1, 4, 3)
    plot_probability_distribution(
        y_true, y_pred_proba, threshold, class_labels)
    # fourth chart
    plt.subplot(1, 4, 4)
    plot_confusion_matrix(
        y_true, y_pred_proba, threshold, class_labels, normalize)


# Models benchmark

def model_benchmark(x, y, models):
    skf = StratifiedKFold(random_state=42, n_splits=3, shuffle=True)
    result_list = []
    for m in models:
        try:
            models[m].fit(x, y)
            accuracy_train = round(models[m].score(x, y) * 100, 2)
            accuracy_cross = cross_val_score(
                                models[m], x, y,
                                scoring='accuracy', cv=skf.split(x, y))
            auc_cross = cross_val_score(
                            models[m], x, y,
                            scoring='roc_auc', cv=skf.split(x, y))

            result_list.append({'model': models[m].__class__.__name__,
                                'train-score': accuracy_train,
                                'cross-score': '{:.3f} (+-{:.3f})'.format(
                                    accuracy_cross.mean(),
                                    accuracy_cross.std()),
                                'auc': '{:.3f} (+-{:.3f})'.format(
                                    auc_cross.mean(),
                                    auc_cross.std())}
                               )
        except Exception:
            print('{} model error'.format(models[m].__class__.__name__))

    result_df = pd.DataFrame(result_list)[['model', 'train-score',
                                           'cross-score', 'auc']]
    return result_df.sort_values(by='cross-score', ascending=False)


__version__ = '0.0.7'
