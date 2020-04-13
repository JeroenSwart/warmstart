from statsmodels.tsa.stattools import adfuller, acf, kpss
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from scipy.stats import spearmanr


def size(time_series):
    """Metafeature function returning the number of samples in the dataset.

    Args:
        time_series (pd.DataFrame): the dataset for which to optimize the superparameters.

    Returns:
        float

    """
    return len(time_series)


def maxminvar(time_series):
    """Metafeature function returning the maximum/minimum ratio of the variance, with a rolling window of 240 samples.

    Args:
        time_series (pd.DataFrame): the dataset for which to optimize the superparameters.

    Returns:
        float

    """
    rolling_var = time_series["endogenous"].rolling(24 * 10).var()
    return rolling_var.max() / rolling_var.min()


def adf(time_series):
    """Metafeature function returning the p-value of an Augmented Dickey-Fuller test.

    Args:
        time_series (pd.DataFrame): the dataset for which to optimize the superparameters.

    Returns:
        float

    """
    return adfuller(time_series["endogenous"])[1]


def stat_test(time_series):
    """Metafeature function returning the test statistic of a Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.

    Args:
        time_series (pd.DataFrame): the dataset for which to optimize the superparameters.

    Returns:
        float

    """
    return kpss(time_series["endogenous"], lags="auto")[0]


def cumac(time_series):
    """Metafeature function returning the cumulative abolute autocorrelation of lags 1 to 48.

    Args:
        time_series (pd.DataFrame): the dataset for which to optimize the superparameters.

    Returns:
        float

    """
    return sum(abs(acf(time_series["endogenous"], fft=True, nlags=48)))


def total_splits(time_series):
    """Metafeature function returning the number of splits after training an XGBoost model with 1000 trees and a maximum
    depth of 1000.

    Args:
        time_series (pd.DataFrame): the dataset for which to optimize the superparameters.

    Returns:
        float

    """
    xg_reg = xgb.XGBRegressor(
        objective="reg:squarederror", n_trees=1000, max_depth=1000
    )
    xg_reg.fit(
        time_series.drop(columns=["endogenous"]),
        pd.DataFrame(data=time_series["endogenous"]),
    )
    return sum(xg_reg.get_booster().get_score(importance_type="weight").values())


def feature_xgb_interactions(time_series):
    """Metafeature function representing feature importance plus XGBoost model interactions. The mean absolute error of a
    zero forecast is divided by the mean absolute error of a trained xgboost model with one tree and maximum depth of
    one.

    Args:
        time_series (pd.DataFrame): the dataset for which to optimize the superparameters.

    Returns:
        float

    """
    # split time_series in train and test
    train, test = train_test_split(time_series)
    ex_train = train.drop(columns=["endogenous"])
    ex_test = test.drop(columns=["endogenous"])
    end_train = pd.DataFrame(data=train["endogenous"])
    end_test = pd.DataFrame(data=test["endogenous"])

    # XGBoost regression, train and predict
    xg_reg = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=1, max_depth=1000, learning_rate=1
    )
    xg_reg.fit(ex_train, end_train)
    xgb_preds = pd.Series(data=xg_reg.predict(ex_test), index=end_test.index)

    # mae of xgb and initial
    mae_xgb = mean_absolute_error(end_test, xgb_preds)
    mae_initial = mean_absolute_error(end_test, np.zeros(len(end_test)))

    return mae_xgb / mae_initial


def xgb_feature_interactions_minus_linear_relations(time_series):
    """Metafeature function returning the importance of feature interactions in XGBoost minus linear correlations.

    Args:
        time_series (pd.DataFrame): the dataset for which to optimize the superparameters.

    Returns:
        float

    """
    # split time_series in train and test
    length = len(time_series)
    ex_train = time_series.drop(columns=["endogenous"])[: int(length * 2 / 3)]
    ex_test = time_series.drop(columns=["endogenous"])[int(length * 2 / 3) :]
    end_train = pd.DataFrame(data=time_series["endogenous"])[: int(length * 2 / 3)]
    end_test = pd.DataFrame(data=time_series["endogenous"])[int(length * 2 / 3) :]

    # XGBoost regression, train and predict
    xg_reg = xgb.XGBRegressor(
        objective="reg:squarederror", n_trees=1000, max_depth=1000
    )
    xg_reg.fit(ex_train, ex_train)
    xgb_preds = pd.Series(data=xg_reg.predict(ex_test), index=end_test.index)

    # Linear regression, train and predict
    lreg = LinearRegression().fit(ex_train, end_train)
    lreg_preds = pd.Series(data=lreg.predict(ex_test), index=end_test.index)

    # calculate mape ratio of xgb and linear regression
    ratio = mean_absolute_error(xgb_preds, end_test) / mean_absolute_error(
        lreg_preds, end_test
    )

    return ratio


def pca_rank_cor(time_series):
    """Metafeature function returning the feature importance. A principal component analysis decorrelates the features,
    after which the univariate feature importance is calculated by Spearman's rank correlation. The function returns
    a summation of these feature importances.

    Args:
        time_series (pd.DataFrame): the dataset for which to optimize the superparameters.

    Returns:
        float

    """
    # take exogenous frame
    exogenous = time_series.drop(columns=["endogenous"])
    n_feats = len(exogenous.columns)

    # standardize exogenous frame
    standard_x = StandardScaler().fit_transform(exogenous)

    # perform pca
    pcs = PCA().fit_transform(standard_x)

    # calculate rank correlation between target and PC's
    return (
        sum(
            [
                abs(spearmanr(pcs[:, i], time_series["endogenous"]).correlation)
                for i in range(n_feats)
            ]
        )
        / n_feats
    )


def pca_fisher_score(time_series):
    """Metafeature function returning the feature importance. A principal component analysis decorrelates the features,
    after which the univariate feature importance is calculated by the Fisher score. The function returns a summation of
    these feature importances.

    Args:
        time_series (pd.DataFrame): the dataset for which to optimize the superparameters.

    Returns:
        float

    """
    # take exogenous frame
    exogenous = time_series.drop(columns=["endogenous"])
    n_feats = len(exogenous.columns)

    # standardize exogenous frame
    standard_x = StandardScaler().fit_transform(exogenous)

    # perform pca
    pcs = PCA().fit_transform(standard_x)

    return sum(abs(f_regression(pcs, time_series["endogenous"])[0])) / n_feats


def intrinsic_dimensionality(time_series):
    """Metafeature function returning the intrinsic dimensionality. This is calculated by performing PCA and multiplying
    the components by their feature importance, resulting in a vector of the contribution per feature. The skewness of
    this vector is then calculatd.

    Note:
        This is an experimental metafeature, with no evidence of usefulness.

    Args:
        time_series (pd.DataFrame): the dataset for which to optimize the superparameters.

    Returns:
        float

    """
    # take exogenous frame
    exogenous = time_series.drop(columns=["endogenous"])
    n_feats = len(exogenous.columns)

    # standardize exogenous frame
    standard_x = StandardScaler().fit_transform(exogenous)

    # perform pca
    pca = PCA().fit(standard_x)
    transf_pca = pd.DataFrame(pca.fit_transform(standard_x))

    # get feature importance (rank correlation)
    ft_imp_rank = [
        abs(spearmanr(transf_pca[i], time_series["endogenous"]).correlation)
        for i in range(n_feats)
    ]

    # multiply explained variance per PC with its feature importance
    weighted_expl_var = pca.explained_variance_ * ft_imp_rank

    # scale percentage of total & sort
    scaled_weighted_expl_var = weighted_expl_var / sum(weighted_expl_var)
    sorted = np.sort(scaled_weighted_expl_var)

    # summate disunity, to the square of the nr of features away
    disunity = 0
    for i in range(len(sorted) - 1):
        disunity += sorted[i + 1] ** i

    return disunity


def unit(time_series):
    """Dummy metafeature function, returning one.

    Args:
        time_series (pd.DataFrame): the dataset for which to optimize the superparameters.

    Returns:
        float

    """
    return 1
