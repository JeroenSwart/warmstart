from statsmodels.tsa.stattools import adfuller, acf


def size(time_series):
    return len(time_series)


def endogenous_mean(time_series):
    return time_series['endogenous'].mean()


def maxminvar(time_series):
    rolling_var = time_series['endogenous'].rolling(24*10).var()
    maxminvar = rolling_var.max()/rolling_var.min()
    return maxminvar


def adf(time_series):
    adf = adfuller(time_series['endogenous'])[0]
    return adf


def cumac(time_series):
    cumac = sum(abs(acf(time_series['endogenous'], fft=True, nlags=48)))
    return cumac
