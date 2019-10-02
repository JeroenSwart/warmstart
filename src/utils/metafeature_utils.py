def size(time_series):
    return len(time_series)


def endogenous_mean(time_series):
    return time_series['endogenous'].mean()
