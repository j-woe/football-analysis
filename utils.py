from numpy import percentile
from pandas import DataFrame, Series


def get_other_value(values: list, current: any) -> any:
    return (set(values) - {current}).pop()


def n_bins_labels(
    series: Series,
    bin_count: int,
    equal_bin_sizes: bool
) -> tuple:
    if equal_bin_sizes:
        bins = [
            percentile(series, i / bin_count * 100)
            for i in range(bin_count + 1)
        ]
    else:
        bins = [
            series.min() + (series.max() - series.min()) * i / bin_count
            for i in range(bin_count + 1)
        ]
    labels = [
        f'{bins[i - 1]:.2f} - {bins[i]:.2f}' for i in range(1, len(bins))
    ]
    return bins, labels


def filter_df(df: DataFrame, **kwargs) -> DataFrame:
    filter = True
    for col, val in kwargs.items():
        if val is not None:
            if isinstance(val, (list, tuple)):
                filter &= (df[col].isin(val))
            else:
                filter &= (df[col] == val)
    return df[filter].copy()
