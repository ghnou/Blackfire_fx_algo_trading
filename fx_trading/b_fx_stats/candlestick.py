import os
import talib
import numpy as np
import pandas as pd
import fx_trading.utils.constants as cst
from fx_trading.b_fx_stats.format_stats import BuildStrategyReport
from fx_trading.utils.data_interact import read_data, save_data
from fx_trading.utils.parallel_tasks import AsyncMP


CANDLE_NAMES = talib.get_function_groups()['Pattern Recognition']
CANDLE_NAMES = ['CDLENGULFING']


def add_candlestick_pattern(df):
    """
    Description:
    ------------

    Add Candle stick pattern to the price.
    """
    op = df[cst.OPEN]
    hi = df[cst.HIGH]
    lo = df[cst.LOW]
    cl = df[cst.CLOSE]

    for candle in CANDLE_NAMES:
        df[candle] = getattr(talib, candle)(op, hi, lo, cl)

    return df


def fast_prepare_data(df, fx_pair, forecast_period):

    df.reset_index(inplace=True)
    df[CANDLE_NAMES] = df[CANDLE_NAMES].shift(1)

    _df = df[[cst.LOW, cst.HIGH, cst.OPEN]].copy()
    header = ['FORECAST_PERIOD_LOW', 'FORECAST_PERIOD_HIGH', 'BEAR_STOP', 'BULL_STOP', 'BEAR_TP', 'BULL_TP',
              'FORECAST_PERIOD_ENTRY_BEAR', 'FORECAST_PERIOD_ENTRY_BULL']
    _df.loc[:, header] = np.NaN
    x = _df.values

    for pos in range(0, len(x) - forecast_period):
        subset = x[pos: pos + forecast_period]

        forecast_period_low = np.nanargmin(subset[:, 0])
        forecast_period_high = np.nanargmax(subset[:, 1])

        bear_tp = (subset[0, 2] - subset[forecast_period_low, 0]) * cst.FX_PIP[fx_pair]
        bull_tp = (subset[forecast_period_high, 1] - subset[0, 2]) * cst.FX_PIP[fx_pair]

        bear_stop = (np.nanmax(subset[:forecast_period_low + 1, 1]) - subset[0, 2]) * cst.FX_PIP[fx_pair]
        bull_stop = (subset[0, 2] - np.nanmin(subset[:forecast_period_high + 1, 0])) * cst.FX_PIP[fx_pair]

        bear_entry_period = np.nanargmax(subset[:forecast_period_low + 1, 1])
        bull_entry_period = np.nanargmin(subset[:forecast_period_high + 1, 0])

        x[pos, -8:] = [forecast_period_low + 1, forecast_period_high + 1, bear_stop, bull_stop, bear_tp, bull_tp,
                       bear_entry_period + 1, bull_entry_period + 1]

    df.loc[:, header] = x[:, -8:]
    df.set_index([cst.FX_PAIR, cst.DATE], inplace=True)

    return df.dropna(subset=['BULL_TP'])


def prepare_data_for_stats(df, fx_pair, forecast_period):
    """
    Description:
    ------------
    Prepare the data set to compute the stats on accuracy.
    """
    df.reset_index(inplace=True)
    df[CANDLE_NAMES] = df[CANDLE_NAMES].shift(1)

    df['FORECAST_PERIOD_LOW'] = df[cst.LOW].rolling(forecast_period).apply(
        lambda x: x.idxmin()).shift(-forecast_period + 1)
    df['FORECAST_PERIOD_HIGH'] = df[cst.HIGH].rolling(forecast_period).apply(
        lambda x: x.idxmax()).shift(-forecast_period + 1)

    df['BEAR_STOP'] = np.NaN
    df['BULL_STOP'] = np.NaN

    h = [cst.LOW, cst.HIGH, 'FORECAST_PERIOD_HIGH', 'FORECAST_PERIOD_LOW', 'BEAR_STOP', 'BULL_STOP']

    for i, row in df[h].iterrows():

        if ~np.isnan(row['FORECAST_PERIOD_HIGH']):
            df.loc[i, 'BULL_STOP'] = df[h].iloc[i:int(row['FORECAST_PERIOD_HIGH']) + 1, 0].min()
        if ~np.isnan(row['FORECAST_PERIOD_LOW']):
            df.loc[i, 'BEAR_STOP'] = df[h].iloc[i:int(row['FORECAST_PERIOD_HIGH']) + 1, 1].max()

    df['BEAR_TP'] = df[cst.LOW].rolling(forecast_period).min().shift(-forecast_period + 1)
    df['BEAR_TP'] = (df[cst.OPEN] - df['BEAR_TP']) * cst.FX_PIP[fx_pair]
    df['BEAR_STOP'] = (df['BEAR_STOP'] - df[cst.OPEN]) * cst.FX_PIP[fx_pair]

    df['BULL_TP'] = df[cst.HIGH].rolling(forecast_period).max().shift(-forecast_period + 1)
    df['BULL_TP'] = (df['BULL_TP'] - df[cst.OPEN]) * cst.FX_PIP[fx_pair]
    df['BULL_STOP'] = (df[cst.OPEN] - df['BULL_STOP']) * cst.FX_PIP[fx_pair]

    df.set_index([cst.FX_PAIR, cst.DATE], inplace=True)

    return df.dropna(subset=['BULL_TP'])


def build_max_gain_stats(df, pip_target):
    """
    Description:
    ------------
    Compute the max potential gain as well as the RR ratio for an entry at the
    closing price.
    """

    df = df.copy()
    stats = {}
    df['YEAR'] = df[cst.DATE].dt.strftime('%Y').astype(str)
    group_by = [cst.FX_PAIR, 'Candle_type', 'YEAR']
    group_by_total = [cst.FX_PAIR, 'Candle_type']
    index = [cst.FX_PAIR, 'Candle_type', 'YEAR']
    percentiles = [0.5]

    dist = df.groupby(group_by)['TP'].describe(
        percentiles=percentiles
    ).reset_index()

    tot = df.groupby(group_by_total)['TP'].describe(
        percentiles=percentiles
    ).reset_index()
    tot['YEAR'] = 'Total'

    dist = pd.concat([dist, tot]).set_index(index)
    stats['Distribution'] = dist[['count', 'mean', '50%', 'std']].rename(
        columns={'count': 'N. Obs', 'mean': 'Avg.', '50%': 'Med', 'std': 'Std'})
    stats = pd.concat(stats, axis=1)

    accuracy = {}
    for i in pip_target:
        _df = df[df['TP'] >= i]

        tot = _df.groupby(group_by_total)[['TP', 'STOP']].agg({'TP': 'count', 'STOP': 'mean'})
        tot.rename(columns={'TP': 'Accuracy Target', 'STOP': 'Stop Loss'},
                   inplace=True)
        tot.reset_index(inplace=True)
        tot['YEAR'] = 'Total'

        _df = _df.groupby(group_by)[['TP', 'STOP']].agg(
            {'TP': 'count', 'STOP': 'mean'}).reset_index()
        _df.rename(columns={'TP': 'Accuracy Target', 'STOP': 'Stop Loss'},
                   inplace=True)
        accuracy[str(i)] = pd.concat([_df, tot]).set_index(index)

    tp = {}
    for i in pip_target:
        rr = []
        for j in [1, 2, 3]:
            _df = df[(df['TP'] >= i) & (df['STOP'] < i * j)]
            tot = _df.groupby(group_by_total)['TP'].count().to_frame('RR > {}'.format(round(1/j, 1))).reset_index()
            tot['YEAR'] = 'Total'

            _df = _df.groupby(group_by)[['TP']].count().reset_index()
            _df.rename(columns={'TP': 'RR > {}'.format(round(1/j, 1))}, inplace=True)

            rr.append(pd.concat([_df, tot]).set_index(index))
        tp[str(i)] = pd.concat(rr, axis=1)
    accuracy = pd.concat(accuracy, axis=1)
    tp = pd.concat(tp, axis=1)

    stats = pd.concat([stats, accuracy.swaplevel(axis=1), tp.swaplevel(axis=1)], axis=1)
    stats = stats[stats[('Distribution', 'N. Obs')] != 0]
    stats['Accuracy Target'] = (stats['Accuracy Target'].div(stats[('Distribution', 'N. Obs')], axis=0) * 100).fillna(0)
    stats['RR > 1.0'] = (stats['RR > 1.0'].div(stats[('Distribution', 'N. Obs')], axis=0) * 100).fillna(0).astype(int)
    stats['RR > 0.5'] = (stats['RR > 0.5'].div(stats[('Distribution', 'N. Obs')], axis=0) * 100).fillna(0).astype(int)
    stats['RR > 0.3'] = (stats['RR > 0.3'].div(stats[('Distribution', 'N. Obs')], axis=0) * 100).fillna(0).astype(int)

    stats['Accuracy Target'] = stats['Accuracy Target'].astype(int)
    stats['Distribution'] = stats['Distribution'].fillna(0).astype(int)
    stats['Stop Loss'] = stats['Stop Loss'].fillna(0).astype(int)
    h = ['Distribution', 'Accuracy Target', 'Stop Loss', 'RR > 1.0', 'RR > 0.5', 'RR > 0.3']

    stats = stats.loc[:, (h, slice(None))].round(0)

    return stats


def run_max_gain_stats(df, pip_target):
    """
    Description:
    ------------
    Compute the accuracy of the signals at the entry price based on the candle pattern.

    """

    header = ['TP', 'STOP', 'PERIOD_TP']
    stats = {}

    for candle in CANDLE_NAMES:

        df['Candle_type'] = np.NaN
        df.loc[df[candle].fillna(-1) > 0, 'Candle_type'] = cst.BULLISH
        df.loc[df[candle].fillna(-1) > 0, 'TP'] = df['BULL_TP']
        df.loc[df[candle].fillna(-1) > 0, 'STOP'] = df['BULL_STOP']
        df.loc[df[candle].fillna(-1) > 0, 'PERIOD_TP'] = df['FORECAST_PERIOD_HIGH']

        df.loc[df[candle].fillna(1) < 0, 'Candle_type'] = cst.BEARISH
        df.loc[df[candle].fillna(1) < 0, 'TP'] = df['BEAR_TP']
        df.loc[df[candle].fillna(1) < 0, 'STOP'] = df['BEAR_STOP']
        df.loc[df[candle].fillna(1) < 0, 'PERIOD_TP'] = df['FORECAST_PERIOD_LOW']

        _df = df.copy()
        _df.dropna(subset=['Candle_type'], inplace=True)
        _df.set_index('Candle_type', inplace=True, append=True)

        if len(_df) != 0:
            st = build_max_gain_stats(_df[header].reset_index(), pip_target)
            stats[('Candlestick_MaxG', candle)] = st

    stats = pd.concat(stats)
    stats.index.names = ['Stats', 'Candle_name', cst.FX_PAIR, 'Candle_type', cst.DATE]

    return stats.reset_index()


def get_optimal_entry_stats(df, pip_target):

    stats = {}
    df['YEAR'] = df[cst.DATE].dt.strftime('%Y').astype(str)
    group_by = [cst.FX_PAIR, 'Candle_type', 'YEAR']
    group_by_total = [cst.FX_PAIR, 'Candle_type']
    index = [cst.FX_PAIR, 'Candle_type', 'YEAR']

    for e in [10]:

        df['TARGET'] = df['TP'] + e
        df['NEW_STOP'] = df['STOP'] - e
        df.loc[df['NEW_STOP'] < 0, 'TARGET'] = np.NaN
        df.loc[df['NEW_STOP'] < 0, 'NEW_STOP'] = np.NaN

        accuracy = {}
        count = df.groupby(group_by)[['TARGET']].count().reset_index()
        tot = df.groupby(group_by_total)[['TARGET']].count().reset_index()
        tot['YEAR'] = 'Total'

        accuracy['N. Obs'] = pd.concat([count, tot]).set_index(index)['TARGET']

        count = df.groupby(group_by)[['PERIOD_TP']].mean().reset_index()
        tot = df.groupby(group_by_total)[['PERIOD_TP']].mean().reset_index()
        tot['YEAR'] = 'Total'
        accuracy['Avg. Per'] = pd.concat([count, tot]).set_index(index)['PERIOD_TP']

        for i in pip_target:
            _df = df[(df['TARGET'] >= i) & (df['NEW_STOP'] < 20)]
            tot = _df.groupby(group_by_total)[['TARGET', 'NEW_STOP']].agg(
                {'TARGET': 'count', 'NEW_STOP': 'mean'}).reset_index()
            tot['YEAR'] = 'Total'
            tot.rename(columns={'TARGET': 'Accuracy Target', 'NEW_STOP': 'Stop Loss'},
                       inplace=True)

            _df = _df.groupby(group_by)[['TARGET', 'NEW_STOP']].agg(
                {'TARGET': 'count', 'NEW_STOP': 'mean'}).reset_index()

            _df.rename(columns={'TARGET': 'Accuracy Target', 'NEW_STOP': 'Stop Loss'},
                       inplace=True)
            accuracy[str(i)] = pd.concat([_df, tot]).set_index(index)['Accuracy Target']

        accuracy = pd.concat(accuracy, axis=1)

        accuracy = accuracy[accuracy['N. Obs'] != 0]
        accuracy[list(map(str, pip_target))] = (100 * accuracy[list(map(str, pip_target))].div(
            accuracy['N. Obs'], axis=0)).round(1)
        stats[f'Open - {e}'] = accuracy

    return pd.concat(stats, axis=1).round(0)


def run_potential_entry(df, pip_target):

    header = ['TP', 'STOP', 'PERIOD_TP', 'PERIOD_ENTRY']
    stats = {}
    for candle in CANDLE_NAMES:

        df['Candle_type'] = np.NaN
        df.loc[df[candle].fillna(-1) > 0, 'Candle_type'] = cst.BULLISH
        df.loc[df[candle].fillna(-1) > 0, 'TP'] = df['BULL_TP']
        df.loc[df[candle].fillna(-1) > 0, 'STOP'] = df['BULL_STOP']
        df.loc[df[candle].fillna(-1) > 0, 'PERIOD_TP'] = df['FORECAST_PERIOD_HIGH']
        df.loc[df[candle].fillna(-1) > 0, 'PERIOD_ENTRY'] = df['FORECAST_PERIOD_ENTRY_BULL']

        df.loc[df[candle].fillna(1) < 0, 'Candle_type'] = cst.BEARISH
        df.loc[df[candle].fillna(1) < 0, 'TP'] = df['BEAR_TP']
        df.loc[df[candle].fillna(1) < 0, 'STOP'] = df['BEAR_STOP']
        df.loc[df[candle].fillna(1) < 0, 'PERIOD_TP'] = df['FORECAST_PERIOD_LOW']
        df.loc[df[candle].fillna(1) < 0, 'PERIOD_ENTRY'] = df['FORECAST_PERIOD_ENTRY_BEAR']

        _df = df.copy()
        _df.dropna(subset=['Candle_type'], inplace=True)
        _df.set_index('Candle_type', inplace=True, append=True)

        if len(_df) != 0:
            st = get_optimal_entry_stats(_df[header].reset_index(), pip_target)
            stats[('Candlestick_Entry', candle)] = st

    stats = pd.concat(stats)
    stats.index.names = ['Stats', 'Candle_name', cst.FX_PAIR, 'Candle_type', cst.DATE]

    return stats.reset_index()


def bulk_create_sav_dir(fx_pairs):

    for i in fx_pairs:
        path = cst.FX_STATS_CANDLESTICK.format(cst.ASSET[i], i)
        if not os.path.exists(path):
            os.makedirs(path)


def build_candlestick_pattern_analysis(fx_pair, date_range, path=cst.FX_15MIN_DATA_PATH,
                                       forecasting_period=16):
    """
    Description:
    ------------
    Compute some simple stats on the predictive power of the candlestick

    """
    fx_quote = read_data(path, fx_pair, date_range)
    fx_quote[cst.FX_PAIR] = fx_quote[cst.FX_PAIR].astype(str)
    fx_quote = fx_quote.set_index(cst.FX_PAIR, append=True).swaplevel(axis=0)
    fx_quote.sort_index(inplace=True)
    fx_quote = fx_quote[[cst.OPEN, cst.HIGH, cst.LOW, cst.CLOSE]]
    return fx_quote
    #############################################################################
    # Add candlestick pattern
    #############################################################################
    params = [(df,) for (name, df) in fx_quote.groupby(level=0)]
    fx_quote = pd.concat(AsyncMP().exec(params, add_candlestick_pattern))
    del params

    #############################################################################
    # Prepare data set for the stats
    #############################################################################
    params = [(df, name, forecasting_period) for (name, df) in fx_quote.groupby(level=0)]
    fx_quote = pd.concat(AsyncMP().exec(params, fast_prepare_data))
    del params
    return fx_quote
    #############################################################################
    # Stats for each FX pair
    #############################################################################

    bulk_create_sav_dir(fx_pair)
    per_fx_stats = run_potential_entry(fx_quote, pip_target=[10])

    per_fx_stats.groupby(cst.FX_PAIR).apply(
        lambda x: save_data(
            x,
            path=cst.FX_STATS_CANDLESTICK.format(cst.ASSET[x.name], x.name) + f'potential_entry_plus.parquet',
            partition_cols=None
        )
    )

    per_fx_stats = run_max_gain_stats(fx_quote, pip_target=[10])

    per_fx_stats.groupby(cst.FX_PAIR).apply(
        lambda x: save_data(
            x,
            path=cst.FX_STATS_CANDLESTICK.format(cst.ASSET[x.name], x.name) + f'max_g_plus.parquet',
            partition_cols=None
        )
    )

    return per_fx_stats


def report(fx_pair):
    BuildStrategyReport(fx_pair).run()


if __name__ == '__main__':

    DATE_RANGE = list(range(2022, 2023))
    FX_PAIR = list(cst.FX_PIP.keys())
    FX_PAIR = ['eurgbp']
    for i in range(0, len(FX_PAIR), 7):
        pairs = FX_PAIR[i: i + 7]
        print(pairs)
        df = build_candlestick_pattern_analysis(pairs, DATE_RANGE, path=cst.FX_15MIN_DATA_PATH)

    # bullish, bearish = {}, {}
    #
    # for pair in FX_PAIR:
    #     if cst.ASSET[pair]==cst.FX:
    #         path = cst.FX_STATS_CANDLESTICK.format(cst.ASSET[pair], pair) + f'potential_entry_plus.parquet'
    #         df = pd.read_parquet(path).drop(columns=[('Stats', ), ('FX_PAIR',)])\
    #             .set_index([('Candle_name',), ('Candle_type', ), ('DATE', )])
    #         bullish[pair] = df[('Open - 10', 'N. Obs')].xs('2022', level=-1).xs(cst.BULLISH, level=-1)
    #         bearish[pair] = df[('Open - 10', 'N. Obs')].xs('2022', level=-1).xs(cst.BEARISH, level=-1)
    #
    # path = 'D:/Blackfire_Projects/Blackfire_fx_algo_trading/fx_stats/FOREX/'
    # bullish = pd.concat(bullish, axis=1).sort_index().sort_index(axis=1)
    # # bullish.to_excel(path + 'Candlestick_bull_statsn2.xlsx')
    # bearish = pd.concat(bearish, axis=1).sort_index().sort_index(axis=1)
    # bearish.to_excel(path + 'Candlestick_bear_statsn2.xlsx')
    # # FX_PAIR = ['eurusd']
    # print(CANDLE_NAMES)

    # for fx_pair in cst.FX_PIP:
    #     build_candlestick_pattern_analysis(fx_pair, DATE_RANGE, path=cst.FX_15MIN_DATA_PATH)
    #     report(fx_pair)
    #     print(fx_pair)

    # params = [(fx_pair,) for fx_pair in cst.FX_PIP]
    # fx_quote = pd.concat(AsyncMP().exec(params, report))


    # df[df[('Distribution', 'count')] > 600].xs('Total', level=4).sort_values(('Accuracy Direction', 15))