from __future__ import (absolute_import, division, print_function, absolute_import, unicode_literals)
import argparse
import fx_trading.utils.constants as cst
import pandas as pd
import numpy as np
from numba import njit, prange
import talib
from pathlib import Path
from fx_trading.utils.data_interact import read_data
from time import time
from matplotlib import pyplot as plt

import ta
from fx_trading.utils.fast_back_test import FastBackTesting

RAW_DATA_PATH = str(Path(__file__).parent)


class Strategy(FastBackTesting):

    HIGH_PROBABILITIES_CANDLES = [
        'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE', 'CDLADVANCEBLOCK', 'CDLBELTHOLD',
        'CDLCLOSINGMARUBOZU', 'CDLDARKCLOUDCOVER', 'CDLDOJISTAR', 'CDLENGULFING',
        'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLHANGINGMAN',
        'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD',
        'CDLLONGLINE', 'CDLMARUBOZU', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE',
        'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLTHRUSTING', 'CDLXSIDEGAP3METHODS'
    ]

    def __init__(self, fx_data_dict, re_sample_frequency, price_for_signal, signals,
                 entry_pip, stop_loss, take_profit, valid, lot_size, signal_to_test):
        n_signals = 0
        for i in signals:
            n_signals += len(signals[i])

        super().__init__(
            fx_data_dict=fx_data_dict, re_sample_frequency=re_sample_frequency,
            price_for_signal=price_for_signal, n_signals=n_signals, entry_pip=entry_pip,
            stop_loss=stop_loss, take_profit=take_profit, valid=valid, lot_size=lot_size
        )
        self.signal_frequency = signals
        n_labels = len(self.label_data)

        for s in signals:
            for freq in signals[s]:
                self.label_data[f'{s}_{freq}'] = n_labels
                self.label_data_r[n_labels] = f'{s}_{freq}'
                n_labels += 1
        self.add_signals()
        self.signal_to_test = signal_to_test

    def macd_diff(self, freq):
        """
        Description:
        ------------
        http://hotforex.yolasite.com/resources/5pipscalp.pdf

        """
        to_fill = []
        for fx_pair in self.fx_manager:
            df = self.fx_manager[fx_pair]
            for freq in freq:
                close_pos = self.label_data[f'CLOSE_{freq}']
                new_bar_pos = self.label_data[f'NEW_BAR_{freq}']
                _df = pd.DataFrame(df[df[:, new_bar_pos] == -1])

                s = ta.trend.macd_diff(_df[close_pos],  window_slow=26, window_fast=12, window_sign=9)
                s = s.to_frame('current')
                s['previous'] = s.shift(1)
                s['cross'] = 0
                s.loc[(s['current'] > 0) & (s['previous'] < 0), 'cross'] = 100
                s.loc[(s['current'] < 0) & (s['previous'] > 0), 'cross'] = -100
                pos = self.label_data[f'MACD_DIFF_{freq}']
                self.fx_manager[fx_pair][:, pos] = 0
                self.fx_manager[fx_pair][_df[0].astype(np.int64), pos] = s['cross'].values

    def macd_sc(self, freq):
        """
        Description:
        ------------
        https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd
        https://www.learn-forextrading.org/2015/04/1minute-trend-momentum-scalping-strategy.html?m=0
        """

        for fx_pair in self.fx_manager:
            df = self.fx_manager[fx_pair]
            for freq in freq:
                open_pos = self.label_data[f'OPEN_{freq}']
                high_pos = self.label_data[f'HIGH_{freq}']
                low_pos = self.label_data[f'LOW_{freq}']
                close_pos = self.label_data[f'CLOSE_{freq}']
                new_bar_pos = self.label_data[f'NEW_BAR_{freq}']
                _df = pd.DataFrame(df[df[:, new_bar_pos] == -1])
                _df['ema'] = ta.trend.EMAIndicator(_df[close_pos], 10).ema_indicator()
                _df['macd'] = ta.trend.MACD(_df[close_pos], window_slow=26, window_fast=12, fillna=False).macd_diff()
                _df['rsi'] = ta.momentum.RSIIndicator(_df[close_pos], window=14).rsi()
                bollinger = ta.volatility.BollingerBands(_df[close_pos])
                _df['bband_mavg'] = bollinger.bollinger_mavg()
                _df['candle'] = getattr(talib, 'CDLENGULFING')(_df[open_pos], _df[high_pos], _df[low_pos], _df[close_pos])

                _df['signal'] = 0
                mask = (_df['ema'] > _df['bband_mavg']) & (_df['rsi'] > 50) & (_df['macd'] > 0) & (_df['macd'].shift(1) < 0) & (_df['candle'] == 100)
                _df.loc[mask, 'signal'] = 100
                mask = (_df['ema'] < _df['bband_mavg']) & (_df['rsi'] < 50) & (_df['macd'] < 0) & (_df['macd'].shift(1) > 0) & (_df['candle'] == -100)
                _df.loc[mask, 'signal'] = -100

                pos = self.label_data[f'MACD_SC_{freq}']
                self.fx_manager[fx_pair][:, pos] = 0
                self.fx_manager[fx_pair][_df[0].astype(np.int64), pos] = _df['signal'].values

    def custom_acc(self, freq):

        for fx_pair in self.fx_manager:
            df = self.fx_manager[fx_pair]
            for freq in freq:
                close_pos = self.label_data[f'CLOSE_{freq}']
                new_bar_pos = self.label_data[f'NEW_BAR_{freq}']
                _df = pd.DataFrame(df[df[:, new_bar_pos] == -1])
                _df['acc'] = _df[close_pos] - 2 * _df[close_pos].shift(1) + _df[close_pos].shift(2)
                _df['acc_pos'] = 0
                _df.loc[_df['acc'] > 0, 'acc_pos'] = _df['acc']
                _df['acc_neg'] = 0
                _df.loc[_df['acc'] < 0, 'acc_neg'] = - _df['acc']
                _df[['acc_pos', 'acc_neg']] = _df[['acc_pos', 'acc_neg']].cumsum()
                _df[['acc_pos', 'acc_neg']] = _df[['acc_pos', 'acc_neg']].rolling(21).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std())
                _df['signal'] = _df['acc_pos'] - _df['acc_neg']
                _df['cross'] = 0

                mask = (_df['signal'] < 0)
                _df.loc[mask, 'cross'] = -100
                mask = (_df['signal'] > 0)
                _df.loc[mask, 'cross'] = 100
                print(_df)
                pos = self.label_data[f'CUSTOM_ACC_{freq}']
                self.fx_manager[fx_pair][:, pos] = 0
                self.fx_manager[fx_pair][_df[0].astype(np.int64), pos] = _df['cross'].values

        pass

    def super_trend(self, freq, look_back=10, multiplier=3):

        for fx_pair in self.fx_manager:
            df = self.fx_manager[fx_pair]
            for freq in freq:
                open_pos = self.label_data[f'OPEN_{freq}']
                high_pos = self.label_data[f'HIGH_{freq}']
                low_pos = self.label_data[f'LOW_{freq}']
                close_pos = self.label_data[f'CLOSE_{freq}']
                new_bar_pos = self.label_data[f'NEW_BAR_{freq}']
                _df = pd.DataFrame(df[df[:, new_bar_pos] == -1])

                # Compute ATR
                high_minus_low = _df[high_pos] - _df[low_pos]
                high_minus_close = _df[high_pos] - _df[close_pos]
                close_minus_low = _df[close_pos] - _df[low_pos]
                _df['TR'] = np.maximum.reduce([high_minus_low.abs(), high_minus_close.abs(), close_minus_low.abs()])
                _df['ATR'] = ta.trend.EMAIndicator(_df['TR'], look_back).ema_indicator()
                _df['ATR2'] = ta.trend.EMAIndicator(_df['TR'], 20).ema_indicator()

                # H/L AVG AND BASIC UPPER & LOWER BAND
                hl_avg = (_df[high_pos] + _df[low_pos]) / 2
                final_upperband = _df['U_BANDS'] = (hl_avg + multiplier * _df['ATR'])
                final_lowerband = _df['L_BANDS'] = (hl_avg - multiplier * _df['ATR'])

                final_upperband2 = (hl_avg + 5 * _df['ATR2'])
                final_lowerband2 = (hl_avg - 5 * _df['ATR2'])

                # initialize Supertrend column to True
                supertrend = [True] * len(_df)
                supertrend2 = [True] * len(_df)

                for i in range(1, len(_df.index)):
                    curr, prev = i, i - 1

                    # if current close price crosses above upperband
                    if _df[close_pos][curr] > final_upperband[prev]:
                        supertrend[curr] = True
                    # if current close price crosses below lowerband
                    elif _df[close_pos][curr] < final_lowerband[prev]:
                        supertrend[curr] = False
                    # else, the trend continues
                    else:
                        supertrend[curr] = supertrend[prev]

                        # adjustment to the final bands
                        if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                            final_lowerband[curr] = final_lowerband[prev]
                        if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                            final_upperband[curr] = final_upperband[prev]

                    # to remove bands according to the trend direction
                    if supertrend[curr] == True:
                        final_upperband[curr] = np.nan
                    else:
                        final_lowerband[curr] = np.nan

                    # if current close price crosses above upperband
                    if _df[close_pos][curr] > final_upperband2[prev]:
                        supertrend2[curr] = True
                    # if current close price crosses below lowerband
                    elif _df[close_pos][curr] < final_lowerband2[prev]:
                        supertrend2[curr] = False
                    # else, the trend continues
                    else:
                        supertrend2[curr] = supertrend2[prev]

                        # adjustment to the final bands
                        if supertrend2[curr] == True and final_lowerband2[curr] < final_lowerband2[prev]:
                            final_lowerband2[curr] = final_lowerband2[prev]
                        if supertrend2[curr] == False and final_upperband2[curr] > final_upperband2[prev]:
                            final_upperband2[curr] = final_upperband2[prev]

                    # to remove bands according to the trend direction
                    if supertrend2[curr] == True:
                        final_upperband2[curr] = np.nan
                    else:
                        final_lowerband2[curr] = np.nan

                _df['SUPER_TREND'] = supertrend
                _df['SUPER_TREND2'] = supertrend2
                _df['FU_BANDS'] = final_upperband
                _df['FL_BANDS'] = final_lowerband

                _df['SPREAD'] = _df['FU_BANDS'] - _df[close_pos]
                _df.loc[_df['FU_BANDS'].isna(), 'SPREAD'] = _df[close_pos] - _df['FL_BANDS']
                _df['SPREAD'] *= 10000
                _df['signal'] = 0
                _df.loc[(_df['SUPER_TREND']==True) & (_df['SUPER_TREND'] != _df['SUPER_TREND'].shift(1)), 'signal'] = 100
                _df.loc[(_df['SUPER_TREND']==False) & (_df['SUPER_TREND'] != _df['SUPER_TREND'].shift(1)), 'signal'] = -100
                _df.loc[_df['SUPER_TREND'] != _df['SUPER_TREND2'], 'signal'] = 0
                # _df.loc[(_df['FU_BANDS'].isna()) & (~_df['FL_BANDS'].isna()), 'signal'] = 100
                # _df.loc[(_df['FL_BANDS'].isna()) & (~_df['FU_BANDS'].isna()), 'signal'] = -100

                pos = self.label_data[f'SUPER_TREND_{freq}']
                self.fx_manager[fx_pair][:, pos] = 0
                self.fx_manager[fx_pair][_df[0].astype(np.int64), pos] = _df['signal'].values

    def upside_vol(self, freq):
        for fx_pair in self.fx_manager:
            df = self.fx_manager[fx_pair]
            for freq in freq:
                open_pos = self.label_data[f'OPEN_{freq}']
                high_pos = self.label_data[f'HIGH_{freq}']
                low_pos = self.label_data[f'LOW_{freq}']
                close_pos = self.label_data[f'CLOSE_{freq}']
                new_bar_pos = self.label_data[f'NEW_BAR_{freq}']
                _df = pd.DataFrame(df[df[:, new_bar_pos] == -1])
                _df['close_diff'] = _df[close_pos].diff() * 10_000
                _df.loc[_df['close_diff'] > 0, 'BLK+'] = (_df[high_pos] - _df[low_pos])
                _df.loc[_df['close_diff'] < 0, 'BLK-'] = (_df[high_pos] - _df[low_pos])

                _df['VOL'] = ((_df[high_pos] - _df[low_pos]) ** 2).rolling(21, min_periods=1).sum()
                _df['VOL+'] = (_df['BLK+']**2).rolling(21, min_periods=1).sum() / _df['VOL']
                _df['VOL-'] = (_df['BLK-']**2).rolling(21, min_periods=1).sum() / _df['VOL']

                # _df['VOL2'] = ((_df[high_pos] - _df[low_pos]) ** 2).rolling(9, min_periods=1).sum()
                # _df['VOL2+'] = (_df['BLK+'] ** 2).rolling(9, min_periods=1).sum() / _df['VOL']
                # _df['VOL2-'] = (_df['BLK-'] ** 2).rolling(9, min_periods=1).sum() / _df['VOL']
                _df['VOL_BLK'] = _df['VOL+'] - _df['VOL-']
                # _df['VOL_BLK2'] = _df['VOL2+'] - _df['VOL2-']
                # _df['VOL_BLK'] = _df['VOL_BLK2'] - _df['VOL_BLK1']

                _df['DIR_BLK'] = _df['BLK+'].rolling(14, min_periods=1).sum() - _df['BLK-'].rolling(14, min_periods=1).sum()

                _df['VOL_BLK'].plot()
                print(_df[['DIR_BLK', 'VOL_BLK']].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))
                plt.show()

                print(_df)

    def candle_stick(self, freq, candle='CDLENGULFING'):

        for fx_pair in self.fx_manager:
            df = self.fx_manager[fx_pair]
            for freq in freq:
                open_pos = self.label_data[f'OPEN_{freq}']
                high_pos = self.label_data[f'HIGH_{freq}']
                low_pos = self.label_data[f'LOW_{freq}']
                close_pos = self.label_data[f'CLOSE_{freq}']
                new_bar_pos = self.label_data[f'NEW_BAR_{freq}']
                _df = pd.DataFrame(df[df[:, new_bar_pos] == -1])
                pattern = getattr(talib, candle)(_df[open_pos], _df[high_pos], _df[low_pos], _df[close_pos])
        pass

    def cci(self, freq):
        for fx_pair in self.fx_manager:
            df = self.fx_manager[fx_pair]
            for freq in freq:
                open_pos = self.label_data[f'OPEN_{freq}']
                high_pos = self.label_data[f'HIGH_{freq}']
                low_pos = self.label_data[f'LOW_{freq}']
                close_pos = self.label_data[f'CLOSE_{freq}']
                new_bar_pos = self.label_data[f'NEW_BAR_{freq}']
                _df = pd.DataFrame(df[df[:, new_bar_pos] == -1])
                _df['cci'] = ta.trend.CCIIndicator(_df[high_pos], _df[low_pos], _df[close_pos], window=20).cci()
                _df['cci_avg'] = _df['cci'].rolling(10).mean()

                _df['signal'] = 0
                mask = (_df['cci'] > -100) & (_df['cci_avg'] < -100) & (_df['cci'].shift(1) < -100)
                _df.loc[mask, 'signal'] = 100
                mask = (_df['cci'] < 100) & (_df['cci_avg'] > 100) & (_df['cci'].shift(1) > 100)
                _df.loc[mask, 'signal'] = -100

    def composite_signal(self, freq):

        for fx_pair in self.fx_manager:
            df = self.fx_manager[fx_pair]
            for freq in freq:
                new_bar_pos = self.label_data[f'NEW_BAR_{freq}']
                _df = pd.DataFrame(df[df[:, new_bar_pos] == 1])
                SIGNALS = [f'MACD_SC_{freq}', f'SUPER_TREND_{freq}']
                SIGNALS = [self.label_data[i] for i in SIGNALS]
                _df['sum'] = _df[SIGNALS].sum(axis=1)
                _df.loc[_df['sum'] == 200, 'signal'] = 100
                _df.loc[_df['sum'] == -200, 'signal'] = -100

                pos = self.label_data[f'COMPOSITE_{freq}']
                self.fx_manager[fx_pair][:, pos] = 0
                self.fx_manager[fx_pair][_df[0].astype(np.int64) - 1, pos] = _df['signal'].values

    def add_signals(self):

        signal_map = {
            'MACD_DIFF': self.macd_diff, 'MACD_SC': self.macd_sc, 'CUSTOM_ACC': self.custom_acc,
            'SUPER_TREND': self.super_trend, 'UPSIDE_VOL': self.upside_vol, 'CCI': self.cci,

            'COMPOSITE': self.composite_signal
        }

        for i in self.signal_frequency:
            signal_map.get(i)(self.signal_frequency[i])


def parse_args(pargs=None):

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Back-test FOREX Strategies')

    parser.add_argument('--fx_pair', '-fx', required=False, default='eurgbp',
                        choices=list(cst.FX_PIP.keys()), help='FX pair to backtest')
    parser.add_argument('--backtest_column', '-target_columns', required=False, default=cst.R_BID,
                        choices=[cst.R_BID, cst.R_ASK], help='Columns used for the back test.')
    # parser.add_argument('--year', '-y', required=False, default=[2022])

    parser.add_argument('--entry_pip', '-entry', required=False, default=0)
    parser.add_argument('--take_profit', '-tp', required=False, default=5)
    parser.add_argument('--stop_loss', '-sl', required=False, default=50)
    parser.add_argument('--valid', '-vld', required=False, default=14400)
    parser.add_argument('--lot_size', '-ls', required=False, default=200_000)
    #
    # parser.add_argument('--indicator', required=False, action='store', default='CDLENGULFING',
    #                     choices=FastHighProbCandleStick.HIGH_PROBABILITIES_CANDLES,
    #                     help=('Which indicator pair to show together'))

    return parser.parse_args()


def run_strategy(pair, args=None):

    args = parse_args(args)
    data = pd.read_parquet(RAW_DATA_PATH + '/sample.parquet')

    # data = read_data(cst.FX_TICK_CHUNK_DATA_PATH, fx_pair=[pair], date_range=[2022])
    data['TIMESTAMP'] = data.index.astype(np.int64) // 10 ** 9
    cols = ['TIMESTAMP', 'BID', 'ASK']

    signals = {'SUPER_TREND': ['5MIN'], 'MACD_SC': ['1MIN'], 'COMPOSITE': ['1MIN']}
    signals = {'CCI': ['15MIN']}
    t = time()
    bt = Strategy(
        fx_data_dict={pair: data[cols].values}, re_sample_frequency=['15MIN'],
        price_for_signal=args.backtest_column, signals=signals, entry_pip=args.entry_pip,
        stop_loss=args.stop_loss, take_profit=args.take_profit, valid=args.valid,
        lot_size=args.lot_size, signal_to_test={'CCI': ['15MIN']}
    )
    strategy_result = bt.run()

    print(time() - t)
    print(strategy_result)
    strategy_result.to_csv('super_trend2.csv')
    return strategy_result


if __name__ == '__main__':
    strategy_result = run_strategy(pair='eurgbp', args=None)