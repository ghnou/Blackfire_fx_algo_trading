from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import argparse
import datetime
import backtrader as bt
import fx_trading.utils.constants as cst
from fx_trading.utils.data_interact import read_data
import pandas as pd
import numpy as np
from backtrader.indicators import EMA

class MACD(bt.Indicator):
    lines = ('long', 'short')
    params = (('long_period', 251), ('short_period', 21))

    def next(self):
        y = self.data.get(size=self.p.long_period)
        x = list(range(self.p.long_period))
        self.lines.long[0] = np.arctan(np.polyfit(x, y, 1)[0])

        y = self.data.get(size=self.p.short_period)
        x = list(range(self.p.short_period))
        self.lines.short[0] = np.arctan(np.polyfit(x, y, 1)[0])


class TradeClosed(bt.analyzers.Analyzer):
    """
    Analyzer returning closed trade information.
    """

    def start(self):
        super(TradeClosed, self).start()

    def create_analysis(self):

        self.rets = {}
        self.vals = tuple()

    def notify_trade(self, trade):
        """Receives trade notifications before each next cycle"""
        if trade.isclosed:
            self.vals = (
                self.strategy.datetime.datetime(),
                trade.tradeid,
                trade.data._name,
                round(trade.pnl, 2),
                round(trade.pnlcomm, 2),
                trade.commission,
                trade.baropen,
                trade.barclose,
                trade.dtopen,
                trade.dtclose,
                (trade.dtclose - trade.dtopen),
            )

            self.rets[trade.ref] = self.vals

    def get_analysis(self):
        return self.rets


class HighProbCandleStick(bt.Strategy):

    params = (('ind', 'CDLENGULFING'), ('entry_pip', 0), ('exit_pip', 5), ('stop_loss', 45),
              ('max_spread', 2))

    HIGH_PROBABILITIES_CANDLES = [
        'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE', 'CDLADVANCEBLOCK', 'CDLBELTHOLD',
        'CDLCLOSINGMARUBOZU', 'CDLDARKCLOUDCOVER', 'CDLDOJISTAR', 'CDLENGULFING',
        'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLHANGINGMAN',
        'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD',
        'CDLLONGLINE', 'CDLMARUBOZU', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE',
        'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLTHRUSTING', 'CDLXSIDEGAP3METHODS'
    ]

    def __init__(self):

        self.fx_manager = {}
        self.rets = {}
        self.order_manager = dict()
        self.trade_manager = dict()
        self.n_wins = 0
        self.n_losses = 0
        self.profits = 0
        self.losses = 0
        self.vals = tuple()

        for i, d in enumerate(self.datas):
            name = d._name
            fx_pair, attrib = name.split('_')

            if fx_pair not in self.fx_manager:
                self.fx_manager[fx_pair] = dict()
                self.order_manager[fx_pair] = {}
                self.trade_manager[fx_pair] = {}
            self.fx_manager[fx_pair][attrib] = d

            if attrib == 'bar15MIN':

                self.fx_manager[fx_pair]['pattern'] = getattr(bt.talib, self.p.ind)(
                    d.open, d.high, d.low, d.close
                )
                # self.fx_manager[fx_pair]['ATR'] = bt.indicators.AverageTrueRange(d, subplot=True)
                # self.fx_manager[fx_pair]['SMA'] = bt.indicators.EMA(d, period=50)
                # self.fx_manager[fx_pair]['SMA_CT'] = bt.indicators.EMA(d, period=9)
            if attrib == 'bar1H':
                # self.fx_manager[fx_pair]['pattern4H'] = getattr(bt.talib, self.p.ind)(
                #     d.open, d.high, d.low, d.close
                # )
                pass
                # self.fx_manager[fx_pair]['ATR'] = bt.indicators.AverageTrueRange(d, subplot=True)
                # self.fx_manager[fx_pair]['Support'] = bt.indicators.PivotPoint(d).s1()
                # self.fx_manager[fx_pair]['Resistance'] = bt.indicators.PivotPoint(d).r1()
                # self.fx_manager[fx_pair]['Trend_long'] = MACD(d).long_period
                # self.fx_manager[fx_pair]['Trend_Short'] = MACD(d).short_period

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.data.datetime[0]
        if isinstance(dt, float):
            dt = bt.num2date(dt)
        print('%s, %s' % (dt.isoformat(), txt))

    def start(self):
        self.counter = 0

    def notify_order(self, order):

        # if order.status in [order.Submitted]:
        #     # Buy/Sell order submitted/accepted to/by broker - Nothing to do
        #     self.log('ORDER SUBMITTED', dt=order.created.dt)
        #     self.order = order
        #     return
        #
        # if order.status in [order.Accepted]:
        #     # Buy/Sell order submitted/accepted to/by broker - Nothing to do
        #     self.log('ORDER ACCEPTED', dt=order.created.dt)
        #     self.order = order
        #     return
        #
        if order.status in [order.Expired]:
            print('{}: Order ref: {} / Type {} / Status {}'.format(
                self.data.datetime.datetime(0),
                order.ref, 'Buy' * order.isbuy() or 'Sell',
                order.getstatusname()))

        if order.status in [order.Canceled]:
            print('{}: Order ref: {} / Type {} / Status {}'.format(
                self.data.datetime.datetime(0),
                order.ref, 'Buy' * order.isbuy() or 'Sell',
                order.getstatusname()))

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'Order ref: %1d / BUY EXECUTED, Price: %.5f, Cost: %.5f, Comm %.5f' %
                    (order.ref, order.executed.price, order.executed.value, order.executed.comm))

            else:  # Sell
                self.log(
                    'Order ref: %1d / SELL EXECUTED, Price: %.5f, Cost: %.5f, Comm %.5f' %
                    (order.ref, order.executed.price, order.executed.value, order.executed.comm))

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f\n' %
                 (trade.pnl, trade.pnlcomm))

        if trade.pnl > 0:
            self.n_wins += 1
            self.profits += trade.pnl
        else:
            self.n_losses += 1
            self.losses += trade.pnl
        self.vals = (
            self.data.datetime.datetime(),
            trade.tradeid,
            trade.data._name,
            round(trade.pnl, 2),
            round(trade.pnlcomm, 2),
            10_000 * abs(
                self.trade_manager['eurgbp'][trade.tradeid][0] - self.trade_manager['eurgbp'][trade.tradeid][1]),
            trade.commission,
            trade.baropen,
            trade.barclose,
            trade.dtopen,
            trade.dtclose,
            (trade.dtclose - trade.dtopen),
        )

        #
        self.rets[trade.ref] = self.vals

    @staticmethod
    def pending_orders(dict_orders):

        pending_orders = {}
        for order_ref in dict_orders:
            order = dict_orders[order_ref]
            if order[0].status in [1, 2]:
                pending_orders[order[0].ref] = order

        return pending_orders

    @staticmethod
    def active_orders(dict_orders):

        active_orders = {}

        for order_ref in dict_orders:
            order = dict_orders[order_ref]
            main = order[0]
            sl = order[1]
            tp = order[2]
            tp = sl if tp is None else tp

            if (main.status in [4]) & (sl.status in [1, 2] or tp.status in [1, 2]):
                active_orders[main.ref] = order

        return active_orders

    def get_max_loss(self, current_price, fx_pair):

        dict_active_orders = self.active_orders(self.order_manager[fx_pair])

        for order_ref in dict_active_orders:
            order = dict_active_orders[order_ref]
            main = order[0]
            sl = order[1]
            tp = order[2]
            tp = sl if tp is None else tp

            if main.tradeid not in self.trade_manager[fx_pair]:
                self.trade_manager[fx_pair][main.tradeid] = [main.executed.price, main.executed.price]

            if (main.ordtype == 0) & (self.trade_manager[fx_pair][main.tradeid][1] > current_price):
                self.trade_manager[fx_pair][main.tradeid][1] = current_price
            elif (main.ordtype == 1) & (self.trade_manager[fx_pair][main.tradeid][1] < current_price):
                self.trade_manager[fx_pair][main.tradeid][1] = current_price

    ###########################################################################################################
    # Order type
    ###########################################################################################################

    def send_protected_order(self, type, fx_pair, data, valid):

        price = data.open[0]

        if type == 'BUY':
            p1 = price - self.p.entry_pip / cst.FX_PIP[fx_pair]
            p2 = p1 - self.p.stop_loss / cst.FX_PIP[fx_pair]

            mainside = self.buy(price=p1, exectype=bt.Order.Limit, transmit=False, valid=valid, size=200_000,
                                tradeid=len(self))
            lowside = self.sell(price=p2, size=mainside.size, exectype=bt.Order.Stop,
                                transmit=True, parent=mainside, tradeid=len(self))

            txt = ','.join(
                ['Open Price: @ %.5f' % price,
                 'Buy Limit order Ref: %1d  @ %.5f' % (mainside.ref, p1),
                 'SL Ref: %1d  @ %.5f' % (lowside.ref, p2)
                 ])
            self.log(txt)
        elif type == 'SELL':

            p1 = price + self.p.entry_pip / cst.FX_PIP[fx_pair]
            p2 = p1 + self.p.stop_loss / cst.FX_PIP[fx_pair]

            mainside = self.sell(price=p1, exectype=bt.Order.Limit, transmit=False, valid=valid, size=200_000,
                                 tradeid=len(self))
            lowside = self.buy(price=p2, size=mainside.size, exectype=bt.Order.Stop,
                               transmit=True, parent=mainside, tradeid=len(self))

            txt = ','.join(
                ['Open Price: @ %.5f' % price,
                 'Sell Limit order Ref: %1d  @ %.5f' % (mainside.ref, p1),
                 'SL Ref: %1d  @ %.5f' % (lowside.ref, p2)
                 ])
            self.log(txt)

        return [mainside, lowside, None]

    def send_new_bracket_order(self, type, fx_pair, data, valid, loss):

        price = data.open[0]

        if type == 'BUY':
            p1 = price - self.p.entry_pip / cst.FX_PIP[fx_pair]
            p2 = p1 - self.p.stop_loss / cst.FX_PIP[fx_pair]
            p3 = p1 + self.p.exit_pip / cst.FX_PIP[fx_pair]

            mainside = self.buy(price=p1, exectype=bt.Order.Limit, transmit=False, valid=valid, size=200_000,
                                tradeid=len(self))
            lowside = self.sell(price=p2, size=mainside.size, exectype=bt.Order.Stop,
                                transmit=False, parent=mainside, tradeid=len(self))
            highside = self.sell(price=p3, size=mainside.size, exectype=bt.Order.Limit,
                                 transmit=True, parent=mainside, tradeid=len(self))

            txt = ','.join(
                ['Open Price: @ %.5f' % price,
                 'Buy Limit order Ref: %1d  @ %.5f' % (mainside.ref, p1),
                 'TP Ref: %1d  @ %.5f' % (highside.ref, p3),
                 'SL Ref: %1d  @ %.5f' % (lowside.ref, p2)
                 ])
            self.log(txt)
        elif type == 'SELL':

            p1 = price + self.p.entry_pip / cst.FX_PIP[fx_pair]
            p2 = p1 + self.p.stop_loss / cst.FX_PIP[fx_pair]
            p3 = p1 - self.p.exit_pip / cst.FX_PIP[fx_pair]

            mainside = self.sell(price=p1, exectype=bt.Order.Limit, transmit=False, valid=valid, size=200_000,
                                 tradeid=len(self))
            lowside = self.buy(price=p2, size=mainside.size, exectype=bt.Order.Stop,
                               transmit=False, parent=mainside, tradeid=len(self))
            highside = self.buy(price=p3, size=mainside.size, exectype=bt.Order.Limit,
                                transmit=True, parent=mainside, tradeid=len(self))

            txt = ','.join(
                ['Open Price: @ %.5f' % price,
                 'Sell Limit order Ref: %1d  @ %.5f' % (mainside.ref, p1),
                 'TP Ref: %1d  @ %.5f' % (highside.ref, p3),
                 'SL Ref: %1d  @ %.5f' % (lowside.ref, p2)
                 ])
            self.log(txt)

        return [mainside, lowside, highside]

    ###########################################################################################################
    # Manage Pending orders
    ###########################################################################################################
    def cancel_pending_orders(self, type, fx_pair):

        to_remove = []
        dict_pending_orders = self.pending_orders(self.order_manager[fx_pair])
        for order_ref in dict_pending_orders:
            bracket = dict_pending_orders[order_ref]
            if (bracket[0].ordtype == 0) & (type == 'BUY'):
                self.cancel(bracket[0])
                to_remove.append(order_ref)
                self.log('Manual Cancel of order ref %1d' % order_ref)
            if (bracket[0].ordtype == 1) & (type == 'SELL'):
                self.cancel(bracket[0])
                to_remove.append(order_ref)
                self.log('Manual Cancel of order ref %1d' % order_ref)

        for order_ref in to_remove:
            remove_key = self.order_manager[fx_pair].pop(order_ref, None)

    ###########################################################################################################
    # Manage Active Trade
    ###########################################################################################################

    def __trailing_sl(self, bracket, new_sl, trade_order):

        # Cancel Previous SL
        self.cancel(bracket[1])
        before = bracket[1].created.price

        # need to replace both stopOrder and profitOrder
        stopOrder = trade_order(
            price=new_sl, size=bracket[0].size, exectype=bt.Order.Stop,
            transmit=True, tradeid=bracket[0].tradeid)

        bracket[1] = stopOrder

        self.log('Move Stop loss ref @ %1d from %.5f to %.5f' % (bracket[0].ref, before, new_sl))
        txt = ','.join(['New SL Ref: %1d  @ %.5f' % (stopOrder.ref, new_sl)])
        self.log(txt)

        return bracket

    def trailing_sl(self, current_price, fx_pair, stop_buy, stop_sell):

        dict_active_orders = self.active_orders(self.order_manager[fx_pair])

        for order_ref in dict_active_orders:
            bracket = dict_active_orders[order_ref]
            if (bracket[0].ordtype == 0) & (current_price > bracket[0].executed.price):
                bracket = self.__trailing_sl(bracket, stop_buy, self.sell)
                self.order_manager[fx_pair][bracket[0].ref] = bracket

            if (bracket[0].ordtype == 1) & (current_price < bracket[0].executed.price):
                bracket = self.__trailing_sl(bracket, stop_sell, self.buy)
                self.order_manager[fx_pair][bracket[0].ref] = bracket

    def move_bracket_sl(self, bracket, tp, sl):

        # cancel stop order automatically cancels the profit order of the bracket too
        self.cancel(bracket[1])
        self.cancel(bracket[2])

        before = bracket[1].created.price
        stopOrder = profitOrder = None

        # need to replace both stopOrder and profitOrder
        if bracket[0].ordtype == 0:  # buy
            profitOrder = self.sell(
                price=bracket[2].created.price, size=bracket[0].size, exectype=bt.Order.Limit,
                transmit=False)

            stopOrder = self.sell(
                price=bracket[0].executed.price, size=bracket[0].size, exectype=bt.Order.Stop,
                transmit=True, oco=profitOrder)

        if bracket[0].ordtype == 1:  # sell
            profitOrder = self.buy(
                price=bracket[2].created.price, size=bracket[0].size, exectype=bt.Order.Limit,
                transmit=False)

            stopOrder = self.buy(
                price=bracket[0].executed.price, size=bracket[0].size, exectype=bt.Order.Stop,
                transmit=True, oco=profitOrder)

        bracket[1] = stopOrder
        bracket[2] = profitOrder

        Helper.log(self, 'Move stoploss to breakeven for ',
                   'long' if bracket[0].ordtype == 0 else 'short',
                   'position from ',
                   before, 'to',
                   bracket[1].created.price)

    def manage_new_orders(self, fx_pair, type, allow_multiple_active, allow_sideway_trade):

        post = True
        if len(self.order_manager[fx_pair]) == 0:
            return post

        if not allow_multiple_active:
            self.cancel_pending_orders(type, fx_pair)

        dict_active_orders = self.pending_orders(self.order_manager[fx_pair])
        for order_ref in dict_active_orders:
            if (dict_active_orders[order_ref][0].ordtype == 0) & (type == 'BUY') & (not allow_multiple_active):
                post = False
                break

        if not allow_sideway_trade:
            side_type = 'BUY' if type == 'SELL' else 'SELL'
            self.cancel_pending_orders(side_type, fx_pair)
            pass

        return post
        if pattern == cst.BULLISH:
            # Cancel or manage existing sell orders
            sell_orders = self.order_manager[fx_pair]['SELL']
            for brackets in self.pending_orders(sell_orders):
                self.log('Change in Pattern cancel order ref: %1d' % brackets[0].ref)
                self.cancel(brackets[0])
            # Buy only if not orders:
            buy_orders = self.active_orders(self.order_manager[fx_pair]['BUY'])
            if len(buy_orders) == 0:
                for brackets in self.pending_orders(self.order_manager[fx_pair]['BUY']):
                    self.log('Posting new BUY orders ref: %1d' % brackets[0].ref)
                    self.cancel(brackets[0])
                post = True

        elif pattern == cst.BEARISH:
            buy_orders = self.order_manager[fx_pair]['BUY']
            for brackets in self.pending_orders(buy_orders):
                self.log('Change in Pattern cancel order ref: %1d' % brackets[0].ref)
                self.cancel(brackets[0])

            # Buy only if not orders:
            sell_orders = self.active_orders(self.order_manager[fx_pair]['SELL'])
            if len(sell_orders) == 0:
                for brackets in self.pending_orders(self.order_manager[fx_pair]['SELL']):
                    self.log('Posting new SELL orders ref: %1d' % brackets[0].ref)
                    self.cancel(brackets[0])
                post = True

        return post

    def next(self):

        for fx_pair in self.fx_manager:

            bid = self.fx_manager[fx_pair]['bid']
            ask = self.fx_manager[fx_pair]['ask']
            bar = self.fx_manager[fx_pair]['bar15MIN']
            pattern = self.fx_manager[fx_pair]['pattern']
            spread = (ask.close[0] - bid.close[0]) * cst.FX_PIP[fx_pair]

            txt = ','.join(
                [self.data.datetime.datetime().isoformat(), '%04d' % len(self),
                 '%04d' % len(bid), '%04d' % len(bar),
                 '%.5f' % bar.open[0], '%.5f' % bar.high[0],
                 '%.5f' % bar.low[0], '%.5f' % bar.close[0],
                 # '%.5f' % bar.open[0], '%.5f' % bar.close[0],
                  '%.1f' % spread,
                 '%.04d' % pattern[0]]
            )

            self.get_max_loss(bid.close[0], fx_pair)

            if self.counter != len(bar):
                # print(txt)
                ######################################################################################
                # Manage current active position TP and SL.
                ######################################################################################
                # self.trailing_sl(bid.open[0], fx_pair, bar.low[0], bar.high[0])
                valid = datetime.timedelta(minutes=15*16)
                candle_pattern_size = np.round(abs(bar.high[0] - bar.low[0]) * 10_000, 0)

                post = False
                if pattern[0] >= 100:
                    print('')
                    self.log('BULLISH {}, Spread: {}'.format(self.p.ind, candle_pattern_size))
                    # post = self.manage_new_orders(fx_pair, 'BUY', True, True)
                    # if post:
                    bracket = self.send_new_bracket_order('BUY', fx_pair, bid, valid, loss=bar.low[0])
                    self.order_manager[fx_pair][bracket[0].ref] = bracket

                if pattern[0] <= -100:
                    print('')
                    self.log('BEARISH {}, Spread: {}'.format(self.p.ind, candle_pattern_size))
                    # post = self.manage_new_orders(fx_pair, 'SELL',  True, True)
                    # if post:
                    bracket = self.send_new_bracket_order('SELL', fx_pair, bid, valid, loss=bar.high[0])
                    self.order_manager[fx_pair][bracket[0].ref] = bracket

                self.counter = len(bar)

    def stop(self):

        df = pd.DataFrame(self.rets).T
        print(df)
        print(df[df[3] < 0])
        df.to_csv('stats_max_losses.csv')
        win_ratio = self.n_wins / (self.n_wins + self.n_losses)
        avg_profit = self.profits / self.n_wins
        avg_loss = self.losses / self.n_losses
        profit_loss_ratio = avg_profit / avg_loss
        expected_profit_per_trade = win_ratio * avg_profit + (1 - win_ratio) * avg_loss
        print(f"win_ratio: {win_ratio:.2f}, "
              f"profit_loss_ratio: {profit_loss_ratio:.2f}, "
              f"expected_profit_per_trade: {expected_profit_per_trade:.2f}")



    def get_analysis(self):
        return self.rets

def runstrat(indicator, args=None):

    args = parse_args(args)
    cerebro = bt.Cerebro()
    
    # data = read_data(cst.FX_TICK_CHUNK_DATA_PATH, args.fx_pair, args.year)
    data = pd.read_parquet('sample.parquet').loc['2022-01-02': '2022-01-03']

    data_bid = bt.feeds.PandasData(
        dataname=data, open=0, high=0, low=0, close=0, name='eurgbp_bid',
        timeframe=bt.TimeFrame.Ticks, compression=1, plot=False)

    cerebro.adddata(data_bid)
    data_ask = bt.feeds.PandasData(
        dataname=data, open=1, high=1, low=1, close=1, name='eurgbp_ask',
        timeframe=bt.TimeFrame.Ticks, compression=1, plot=False)
    cerebro.adddata(data_ask, )
    cerebro.resampledata(data_bid, timeframe=bt.TimeFrame.Minutes, compression=15, name='eurgbp_bar15MIN')

    cerebro.broker.setcash(10_000 * 500)
    # cerebro.addstrategy(HighProbCandleStick, ind=args.ind)
    cerebro.addstrategy(HighProbCandleStick, ind=indicator)
    # cerebro.addanalyzer(TradeClosed, _name="trade_closed")

    thestrats = cerebro.run(runonce=False, stdstats=True)
    # thestrat = thestrats[0]

    # df = pd.DataFrame(cerebro.rets).T
    # df.to_csv('result_root_{}.csv'.format(indicator))

    # if args.plot:
    pkwargs = dict(style='bar')
    # if args.plot is not True:  # evals to True but is not True
    #     npkwargs = eval('dict(' + args.plot + ')')  # args were passed
    #     pkwargs.update(npkwargs)

    # cerebro.plot(volume=False, **pkwargs)

    # return df

def parse_args(pargs=None):

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Back-test High probability candlesticks patterns')
    
    parser.add_argument('--fx_pair', '-fx', required=False, default=['eurgbp'],
                        choices=list(cst.FX_PIP.keys()), help='FX pair to backtest')
    parser.add_argument('--year', '-y', required=False, default=[2022])

    parser.add_argument('--ind', required=False, action='store',
                        default='CDLENGULFING',
                        choices=HighProbCandleStick.HIGH_PROBABILITIES_CANDLES,
                        help=('Which indicator pair to show together'))

    parser.add_argument('--plot', '-p', nargs='?', required=False,
                        metavar='kwargs', const=True,
                        help=('Plot the read data applying any kwargs passed\n'
                              '\n'
                              'For example (escape the quotes if needed):\n'
                              '\n'
                              '  --plot style="candle" (to plot candles)\n'))
    if pargs is not None:
        return parser.parse_args(pargs)

    return parser.parse_args()



    parser.add_argument('--use-next', required=False, action='store_true',
                        help=('Use next (step by step) '
                              'instead of once (batch)'))
    # Plot options
    parser.add_argument('--plot', '-p', nargs='?', required=False,
                        metavar='kwargs', const=True,
                        help=('Plot the read data applying any kwargs passed\n'
                              '\n'
                              'For example (escape the quotes if needed):\n'
                              '\n'
                              '  --plot style="candle" (to plot candles)\n'))

    if pargs is not None:
        return parser.parse_args(pargs)

    return parser.parse_args()

if __name__ == '__main__':

    a = []
    for ind in ['CDLENGULFING']:
        # df = pd.read_csv('result_root_{}.csv'.format(ind))
        # df['CANDLE'] = ind
        # a.append(df)
        runstrat(ind)
    # pd.concat(a).to_csv('root.csv')