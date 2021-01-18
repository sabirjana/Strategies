# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 19:45:58 2021

@author: Sabir Jana
Momentum Strategy - Based on Andreas F. Clenow’s book Stocks on the Move: 
    Beating the Market with Hedge Fund Momentum Strategy
We will use nifty200 as univ. with 20 stocks 
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import datetime
import pandas as pd
import numpy as np
idx = pd.IndexSlice

import matplotlib.pyplot as plt
import seaborn as sns
import backtrader as bt
import pyfolio as pf
import collections
from scipy.stats import linregress
sns.set_style('whitegrid')
np.random.seed(42)
bt.__version__

# Calculate momentum
def momentum_func(self, the_array):
    r = np.log(the_array)
    slope, _, rvalue, _, _ = linregress(np.arange(len(r)), r)
    annualized = (1 + slope) ** 252
    return annualized * (rvalue ** 2)

class Momentum(bt.ind.OperationN):
    lines = ('trend',)
    params = dict(period=126)
    func = momentum_func
    
class StrategyEqWt(bt.Strategy):
    params = dict(
        # parametrize the Momentum and its period
        momentum=Momentum,
        momentum_period=126,
        num_positions=30,
        rebalance_days = [1,4],

        printlog=False,
        reserve=0.00  # 5% reserve capital
    )

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function '''
        if self.params.printlog or doprint:
            dt = dt or self.data.datetime[0]
            if isinstance(dt, float):
                dt = bt.num2date(dt)
            print("%s, %s" % (dt.isoformat(), txt))

    def __init__(self):
        self.securities = self.datas
        self.inds = collections.defaultdict(dict)
        for d in self.datas:
            self.inds[d]['mom'] = self.p.momentum(d, period=self.p.momentum_period)

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def rebalance(self):
        rankings = list(self.securities)
        rankings.sort(key=lambda s: self.inds[s]['mom'][0], reverse=True)

        # allocation perc per stock
        # reserve kept to make sure orders are not rejected due to
        # margin. Prices are calculated when known (close), but orders can only
        # be executed next day (opening price). Price can gap upwards
        pos_size = (1.0 - self.p.reserve) / self.p.num_positions

        # Sell stocks no longer meeting ranking filter.
        for i, d in enumerate(rankings):
            if self.getposition(d).size:
                if i > self.p.num_positions:
                    self.close(d)
                    self.log('Leave {} - Rank {:.2f}'.format(d._name, i)) 
        
        # Buy and rebalance stocks with remaining cash
        for i, d in enumerate(rankings[:self.p.num_positions]):
            cash = self.broker.get_cash()
            value = self.broker.get_value()
            if cash <= 0:
                break
            if not self.getposition(d).size:
                self.order_target_percent(d, target=pos_size)
                self.log('Buy {} - Rank {:.2f}'.format(d._name, i)) 

        # Final portfolio
        portfolio = []
        for i, d in enumerate(rankings):
            if self.getposition(d).size:
                if i < self.p.num_positions:
                    portfolio.append(d._name)
        self.log('Portfolio - %s'% portfolio)

    def next_open(self):
        dt = self.data.datetime.datetime()
        if dt.weekday() in self.p.rebalance_days:
            self.rebalance()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price, order.executed.value, order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price, order.executed.value,order.executed.comm))

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None
    
    def stop(self):
        self.log('| %2d | %2d |  %.2f |' %
                 (self.p.momentum_period, self.p.num_positions, self.broker.getvalue()), doprint=False)
        

def main():
    # Model Settings
    startcash = 500000
    momentum_period = 126 #days
    num_positions = 20
    reserve = 0.05
    printlog=False
    
    
    # Commission and Slippage Settings
    commission = 0.0025

    from_date=input('start date in format yyyy-mm-dd:')
    to_date=input('end date in format yyyy-mm-dd:')
    
    fromdate=datetime.datetime.strptime(from_date, '%Y-%m-%d')
    todate=datetime.datetime.strptime(to_date, '%Y-%m-%d')
    
    DATA_STORE = '../../Data-Daily/india_asset.h5'
    
    with pd.HDFStore(DATA_STORE) as store:
        nifty200_m = (store['/nse/nifty200/metadata'])

    tickers = nifty200_m.dropna().symbol.to_list()
#    print(len(tickers))
    
    columns = ['close','high','low','open','volume']

    with pd.HDFStore(DATA_STORE) as store:
        prices = store['/ind/nifty500_investing/prices'].loc[idx[tickers, :], columns]

    # remove tickers where we have less than 10 years of data.
    min_obs = 2520
    nobs = prices.groupby(level='ticker').size()
    keep = nobs[nobs>min_obs].index
    prices = prices.loc[idx[keep,:], :]
#    prices.info()

    prices.index.unique(level='ticker')
    
    cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)
#    cerebro.broker.set_coc(True)
    cerebro.broker.setcash(startcash)
    cerebro.broker.setcommission(commission=commission)
    
    # Add securities as datas1:
    for ticker, data in prices.groupby(level=0):
        if ticker in tickers:
            print(f"Adding ticker: {ticker}")
            data = bt.feeds.PandasData(dataname=data.droplevel(level=0),
                                       name=str(ticker),
                                       fromdate=fromdate,
                                       todate=todate,
                                       plot=False)
            cerebro.adddata(data)
            
    cerebro.addanalyzer(bt.analyzers.Returns, _name='pfreturn')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='pfdrawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='pfsharpe')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    
    cerebro.addstrategy(StrategyEqWt,
                        momentum_period = momentum_period,
                        num_positions = num_positions,
                        printlog = printlog,
                        reserve = reserve
                       )
    
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    # Run the strategy. Results will be output from stop.
    results_eq_wts = cerebro.run()
    results_eq_wt = results_eq_wts[0]
    
    pyfoliozer = results_eq_wt.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()

    transactions.to_csv('data/transactions.csv')
    positions.to_csv('data/positions.csv')
    returns.to_csv('data/returns.csv')

    # Print out the return
    print('\nPortfolio Return:', results_eq_wt.analyzers.pfreturn.get_analysis())

    # Print out the drawdown
    print('\nPortfolio Drawdown:', results_eq_wt.analyzers.pfdrawdown.get_analysis())
    
    # Print out the sharpe
    print('\nPortfolio Sharpe ratio:', results_eq_wt.analyzers.pfsharpe.get_analysis())
    
    # Print out the final result
    print('\nFinal Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
if __name__ == '__main__':
    main()