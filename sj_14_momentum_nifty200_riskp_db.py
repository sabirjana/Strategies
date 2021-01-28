# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 19:45:58 2021

@author: Sabir Jana
Momentum Strategy - Based on Andreas F. Clenow’s book Stocks on the Move: 
Beating the Market with Hedge Fund Momentum Strategy
We will use nifty200 as univ. with 20 stocks and risk parity weights
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import pandas as pd
import numpy as np
idx = pd.IndexSlice

import backtrader as bt
#import pyfolio as pf
import collections
from scipy.stats import linregress
import pymysql
np.random.seed(42)
bt.__version__

tickers = ['ACC', 'AUFI', 'ARTI', 'ABOT', 'ADEL', 'ADAG', 'ADNA', 'APSE', 'ADAI', 'ADTB', 'ADIA', 'AJPH', 'ALEM', 'ALKE', 
           'AMAR', 'ABUJ', 'APLH', 'APLO', 'ASOK', 'ASPN', 'ARBN', 'AVEU', 'AXBK', 'BAJA', 'BJFN', 'BJFS', 'BJAT', 'BLKI', 
           'BANH', 'BOB', 'BOI', 'BATA', 'BRGR', 'BAJE', 'BFRG', 'BHEL', 'BPCL', 'BRTI', 'BHRI', 'BION', 'BBRM', 'BOSH', 
           'BRIT', 'CESC', 'CADI', 'CNBK', 'CAST', 'CHLA', 'CIPL', 'CTBK', 'COAL', 'NITT', 'COLG', 'CCRI', 'CORF', 'CROP', 
           'CUMM', 'DLF', 'DABU', 'DALB', 'INDB', 'DIVI', 'DLPA', 'REDY', 'EDEL', 'EMAM', 'ENDU', 'ESCO', 'EXID', 'FED', 
           'FOHE', 
           'FRTL', 'GAIL', 'GMRI', 'GENA', 'GLEN', 'GODE', 'GOCP', 'GODI', 'GODR', 'GRAS', 'GGAS', 'GSPT', 'HCLT', 'HDFA', 
           'HDBK', 'HDFL', 'HVEL', 'HROM', 'HALC', 'HPCL', 'HLL', 'HZNC', 'HUDC', 'HDFC', 'ICBK', 'ICIL', 'ICIR', 'ICCI', 
           'IDFB', 'ITC', 'INBF', 'IHTL', 'IOC', 'INIR', 'IGAS', 'INBK', 'INED', 'INFY', 'INGL', 'IPCA', 'JSWE', 'JSTL', 
           'JNSP', 'JUBI', 'KTKM', 'LTFH', 'LTEH', 'LICH', 'LRTI', 'LART', 'LUPN', 'MRF', 'MGAS', 'MMFS', 'MAHM', 'MNFL', 
           'MRCO', 'MRTI', 'MAXI', 'MINT', 'MOSS', 'MBFL', 'MUTT', 'NATP', 'NMDC', 'NTPC', 'NALU', 'NAFL', 'NEST', 'RELL', 
           'OEBO', 'ONGC', 'OILI', 'ORCL', 'PIIL', 'PAGE', 'PLNG', 'PFIZ', 'PIDI', 'PIRA', 'POLC', 'PWFC', 'PGRD', 'PREG', 
           'PROC', 'PNBK', 'RATB', 'RECM', 'REXP', 'RELI', 'SBIL', 'SRFL', 'SANO', 'SHCM', 'SRTR', 'SIEM', 'SBI', 'SAIL', 
           'SUN', 'SUTV', 'SYNN', 'TVSM', 'TTCH', 'TCS', 'TAGL', 'TAMO', 'TTPW', 'TISC', 'TEML', 'TRCE', 'TITN', 'TORP', 
           'TOPO', 'TREN', 'UPLL', 'ULTC', 'UNBK', 'UBBW', 'UNSP', 'VGUA', 'VARB', 'VODA', 'VOLT', 'WHIR', 'WIPR', 'ZEE']

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

vola_window = 21
# we take a 126-day time series of closing prices, 
# calculate the daily returns, and take a mean of 21 days rolling window of standards deviation.
def volatility(ts):
    std = ts.pct_change().dropna().rolling(vola_window).std().iloc[-1]
    return std

class StrategyRiskparity(bt.Strategy):
    params = dict(
        # parametrize the Momentum and its period
        momentum=Momentum,
        momentum_period=126,
        num_positions=30,
        rebalance_days = [1,4],

        printlog=True,
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
        for d in self.securities:
            self.inds[d]['mom'] = self.p.momentum(d, period=self.p.momentum_period)

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def rebalance(self):
        rankings = list(self.securities)
        rankings.sort(key=lambda s: self.inds[s]['mom'][0], reverse=True)

        # Sell stocks no longer meeting ranking filter and create list of kept positions
        kept_positions = []
        for i, d in enumerate(rankings):
            if self.getposition(d).size:
                if i > self.p.num_positions:
                    self.close(d)
                    self.log('Leave {} - Rank {:.2f}'.format(d._name, i))
                elif i < self.p.num_positions:
                    kept_positions.append(d._name)
        self.log('kept_positions - %s'% kept_positions)
        
        # Based on kept position and new ranking identify new long positions to add
        new_positions = []
        for i, d in enumerate(rankings[:self.p.num_positions]):
            if d._name not in (kept_positions):
                new_positions.append(d._name)
        self.log('new_positions - %s'% new_positions)
        
        # Calculate volatility table
        hist = pd.DataFrame()
        for d in self.securities:
            if d._name in (new_positions):
                hist[d._name] = d.close.get(size=self.p.momentum_period)
        vola_table = hist.apply(volatility)
        self.log('vola_table - %s'% vola_table)
        
        # Calculate weights based on volatility 
        inv_vola_table = 1 / vola_table 
        sum_inv_vola = np.sum(inv_vola_table)         
        vola_target_weights = inv_vola_table / sum_inv_vola
        self.log('vola_target_weights - %s'% vola_target_weights)
        
        # Buy and rebalance stocks with remaining cash
        for i, d in enumerate(rankings[:self.p.num_positions]):
            cash = self.broker.get_cash()
            value = self.broker.get_value()
            if cash <= 0:
                break
            if not self.getposition(d).size:
                weight = vola_target_weights[d._name]
                self.order_target_percent(d, target=weight)
                self.log('Buy {} - Rank {:.2f}'.format(d._name, i)) 

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
                    (order.executed.price,order.executed.value,order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price, order.executed.value, order.executed.comm))

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None
    
    def stop(self):
        self.log('| %2d | %2d |  %.2f |' %
                 (self.p.momentum_period, self.p.num_positions, self.broker.getvalue()),doprint=True)          


def get_db_conn(p_uid='', p_pwd='', p_host='', p_database='tradesoft', p_port=3306):
    pconn = pymysql.connect( user=p_uid, password=p_pwd, host=p_host, database=p_database, port=p_port )
    pconn.autocommit = False
    return pconn

def historical_price(scripts, db_conn):
    columns = ['close','high','low','open','volume']
    fmt='%Y%m%d %H:%M:%S'
    def data(ticker):
        print(f'Processing...{ticker}')
        v_sql_stmt = f"SELECT * FROM t_quotes_eod WHERE ticker = '{ticker}'" 
        df_quotes = pd.read_sql(v_sql_stmt, db_conn, parse_dates={'trade_date':fmt})
        df_quotes = df_quotes.set_index('trade_date')
        df_quotes.index.name = 'date'
        df_quotes = df_quotes[columns]
        df_quotes = df_quotes.sort_values(by='date', ascending=True)
        df_quotes = df_quotes[~df_quotes.index.duplicated()]
        return df_quotes
    datas = map(data, scripts)
    return(pd.concat(datas, keys=tickers, names=['ticker', 'date']))
               
def main():
    # Model Settings
    startcash = 500000
    momentum_period = 126 #days
    num_positions = 20
    reserve = 0.05
    printlog=False
    
    # Commission and Slippage Settings
    commission = 0.0025

    db_conn = get_db_conn(p_uid='freedbtech_tradesoft', p_pwd='tradesoft', p_host='freedb.tech', 
                          p_database='freedbtech_tradesoft', p_port=3306)
    
    prices = historical_price(tickers, db_conn)
    db_conn.close()

    # remove tickers where we have less than 10 years of data.
    min_obs = 2520
    nobs = prices.groupby(level='ticker').size()
    keep = nobs[nobs>min_obs].index
    prices = prices.loc[idx[keep,:], :]
#    prices.info()

    print('Number of tickers with min 10 years history= ', prices.index.unique(level='ticker'))
    print('latest Date: ', prices.loc['ACC'].index[-1])
    print('latest Date - 126 Trading days: ', prices.loc['ACC'].index[-126])

    from_date=input('start date in format yyyy-mm-dd:')
    to_date=input('end date in format yyyy-mm-dd:')
    
    fromdate=datetime.datetime.strptime(from_date, '%Y-%m-%d')
    todate=datetime.datetime.strptime(to_date, '%Y-%m-%d')
    
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
    
    cerebro.addstrategy(StrategyRiskparity,
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