# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:15:27 2024

@author: mathi
"""

import pandas as pd
import numpy as np
import pyanomaly.analytics as pyanalytics
import pyanomaly.portfolio as portfolio
from pyanomaly import tcost

####################
### NYSE BUCKETS ###
####################

def NYSE_median_ME(df):
    """
    Calculate the median ME for each month for companies listed on the NYSE.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data.
    
    Returns
    -------
    pandas DataFrame
        DataFrame containing the median ME for each month.
    """

    NYSE_size = (df
                 .groupby(['month_end'])['ME']
                 .median()
                 .to_frame()
                 .reset_index()
                 .rename(columns = {'ME': 'ME_median'}))
    return NYSE_size

def bucket_names(variable, bins):
    """
    Generate labels and names for bucketing.
    
    Parameters
    ----------
    variable : str
        Name of the variable (e.g. B/M).
    bins : list of float
        List of percentiles for calculating breakpoints for allocation.
    
    Returns
    -------
    list of str, list of str
        Labels and names for bucketing.
    """

    names = ['month_end']
    labels = ['month_end']
    for i in bins:
        j = round(i * 100)
        labels.append(f'{j}%')
        names.append(f'{variable}_{j}')
    return labels, names

def NYSE_ratio(variable, bins, NYSE_df):
    """
    Calculate the ratio for a variable (e.g. B/M) based on percentiles for companies listed on the NYSE.
    
    Parameters
    ----------
    variable : str
        Name of the variable.
    bins : list of float
        List of percentiles for calculating breakpoints for allocation.
    NYSE_df : pandas DataFrame
        DataFrame containing the data.
    
    Returns
    -------
    pandas DataFrame
        DataFrame containing the ratio for the variable.
    """

    labels, names = bucket_names(variable, bins)   

    NYSE_ratio = (NYSE_df
                  .groupby(['month_end'])[variable]
                  .describe(percentiles = bins)
                  .reset_index())[labels]
    NYSE_ratio.columns = names   
    return NYSE_ratio

def size_bucket(row):
    """
    Assign size bucket based on ME and median ME:
        If ME is below median for a row (month), the stock is assigned S, and otherwise B
    
    Parameters
    ----------
    row : pandas Series
        Row of data.
    
    Returns
    -------
    str
        Size bucket.
    """

    if 0 < row['ME'] <= row['ME_median']:
        value = 'S'
        
    elif row['ME'] > row['ME_median']:
        value = 'B'
        
    else:
        value = 'missing'
        
    return value

############################
### PORTFOLIO ALLOCATION ###
############################

def bucket(row, variable, low, mid, high, names):
    """
    Assign bucket value based on the value of a variable relative to the low, mid, and high variables:
        If the variable is not valid, 'missing' is assigned.
    
    Parameters
    ----------
    row : pandas Series
        Row (month) of data.
    variable : str
        Name of the variable (e.g. B/M).
    low : str
        Value to assign if variable is less than or equal to the first percentile.
    mid : str
        Value to assign if variable is less than or equal to the second percentile.
    high : str
        Value to assign if variable is greater than the second percentile.


    names : list of str
        Names used for comparison.
    
    Returns
    -------
    str
        Assigned bucket value.
    """

    if row[variable] <= row[names[1]]:
        value = low
    
    elif row[variable] <= row[names[2]]:
        value = mid
    
    elif row[variable] > row[names[2]]:
        value = high
        
    else:
        value = 'missing'
    
    return value

def allocation(variable, allocation_df, crsp_df, low, mid, high, bins, factor_type = 'FF', crit1 = 'ME', crit2 = 'ME', crit3 = 'ME'):
    """
    Allocate assets based on variable values.
    
    Parameters
    ----------
    variable : str
        Name of the variable.
    allocation_df : pandas DataFrame
        DataFrame containing allocation data.
    crsp_df : pandas DataFrame
        DataFrame containing CRSP data.
    low : str
        Value to assign if variable is less than or equal to the first percentile.
    mid : str
        Value to assign if variable is less than or equal to the second percentile.
    high : str
        Value to assign if variable is greater than the second percentile.
    bins : list of float
        List of percentiles for calculating breakpoints for allocation.
    factor_type : str, optional
        Type of factor, 'FF' (i.e. HML, CMA etc.) or 'other' (UMD, STR, LTR), defaulted to 'FF'.
    crit1 : str, optional
        Criterion 1, defaulted to 'ME'.
    crit2 : str, optional
        Criterion 2, defaulted to 'ME'.
    crit3 : str, optional
        Criterion 3, defaulted to 'ME'.
        
    NOTE: Crits all default to ME. This because all variables require ME > 0, while only some variables have differing criterias (such as BE>0)
    
    Returns
    -------
    pandas DataFrame
        DataFrame containing allocated assets.
    """

    allocation = allocation_df.copy()
    labels, names = bucket_names(variable, bins)   
    
    if factor_type == 'FF':
        allocation[variable] = np.where((allocation['age'] >= 2) &
                                        (allocation[crit1] > 0) &
                                        (allocation[crit2] > 0) &
                                        (allocation[crit3] > 0),
                                        allocation.apply(bucket, args = (variable, low, mid, high, names), axis = 1), 'missing')  
        
        allocation.loc[:, 'FF_year'] = allocation['month_end'].dt.year
        
        data = crsp_df.merge(allocation[['permno', 'FF_year', 'size', variable]], how = 'left', on = ['permno', 'FF_year'])
    
    elif factor_type == 'other':
        allocation[variable] = np.where(allocation['ME'] > 0, allocation.apply(bucket, args = (variable, low, mid, high, names), axis = 1), 'missing')
        allocation['size'] = allocation.groupby('permno')['size'].shift(1)        
        data = crsp_df.merge(allocation[['permno', 'month_end', 'size', variable]], how = 'left', on = ['permno', 'month_end'])

    data = data[(data['weight'] > 0) & (data[variable] != 'missing')]

    return data

def allocation_filtered(variable, allocation_df, crsp_df, low, mid, high, bins, factor_type = 'FF', crit1 = 'ME', crit2 = 'ME', crit3 = 'ME'):
    """
    Allocate assets based on variable values.
    
    Parameters
    ----------
    variable : str
        Name of the variable.
    allocation_df : pandas DataFrame
        DataFrame containing allocation data.
    crsp_df : pandas DataFrame
        DataFrame containing CRSP data.
    low : str
        Value to assign if variable is less than or equal to the first percentile.
    mid : str
        Value to assign if variable is less than or equal to the second percentile.
    high : str
        Value to assign if variable is greater than the second percentile.
    bins : list of float
        List of percentiles for calculating breakpoints for allocation.
    factor_type : str, optional
        Type of factor, 'FF' (i.e. HML, CMA etc.) or 'other' (UMD, STR, LTR), defaulted to 'FF'.
    crit1 : str, optional
        Criterion 1, defaulted to 'ME'.
    crit2 : str, optional
        Criterion 2, defaulted to 'ME'.
    crit3 : str, optional
        Criterion 3, defaulted to 'ME'.
        
    NOTE 1: Crits all default to ME. This because all variables require ME > 0, while only some variables have differing criterias (such as BE>0)
    NOTE 2: This function differs from allocation as it removes all rows where either of the filtering flags are activated
        
    Returns
    -------
    pandas DataFrame
        DataFrame containing filtered allocated assets.
    """
    
    allocation = allocation_df.copy()
    labels, names = bucket_names(variable, bins)   
   
    if factor_type == 'FF':
        allocation[variable] = np.where((allocation['age'] >= 2) &
                                        (allocation[crit1] > 0) &
                                        (allocation[crit2] > 0) &
                                        (allocation[crit3] > 0),
                                        allocation.apply(bucket, args = (variable, low, mid, high, names), axis = 1), 'missing')  
        
        allocation.loc[:, 'FF_year'] = allocation['month_end'].dt.year
        
        data = crsp_df.merge(allocation[['permno', 'FF_year', 'size', variable]], how = 'left', on = ['permno', 'FF_year'])
    
    elif factor_type == 'other':
        allocation[variable] = np.where(allocation[crit1] > 0, allocation.apply(bucket, args = (variable, low, mid, high, names), axis = 1), 'missing')       
        allocation['size'] = allocation.groupby('permno')['size'].shift(1)        
        data = crsp_df.merge(allocation[['permno', 'month_end', 'size', variable]], how = 'left', on = ['permno', 'month_end'])

    data = data[(data['weight'] > 0) & 
                (data[['microcap_flag', 'penny_flag', 'volume_flag']].sum(axis=1) == 0) &
                (data[variable] != 'missing')]

    return data

##############################
### TREND FACTORS (CUMRET) ###
##############################

def cumret_calc(crsp_df, cumret_window, cumret_lag, bins, filtered = 'yes'):
    """
    Calculate cumulative returns for CRSP data.
    
    Parameters
    ----------
    crsp_df : pandas DataFrame
        DataFrame containing CRSP data.
    cumret_window : int
        Rolling window for calculating cumulative returns.
    cumret_lag : int
        Lag for shifting cumulative returns.
    bins : list of float
        List of percentiles for calculating breakpoints for allocation.
    filtered : str, optional
        Whether to include filtered data, 'yes' or 'no', defaulted to 'yes'.
    
    Returns
    -------
    pandas DataFrame
        DataFrame containing calculated cumulative returns.
    """

    data = crsp_df[(crsp_df['weight'] > 0) & ((crsp_df['share_code'] == 10) | (crsp_df['share_code'] == 11))].copy().sort_values(['permno','month_end']).set_index('month_end')
    data['adj_ret'] = data['adj_ret'].fillna(0)
    data['logret'] = np.log(1+data['adj_ret'])
    
    logret = data.groupby('permno')['logret'].rolling(cumret_window, min_periods=cumret_window).sum()
    logret = logret.reset_index()   
    data = data.reset_index().drop(columns=['logret'])

    logret = pd.merge(data, logret,how='left',on=['month_end','permno'])
    logret['raw_cumret']=np.exp(logret['logret'])-1
    logret['cumret']=logret.groupby(['permno'])['raw_cumret'].shift(cumret_lag)
    logret = logret.dropna(axis=0, subset = ['cumret', 'ME'])
    
    if filtered == 'no':
        NYSE = logret[(logret['ME'] > 0) & (logret['exchange_code']==1)]
    else:
        NYSE = logret[(logret['ME'] > 0) &
                        (logret['exchange_code']==1) &
                        (logret[['microcap_flag', 'penny_flag', 'volume_flag']].sum(axis=1) == 0)]
        
        data = data[data[['microcap_flag', 'penny_flag', 'volume_flag']].sum(axis=1) == 0]
    
    data_1 = (NYSE_median_ME(NYSE)
                  .merge(NYSE_ratio('cumret', bins, NYSE), how = 'inner', on = 'month_end')
                  .merge(logret, how = 'right', on = ['month_end']))

    data_1['size'] = np.where(data_1['ME'] > 0, data_1.apply(size_bucket, axis = 1), 'missing')
    
    return data_1

def stock_mom_ports(df, rf, mom_window, mom_lag, buckets = 10, ts=0, weighting_scheme = 'VW'):
    """
    Constructs [buckets] portfolios sorted on momentum.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing data.
    rf : pandas DataFrame / Series
        1-month risk-free rate for every period.
    mom_window : int
        Size of the rolling window for calculating momentum.
    mom_lag : int
        Lag for shifting momentum.
    buckets : int, optional
        Number of buckets for sorting stocks, defaulted to 10.
    ts : int, optional
        Transaction cost, defaulted to 0.
    weighting_scheme : str, optional
        Weighting scheme for portfolio construction, 'VW' for value weighting, defaulted to 'VW'.
    
    Returns
    -------
    pandas DataFrame
        DataFrame containing constructed momentum portfolio.
    """

    mom_data = df[(df['weight'] > 0) & ((df['share_code'] == 10) | (df['share_code'] == 11))].copy().sort_values(['permno','month_end']).set_index('month_end')
    mom_data['adj_ret'] = mom_data['adj_ret'].fillna(0)
    mom_data['logret'] = np.log(1+mom_data['adj_ret'])
        
    UMDr = mom_data.groupby('permno')['logret'].rolling(mom_window, min_periods=mom_window).sum()
    UMDr = UMDr.reset_index()   
    mom_data = mom_data.reset_index().drop(columns=['logret'])

    UMDr = pd.merge(mom_data, UMDr,how='left',on=['month_end','permno'])
    UMDr['raw_cumret']=np.exp(UMDr['logret'])-1
    UMDr['cumret']=UMDr.groupby(['permno'])['raw_cumret'].shift(mom_lag)
    UMDr = UMDr.dropna(axis=0, subset = ['cumret', 'ME'])
                
    UMDr = UMDr[(UMDr[['microcap_flag', 'penny_flag', 'volume_flag']].sum(axis=1) == 0) & (UMDr['ME'] > 0)]
    UMDr['q'] = UMDr.groupby('month_end')['cumret'].transform(lambda x: pd.qcut(x, buckets, labels = [str(i) for i in range(1, buckets + 1)]))
    UMDr = UMDr[UMDr['month_end'] >= '1963-07-31']  
    
    # Value weighting
    sum_w = UMDr.groupby(['month_end', 'q'])['lag_ME'].sum().reset_index()
    UMDr = sum_w.rename(columns={'lag_ME': 'sum_weight'}).merge(UMDr, on=['month_end', 'q'], how='right')

    if weighting_scheme == 'VW':
        UMDr['weight'] = UMDr['lag_ME'] / UMDr['sum_weight']
        weight_indicator = 'weight'
    else:
        weight_indicator = None    
    
    umdports = pyanalytics.make_position(UMDr.set_index(['month_end', 'permno']), 'adj_ret', weight_indicator, pf_col='q')
    
    winners = pyanalytics.make_position(umdports[umdports['q'] == str(buckets)].reset_index().set_index(['date', 'id']), 'ret', 'wgt')
    losers = pyanalytics.make_position(umdports[umdports['q']=='1'].reset_index().set_index(['date', 'id']), 'ret', 'wgt')
    highlow = get_port_objects(pyanalytics.make_long_short_portfolio(lposition=winners, sposition=losers, rf=rf, costfcn=ts, name='Winners-Losers'))
    
    output_ports = highlow.rename('Winners-Losers').to_frame()

    for i in range(1, buckets+1):
        temp_port = pyanalytics.make_portfolio(umdports[umdports['q'] == str(i)].reset_index().set_index(['date', 'id']), 'ret', weight_col='wgt', rf=rf, costfcn=ts, name=f'{i}')
        temp_port = get_port_objects(temp_port, ret_type = 'netexret')
        output_ports = output_ports.merge(temp_port.rename(i).to_frame(), how = 'left', left_index=True, right_index=True)
    
    return output_ports

##################
### BAB FACTOR ###
##################

def BAB_factor(df, rf, mkt, filtering = 'yes', apply_tc = 'no'):
    """
    Constructs the Betting Against Beta (BAB) factor using the methodology described in AQR's Betting-Against-Beta paper.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing data.
    rf : pandas DataFrame
        Risk-free rate data.
    mkt : pandas DataFrame
        DataFrame containing the value-weighted market portfolio.
    filtering : str, optional
        Filtering option, 'yes' to apply filtering based on screens, 'no' otherwise. Defaults to 'yes'.
    
    Returns
    -------
    pandas DataFrame
        DataFrame containing the returns of the BAB factor.
    """
    
    rolling_corr_window = 60
    min_corr_window = 36
    
    rolling_std_window = 12
    min_std_window = 6
                
    mkt['std_est'] = mkt['MKTRF'].rolling(rolling_std_window).std().shift(1)   
    df = df[pd.to_numeric(df['adj_ret'], errors='coerce').notnull()]
    df = df.merge(mkt[['date', 'MKTRF', 'std_est']], on='date', how='left').merge(rf, on = 'date', how = 'left')    
    df['ret'] = df['adj_ret'] - df['rf']
    
    # Function to estimate rolling 60 month correlations with minimum 36 non-missing datapoints
    def roll_corr(x):
        return pd.DataFrame(x['ret'].rolling(rolling_corr_window, min_periods=min_corr_window).corr(x['MKTRF']))
    
    # Function to estimate rolling standard deviation over previous roll_std months
    def roll_std(x):
        return pd.DataFrame(x['ret'].rolling(rolling_std_window, min_periods=min_std_window).std())
    
    df['corr_est'] = df.groupby('permno')[['ret', 'MKTRF']].apply(roll_corr).reset_index(drop=True)
    df['corr_est'] = df.groupby('permno')['corr_est'].shift(1)
    df['permno_std_est'] = df.groupby('permno')[['ret', 'MKTRF']].apply(roll_std).reset_index(drop=True)
    df['permno_std_est'] = df.groupby('permno')[['permno_std_est']].shift(1)   
    df = df.dropna(how='any')
    
    # eq 14 - Estimation of betas
    df['beta_est'] = df['corr_est']*df['permno_std_est'].div(df['std_est'])
    
    # eq. 15 - beta shrinking
    df['beta_est'] = 0.6 * df['beta_est'] + 0.4
    
    # Filter based on screens
    if filtering == 'yes':
        df = df[(df[['microcap_flag', 'penny_flag', 'volume_flag']].sum(axis=1) == 0)]
    
    # Divide high/low based on median beta
    df['q'] = df.groupby('date')['beta_est'].transform(lambda x: pd.qcut(x, 2, labels = range(0, 2)))
    
    # eq. 16 - rank betas and calculate the weights
    df['rank'] = df.groupby('date')['beta_est'].rank()
    
    # eq. 16 - z_bar
    df['rank_avg'] = df.groupby('date')['rank'].transform('mean')

    # eq. 16 - abs(z-z_bar)
    df['weight'] = abs(df['rank']-df['rank_avg'])
    
    # eq. 16 - constant k
    df['k'] = 2/(df.groupby('date')['weight'].transform('sum')).copy()

    # eq. 16 - final weights
    df['weight'] = df['weight'] * df['k']
       
    # eq 17. - beta_H and beta_L
    df['dot'] = (df['beta_est'] * df['weight'])
    
    # eq 17. - r_H and r_L
    df['ret'] = (df['ret']*df['weight'])
    
    # eq 17. - aggregate betas and rets
    bab_beta = df.groupby(['date', 'q'])[['dot', 'ret']].sum().reset_index()

    # eq 17. - 1 / dot
    bab_beta['inv'] = 1/bab_beta['dot']
    
    # eq 17. - weight multiplied by return
    bab_beta['w_r'] = bab_beta['inv']*bab_beta['ret']
    
    # eq 17. - Long / Short positions and BAB factor
    bab = bab_beta.pivot(index = 'date', columns = 'q', values = 'w_r').rename(columns = {0: 'BABLong', 1: 'BABShort'}).reset_index()
    
    if apply_tc != 'no':    
        bab['BAB'] = (bab['BABLong'] - bab['BABShort']) - 0.002
    else:
        bab['BAB'] = bab['BABLong'] - bab['BABShort']
    return bab

##################
### PORTFOLIOS ###
##################

def market_port_class(df, weight, rf, name, apply_tc = 'no'):
    """
    Creates a market portfolio object.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing data.
    weight : str
        Column name representing the weight.
    rf : pandas DataFrame
        Risk-free rate data.
    name : str
        Name of the portfolio.
    tc : str, optional
        Apply transaction costs. Defaults to 'no'.
    
    Returns
    -------
    portfolio.Portfolio
        Market portfolio object.
    """
    if apply_tc != 'no':
        params = df[df[weight] > 0].copy().drop_duplicates(subset=['month_end', 'permno'])[['permno', 'month_end', weight, 'adj_ret']].dropna()[['permno', 'month_end']].rename(columns={'month_end':'date', 'permno':'id'})
        params['buy_linear'] = 0.0002  # Default value of 2 bps
        params['sell_linear'] = 0.002 + 0.0002 # Default value of 20 bps + Brokerage
        
        # Update transaction costs for the specified periods
        params.loc[params['date'] < '2004-12', 'buy_linear'] = 0.0004 # 4 bps
        params.loc[params['date'] < '1983-12', 'buy_linear'] = 0.0006 # 6 bps
        params.loc[params['date'] < '2004-12', 'sell_linear'] = 0.0025 + 0.0004 # 25 bps
        params.loc[params['date'] < '1983-12', 'sell_linear'] = 0.0030 + 0.0006 # 30 bps + Brokerage

        params = params.set_index(['date', 'id'])
        tc = tcost.TransactionCost(params=params)
    else:
        tc = 0     

    # Make Portfolio
    df = df[df[weight] > 0].drop_duplicates(subset=['month_end', 'permno'])[['permno', 'month_end', weight, 'adj_ret']].dropna()
    sum_w = df.groupby(['month_end'])[weight].sum().reset_index()
    df = sum_w.rename(columns={weight: 'sum_weight'}).merge(df, on=['month_end'], how='right')
    df['weight'] = df[weight] / df['sum_weight']    
    data = pyanalytics.make_position(df.set_index(['month_end', 'permno']), 'adj_ret', 'weight')

    portfolio_object = portfolio.Portfolio('MKTRF', data, rf=rf, costfcn=tc)
    return portfolio_object

def port_class(df, sort, weight, rf, name, legs = False, apply_tc = 'no', smb = 'no', weighting_scheme = 'VW'):
    """
    Creates a portfolio object.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing data.
    sort : str
        Column name for sorting.
    weight : str
        Column name representing the weight.
    rf : pandas DataFrame
        Risk-free rate data.
    name : str
        Name of the portfolio.
    legs : bool, optional
        Flag indicating whether to return legs separately. Defaults to False.
    tc : str, optional
        Apply transaction costs. Defaults to 'no'.
    smb : str, optional
        Flag indicating whether to calculate SMB factor. Defaults to 'no'.
    weighting_scheme : str, optional
        Weighting scheme for the portfolios. Defaults to 'VW'.
    
    Returns
    -------
    portfolio.Portfolio or tuple
        Portfolio object or tuple of short and long leg portfolios.
    """
    
    if apply_tc != 'no':
        params = df[(df[weight] > 0) & (df['month_end'] >= '1963-07-31')].copy().drop_duplicates(subset=['month_end', 'permno'])[['permno', 'month_end', weight, 'adj_ret', 'size', sort]].dropna()[['permno', 'month_end']].rename(columns={'month_end':'date', 'permno':'id'})
        params['buy_linear'] = 0.0002  # Default value of 2 bps
        params['sell_linear'] = 0.002 + 0.0002 # Default value of 20 bps + Brokerage
        
        # Update transaction costs for the specified periods
        params.loc[params['date'] < '2004-12', 'buy_linear'] = 0.0004 # 4 bps
        params.loc[params['date'] < '1983-12', 'buy_linear'] = 0.0006 # 6 bps
        params.loc[params['date'] < '2004-12', 'sell_linear'] = 0.0025 + 0.0004 # 25 bps
        params.loc[params['date'] < '1983-12', 'sell_linear'] = 0.0030 + 0.0006 # 30 bps + Brokerage
        params = params.set_index(['date', 'id'])
        tc = tcost.TransactionCost(params=params)
    else:
        tc = 0

    df = df[(df[weight] > 0) & (df['month_end'] >= '1963-07-31')].drop_duplicates(subset=['month_end', 'permno'])[['permno', 'month_end', weight, 'adj_ret', 'size', sort]].dropna()
    sum_w = df.groupby(['month_end', 'size', sort])[weight].sum().reset_index()
    df = sum_w.rename(columns={weight: 'sum_weight'}).merge(df, on=['month_end', 'size', sort], how='right')
    df['allocation'] = df['size'] + df[sort]

    if smb == 'no':
        if weighting_scheme == 'VW':
            df['weight'] = (df[weight] / df['sum_weight']) / 2 # 2 BUCKETS
            weight_indicator = 'weight'
        else:
            weight_indicator = None

        # Make Portfolios Based on weighting scheme. weight = None is equal-weighted
        long_ports = pyanalytics.make_position(df[df['allocation'].isin(['S3', 'B3'])].set_index(['month_end', 'permno']), 'adj_ret', weight_indicator)
        short_ports = pyanalytics.make_position(df[df['allocation'].isin(['S1', 'B1'])].set_index(['month_end', 'permno']), 'adj_ret', weight_indicator)

    else:
        if weighting_scheme == 'VW':
            df['weight'] = (df[weight] / df['sum_weight']) / 3 # 3 BUCKETS
            weight_indicator = 'weight'
        else:
            weight_indicator = None
            
        # Make Portfolios Based on weighting scheme. weight = None is equal-weighted
        long_ports = pyanalytics.make_position(df[df['allocation'].isin(['S3', 'S2', 'S1'])].set_index(['month_end', 'permno']), 'adj_ret', weight_indicator)
        short_ports = pyanalytics.make_position(df[df['allocation'].isin(['B3', 'B2', 'B1'])].set_index(['month_end', 'permno']), 'adj_ret', weight_indicator)
    
    if legs == False:
        portfolio_object = pyanalytics.make_long_short_portfolio(lposition=long_ports, sposition=short_ports, rf=rf, costfcn=tc, name=name)   
        return portfolio_object

    else:        
        long_leg = pyanalytics.make_portfolio(long_ports.reset_index().set_index(['date', 'id']), 'ret', weight_col='wgt', rf=rf, costfcn=tc, name=f'{name}_L')
        short_leg = pyanalytics.make_portfolio(short_ports.reset_index().set_index(['date', 'id']), 'ret', weight_col='wgt', rf=rf, costfcn=tc, name=f'{name}_S')
        return short_leg, long_leg

def get_port_objects(df, get_object = 'returns', logscale = False, ret_type = 'grossret', percentage=True):
    """
    Returns the desired data from the portfolio object.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame containing portfolio data.
    get_object : {'returns', 'performance', 'summary'}, optional
        Type of data to retrieve. Defaults to 'returns'.
    logscale : bool, optional
        Flag indicating whether to use log scale. Defaults to False.
    ret_type : {'grossret', 'netret'}, optional
        Type of return. Defaults to 'grossret'.
    percentage : bool, optional
        Flag indicating whether to return data in percentages or absolute values. Defaults to True.
    
    Returns
    -------
    object
        Requested data from the portfolio object.
    """

    if get_object == 'returns':
        return df.eval(logscale=False, return_type=ret_type, annualize_factor=12, percentage=percentage)[1][ret_type]
    
    elif get_object == 'performance':
        return df.eval(logscale=False, return_type=ret_type, annualize_factor=12, percentage=percentage)[1]
    
    elif get_object == 'summary':
        return df.eval(logscale=False, return_type=ret_type, annualize_factor=12, percentage=percentage)[0]
    
    else:
        raise KeyError("Insert valid return object")
        
############################
### TRANSACTIONS COST ######
############################      
        
def additional_net(factor_list, additional_factors):
    test = factor_list.copy()
    tc_cost = test[0]['grossexret'] - test[0]['netexret']
    tc_cost = tc_cost.to_frame()
            
    for i in range(1, len(test)):
        tc_temp = test[i]['grossexret'] - test[i]['netexret'] 
        tc_temp.name = i   
        tc_temp.index = pd.to_datetime(tc_temp.index)
        tc_cost = tc_cost.merge(tc_temp, left_index=True, right_index=True, how = 'left')
    tc_cost_avg = tc_cost.mean(axis=1).to_frame()
    add_fac = additional_factors.copy().set_index('date')
            
    for i in add_fac.columns:
        add_fac[i] = np.where(add_fac[i].notna(), add_fac[i].subtract(tc_cost_avg[0]), np.nan)
    add_fac = add_fac.reset_index()
    return add_fac
        
def gen_tc_df(factor_list, BAB_cost):
    tc_cost = factor_list[0]['grossexret'] - factor_list[0]['netexret']
    tc_cost = tc_cost.to_frame()
                        
    for i in range(1, len(factor_list)):
        tc_temp = factor_list[i]['grossexret'] - factor_list[i]['netexret'] 
        tc_temp.name = i   
        tc_temp.index = pd.to_datetime(tc_temp.index)
        tc_cost = tc_cost.merge(tc_temp, left_index=True, right_index=True, how = 'left')
    tc_cost.columns = ['SMB', 'HML', 'UMD', 'RMW', 'CMA', 'ACC', 'NSI', 'STR', 'LTR', 'EP', 'CFP']
    tc_cost_avg = tc_cost[['SMB', 'HML', 'RMW', 'CMA', 'ACC', 'NSI', 'EP', 'CFP']].mean(axis=1).to_frame()   
    tc_cost['QMJ'] = tc_cost_avg
    tc_cost['LIQ'] = tc_cost_avg
    tc_cost['RES'] = tc_cost_avg
    tc_cost = tc_cost.merge(BAB_cost, left_index=True, right_index=True, how = 'left')
    return tc_cost
        
############################
### WRITE TO EXCEL SHEET ###
############################

def write_excel(filename , sheetname, dataframe):
    """
    Write a DataFrame to an Excel file.
    
    Parameters
    ----------
    filename : str
        Name of the Excel file.
    sheetname : str
        Name of the sheet within the Excel file.
    dataframe : DataFrame
        DataFrame to be written to the Excel file.
    
    Returns
    -------
    None (prints whether worksheet exists)
    """

    with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer: 
        workBook = writer.book
        try:
            workBook.remove(workBook[sheetname])
        except:
            print(f'Worksheet created: {sheetname}')
        finally:
            dataframe.to_excel(writer, sheet_name=sheetname, index=True)