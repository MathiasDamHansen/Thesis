# -*- coding: utf-8 -*-
"""
Data Statistics

@author: Mathias Dam Hansen
"""

import pandas as pd
import numpy as np
import Model_CRSP.__factor_functions__ as fcs

################################
### COUNT NO STOCKS AND ROWS ###
################################

def stock_number_time(label, df):
    """
    Calculate the number of unique permnos over time.
    
    Parameters
    ----------
    label : str
        Label for the column representing the count of unique permnos.
    df : pandas DataFrame
        DataFrame containing the data to calculate permno numbers.
    
    Returns
    -------
    pandas DataFrame
        DataFrame containing the count of unique permnos over time.
    """

    stocks_over_time = df.groupby('date')['permno'].nunique().reset_index().rename(columns = {'permno': label})   
    return stocks_over_time

def stock_number_total(label, df):
    """
    Calculate the total number of unique permnos for the entire sample.
    
    Parameters
    ----------
    label : str
        Label for the total count of unique permnos.
    df : pandas DataFrame
        DataFrame containing the data to calculate permno numbers.
    
    Returns
    -------
    tuple
        Tuple containing the label and the total count of permnos stocks.
    """

    stocks_total = df['permno'].nunique()
    stock_tuple = (label, stocks_total)
    return stock_tuple 

def row_count(label, df):
    """
    Calculate the total number of rows in the DataFrame.
    
    Parameters
    ----------
    label : str
        Label for the total count of rows.
    df : pandas DataFrame
        DataFrame containing the data to count rows.
    
    Returns
    -------
    tuple
        Tuple containing the label and the total count of rows.
    """

    row_count = df.shape[0]
    row_tuple = (label, row_count)

    return row_tuple

#######################
### DELISTING STATS ###
#######################

def delisting_stats(ret_df, delist_df, output_file):
    """
    Compute statistics related to delistings and write them to an Excel file (output_file).
    
    Parameters
    ----------
    ret_df : pandas DataFrame
        DataFrame containing the returns data.
    delist_df : pandas DataFrame
        DataFrame containing the delisting data.
    output_file : str
        Path to the output Excel file.
    
    Returns
    -------
    pandas DataFrame
        DataFrame containing delisting statistics.
    """

    ret_df.loc[:, 'permno'] = ret_df['permno'].astype(int)
    delist_df.loc[:, 'permno'] = delist_df['permno'].astype(int)
    
    delistings = delist_df[delist_df['permno'].isin(ret_df['permno'].unique())].dropna()
    delistings = delistings[(delistings['delist_ret'].notna()) & (delistings['delist_ret'] != 0) & (delistings['delist_code'] != 100)]
    delistings = delistings.merge(ret_df[['permno', 'month_end', 'ME']], on = ['permno', 'month_end'], how = 'left')
    
    mergers = delistings[delistings['delist_code'].astype(str).str[0] == '2'].shape[0]
    exchange = delistings[delistings['delist_code'].astype(str).str[0] == '3'].shape[0]
    liquidations = delistings[delistings['delist_code'].astype(str).str[0] == '4'].shape[0]
    dropped = delistings[delistings['delist_code'].astype(str).str[0] == '5'].shape[0]
    expirations = delistings[delistings['delist_code'].astype(str).str[0] == '6'].shape[0]
    foreign_listing = delistings[delistings['delist_code'].astype(str).str[0] == '9'].shape[0]  
    
    delisting_stats = pd.DataFrame({'delistings': delistings['permno'].nunique(),
                                    'avg. delisting ret': delistings['delist_ret'].mean(),
                                    'count_pos_ret': delistings[delistings['delist_ret'] > 0].shape[0],
                                    'count_neq_ret': delistings[delistings['delist_ret'] < 0].shape[0],
                                    'avg_pos_ret': delistings[delistings['delist_ret'] > 0]['delist_ret'].mean(),
                                    'avg_neq_ret': delistings[delistings['delist_ret'] < 0]['delist_ret'].mean(),
                                    'mergers': mergers,
                                    'exchange_change': exchange,
                                    'liquidation': liquidations,
                                    'dropped': dropped,
                                    'expirations': expirations,
                                    'foreign_listing': foreign_listing}, index = ['index'])
    
    fcs.write_excel(output_file, 'delisting_data', delisting_stats)
    
    return delisting_stats

#################
### SIC STATS ###
#################

def sic_stats(crsp_data, output_file):
    """
    Calculate statistics related to SIC codes and write them to output_file.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the SIC code data.
    output_file : str
        Path to the output Excel file.
    
    Returns
    -------
    pandas DataFrame
        DataFrame containing SIC code statistics.
        
    """
       
    df = crsp_data.copy()
    agri = df[df['SIC_code'].astype(str).str.len() == 3]
    df = df[df['SIC_code'].astype(str).str.len() == 4]

    # SPLIT FIRST AND SECOND DIGIT
    df['first_digit'] = df['SIC_code'].astype(str).str[0].astype(int)
    df['second_digit'] = df['SIC_code'].astype(str).str[1].astype(int)

    # DEFINE SIC GROUPS
    mining = df[(df['first_digit'] == 1) & (df['second_digit'].between(0, 4))]
    construction = df[(df['first_digit'] == 1) & (df['second_digit'].between(5, 6))]
    manufacturing = df[df['first_digit'].isin([2, 3])]
    trans_publicutil = df[df['first_digit'] == 4]
    wholesale = df[(df['first_digit'] == 5) & (df['second_digit'].isin([0, 1]))]
    retail = df[(df['first_digit'] == 5) & (df['second_digit'] > 1)]
    finance = df[(df['first_digit'] == 6) & (df['second_digit'].between(0, 7))]
    services = df[df['first_digit'].isin([7, 8])]
    public_admin = df[df['first_digit'] == 9]
   
    sic_stats = pd.DataFrame({'agri': agri['permno'].nunique(),
                              'mining': mining['permno'].nunique(),
                              'construction': construction['permno'].nunique(),
                              'manufacturing': manufacturing['permno'].nunique(),
                              'trans_publicutil': trans_publicutil['permno'].nunique(),
                              'wholesale': wholesale['permno'].nunique(),
                              'retail': retail['permno'].nunique(),
                              'finance': finance['permno'].nunique(),
                              'services': services['permno'].nunique(),
                              'public_admin': public_admin['permno'].nunique()}, index = ['index'])   
    fcs.write_excel(output_file, 'Figure 9', sic_stats)
   
    return sic_stats

#################
### ME filter ###
#################

def ME_review(df, output_file):
    """
    Compute median and mean ME and write to output_file.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data to analyse.
    output_file : str
        Path to the output Excel file.
    
    Returns
    -------
    None
    """
    
    ME_filter = df.copy()[['month_end', 'ME', 'microcap_excl', 'exchange_code', 'microcap_incl', 'microcap_flag']]
    
    ME_mean = (ME_filter
               .groupby('month_end')['ME']
               .mean()
               .to_frame()
               .rename(columns = {'ME': 'ME_mean'}))
    
    ME_median = (ME_filter
                 .groupby('month_end')['ME']
                 .median()
                 .to_frame()
                 .rename(columns = {'ME': 'ME_median'}))
    
    ME_stats = ME_mean.merge(ME_median, left_index=True, right_index=True)

    fcs.write_excel(output_file, 'Figure 6', ME_stats)

def ME_filter_review(df, output_file):
    """
    Compute summary statistics on the effect of the ME filter and write to output_file.
    
    Specifically, compute which exchanges are affected, the percentage and number of exclusions from each exchange, and display the result in a DataFrame.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data to analyse.
    output_file : str
        Path to the output Excel file.
    
    Returns
    -------
    None
    """
    
    ME_filter = df[['month_end', 'ME', 'microcap_excl', 'exchange_code', 'microcap_incl', 'microcap_flag']].copy()
    
    ME_filter = ME_filter.dropna(subset = 'microcap_excl')
    exclusions = ME_filter[ME_filter['microcap_flag'] == 1]
    exclusions = exclusions.groupby('month_end').agg(count = ('microcap_flag', 'count'), microcap_excl = ('microcap_excl', 'first'), microcap_incl = ('microcap_incl', 'first')).astype(int)
    
    exchange_exclusions = pd.DataFrame().reindex(exclusions.index)
    exchange_tot_stocks = pd.DataFrame().reindex(exclusions.index)
    exchange_exclusions_pct = pd.DataFrame().reindex(exclusions.index)
    
    for i in [1, 2, 3]:
        subset = ME_filter[ME_filter['exchange_code'] == i]
        subset_count = subset.groupby('month_end')['microcap_flag'].count().astype(int)
        subset_sum = subset.groupby('month_end')['microcap_flag'].sum().astype(int)
        exchange_exclusions[i] = subset_sum
        exchange_tot_stocks[i] = subset_count
        exchange_exclusions_pct[i] = subset_sum / subset_count.astype(float)       
    
    exchange_exclusions = exchange_exclusions.rename(columns = {1: 'NYSE_excl',
                                                                2: 'AMEX_excl',
                                                                3: 'NASDAQ_excl'})
    
    exchange_tot_stocks = exchange_tot_stocks.rename(columns = {1: 'NYSE_sum',
                                                                2: 'AMEX_sum',
                                                                3: 'NASDAQ_sum'})
    
    exchange_exclusions_pct = exchange_exclusions_pct.rename(columns = {1: 'NYSE_excl_pct',
                                                                        2: 'AMEX_excl_pct',
                                                                        3: 'NASDAQ_excl_pct'})

    ME_filter_review = (exclusions
                        .merge(exchange_exclusions, left_index=True, right_index=True)
                        .merge(exchange_tot_stocks, left_index=True, right_index=True)
                        .merge(exchange_exclusions_pct, left_index=True, right_index=True)).rename(columns = {'count': 'total exclusions'})

    fcs.write_excel(output_file, 'Figure 12', ME_filter_review[ME_filter_review.index >= '1963-06-30'])

def microcap_ME(df, output_file):
    """
    Compute summary statistics on the effect of the ME filter and write to output_file.
    
    Specifically, compute which exchanges are affected, the percentage and number of exclusions from each exchange, and display the result in a DataFrame.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data to analyse.
    output_file : str
        Path to the output Excel file.
    
    Returns
    -------
    None
    """

    ME_df = (df.groupby('month_end')['ME'].sum().to_frame().rename(columns={'ME': 'ME_sum_incl'})
             .merge(df.groupby('month_end')['permno'].nunique().to_frame().rename(columns={'permno': 'permno_count_incl'}), left_index=True,right_index=True)
             .merge(df[df['microcap_flag'] == 0].groupby('month_end')['ME'].sum().to_frame().rename(columns={'ME': 'ME_sum_excl'}), left_index=True,right_index=True)
             .merge(df[df['microcap_flag'] == 0].groupby('month_end')['permno'].nunique().to_frame().rename(columns={'permno': 'permno_count_excl'}), left_index=True,right_index=True))
             
    ME_df['smallcaps_%total_stocks'] = abs(ME_df['permno_count_excl'] / ME_df['permno_count_incl'] - 1) * 100
    ME_df['smallcaps_%total_marketcap'] = abs(ME_df['ME_sum_excl'] / ME_df['ME_sum_incl'] - 1) * 100
    fcs.write_excel(output_file, 'misc', ME_df[ME_df.index >= '1963-06-30'][['smallcaps_%total_stocks', 'smallcaps_%total_marketcap']].mean())

####################
### MAX DRAWDOWN ###
####################

def max_drawdown(ret):
    """
    Computes the maximum drawdown of each ticker in the input data and when these events occurred.
    
    Parameters
    ----------
    ret : pandas DataFrame or pandas Series
        Returns data as a DataFrame or a Series. The returns data can be in any time frame such as daily, monthly, etc...
    
    Returns
    -------
    Tuple
        A tuple containing two items:
        - maximum drawdown: Series if ret is a DataFrame, Scalar if ret is a Series.
        - index of the max drawdown occurrence: Series if ret is a DataFrame, Scalar if ret is a Series.
    """
    
    wealth_index = (1 + ret).cumprod()
    peak = wealth_index.cummax() 
    drawdown = (wealth_index - peak) / peak
    return round(drawdown.min(), 4), drawdown.idxmin()

######################################
### Historical Value-at-Risk (VaR) ###
######################################
        
def VaR_historical(ret, alpha = 5):
    """
    Computes the 1-period Value at Risk using historical data, at a specified confidence level.
    
    Parameters
    ----------
    ret : pandas DataFrame or pandas Series
        Returns data as a DataFrame or a Series. The returns data can be in any time frame such as daily, monthly, etc...
    alpha : int
        Confidence level in integer. 5% should be input as 5.
    
    Returns
    -------
    Scalar or Series
        Value at risk: Scalar for Series input, Series for DataFrame input.
    """
    
    return -ret.quantile(alpha/100)

##############################################################
### AVERAGE SUMMARY STATISTICS ACROSS ALL STOCKS / PERMNOS ###
##############################################################

def summary_stats_grouped(ret, freq): 
    """
    Calculate summary statistics for each permno in the returns data. Display the average summary statistic for all permnos in a DataFrame.
    
    Parameters
    ----------
    ret : pandas DataFrame or Series
        DataFrame or Series containing the returns data.
    freq : int
        Frequency of the returns in terms of the number of periods per year.
    
    Returns
    -------
    pandas DataFrame
        DataFrame containing summary statistics.
    """

    grouped = ret.groupby('permno')
    
    if isinstance(ret, pd.Series):
        ret = pd.DataFrame(ret)
               
    ann_ret = grouped.apply(lambda x: x.mean()*freq, include_groups = False)
    ann_vol = grouped.apply(lambda x: x.std()*np.sqrt(freq), include_groups = False)
    avg_ret = grouped.apply(lambda x: x.mean(), include_groups = False)
    avg_std = grouped.apply(lambda x: x.std(), include_groups=False)
    skew = grouped.apply(lambda x: x.skew(), include_groups=False)
    kurt = grouped.apply(lambda x: x.kurt(), include_groups=False)
    var_hist = grouped.apply(lambda x: VaR_historical(x), include_groups=False)
    maxDD = grouped.apply(lambda x: max_drawdown(x)[0], include_groups=False)

    result = pd.concat([ann_ret, ann_vol, avg_ret, avg_std, skew, kurt, var_hist, maxDD], axis=1)
    result.columns = ['Annualised ret', 'Annualised vol', 'Monthly ret', 'Monthly vol', 'Skewness', 'Kurtosis', 'Hist VaR', 'Max drawdown']
    
    return result.mean()