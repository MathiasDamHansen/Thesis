# -*- coding: utf-8 -*-
"""
This document runs the first part of the code related to data screening and factor construction

@author: Mathias Dam Hansen
"""

import os
import time
import pandas as pd
import numpy as np
import datetime as dt
from pandas.tseries.offsets import YearEnd, MonthEnd, MonthBegin
import scipy.stats as stats

#############
### PATHS ###
#############

# DIRECTORY
directory = r'C:\Users\mathi\Desktop\Thesis model'
os.chdir(directory)

# INTERMEDIATE DATA FILE (FACTOR PORTFOLIOS --> FACTOR MOMENTUM PORTFOLIOS)
data_file = 'Intermediate_data.xlsx'

# FINAL OUTPUT FILE
output_file = 'Tables&Figures.xlsx'

######################
### CUSTOM MODULES ###
######################

import supporting_modules.__data_filtering__ as filtering
import supporting_modules.__port_stats__ as port_stats
import supporting_modules.__factor_functions__ as fcs
import supporting_modules.__data_stats__ as data_stats

########################
### TIMING FUNCTIONS ###
########################

def time_read():
    return time.strftime("%H:%M:%S", time.localtime(time.time()))

################
### SETTINGS ###
################

pd.set_option('display.max_columns', 5)
import warnings
warnings.filterwarnings('ignore')
bins = [0.3, 0.7]

#################
### LOAD DATA ###
#################

print('INITIATE PART ONE OF THE MODEL. ESTIMATED TIME IS 40 MINS:', time_read())

compustat = pd.read_csv(os.path.join(directory, 'import_files/compustat.csv'), index_col = 0, header = 0)
crsp_ret = pd.read_csv(os.path.join(directory, 'import_files/crsp_ret.csv'), index_col = 0, header = 0)
crsp_delist = pd.read_csv(os.path.join(directory, 'import_files/crsp_delist.csv'), index_col = 0, header = 0)
crsp_compustat_link = pd.read_csv(os.path.join(directory, 'import_files/crsp_compustat.csv'), index_col = 0, header = 0)

# format date
compustat['date'] = pd.to_datetime(compustat['date'])
crsp_ret['date'] = pd.to_datetime(crsp_ret['date'])
crsp_delist['delist_date'] = pd.to_datetime(crsp_delist['delist_date'])
crsp_compustat_link['link_date_first'] = pd.to_datetime(crsp_compustat_link['link_date_first'])
crsp_compustat_link['link_date_last'] = pd.to_datetime(crsp_compustat_link['link_date_last'])

# format merge elements
crsp_ret[['permco', 'permno', 'share_code', 'exchange_code']] = crsp_ret[['permco', 'permno', 'share_code', 'exchange_code']].astype(int)
crsp_delist['permno'] = crsp_delist['permno'].astype(int)

####################
### MERGING DATA ###
####################

# Align dates
compustat['month_end'] = compustat['date'] + MonthEnd(0)
crsp_ret['month_end'] = crsp_ret['date'] + MonthEnd(0)
crsp_delist['month_end'] = crsp_delist['delist_date'] + MonthEnd(0)
crsp_delist['delist_month_end'] = crsp_delist['delist_date'] + MonthEnd(0)
crsp_compustat_link['link_date_first'] = crsp_compustat_link['link_date_first'] + MonthEnd(0)
crsp_compustat_link['link_date_last'] = crsp_compustat_link['link_date_last'] + MonthEnd(0)

# Merge price and delisting data
crsp_data = crsp_ret.merge(crsp_delist, how = 'left', on = ['permno', 'month_end'])

#########################
### Delisting returns ###
#########################

# Adjust for delistings
crsp_data['ret'] = np.where(crsp_data['ret'] == -0.99, np.nan, crsp_data['ret'])
crsp_data['adj_ret'] = (1 + crsp_data['ret'].fillna(0)) * (1 + crsp_data['delist_ret'].fillna(0)) - 1

###################
### BOOK EQUITY ###
###################

# Preferred Equity: Redemption, liquidation, or par value (in that order)
compustat['PE'] = np.where(compustat['pref_redemption'].isnull(), compustat['pref_liquidation'], compustat['pref_redemption'])
compustat['PE'] = np.where(compustat['PE'].isnull(), compustat['pref_par'], compustat['PE'])
compustat['PE'] = np.where(compustat['PE'].isnull(), 0, compustat['PE'])   
    
# Tax deferral https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
compustat['tax_deferral'] = np.where(compustat['tax_inv_credit'].isnull(), compustat['tax_deferral_BS'].fillna(0) + compustat['inv_tax_credit'].fillna(0), compustat['tax_inv_credit'])
compustat['tax_deferral'] = np.where(compustat['tax_deferral'].isnull(), 0, compustat['tax_deferral'])

# Book Value of Equity: stockholder_equity or common_equity (in that order)
compustat['BVE_prox'] = np.where(compustat['stockholder_equity'].isnull(), compustat['stockholder_equity'] - compustat['PE'], compustat['common_equity'])
compustat['BVE_prox'] = np.where(compustat['BVE_prox'].isnull(), compustat['Assets'] - compustat['Liabilities'], compustat['BVE_prox'])

compustat['BE'] = np.where(compustat['fiscal_year'] < 1993, compustat['BVE_prox'] + compustat['tax_deferral'] - compustat['PE'], compustat['BVE_prox'] + compustat['PE'])
compustat['BE'] = np.where(compustat['BE'] > 0, compustat['BE'], 0)

#####################
### MARKET EQUITY ###
#####################

# FREEFLOW ME FOR EACH PERMNO
crsp_data['ME'] = crsp_data['price'].abs() * crsp_data['shares_outstanding']

# MAX ME ACROSS PERMNOS WITH SAME PERMCO
ME_max = crsp_data.groupby(['month_end','permco'])['ME'].max().reset_index()

# SUM ME ACROSS PERMNOS WITH SAME PERMCO
ME_sum = crsp_data.groupby(['month_end', 'permco'])['ME'].sum().reset_index()

# MERGE MAX & SUM
crsp_data = (crsp_data
             .merge(ME_max, how = 'inner', on = ['month_end', 'permco', 'ME']) # MERGE WITH CRSP DATA (INNER: DELETES PERMNOS != MAX ME)
             .drop(['ME'], axis = 1)
             .merge(ME_sum, how = 'inner', on = ['month_end','permco']) # MERGE WITH MAX ME DATA (LEAVING ONLY MAJOR PERMNO WITH ME-SUM OF ALL PERMNO BELONGING TO A PERMCO)
             .sort_values(by = ['permno', 'month_end']) # SORT AND DROP POTENTIAL DUPLICATES
             .drop_duplicates()
             )

del ME_max, ME_sum

# REMOVE NON-COMMON STOCK
crsp_data = crsp_data[crsp_data['share_code'].isin([10, 11])]

# DATA STATS - INPUT FOR WATERFALL CHART
label = 'Common Stock'
stocks_total = [data_stats.stock_number_total(label, crsp_data)]
row_total = [data_stats.row_count(label, crsp_data)]
stocks_time = data_stats.stock_number_time(label, crsp_data)

#######################
### DECEMBER VALUES ###
#######################

# ALIGN DECEMBER VALUES WITH JULY-JUNE DATE
crsp_data['year'] = crsp_data['month_end'].dt.year
crsp_data_dec = crsp_data[crsp_data['month_end'].dt.month == 12].rename(columns = {'ME': 'ME_dec'})
crsp_data_dec['year'] = crsp_data_dec['year'] + 1
crsp_data_dec = crsp_data_dec[['permno', 'year', 'ME_dec', 'shares_outstanding']]

# JULY-JUNE DATES
crsp_data['FF_date'] = crsp_data['month_end'] + MonthEnd(-6)
crsp_data['FF_year'] = crsp_data['FF_date'].dt.year
crsp_data['FF_month'] = crsp_data['FF_date'].dt.month

############################################
### CUMULATIVE RETURNS AND MARKET EQUITY ###
############################################

# CUM RET (NO DIV)
crsp_data['1+retx'] = 1 + crsp_data['retx']
crsp_data = crsp_data.sort_values(by = ['permno', 'month_end'])

# CUMRET, LAG CUMRET AND LAG ME
crsp_data['cumulative_retx'] = crsp_data.groupby(['permno', 'FF_year'])['1+retx'].cumprod()
crsp_data['lag_cumulative_retx'] = crsp_data.groupby(['permno'])['cumulative_retx'].shift(1)
crsp_data['lag_ME'] = crsp_data.groupby(['permno'])['ME'].shift(1)

# LAG ME: IF FIRST YEAR OF PERMNO, USE DISCOUNTED ME (ME/(1 + retx)), ELSE USE LAGGED ME
crsp_data['lag_ME'] = np.where(crsp_data.groupby(['permno']).cumcount() == 0, crsp_data['ME'] / crsp_data['1+retx'], crsp_data['lag_ME'])

# BASE ME
base_ME = crsp_data[crsp_data['FF_month'] == 1][['permno', 'FF_year', 'lag_ME']].rename(columns = {'lag_ME': 'base_ME'})

# MERGE RESULTS BACK
crsp_data = crsp_data.merge(base_ME, how = 'left', on = ['permno', 'FF_year'])

# WEIGHT IN ANNUALLY ALLOCATED PORTFOLIOS IS LAGGED ME
crsp_data['weight'] = np.where(crsp_data['FF_month'] == 1, crsp_data['lag_ME'], crsp_data['base_ME'] * crsp_data['lag_cumulative_retx'])

del base_ME

# COPY CRSP_DATA WITH NO FILTERS FOR STATS ON THE RAW SAMPLE
crsp_data_nofilters_long = crsp_data[['permno', 'date', 'share_code', 'exchange_code', 'month_end', 'ME', 'lag_ME', 'weight', 'adj_ret']].copy()
crsp_data_nofilters_norm = crsp_data[crsp_data['month_end'] > '01-06-1962'][['permno', 'permco', 'date', 'share_code', 'exchange_code', 'SIC_code', 'year', 'month_end', 'ME', 'lag_ME', 'weight', 'adj_ret', 'FF_year', 'FF_month']].copy()

###############
### FILTERS ###
###############

print('FILTERS:', time_read())

# ME filter
ME_decile = filtering.ME_filter(crsp_data[['month_end', 'ME', 'exchange_code']])
crsp_data = crsp_data.merge(ME_decile[['month_end', 'microcap_excl', 'microcap_incl']], on = 'month_end', how = 'left')
crsp_data = crsp_data.groupby('permno').apply(filtering.ME_flag, include_groups = False).reset_index().drop('level_1', axis = 1)

# WRITE STATS ON ME TO OUTPUT FILE
data_stats.microcap_ME(crsp_data.copy(), output_file)
data_stats.ME_filter_review(crsp_data.copy(), output_file)

# Penny filter
penny_threshold = 1
crsp_data = filtering.penny_filter(crsp_data, penny_threshold)

# AMV filter
AMV_threshold = 0.05
AMV_window = 12
crsp_data = filtering.volume(crsp_data, AMV_window, AMV_threshold)

# Extreme return
max_return = 9
crsp_data = filtering.extreme_return(crsp_data, max_return)

# Extreme return correction
upper_limit = 3
lower_limit = -0.5
crsp_data = crsp_data.groupby('permno').apply(filtering.extreme_return_correction, upper_limit, lower_limit, include_groups=False).reset_index().drop('level_1', axis = 1)

# Zero return filter
crsp_data = crsp_data.groupby('permno').apply(filtering.zero_pct_returns_filter, include_groups=False).reset_index().drop('level_1', axis = 1)

crsp_data_long = crsp_data[['permno', 'date', 'share_code', 'exchange_code', 'volume_flag', 'microcap_flag', 'penny_flag', 'month_end', 'ME', 'lag_ME', 'weight', 'adj_ret']].copy()
crsp_data = crsp_data[crsp_data['month_end'] > '01-12-1962'][['permno', 'permco', 'date', 'share_code', 'exchange_code', 'SIC_code', 'volume_flag', 'microcap_flag', 'penny_flag', 'extreme_ret_correction_flag', 'extreme_ret_flag', 'year', 'month_end', 'ME', 'lag_ME', 'weight', 'adj_ret', 'FF_year', 'FF_month']]

#########################
### EFFECT OF FILTERS ###
#########################

print('EFFECT OF FILTERS:', time_read())

### CALCULATE EFFECT OF EACH FILTER ###
effect = crsp_data[crsp_data['month_end'] > '01-06-1963'].copy()

# delisting_stats
crsp_data_full_sample = crsp_data.copy()
crsp_data_delistings = crsp_delist.copy()
sample_filtered = crsp_data_full_sample[(crsp_data_full_sample['microcap_flag'] == 0) & (crsp_data_full_sample['penny_flag'] == 0)]
delisting_output = data_stats.delisting_stats(crsp_data_full_sample, crsp_data_delistings, output_file)

effect['overlap'] = np.where((effect['microcap_flag'] == 1) & (effect['penny_flag'] == 1), 1, 0)
tot_overlap = effect['overlap'].sum() / effect['microcap_flag'].sum()

# ME
effect = effect[effect['microcap_flag'] == 0]
label = 'Microcap Stocks'
stocks_total.append(data_stats.stock_number_total(label, effect))
row_total.append(data_stats.row_count(label, effect))
stocks_time = stocks_time.merge(data_stats.stock_number_time(label, effect), on = 'date')

# Volume
effect = effect[effect['volume_flag'] == 0]
label = 'Volume'
stocks_total.append(data_stats.stock_number_total(label, effect))
row_total.append(data_stats.row_count(label, effect))
stocks_time = stocks_time.merge(data_stats.stock_number_time(label, effect), on = 'date')

# Penny
effect = effect[effect['penny_flag'] == 0]
label = 'Penny Stocks'
stocks_total.append(data_stats.stock_number_total(label, effect))
row_total.append(data_stats.row_count(label, effect))
stocks_time = stocks_time.merge(data_stats.stock_number_time(label, effect), on = 'date')

# Delistings
label = 'Delistings'
stocks_total.append(data_stats.stock_number_total(label, effect))
row_total.append(data_stats.row_count(label, effect))
stocks_time = stocks_time.merge(data_stats.stock_number_time(label, effect), on = 'date')

# Data availability
effect = filtering.min_obs(effect, 12)
label = 'Observations'
stocks_total.append(data_stats.stock_number_total(label, effect))
row_total.append(data_stats.row_count(label, effect))
stocks_time = stocks_time.merge(data_stats.stock_number_time(label, effect), on = 'date')

data_stats.summary_stats_grouped(effect[effect['month_end']>'01-07-1963'][['permno', 'adj_ret']], 12)
data_stats.summary_stats_grouped(crsp_data[['permno', 'adj_ret']], 12)

# Extreme ret flag
label = 'Extreme ret'
stocks_total.append(data_stats.stock_number_total(label, effect))
row_total.append(data_stats.row_count(label, effect))
stocks_time = stocks_time.merge(data_stats.stock_number_time(label, effect), on = 'date')

### CALCULATE SUMMARY AND SIC STATS ###
sample_stats_filtered = data_stats.summary_stats_grouped(effect[effect['month_end']>'01-07-1963'][['permno', 'adj_ret']], 12)
sample_stats_raw = data_stats.summary_stats_grouped(crsp_data[['permno', 'adj_ret']], 12)

data_stats.sic_stats(effect, output_file)

########################################
### Loading link data CRSP/COMPUSTAT ###
########################################

# Years in Compustat
compustat = compustat.sort_values(by = ['gvkey', 'month_end'])
compustat['age'] = compustat.groupby(['gvkey']).cumcount()

# Sets link date last to today if security is still in compustat
crsp_compustat_link['link_date_last'] = crsp_compustat_link['link_date_last'].fillna(pd.to_datetime('today'))

# Merge COMPUSTAT and CRSP
crsp_compustat = compustat.merge(crsp_compustat_link, how = 'left', on = ['gvkey'])
crsp_compustat['month_end'] = crsp_compustat.apply(filtering.JuneScheme, axis = 1)

# Link date bounds
crsp_compustat = crsp_compustat[(crsp_compustat['month_end'] >= crsp_compustat['link_date_first']) & (crsp_compustat['month_end'] <= crsp_compustat['link_date_last'])]
crsp_compustat = crsp_compustat.drop(['date', 'pref_liquidation', 'pref_redemption', 'pref_par', 'fiscal_year', 'common_equity', 'stockholder_equity', 'PE', 'BVE_prox', 'linktype', 'linkprim', 'link_date_first', 'link_date_last'], axis = 1)

###############
### LOAD RF ###
###############

RF = pd.read_excel(os.path.join(directory, 'import_files/RF.xlsx'), header = 0).set_index('date').astype(float).reset_index()
RF['date'] = pd.to_datetime(RF['date'], format = "%Y%m") + MonthEnd(0)
RF.loc[:, 'rf'] = RF['rf'].astype(float)
RF = RF[['date', 'rf']].set_index('date')

# Additional factors (QMJ, LIQ, RES)
additional_factors = pd.read_excel(os.path.join(directory, 'import_files/external_factors.xlsx'), header = 0).set_index('date').astype(float).reset_index()
additional_factors['date'] = pd.to_datetime(additional_factors['date'], format = "%Y%m") + MonthEnd(0)

################################################################
### SPLIT REMAINING FUNCTIONS INTO FILTERED AND NON-FILTERED ###
################################################################

def raw_data_handling(crsp_data, crsp_data_long, crsp_compustat, output_file, data_file):
    """
    Compute all data related to the raw sample and write to the relevant output file
    
    Parameters
    ----------
    crsp_data : pandas DataFrame
        DataFrame containing the crisp_data with filtering columns (flags)
    crsp_compustat : pandas DataFrame
        DataFrame containing the crsp_compustat data
    
    Returns
    -------
    None
    """

    #################
    ### JUNE DATA ###
    #################

    # Merge June and December data
    crsp_compustat_jun = (crsp_data[crsp_data['month_end'].dt.month == 6]
                          .merge(crsp_data_dec, how = 'inner', on = ['permno', 'year'])
                          .sort_values(by = ['permno', 'month_end'])
                          .drop_duplicates()
                          .merge(crsp_compustat, how = 'inner', on = ['permno', 'month_end']))

    ###############
    ### FACTORS ###
    ###############

    # BookRatio, BM
    crsp_compustat_jun['BM'] = np.where(
        (crsp_compustat_jun['BE'] > 0) &
        (crsp_compustat_jun['ME'] > 0) &
        (crsp_compustat_jun['ME_dec'] > 0),
        crsp_compustat_jun['BE'] * 1000 / crsp_compustat_jun['ME_dec'], np.nan)

    # Operating Profitability, OP
    crsp_compustat_jun['xp_allnan'] = (crsp_compustat_jun['COGS'].isnull()) & (crsp_compustat_jun['SGA'].isnull()) & (crsp_compustat_jun['I'].isnull())
    crsp_compustat_jun['profit'] = crsp_compustat_jun['Revenue'] - crsp_compustat_jun['COGS'].fillna(0) - crsp_compustat_jun['I'].fillna(0) - crsp_compustat_jun['SGA'].fillna(0)
    crsp_compustat_jun['OP'] = crsp_compustat_jun['profit'] / (crsp_compustat_jun['BE'] + crsp_compustat_jun['minority_interest'].fillna(0))
    crsp_compustat_jun['OP'] = np.where((crsp_compustat_jun['BE'] > 0) & (crsp_compustat_jun['OP'].notnull()) & (crsp_compustat_jun['Revenue'].notnull()) & (~crsp_compustat_jun['xp_allnan']), crsp_compustat_jun['OP'], np.nan)

    # Investments, INV
    try:
        crsp_compustat_jun['INV'] = np.log(crsp_compustat_jun['Assets']) - np.log(crsp_compustat_jun.groupby(['permno'])['Assets'].shift(1))
    except FloatingPointError:
        crsp_compustat_jun['INV'] = (crsp_compustat_jun['Assets'] / crsp_compustat_jun.groupby(['permno'])['Assets'].shift(1)) - 1

    crsp_compustat_jun['INV'] = np.where(crsp_compustat_jun['INV'].notnull(), crsp_compustat_jun['INV'], np.nan)

    # Earnings to price, EP
    crsp_compustat_jun['ib'] = np.where(crsp_compustat_jun['core_income'].notnull(), crsp_compustat_jun['core_income'], np.nan)
    crsp_compustat_jun['EP'] = np.where(crsp_compustat_jun['ME_dec'] > 0, crsp_compustat_jun['ib'] * 1000 / crsp_compustat_jun['ME_dec'], np.nan)
    crsp_compustat_jun['EP'] = np.where(crsp_compustat_jun['EP'] > 0, crsp_compustat_jun['EP'], np.nan)

    # Cash Flow to price, CFP
    crsp_compustat_jun['cf'] = crsp_compustat_jun['core_income'] + crsp_compustat_jun['tax_deferral_IS'].fillna(0) + crsp_compustat_jun['DA'].fillna(0)
    crsp_compustat_jun['cf'] = np.where(crsp_compustat_jun['cf'].notnull(), crsp_compustat_jun['cf'], np.nan)
    crsp_compustat_jun['CFP'] = np.where((crsp_compustat_jun['BE'] > 0) &
                                         (crsp_compustat_jun['ME'] > 0) &
                                         (crsp_compustat_jun['ME_dec'] > 0),
                                         crsp_compustat_jun['cf'] * 1000 / crsp_compustat_jun['ME_dec'], np.nan)
    crsp_compustat_jun['CFP'] = np.where(crsp_compustat_jun['CFP'] > 0, crsp_compustat_jun['CFP'], np.nan)

    # Accruals, AC
    crsp_compustat_jun['com_stock_adj'] = np.where((crsp_compustat_jun['com_stock'] * crsp_compustat_jun['cum_adj_factor']) > 0, crsp_compustat_jun['com_stock'] * crsp_compustat_jun['cum_adj_factor'], np.nan)
    crsp_compustat_jun['owcap_adj'] = ((crsp_compustat_jun['cur_assets'] - crsp_compustat_jun['cash_st_inv']) - (crsp_compustat_jun['cur_liabilities'].fillna(0) - crsp_compustat_jun['debt_current'].fillna(0))) / crsp_compustat_jun['com_stock_adj']
    crsp_compustat_jun['d_owcap_adj'] = (crsp_compustat_jun['owcap_adj'] - crsp_compustat_jun.groupby(['permno'])['owcap_adj'].shift(1))

    crsp_compustat_jun['AC'] = np.where((crsp_compustat_jun['BE'] > 0) & (crsp_compustat_jun['ME'] > 0), crsp_compustat_jun['d_owcap_adj'] / (crsp_compustat_jun['BE'] / crsp_compustat_jun['com_stock_adj']), np.nan)
    crsp_compustat_jun['AC'] = np.where(crsp_compustat_jun['AC'].notnull(), crsp_compustat_jun['AC'], np.nan)

    # Net Income Shares, NI
    crsp_compustat_jun['ni_com_stock_adj'] = np.where(((crsp_compustat_jun['com_stock'] * crsp_compustat_jun['cum_adj_factor']) > 0), (crsp_compustat_jun['com_stock'] * crsp_compustat_jun['cum_adj_factor']), np.nan)
    try:
        crsp_compustat_jun['NI'] = np.log(crsp_compustat_jun['ni_com_stock_adj']) - np.log(crsp_compustat_jun.groupby(['permno'])['ni_com_stock_adj'].shift(1))
    except FloatingPointError:
        crsp_compustat_jun['NI'] = (crsp_compustat_jun['ni_com_stock_adj'] / crsp_compustat_jun.groupby(['permno'])['ni_com_stock_adj'].shift(1)) - 1
    crsp_compustat_jun['NI'] = np.where((crsp_compustat_jun['NI'].notnull()) &
                                        (crsp_compustat_jun['ME'] > 0) &
                                        (crsp_compustat_jun['BE'] > 0),
                                        crsp_compustat_jun['NI'], np.nan)
    crsp_compustat_jun['NI'] = np.where(crsp_compustat_jun['NI'] > 0, crsp_compustat_jun['NI'], np.nan)

    ####################
    ### NYSE BUCKETS ###
    ####################

    # Filter NYSE stocks for FF factors 
    NYSE_stocks = crsp_compustat_jun[(crsp_compustat_jun['age'] >= 2) &
                                     (crsp_compustat_jun['BE'] > 0) & 
                                     (crsp_compustat_jun['ME'] > 0) &
                                     (crsp_compustat_jun['ME_dec'] > 0) &
                                     (crsp_compustat_jun['exchange_code'] == 1)]

    NYSE_size = fcs.NYSE_median_ME(NYSE_stocks)

    crsp_compustat_jun_1 = (NYSE_size
                            .merge(fcs.NYSE_ratio('BM', bins, NYSE_stocks), how = 'inner', on = 'month_end')
                            .merge(fcs.NYSE_ratio('INV', bins, NYSE_stocks), how = 'inner', on = 'month_end')
                            .merge(fcs.NYSE_ratio('OP', bins, NYSE_stocks), how = 'inner', on = 'month_end' )
                            .merge(fcs.NYSE_ratio('EP', bins, NYSE_stocks), how = 'inner', on = 'month_end')
                            .merge(fcs.NYSE_ratio('CFP', bins, NYSE_stocks), how = 'inner', on = 'month_end')
                            .merge(fcs.NYSE_ratio('AC', bins, NYSE_stocks), how = 'inner', on = 'month_end')
                            .merge(fcs.NYSE_ratio('NI', bins, NYSE_stocks), how = 'inner', on = 'month_end')
                            .merge(crsp_compustat_jun, how = 'right', on = ['month_end']))

    stocks_corr = crsp_compustat_jun[crsp_compustat_jun['month_end'] > '30-06-1963'][['month_end', 'permco', 'ME', 'BM', 'OP', 'INV', 'EP', 'CFP', 'AC', 'NI']].copy().set_index(['permco']).dropna()
    stocks_corr = stocks_corr.groupby('permco').filter(lambda x: len(x) > 2)
    zscores = stocks_corr.groupby('month_end').apply(stats.zscore, axis = 0)
    fcs.write_excel(output_file, 'Table 3', port_stats.corr_and_p(zscores.reset_index().set_index(['permco', 'month_end'])))

    ############################
    ### PORTFOLIO ALLOCATION ###
    ############################

    crsp_compustat_jun_1['size'] = np.where((crsp_compustat_jun_1['age'] >= 2) & (crsp_compustat_jun_1['ME'] > 0), crsp_compustat_jun_1.apply(fcs.size_bucket, axis = 1), 'missing')
    hml = fcs.allocation('BM', crsp_compustat_jun_1, crsp_data, '1', '2', '3', bins, crit1 = 'BE', crit2 = 'ME', crit3 = 'ME_dec')
    cma = fcs.allocation('INV', crsp_compustat_jun_1, crsp_data, '3', '2', '1', bins)
    rmw = fcs.allocation('OP', crsp_compustat_jun_1, crsp_data, '1', '2', '3', bins, crit1 = 'BE', crit2 = 'BE', crit3 = 'BE')
    ep = fcs.allocation('EP', crsp_compustat_jun_1, crsp_data, '1', '2', '3', bins)
    cfp = fcs.allocation('CFP', crsp_compustat_jun_1, crsp_data, '1', '2', '3', bins)
    acc = fcs.allocation('AC', crsp_compustat_jun_1, crsp_data, '3', '2', '1', bins)
    nsi = fcs.allocation('NI', crsp_compustat_jun_1, crsp_data, '3', '2', '1', bins)

    mom = fcs.cumret_calc(crsp_data_long, 11, 2, bins, filtered = 'no')
    umd = fcs.allocation('cumret', mom, crsp_data, '1', '2', '3', bins, factor_type = 'other')

    lt_data = fcs.cumret_calc(crsp_data_long, 47, 13, bins, filtered = 'no')
    lt_reversal = fcs.allocation('cumret', lt_data, crsp_data, '3', '2', '1', bins, factor_type = 'other')

    st_data = fcs.cumret_calc(crsp_data_long, 1, 1, bins, filtered = 'no')
    st_reversal = fcs.allocation('cumret', st_data, crsp_data, '3', '2', '1', bins, factor_type = 'other')

    ##############################
    ### PORTFOLIO CONSTRUCTION ###
    ##############################

    MKTRF = fcs.get_port_objects(fcs.market_port_class(crsp_data[crsp_data['weight'] > 0], 'lag_ME', RF, 'MKTRF'), ret_type = 'netexret')
    HML = fcs.get_port_objects(fcs.port_class(hml, 'BM', 'weight', RF, 'HML'))
    SMB = fcs.get_port_objects(fcs.port_class(hml, 'BM', 'weight', RF, 'SMB', smb = 'yes'))
    CMA = fcs.get_port_objects(fcs.port_class(cma, 'INV', 'weight', RF, 'CMA'))
    RMW = fcs.get_port_objects(fcs.port_class(rmw, 'OP', 'weight', RF, 'RMW'))
    EP = fcs.get_port_objects(fcs.port_class(ep, 'EP', 'weight', RF, 'EP'))
    CFP = fcs.get_port_objects(fcs.port_class(cfp, 'CFP', 'weight', RF, 'CFP'))
    NSI = fcs.get_port_objects(fcs.port_class(nsi, 'NI', 'weight', RF, 'NSI'))
    ACC = fcs.get_port_objects(fcs.port_class(acc, 'AC', 'weight', RF, 'ACC'))
    UMD = fcs.get_port_objects(fcs.port_class(umd, 'cumret', 'lag_ME', RF, 'UMD'))
    LTR = fcs.get_port_objects(fcs.port_class(lt_reversal, 'cumret', 'lag_ME', RF, 'LTR'))
    STR = fcs.get_port_objects(fcs.port_class(st_reversal, 'cumret', 'lag_ME', RF, 'STR'))

    # BAB
    MKTRF_long = fcs.get_port_objects(fcs.market_port_class(crsp_data_long, 'lag_ME', RF, 'MKTRF'), ret_type = 'netexret').reset_index()
    MKTRF_long.columns = ['date', 'MKTRF']
    bab_data = crsp_data_long[['permno', 'month_end', 'adj_ret']]
    bab_data.columns = ['permno', 'date', 'adj_ret']
    BAB = fcs.BAB_factor(bab_data, RF.reset_index(), MKTRF_long, filtering='no')

    factor_longshort = (MKTRF.rename('MKTRF').to_frame()
                           .merge(SMB.rename('SMB').to_frame(), left_index=True, right_index=True)                      
                           .merge(HML.rename('HML').to_frame(), left_index=True, right_index=True)                      
                           .merge(UMD.rename('UMD').to_frame(), left_index=True, right_index=True)                     
                           .merge(RMW.rename('RMW').to_frame(), left_index=True, right_index=True)                      
                           .merge(CMA.rename('CMA').to_frame(), left_index=True, right_index=True)                      
                           .merge(ACC.rename('ACC').to_frame(), left_index=True, right_index=True)                      
                           .merge(NSI.rename('NSI').to_frame(), left_index=True, right_index=True)                      
                           .merge(STR.rename('STR').to_frame(), left_index=True, right_index=True)                      
                           .merge(LTR.rename('LTR').to_frame(), left_index=True, right_index=True)                      
                           .merge(CFP.rename('CFP').to_frame(), left_index=True, right_index=True)                      
                           .merge(EP.rename('EP').to_frame(), left_index=True, right_index=True)                      
                           .merge(BAB.set_index('date')['BAB'].to_frame(), left_index=True, right_index=True)                      
                           .merge(additional_factors.set_index('date'), left_index=True, right_index=True))

    HML_short, HML_long = fcs.get_port_objects(fcs.port_class(hml, 'BM', 'weight', RF, 'HML', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(hml, 'BM', 'weight', RF, 'HML', legs = 'True')[1], ret_type = 'netexret')
    SMB_short, SMB_long = fcs.get_port_objects(fcs.port_class(hml, 'BM', 'weight', RF, 'SMB', legs = 'True', smb = 'yes')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(hml, 'BM', 'weight', RF, 'SMB', legs = 'True', smb = 'yes')[1], ret_type = 'netexret')
    UMD_short, UMD_long = fcs.get_port_objects(fcs.port_class(umd, 'cumret', 'lag_ME', RF, 'UMD', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(umd, 'cumret', 'lag_ME', RF, 'UMD', legs = 'True')[1], ret_type = 'netexret')
    RMW_short, RMW_long = fcs.get_port_objects(fcs.port_class(rmw, 'OP', 'weight', RF, 'RMW', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(rmw, 'OP', 'weight', RF, 'RMW', legs = 'True')[1], ret_type = 'netexret')
    CMA_short, CMA_long = fcs.get_port_objects(fcs.port_class(cma, 'INV', 'weight', RF, 'CMA', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(cma, 'INV', 'weight', RF, 'CMA', legs = 'True')[1], ret_type = 'netexret')
    ACC_short, ACC_long = fcs.get_port_objects(fcs.port_class(acc, 'AC', 'weight', RF, 'ACC', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(acc, 'AC', 'weight', RF, 'ACC', legs = 'True')[1], ret_type = 'netexret')
    NSI_short, NSI_long = fcs.get_port_objects(fcs.port_class(nsi, 'NI', 'weight', RF, 'NSI', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(nsi, 'NI', 'weight', RF, 'NSI', legs = 'True')[1], ret_type = 'netexret')
    STR_short, STR_long = fcs.get_port_objects(fcs.port_class(st_reversal, 'cumret', 'lag_ME', RF, 'STR', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(st_reversal, 'cumret', 'lag_ME', RF, 'STR', legs = 'True')[1], ret_type = 'netexret')
    LTR_short, LTR_long = fcs.get_port_objects(fcs.port_class(lt_reversal, 'cumret', 'lag_ME', RF, 'LTR', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(lt_reversal, 'cumret', 'lag_ME', RF, 'LTR', legs = 'True')[1], ret_type = 'netexret')
    CFP_short, CFP_long = fcs.get_port_objects(fcs.port_class(cfp, 'CFP', 'weight', RF, 'CFP', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(cfp, 'CFP', 'weight', RF, 'CFP', legs = 'True')[1], ret_type = 'netexret')
    EP_short, EP_long = fcs.get_port_objects(fcs.port_class(ep, 'EP', 'weight', RF, 'EP', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(ep, 'EP', 'weight', RF, 'EP', legs = 'True')[1], ret_type = 'netexret')

    factor_legs = (SMB_long.rename('SMB_long').to_frame()
                           .merge(SMB_short.rename('SMB_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(HML_long.rename('HML_long').to_frame(), left_index=True, right_index=True)
                           .merge(HML_short.rename('HML_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(UMD_long.rename('UMD_long').to_frame(), left_index=True, right_index=True)                      
                           .merge(UMD_short.rename('UMD_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(RMW_long.rename('RMW_long').to_frame(), left_index=True, right_index=True)                      
                           .merge(RMW_short.rename('RMW_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(CMA_long.rename('CMA_long').to_frame(), left_index=True, right_index=True)                      
                           .merge(CMA_short.rename('CMA_short').to_frame(), left_index=True, right_index=True)                       
                           .merge(ACC_long.rename('ACC_long').to_frame(), left_index=True, right_index=True)                      
                           .merge(ACC_short.rename('ACC_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(NSI_long.rename('NSI_long').to_frame(), left_index=True, right_index=True)                      
                           .merge(NSI_short.rename('NSI_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(STR_long.rename('STR_long').to_frame(), left_index=True, right_index=True)
                           .merge(STR_short.rename('STR_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(LTR_long.rename('LTR_long').to_frame(), left_index=True, right_index=True)                      
                           .merge(LTR_short.rename('LTR_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(CFP_long.rename('CFP_long').to_frame(), left_index=True, right_index=True)                      
                           .merge(CFP_short.rename('CFP_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(EP_long.rename('EP_long').to_frame(), left_index=True, right_index=True)                      
                           .merge(EP_short.rename('EP_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(BAB.set_index('date')['BABLong'].rename('BAB_long').to_frame(), left_index=True, right_index=True)
                           .merge(BAB.set_index('date')['BABShort'].rename('BAB_short').to_frame(), left_index=True, right_index=True))
    
    # DATA FOR FURTHER ANALYSIS
    fcs.write_excel(data_file, 'ANOMALIES_NF', factor_longshort)
    fcs.write_excel(output_file, 'Table 6 - Raw Sample', factor_legs.mean(axis=0).to_frame().T*100*12)
    
    # DATA STATS
    stat = port_stats.summary_stats(MKTRF.rename('MKTRF').to_frame(), 12)
    fcs.write_excel(output_file, 'Table 4 - Raw Sample', pd.concat([sample_stats_raw, stat.transpose()], axis=1))   
    
def filtered_data_handling(crsp_data, crsp_data_long, crsp_compustat, output_file, data_file, tc, output='all'):
    """
    Compute all data related to the filtered sample and write to the relevant output file
    
    Parameters
    ----------
    crsp_data : pandas DataFrame
        DataFrame containing the crisp_data with filtering columns (flags)
    crsp_compustat : pandas DataFrame
        DataFrame containing the crsp_compustat data
    
    Returns
    -------
    None
    """

    #################
    ### JUNE DATA ###
    #################
    
    # Merge June and December data
    crsp_compustat_jun = (crsp_data[crsp_data['month_end'].dt.month == 6]
                          .merge(crsp_data_dec, how = 'inner', on = ['permno', 'year'])
                          .sort_values(by = ['permno', 'month_end'])
                          .drop_duplicates()
                          .merge(crsp_compustat, how = 'inner', on = ['permno', 'month_end']))
    
    ###############
    ### FACTORS ###
    ###############
    
    # BookRatio, BM
    crsp_compustat_jun['BM'] = np.where(
        (crsp_compustat_jun['BE'] > 0) &
        (crsp_compustat_jun['ME'] > 0) &
        (crsp_compustat_jun['ME_dec'] > 0),
        crsp_compustat_jun['BE'] * 1000 / crsp_compustat_jun['ME_dec'], np.nan)
    
    # Operating Profitability, OP
    crsp_compustat_jun['xp_allnan'] = (crsp_compustat_jun['COGS'].isnull()) & (crsp_compustat_jun['SGA'].isnull()) & (crsp_compustat_jun['I'].isnull())
    crsp_compustat_jun['profit'] = crsp_compustat_jun['Revenue'] - crsp_compustat_jun['COGS'].fillna(0) - crsp_compustat_jun['I'].fillna(0) - crsp_compustat_jun['SGA'].fillna(0)
    crsp_compustat_jun['OP'] = crsp_compustat_jun['profit'] / (crsp_compustat_jun['BE'] + crsp_compustat_jun['minority_interest'].fillna(0))
    crsp_compustat_jun['OP'] = np.where((crsp_compustat_jun['BE'] > 0) & (crsp_compustat_jun['OP'].notnull()) & (crsp_compustat_jun['Revenue'].notnull()) & (~crsp_compustat_jun['xp_allnan']), crsp_compustat_jun['OP'], np.nan)
    
    # Investments, INV
    try:
        crsp_compustat_jun['INV'] = np.log(crsp_compustat_jun['Assets']) - np.log(crsp_compustat_jun.groupby(['permno'])['Assets'].shift(1))
    except FloatingPointError:
        crsp_compustat_jun['INV'] = (crsp_compustat_jun['Assets'] / crsp_compustat_jun.groupby(['permno'])['Assets'].shift(1)) - 1
    crsp_compustat_jun['INV'] = np.where(crsp_compustat_jun['INV'].notnull(), crsp_compustat_jun['INV'], np.nan)
    
    # Earnings to price, EP
    crsp_compustat_jun['ib'] = np.where(crsp_compustat_jun['core_income'].notnull(), crsp_compustat_jun['core_income'], np.nan)
    crsp_compustat_jun['EP'] = np.where(crsp_compustat_jun['ME_dec'] > 0, crsp_compustat_jun['ib'] * 1000 / crsp_compustat_jun['ME_dec'], np.nan)
    crsp_compustat_jun['EP'] = np.where(crsp_compustat_jun['EP'] > 0, crsp_compustat_jun['EP'], np.nan)
    
    # Cash Flow to price, CFP
    crsp_compustat_jun['cf'] = crsp_compustat_jun['core_income'] + crsp_compustat_jun['tax_deferral_IS'].fillna(0) + crsp_compustat_jun['DA'].fillna(0)
    crsp_compustat_jun['cf'] = np.where(crsp_compustat_jun['cf'].notnull(), crsp_compustat_jun['cf'], np.nan)
    crsp_compustat_jun['CFP'] = np.where((crsp_compustat_jun['BE'] > 0) &
                                         (crsp_compustat_jun['ME'] > 0) &
                                         (crsp_compustat_jun['ME_dec'] > 0),
                                         crsp_compustat_jun['cf'] * 1000 / crsp_compustat_jun['ME_dec'], np.nan)
    crsp_compustat_jun['CFP'] = np.where(crsp_compustat_jun['CFP'] > 0, crsp_compustat_jun['CFP'], np.nan)
    
    # Accruals, AC
    crsp_compustat_jun['com_stock_adj'] = np.where((crsp_compustat_jun['com_stock'] * crsp_compustat_jun['cum_adj_factor']) > 0, crsp_compustat_jun['com_stock'] * crsp_compustat_jun['cum_adj_factor'], np.nan)
    crsp_compustat_jun['owcap_adj'] = ((crsp_compustat_jun['cur_assets'] - crsp_compustat_jun['cash_st_inv']) - (crsp_compustat_jun['cur_liabilities'].fillna(0) - crsp_compustat_jun['debt_current'].fillna(0))) / crsp_compustat_jun['com_stock_adj']
    crsp_compustat_jun['d_owcap_adj'] = (crsp_compustat_jun['owcap_adj'] - crsp_compustat_jun.groupby(['permno'])['owcap_adj'].shift(1))
    crsp_compustat_jun['AC'] = np.where((crsp_compustat_jun['BE'] > 0) & (crsp_compustat_jun['ME'] > 0), crsp_compustat_jun['d_owcap_adj'] / (crsp_compustat_jun['BE'] / crsp_compustat_jun['com_stock_adj']), np.nan)
    crsp_compustat_jun['AC'] = np.where(crsp_compustat_jun['AC'].notnull(), crsp_compustat_jun['AC'], np.nan)
    
    # Net Income Shares, NI
    crsp_compustat_jun['ni_com_stock_adj'] = np.where(((crsp_compustat_jun['com_stock'] * crsp_compustat_jun['cum_adj_factor']) > 0), (crsp_compustat_jun['com_stock'] * crsp_compustat_jun['cum_adj_factor']), np.nan)
    try:
        crsp_compustat_jun['NI'] = np.log(crsp_compustat_jun['ni_com_stock_adj']) - np.log(crsp_compustat_jun.groupby(['permno'])['ni_com_stock_adj'].shift(1))
    except FloatingPointError:
        crsp_compustat_jun['NI'] = (crsp_compustat_jun['ni_com_stock_adj'] / crsp_compustat_jun.groupby(['permno'])['ni_com_stock_adj'].shift(1)) - 1
    crsp_compustat_jun['NI'] = np.where((crsp_compustat_jun['NI'].notnull()) &
                                        (crsp_compustat_jun['ME'] > 0) &
                                        (crsp_compustat_jun['BE'] > 0),
                                        crsp_compustat_jun['NI'], np.nan)
    crsp_compustat_jun['NI'] = np.where(crsp_compustat_jun['NI'] > 0, crsp_compustat_jun['NI'], np.nan)
    
    # Filter out micro_cap_stocks and penny stocks
    crsp_compustat_jun = crsp_compustat_jun[crsp_compustat_jun['microcap_flag'] == 0]
    crsp_compustat_jun = crsp_compustat_jun[crsp_compustat_jun['penny_flag'] == 0]
    
    ####################
    ### NYSE BUCKETS ###
    ####################
       
    # Filter NYSE stocks for FF factors 
    NYSE_stocks = crsp_compustat_jun[(crsp_compustat_jun['age'] >= 2) &
                                     (crsp_compustat_jun['BE'] > 0) & 
                                     (crsp_compustat_jun['ME'] > 0) &
                                     (crsp_compustat_jun['ME_dec'] > 0) &
                                     (crsp_compustat_jun['exchange_code'] == 1)]
    
    NYSE_size = fcs.NYSE_median_ME(NYSE_stocks)
    
    crsp_compustat_jun_1 = (NYSE_size
                            .merge(fcs.NYSE_ratio('BM', bins, NYSE_stocks), how = 'inner', on = 'month_end')
                            .merge(fcs.NYSE_ratio('INV', bins, NYSE_stocks), how = 'inner', on = 'month_end')
                            .merge(fcs.NYSE_ratio('OP', bins, NYSE_stocks), how = 'inner', on = 'month_end' )
                            .merge(fcs.NYSE_ratio('EP', bins, NYSE_stocks), how = 'inner', on = 'month_end')
                            .merge(fcs.NYSE_ratio('CFP', bins, NYSE_stocks), how = 'inner', on = 'month_end')
                            .merge(fcs.NYSE_ratio('AC', bins, NYSE_stocks), how = 'inner', on = 'month_end')
                            .merge(fcs.NYSE_ratio('NI', bins, NYSE_stocks), how = 'inner', on = 'month_end')
                            .merge(fcs.NYSE_ratio('ME', bins, NYSE_stocks), how = 'inner', on = 'month_end')
                            .merge(crsp_compustat_jun, how = 'right', on = ['month_end']))
    
    # Size comparisons NYSE Stocks versus ALL Stocks
    all_size = fcs.NYSE_median_ME(crsp_compustat_jun[(crsp_compustat_jun['age'] >= 2) &
                                     (crsp_compustat_jun['BE'] > 0) & 
                                     (crsp_compustat_jun['ME'] > 0) &
                                     (crsp_compustat_jun['ME_dec'] > 0)]).merge(NYSE_size, on = 'month_end', how = 'left')
    
    all_size.columns = ['date', 'MedianME_all', 'MedianME_NYSE']
    fcs.write_excel(output_file, 'Figure 6', all_size.set_index('date'))
    
    ############################
    ### PORTFOLIO ALLOCATION ###
    ############################
    
    crsp_compustat_jun_1['size'] = np.where((crsp_compustat_jun_1['age'] >= 2) & (crsp_compustat_jun_1['ME'] > 0), crsp_compustat_jun_1.apply(fcs.size_bucket, axis = 1), 'missing')
    
    hml = fcs.allocation_filtered('BM', crsp_compustat_jun_1, crsp_data, '1', '2', '3', bins, crit1 = 'BE', crit2 = 'ME', crit3 = 'ME_dec')
    cma = fcs.allocation_filtered('INV', crsp_compustat_jun_1, crsp_data, '3', '2', '1', bins)
    rmw = fcs.allocation_filtered('OP', crsp_compustat_jun_1, crsp_data, '1', '2', '3', bins, crit1 = 'BE', crit2 = 'BE', crit3 = 'BE')
    ep = fcs.allocation_filtered('EP', crsp_compustat_jun_1, crsp_data, '1', '2', '3', bins)
    cfp = fcs.allocation_filtered('CFP', crsp_compustat_jun_1, crsp_data, '1', '2', '3', bins)
    acc = fcs.allocation_filtered('AC', crsp_compustat_jun_1, crsp_data, '3', '2', '1', bins)
    nsi = fcs.allocation_filtered('NI', crsp_compustat_jun_1, crsp_data, '3', '2', '1', bins)
    
    mom = fcs.cumret_calc(crsp_data_long, 11, 2, bins, filtered = 'yes')
    umd = fcs.allocation_filtered('cumret', mom, crsp_data, '1', '2', '3', bins, factor_type = 'other')
    lt_data = fcs.cumret_calc(crsp_data_long, 47, 13, bins, filtered = 'yes')
    lt_reversal = fcs.allocation_filtered('cumret', lt_data, crsp_data, '3', '2', '1', bins, factor_type = 'other')
    st_data = fcs.cumret_calc(crsp_data_long, 1, 1, bins, filtered = 'yes')
    st_reversal = fcs.allocation_filtered('cumret', st_data, crsp_data, '3', '2', '1', bins, factor_type = 'other')  
    
    ##############################
    ### PORTFOLIO CONSTRUCTION ###
    ##############################
    
    MKTRF = fcs.get_port_objects(fcs.market_port_class(crsp_data[(crsp_data['weight'] > 0) & (crsp_data[['microcap_flag', 'penny_flag', 'volume_flag']].sum(axis=1) == 0)], 'lag_ME', RF, 'MKTRF', apply_tc=tc), ret_type = 'netexret')
    HML = fcs.get_port_objects(fcs.port_class(hml, 'BM', 'weight', RF, 'HML', apply_tc=tc), ret_type = 'netexret')
    SMB = fcs.get_port_objects(fcs.port_class(hml, 'BM', 'weight', RF, 'SMB', smb = 'yes', apply_tc=tc), ret_type = 'netexret')
    UMD = fcs.get_port_objects(fcs.port_class(umd, 'cumret', 'lag_ME', RF, 'UMD', apply_tc=tc), ret_type = 'netexret')
    RMW = fcs.get_port_objects(fcs.port_class(rmw, 'OP', 'weight', RF, 'RMW', apply_tc=tc), ret_type = 'netexret')
    CMA = fcs.get_port_objects(fcs.port_class(cma, 'INV', 'weight', RF, 'CMA', apply_tc=tc), ret_type = 'netexret')
    ACC = fcs.get_port_objects(fcs.port_class(acc, 'AC', 'weight', RF, 'ACC', apply_tc=tc), ret_type = 'netexret')
    NSI = fcs.get_port_objects(fcs.port_class(nsi, 'NI', 'weight', RF, 'NSI', apply_tc=tc), ret_type = 'netexret')
    STR = fcs.get_port_objects(fcs.port_class(st_reversal, 'cumret', 'lag_ME', RF, 'STR', apply_tc=tc), ret_type = 'netexret')
    LTR = fcs.get_port_objects(fcs.port_class(lt_reversal, 'cumret', 'lag_ME', RF, 'LTR', apply_tc=tc), ret_type = 'netexret')
    EP = fcs.get_port_objects(fcs.port_class(ep, 'EP', 'weight', RF, 'EP', apply_tc=tc), ret_type = 'netexret')
    CFP = fcs.get_port_objects(fcs.port_class(cfp, 'CFP', 'weight', RF, 'CFP', apply_tc=tc), ret_type = 'netexret')
    
    # BAB
    MKTRF_long = fcs.get_port_objects(fcs.market_port_class(crsp_data_long, 'lag_ME', RF, 'MKTRF', apply_tc=tc), ret_type = 'netexret').reset_index()
    MKTRF_long.columns = ['date', 'MKTRF']
    bab_data = crsp_data_long[['permno', 'month_end', 'adj_ret', 'microcap_flag', 'penny_flag', 'volume_flag']]
    bab_data.columns = ['permno', 'date', 'adj_ret', 'microcap_flag', 'penny_flag', 'volume_flag']
    BAB = fcs.BAB_factor(bab_data, RF.reset_index(), MKTRF_long, filtering='yes', apply_tc=tc)
    
    # BAB COSTS
    if tc == 'yes':
        BAB_tc = BAB.copy().set_index('date')
        BAB_no_tc = fcs.BAB_factor(bab_data, RF.reset_index(), MKTRF_long, filtering='yes', apply_tc='no').set_index('date')   
        BAB_cost = BAB_no_tc['BAB'].subtract(BAB_tc['BAB'])   
       
    factor_longshort = (MKTRF.rename('MKTRF').to_frame()
                           .merge(SMB.rename('SMB').to_frame(), left_index=True, right_index=True)                      
                           .merge(HML.rename('HML').to_frame(), left_index=True, right_index=True)                      
                           .merge(UMD.rename('UMD').to_frame(), left_index=True, right_index=True)                     
                           .merge(RMW.rename('RMW').to_frame(), left_index=True, right_index=True)                      
                           .merge(CMA.rename('CMA').to_frame(), left_index=True, right_index=True)                      
                           .merge(ACC.rename('ACC').to_frame(), left_index=True, right_index=True)                      
                           .merge(NSI.rename('NSI').to_frame(), left_index=True, right_index=True)                      
                           .merge(STR.rename('STR').to_frame(), left_index=True, right_index=True)                      
                           .merge(LTR.rename('LTR').to_frame(), left_index=True, right_index=True)                      
                           .merge(CFP.rename('CFP').to_frame(), left_index=True, right_index=True)                      
                           .merge(EP.rename('EP').to_frame(), left_index=True, right_index=True)                      
                           .merge(BAB.set_index('date')['BAB'].to_frame(), left_index=True, right_index=True))                      
    
    HML_short, HML_long = fcs.get_port_objects(fcs.port_class(hml, 'BM', 'weight', RF, 'HML', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(hml, 'BM', 'weight', RF, 'HML', legs = 'True')[1], ret_type = 'netexret')
    SMB_short, SMB_long = fcs.get_port_objects(fcs.port_class(hml, 'BM', 'weight', RF, 'SMB', legs = 'True', smb = 'yes')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(hml, 'BM', 'weight', RF, 'SMB', legs = 'True', smb = 'yes')[1], ret_type = 'netexret')
    UMD_short, UMD_long = fcs.get_port_objects(fcs.port_class(umd, 'cumret', 'lag_ME', RF, 'UMD', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(umd, 'cumret', 'lag_ME', RF, 'UMD', legs = 'True')[1], ret_type = 'netexret')
    RMW_short, RMW_long = fcs.get_port_objects(fcs.port_class(rmw, 'OP', 'weight', RF, 'RMW', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(rmw, 'OP', 'weight', RF, 'RMW', legs = 'True')[1], ret_type = 'netexret')
    CMA_short, CMA_long = fcs.get_port_objects(fcs.port_class(cma, 'INV', 'weight', RF, 'CMA', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(cma, 'INV', 'weight', RF, 'CMA', legs = 'True')[1], ret_type = 'netexret')
    ACC_short, ACC_long = fcs.get_port_objects(fcs.port_class(acc, 'AC', 'weight', RF, 'ACC', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(acc, 'AC', 'weight', RF, 'ACC', legs = 'True')[1], ret_type = 'netexret')
    NSI_short, NSI_long = fcs.get_port_objects(fcs.port_class(nsi, 'NI', 'weight', RF, 'NSI', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(nsi, 'NI', 'weight', RF, 'NSI', legs = 'True')[1], ret_type = 'netexret')
    STR_short, STR_long = fcs.get_port_objects(fcs.port_class(st_reversal, 'cumret', 'lag_ME', RF, 'STR', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(st_reversal, 'cumret', 'lag_ME', RF, 'STR', legs = 'True')[1], ret_type = 'netexret')
    LTR_short, LTR_long = fcs.get_port_objects(fcs.port_class(lt_reversal, 'cumret', 'lag_ME', RF, 'LTR', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(lt_reversal, 'cumret', 'lag_ME', RF, 'LTR', legs = 'True')[1], ret_type = 'netexret')
    CFP_short, CFP_long = fcs.get_port_objects(fcs.port_class(cfp, 'CFP', 'weight', RF, 'CFP', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(cfp, 'CFP', 'weight', RF, 'CFP', legs = 'True')[1], ret_type = 'netexret')
    EP_short, EP_long = fcs.get_port_objects(fcs.port_class(ep, 'EP', 'weight', RF, 'EP', legs = 'True')[0], ret_type = 'netexret'), fcs.get_port_objects(fcs.port_class(ep, 'EP', 'weight', RF, 'EP', legs = 'True')[1], ret_type = 'netexret')
    
    factor_legs = (SMB_long.rename('SMB_long').to_frame()
                           .merge(SMB_short.rename('SMB_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(HML_long.rename('HML_long').to_frame(), left_index=True, right_index=True)
                           .merge(HML_short.rename('HML_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(UMD_long.rename('UMD_long').to_frame(), left_index=True, right_index=True)                      
                           .merge(UMD_short.rename('UMD_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(RMW_long.rename('RMW_long').to_frame(), left_index=True, right_index=True)                      
                           .merge(RMW_short.rename('RMW_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(CMA_long.rename('CMA_long').to_frame(), left_index=True, right_index=True)                      
                           .merge(CMA_short.rename('CMA_short').to_frame(), left_index=True, right_index=True)                       
                           .merge(ACC_long.rename('ACC_long').to_frame(), left_index=True, right_index=True)                      
                           .merge(ACC_short.rename('ACC_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(NSI_long.rename('NSI_long').to_frame(), left_index=True, right_index=True)                      
                           .merge(NSI_short.rename('NSI_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(STR_long.rename('STR_long').to_frame(), left_index=True, right_index=True)
                           .merge(STR_short.rename('STR_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(LTR_long.rename('LTR_long').to_frame(), left_index=True, right_index=True)                      
                           .merge(LTR_short.rename('LTR_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(CFP_long.rename('CFP_long').to_frame(), left_index=True, right_index=True)                      
                           .merge(CFP_short.rename('CFP_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(EP_long.rename('EP_long').to_frame(), left_index=True, right_index=True)                      
                           .merge(EP_short.rename('EP_short').to_frame(), left_index=True, right_index=True)                      
                           .merge(BAB.set_index('date')['BABLong'].rename('BAB_long').to_frame(), left_index=True, right_index=True)
                           .merge(BAB.set_index('date')['BABShort'].rename('BAB_short').to_frame(), left_index=True, right_index=True))
    
    if tc == 'yes':
        # GET PERFORMANCE (TURNOVER, COSTS)
        MKTRF = fcs.get_port_objects(fcs.market_port_class(crsp_data[(crsp_data['weight'] > 0) & (crsp_data[['microcap_flag', 'penny_flag', 'volume_flag']].sum(axis=1) == 0)], 'lag_ME', RF, 'MKTRF', apply_tc=tc), get_object = 'performance')
        HML = fcs.get_port_objects(fcs.port_class(hml, 'BM', 'weight', RF, 'HML', apply_tc=tc), get_object = 'performance')
        SMB = fcs.get_port_objects(fcs.port_class(hml, 'BM', 'weight', RF, 'SMB', smb = 'yes', apply_tc=tc), get_object = 'performance')
        UMD = fcs.get_port_objects(fcs.port_class(umd, 'cumret', 'lag_ME', RF, 'UMD', apply_tc=tc), get_object = 'performance')
        RMW = fcs.get_port_objects(fcs.port_class(rmw, 'OP', 'weight', RF, 'RMW', apply_tc=tc), get_object = 'performance')
        CMA = fcs.get_port_objects(fcs.port_class(cma, 'INV', 'weight', RF, 'CMA', apply_tc=tc), get_object = 'performance')
        ACC = fcs.get_port_objects(fcs.port_class(acc, 'AC', 'weight', RF, 'ACC', apply_tc=tc), get_object = 'performance')
        NSI = fcs.get_port_objects(fcs.port_class(nsi, 'NI', 'weight', RF, 'NSI', apply_tc=tc), get_object = 'performance')
        STR = fcs.get_port_objects(fcs.port_class(st_reversal, 'cumret', 'lag_ME', RF, 'STR', apply_tc=tc), get_object = 'performance')
        LTR = fcs.get_port_objects(fcs.port_class(lt_reversal, 'cumret', 'lag_ME', RF, 'LTR', apply_tc=tc), get_object = 'performance')
        EP = fcs.get_port_objects(fcs.port_class(ep, 'EP', 'weight', RF, 'EP', apply_tc=tc), get_object = 'performance')
        CFP = fcs.get_port_objects(fcs.port_class(cfp, 'CFP', 'weight', RF, 'CFP', apply_tc=tc), get_object = 'performance')

        factors_list = [SMB, HML, UMD, RMW, CMA, ACC, NSI, STR, LTR, EP, CFP]

        # GENERATE AVERAGE COST FOR ADDITIONAL FACTORS
        fcs.write_excel(data_file, 'TC', fcs.gen_tc_df(factors_list, BAB_cost))
        additional_factors_net = fcs.additional_net(factors_list, additional_factors)
            
    if tc == 'yes':
        # IF TC, ADD ADDITIONAL FACTORS NET OF TC
        factor_longshort = factor_longshort.merge(additional_factors_net.set_index('date'), left_index=True, right_index=True)
        fcs.write_excel(output_file, 'Table 18 - Panel A', port_stats.summary_stats(factor_longshort, 12).merge(port_stats.ttest_0(factor_longshort), left_index=True, right_index=True, how='left'))

    else:
        # ELSE ADD GROSS ADDITIONAL GROSS OF TC
        factor_longshort = factor_longshort.merge(additional_factors.set_index('date'), left_index=True, right_index=True)

        # DATA FOR ANALYSIS
        fcs.write_excel(data_file, 'ANOMALIES', factor_longshort)
        fcs.write_excel(data_file, 'MOM_10_VW', fcs.stock_mom_ports(crsp_data, RF, 11, 2))
        fcs.write_excel(output_file, 'Table 6 - Filtered Sample', factor_legs.mean(axis=0).to_frame().T*100*12)
            
        # DATA STATS
        stat = port_stats.summary_stats(MKTRF.rename('MKTRF').to_frame(), 12)
        fcs.write_excel(output_file, 'Table 4 - Filtered Sample', pd.concat([sample_stats_filtered, stat.transpose()], axis=1))
    
        # FILTERING PROCES
        fcs.write_excel(output_file, 'Figure 10', stocks_time)
        filtered_rows_stocks = pd.DataFrame(row_total).rename(columns={0: 'label', 1: 'rows'}).set_index('label').merge(pd.DataFrame(stocks_total).rename(columns={0: 'label', 1: 'stocks'}).set_index('label'), left_index=True, right_index=True)
        fcs.write_excel(output_file, 'Figure 11', filtered_rows_stocks)

#######################################
### CALL FILTERED AND RAW FUNCTIONS ###
#######################################

print('PREPARING RAW DATA:', time_read())
raw_data_handling(crsp_data_nofilters_norm, crsp_data_nofilters_long, crsp_compustat.copy(), output_file, data_file)

print('PREPARING FILTERED DATA:', time_read())
filtered_data_handling(crsp_data.copy(), crsp_data_long.copy(), crsp_compustat.copy(), output_file, data_file, tc='no')

print('PREPARING FILTERED DATA WITH TRANSACTION COSTS:', time_read())
filtered_data_handling(crsp_data.copy(), crsp_data_long.copy(), crsp_compustat.copy(), output_file, data_file, tc='yes')

print('DATA IS PROCESSED. OPEN "__main_(2)__.py" TO CONTINUE:', time_read())
