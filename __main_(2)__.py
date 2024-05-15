# -*- coding: utf-8 -*-
"""
This document runs the second part of the code related to factor analysis

@author: Mathias Dam Hansen
"""
import os
import time
import pandas as pd
import scipy.stats as stats
#import statsmodels.api as sm
#import statsmodels as sm_2
#import numpy as np

from sklearn.decomposition import PCA

#############
### PATHS ###
#############

# DIRECTORY
directory = r'C:\Users\mathi\Desktop\Thesis model'
os.chdir(directory)

# NAMES FOR INTERMEDIATE DATA (FACTOR PORTFOLIOS --> FACTOR MOMENTUM PORTFOLIOS)
data_file = 'Intermediate_data.xlsx'

# EXCEL FILE FOR FINAL OUTPUT
output_file = 'Tables&Figures.xlsx'

######################
### CUSTOM MODULES ###
######################

import supporting_modules.__port_stats__ as port_stats
import supporting_modules.__factor_functions__ as fcs
import supporting_modules.__factor_analysis_functions__ as factor_analysis

########################
### TIMING FUNCTIONS ###
########################

def time_read():
    return time.strftime("%H:%M:%S", time.localtime(time.time()))

#################
### LOAD DATA ###
#################

print('INITIATE PART TWO OF THE MODEL. ESTIMATED TIME IS 10 MINS:', time_read())

data_file = 'intermediate_data.xlsx'
output = 'Tables&Figures.xlsx'
pd.set_option('display.max_columns', 5)
import warnings
warnings.filterwarnings('ignore')

# READ ANOMOMALIES
data = pd.read_excel(os.path.join(directory, data_file), header = 0, index_col = 0, parse_dates = True, sheet_name = 'ANOMALIES')
data_nf = pd.read_excel(os.path.join(directory, data_file), header = 0, index_col = 0, parse_dates = True, sheet_name = 'ANOMALIES_NF')
data.index = pd.to_datetime(data.index, format = '%Y%m').to_period('M')
data_nf.index = pd.to_datetime(data_nf.index, format = '%Y%m').to_period('M')

# READ 10 UMD PORTFOLIOS AND WINNERS - LOSERS
UMD_10ports = pd.read_excel(os.path.join(directory, data_file), header = 0, index_col = 0, parse_dates = True, sheet_name = 'MOM_10_VW')
UMD_10ports.index = pd.to_datetime(UMD_10ports.index, format = '%Y%m').to_period('M')

# TC Data
TC_data = pd.read_excel(os.path.join(directory, data_file), header = 0, index_col = 0, parse_dates = True, sheet_name = 'TC')
TC_data.index = pd.to_datetime(TC_data.index, format = '%Y%m').to_period('M')

#########################
### FACTOR STATISTICS ###
#########################

order = ['MKTRF', 'SMB', 'HML', 'UMD', 'RMW', 'CMA', 'ACC', 'NSI', 'LTR', 'STR', 'CFP', 'EP', 'BAB', 'LIQ', 'RES', 'QMJ']

anomalies = data[order].drop(columns=('MKTRF'))
anomalies_nf = data_nf[order].drop(columns=('MKTRF'))

all_factors = data[order]
all_factors_nf = data_nf[order]

port_stat = (port_stats
             .summary_stats(all_factors, 12)
             .merge(port_stats.ttest_0(all_factors), left_index=True, right_index=True, how='left'))

port_stat_NF = (port_stats
                .summary_stats(all_factors_nf, 12)
                .merge(port_stats.ttest_0(all_factors_nf), left_index=True, right_index=True, how='left'))

fcs.write_excel(output, 'Table 5 - Filtered Sample', port_stat)
fcs.write_excel(output, 'Table 5 - Raw Sample', port_stat_NF[['Annualised ret', 'Annualised vol', 'Sharpe ratio']])
fcs.write_excel(output, 'Appendix B', port_stat_NF)

############
### CORR ###
############

fcs.write_excel(output, 'Table 7', port_stats.corr_and_p(all_factors))

##############
### CUMRET ###
##############

# CUMRET BY CATEGORY VOLSCALED
fcs.write_excel(output, 'Figure 13', factor_analysis.cumret_volscaled_category(all_factors))

# CUMRET BY FACTOR VOLSCALED
fcs.write_excel(output, 'Figure 14-18', factor_analysis.cumret_volscaled_factors(all_factors))

# CUMRET UMD
fcs.write_excel(output, 'Figure 21', factor_analysis.cumret(all_factors['UMD'].to_frame()))

# CUMRET BY FM STRATEGY VOLSCALED
#fcs.write_excel(output, 'CUMRET_VOLSCALED_FM_UMD', factor_analysis.cumret_volscaled_fm(anomalies, 12, 1, vol_scale = 'UMD'))
#fcs.write_excel(output, 'CUMRET_VOLSCALED_FM_EW', factor_analysis.cumret_volscaled_fm(anomalies, 12, 1, vol_scale = 'EW'))
#fcs.write_excel(output, 'RET_VOLSCALED_FM_UMD', factor_analysis.volscaled_fm(anomalies, 12, 1, vol_scale = 'UMD'))

####################################
### PAST YEAR RETURNS REGRESSION ###
####################################

# SIGN (12-1 FORMATION)
fcs.write_excel(output, 'Table 8', factor_analysis.regress(anomalies, 12, 1, 'flag'))
#fcs.write_excel(output, 'UNIVARREG_SIGN_12_1FM_NF', factor_analysis.regress(anomalies_nf, 12, 1, 'flag'))

# SIGN (1-1 FORMATION)
fcs.write_excel(output, 'Table 16 - 1-month', factor_analysis.regress(anomalies, 1, 1, 'flag'))

#########################
### FACTOR PORTFOLIOS ###
#########################

# CREATE PORTFOLIOS
EW = factor_analysis.port(anomalies, 'EW', 'EW', 12, 1)[0]
TS = factor_analysis.port(anomalies, 'TS', 'TS', 12, 1)[0]
CS = factor_analysis.port(anomalies, 'CS', 'CS', 12, 1)[0]
FM = EW.merge(TS, left_index=True, right_index=True).merge(CS, left_index=True, right_index=True)

# CALCULATE NUMBER OF TS WINNERS
TSFMW_N = factor_analysis.port(anomalies, 'TS', 'TS', 12, 1)[1]['Winners']

FMUMDMKT = FM[['EW', 'TS', 'CS']].merge(all_factors[['MKTRF', 'UMD']], left_index=True, right_index=True)
FMUMDMKT = FMUMDMKT[['UMD', 'EW', 'TS', 'CS', 'MKTRF']]

fcs.write_excel(output, 'Figure 19', factor_analysis.cumret_volscaled_factors(FMUMDMKT, vol_scale = 'EW'))

##############################
### PORTFOLIOS PERFORMANCE ###
##############################

fcs.write_excel(output, 'Table 9', factor_analysis.port_summary(anomalies, anomalies_nf, 12, 1))
fcs.write_excel(output, 'Figure 20', factor_analysis.decade_performance(anomalies, anomalies_nf, 12, 1))
fcs.write_excel(output, 'Figure 22', FM)

##########################################
### EIGEN PORTFOLIOS EXPLANATORY POWER ###
##########################################

# NORMALISE RETURNS
normalised_returns = anomalies.apply(stats.zscore, nan_policy = 'omit', axis = 0).dropna()

# COVARIANCE MATRIX
cov_matrix = normalised_returns.cov()

# EIGEN DECOMPOSITION USING PCA
pca = PCA()

# EIGENPORTFOLIOS
pca.fit(cov_matrix)
fcs.write_excel(output, 'Appendix C', factor_analysis.explained_var(pca))

################################################
### REGRESSION ON MOMENTUM SORTED PORTFOLIOS ###
################################################

# TIME-SERIES PC MOMENTUM PORTFOLIO    
TSP = factor_analysis.TSPC_port(anomalies, 10, 12, 1).rename(columns={'ret_sign': 'beta'})

# EXTRACT FF5 and FF5 + UMD      
FF5 = all_factors[['MKTRF', 'SMB', 'HML', 'RMW', 'CMA']]
FF5UMD = all_factors[['MKTRF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD']]

# EXOGENOUS COMBINATIONS
combinations = []
combinations.append(('FF5',FF5.rename(columns={'CMA': 'beta'})))
combinations.append(('FF5UMD',FF5UMD.rename(columns={'UMD': 'beta'})))
combinations.append(('FF5TS', FF5.merge(TS['TS'], left_index=True, right_index=True).rename(columns={'TS': 'beta'})))
combinations.append(('FF5CS', FF5.merge(CS['CS'], left_index=True, right_index=True).rename(columns={'CS': 'beta'})))
combinations.append(('FF5PC1-10', FF5.merge(TSP, left_index=True, right_index=True)))
factor_analysis.regress_umd(UMD_10ports, combinations).T

fcs.write_excel(output, 'Table 10', factor_analysis.regress_umd(UMD_10ports, combinations))

#########################
### REGRESSION ON UMD ###
#########################

combinations = []
combinations.append(('FF5',FF5.rename(columns={'CMA': 'beta'})))
combinations.append(('FF5TS', FF5.merge(TS['TS'], left_index=True, right_index=True).rename(columns={'TS': 'beta'})))
combinations.append(('FF5CS', FF5.merge(CS['CS'], left_index=True, right_index=True).rename(columns={'CS': 'beta'})))

for i in range(0, 11):
    TSPC = factor_analysis.TSPC_port(anomalies, i, 12, 1)
    combinations.append((f'PC{i}', FF5.merge(TSPC, left_index=True, right_index=True)))

fcs.write_excel(output, 'Table 11', factor_analysis.regress_alpha(anomalies['UMD'].to_frame(), combinations))

#########################
### REGRESSIONS ON FM ###
#########################

reg = FM[['TS', 'CS']].copy()

for i in range(0, 10):
    TSPC = factor_analysis.TSPC_port(anomalies, i, 12, 1)
    reg[f'PC{i+1}'] = TSPC

combinations = []
combinations.append(('FF5+UMD', FF5UMD))
fcs.write_excel(output, 'Table 12', factor_analysis.regress_alpha(reg, combinations))

#########################
### FORMATION PERIODS ###
#########################

formation_window = []
formation_window.append(('1-3', 3, 1))
formation_window.append(('1-6', 6, 1))
formation_window.append(('1-12', 12, 1))
formation_window.append(('2-12', 11, 2))
formation_window.append(('7-12', 6, 7))
formation_window.append(('13-60', 48, 13))
formation_ports = factor_analysis.formationperiod(anomalies, formation_window)

# STATS BY FORMATION PERIOD
form_stats = (port_stats
             .summary_stats(formation_ports, 12)
             .merge(port_stats.ttest_0(formation_ports), left_index=True, right_index=True, how='left'))
fcs.write_excel(output, 'Table 15 - Stats', form_stats)
#fcs.write_excel(output, 'FORMPERIOD_ret_FM', formation_ports)

# REGRESSIONS VERSUS FF5 AND EW
CSTS_regress = []
CSTS_regress.append(('FF5', FF5))
CSTS_regress.append(('UMD', anomalies['UMD']))
CSTS_regress.append(('EW',EW))
fcs.write_excel(output, 'Table 15 - Alpha', factor_analysis.regress_alpha(formation_ports, CSTS_regress))

#########################
### TRANSACTION COSTS ###
#########################

# ALL FACTORS
EW_tc = factor_analysis.port_tc(anomalies, 'EW', 'EW', 12, 1, TC_data)
TS_tc = factor_analysis.port_tc(anomalies, 'TS', 'TS', 12, 1, TC_data)
CS_tc = factor_analysis.port_tc(anomalies, 'CS', 'CS', 12, 1, TC_data)
FM_tc = EW_tc.merge(TS_tc, left_index=True, right_index=True).merge(CS_tc, left_index=True, right_index=True)
fcs.write_excel(output, 'Table 18 - Panel B', port_stats.summary_stats(FM_tc, 12).merge(port_stats.ttest_0(FM_tc), left_index=True, right_index=True, how='left'))
