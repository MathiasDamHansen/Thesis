# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:18:51 2024

@author: mathi
"""

import os
import time
import pandas as pd
import numpy as np
import datetime as dt
from pandas.tseries.offsets import YearEnd, MonthEnd, MonthBegin
import scipy.stats as stats
import statsmodels.api as sm
import pyanomaly.analytics as pyanomaly
from sklearn.decomposition import PCA

import Model_CRSP.__port_stats__ as port_stats


##################################
### CUM RET BY FACTOR CATEGORY ###
##################################

def cumret_volscaled_category(ret, vol_scale = 'MKTRF'):
    """
    Computes cumulative returns for a portfolio of strategies, scaled by volatility relative to a specified benchmark.
    
    Parameters
    ----------
    ret : pandas DataFrame
        Returns data as a DataFrame. The returns data can be in any time frame such as daily, monthly, etc.
    vol_scale : str, optional
        The column name of the benchmark volatility (default is 'MKTRF').
    
    Returns
    -------
    pandas DataFrame
    A DataFrame containing cumulative returns averaged over factor each category, scaled by volatility.
    """
    
    df = ret.copy().fillna(0)
    
    portfolios = [col for col in df.columns if not col.startswith('MKT')]
    basevol = df[vol_scale].std()
    
    for strategy in portfolios:
        df[strategy] = df[strategy] * basevol / df[strategy].std()
    
    avgret_category = pd.DataFrame().reindex(df.index)
    avgret_category['MKTRF'] = df['MKTRF']
    avgret_category['Size'] = df['SMB']
    avgret_category['Value'] = df[['HML', 'EP', 'CFP']].mean(axis = 1)
    avgret_category['Quality'] = df[['RMW', 'CMA', 'NSI', 'ACC', 'QMJ']].mean(axis = 1)
    avgret_category['Trend'] = df[['UMD', 'LTR', 'STR']].mean(axis = 1)
    avgret_category['Volatility'] = df[['BAB', 'LIQ', 'RES']].mean(axis = 1)
    
    for i in avgret_category.columns:
        name = f'cumret_{i}'
        avgret_category[name] = (1 + avgret_category[i]).cumprod() - 1
           
    return avgret_category

def cumret(ret):
    """
    Computes cumulative returns for each factor in the input data.
    
    Parameters
    ----------
    ret : pandas DataFrame
        Returns data as a DataFrame. The returns data can be in any time frame such as daily, monthly, etc.
    
    Returns
    -------
    pandas DataFrame
        A DataFrame containing cumulative returns for each factor.
    """
    cumret = ret.copy() 
    for factor in cumret:          
        cumret[factor] = (1 + cumret[factor]).cumprod() - 1
       
    return cumret

def cumret_volscaled_factors(ret, vol_scale = 'MKTRF'):
    """
    Computes cumulative returns for a portfolio of factors, scaled by volatility relative to a specified benchmark.
    
    Parameters
    ----------
    ret : pandas DataFrame
        Returns data as a DataFrame. The returns data can be in any time frame such as daily, monthly, etc.
    vol_scale : str, optional
        The column name of the benchmark volatility (default is 'MKTRF').
    
    Returns
    -------
    pandas DataFrame
        A DataFrame containing cumulative returns for each factor, scaled by volatility.
    """
    df = ret.copy()
    portfolios = [col for col in df.columns if col != vol_scale]
    basevol = df[vol_scale].std()
    
    for strategy in portfolios:
        df[strategy] = df[strategy] * basevol / df[strategy].std()
        
    cumret_ports = pd.DataFrame().reindex(df.index)
    
    for strategy in df:          
        cumret_ports[strategy] = (1 + df[strategy]).cumprod() - 1
       
    return cumret_ports

def cumret_volscaled_fm(ret, window, shift, vol_scale = 'EW'):
    """
    Computes cumulative returns for the equal-weighted factor portfolio along with the
    time series and cross sectional factor momentum portolio scaled by volatility relative to a specified benchmark.
    
    Parameters
    ----------
    ret : pandas DataFrame
        Returns data as a DataFrame. The returns data can be in any time frame such as daily, monthly, etc.
    window : int
        The rolling window size for calculating portfolio returns.
    shift : int
        The shift value for calculating portfolio returns.
    vol_scale : str, optional
        The column name of the benchmark volatility (default is 'EW').
    
    Returns
    -------
    pandas DataFrame
        A DataFrame containing cumulative returns for each factor, scaled by volatility.
    """
    df = ret.copy()

    portfolio_types = ['EW', 'TS', 'CS']
    data_sets = {'filters': df}
    
    all_ports = pd.DataFrame()
    for portfolio_type in portfolio_types:
        for data_set_name, data_set in data_sets.items():
            _port, count = port(data_set, portfolio_type, data_set_name,  window, shift)
            _port = _port.fillna(0)
            all_ports = pd.concat([all_ports, _port], axis = 1)
            
    all_ports.columns = ['EW', 'TS', 'TSFMW', 'TSFML', 'CSFM', 'CSFMW', 'CSFML']
        
    if vol_scale in ('EW', 'TS', 'CS'):   
        portfolios = [col for col in all_ports.columns]
    
        basevol = all_ports[vol_scale].std()
    else:
        portfolios = [col for col in all_ports.columns if col.startswith(('EW', 'TS', 'CS'))]
        basevol = df[vol_scale].std()
    
    for strategy in portfolios:
        all_ports[strategy] = all_ports[strategy] * basevol / all_ports[strategy].std()
        
    cumret_ports = pd.DataFrame().reindex(all_ports.index)
    
    for strategy in all_ports:          
        cumret_ports[strategy] = (1 + all_ports[strategy]).cumprod() - 1
       
    return cumret_ports

def volscaled_fm(ret, window, shift, vol_scale = 'EW'):
    """
    Computes EW, TSFM and CSFM monthly returns with volatility scaling based on different portfolio types.
    
    Parameters
    ----------
    ret : pandas DataFrame
        Returns data as a DataFrame. The returns data can be in any time frame such as daily, monthly, etc.
    window : int
        Rolling window size for calculating the portfolio.
    shift : int
        Shift value for calculating the portfolio.
    vol_scale : str, optional
        The column name of the benchmark volatility (default is 'EW').
    
    Returns
    -------
    pandas DataFrame
    A DataFrame containing the portfolio factors with volatility scaling.
    """
    df = ret.copy()

    portfolio_types = ['EW', 'TS', 'CS']
    data_sets = {'filters': df}
    
    all_ports = pd.DataFrame()
    for portfolio_type in portfolio_types:
        for data_set_name, data_set in data_sets.items():
            _port, count = port(data_set, portfolio_type, data_set_name,  window, shift)
            _port = _port.fillna(0)
            all_ports = pd.concat([all_ports, _port], axis = 1)
            
    all_ports.columns = ['EW', 'TS', 'TSFMW', 'TSFML', 'CSFM', 'CSFMW', 'CSFML']
        
    if vol_scale in ('EW', 'TS', 'CS'):   
        portfolios = [col for col in all_ports.columns]
    
        basevol = all_ports[vol_scale].std()
    else:
        portfolios = [col for col in all_ports.columns if col.startswith(('EW', 'TS', 'CS'))]
        basevol = df[vol_scale].std()
    
    for strategy in portfolios:
        all_ports[strategy] = all_ports[strategy] * basevol / all_ports[strategy].std()
    
    all_ports = all_ports.merge(df['UMD'], left_index = True, right_index = True)
    
    return all_ports

##########################
### REGRESS MA OR FLAG ###
##########################

def regress(ret, window, shift, var):
    """
    Computes regression statistics for a given factor against returns data.
    
    Parameters
    ----------
    ret : pandas DataFrame
        Returns data as a DataFrame. The returns data can be in any time frame such as daily, monthly, etc.
    window : int
        Rolling window size for calculating the portfolio.
    shift : int
        Shift value for calculating the portfolio.
    var : str
        The column name of the factor to be regressed against returns.
    
    Returns
    -------
    pandas DataFrame
        A DataFrame containing regression statistics (alpha, beta, t-values, and p-values) for each anomaly.
"""
    df = ret.copy()
    
    df = df.stack().dropna().reset_index()
    df.columns = ['date', 'anomaly', 'ret']
    df = df.sort_values(by=['anomaly','date']).set_index(['anomaly', 'date'])
    df['MA'] = df.groupby('anomaly')['ret'].rolling(window).mean().shift(shift).values
    df = df.dropna(subset='MA')
    df['flag'] = np.where(df['MA'] > 0, 1, 0)

    regress = pd.DataFrame()
    
    def model_stats(model, var, name):
        # Extract coefficients (alpha and beta)
        alpha = model.params['const']  # Intercept (alpha)
        beta = model.params[var]  # Slope (beta)
                
        t_alpha = model.tvalues['const']  # T-value for intercept
        t_beta = model.tvalues[var]  # T-value for slope
        p_alpha = model.pvalues['const']  # P-value for intercept
        p_beta = model.pvalues[var]  # P-value for slope
                           
        index_temp = pd.MultiIndex.from_tuples([(f'{name}', 'alpha'), (f'{name}', 'beta')],
                                               names=['Anomaly', 'coef'])
    
        regression_summary = pd.DataFrame({
            'Estimate': [alpha, beta],
            'T-Value': [t_alpha, t_beta],
            'P-Value': [p_alpha, p_beta]}, index=index_temp)
        
        return regression_summary
    
    for i in df.index.get_level_values(0).unique():
        subset = df[df.index.get_level_values(0) == i]
        model = sm.OLS(subset['ret'], sm.add_constant(subset[var])).fit(cov_type='HC1')

        regress = pd.concat([regress, model_stats(model, var, i)])
    
    no_umd = df[df.index.get_level_values('anomaly') != 'UMD']
    pooled_model = sm.OLS(no_umd['ret'], sm.add_constant(no_umd[var])).fit(cov_type='cluster', cov_kwds={'groups': no_umd.index.get_level_values('date')})
    regress = pd.concat([regress, model_stats(pooled_model, var, 'pooled')])

    return regress

################################
### EXPLANATORY POWER OF PCs ###
################################

def explained_var(pca_obj):
    """
    Computes the cumulative explained variance for each principal component (PC) from a PCA object.
    
    Parameters
    ----------
    pca_obj : sklearn.decomposition.PCA
        A fitted PCA object containing the explained variance ratios.
    
    Returns
    -------
    pandas DataFrame
    A DataFrame with the cumulative explained variance for each PC.
    """
    explained_cumvar = np.cumsum(pca_obj.explained_variance_ratio_)
    explained_var = pd.DataFrame(index=['PC'], columns=['Explained variance'])
    
    # Loop over the range
    for i in range(0, len(explained_cumvar)):
        temp = pd.DataFrame({'Explained variance': [explained_cumvar[i]]}, index=[f'PC{i+1}'])
        explained_var = pd.concat([explained_var, temp])
    
    return explained_var

######################
### REGRESS ON UMD ###
######################

def model_stats(model, endogenous, exogenous):
    """
    Computes summary statistics for a regression model.
    
    Parameters
    ----------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        A fitted regression model.
    endogenous : str
        The name of the dependent variable.
    exogenous : str
        The name of the independent variable.
    
    Returns
    -------
    pandas DataFrame
    A DataFrame containing the estimated coefficients and t-values.
    """
    # Extract coefficients and t-values
    alpha = model.params['const']
    alpha_t = model.tvalues['const']

    beta = model.params['beta']
    beta_t = model.tvalues['beta']
               
    # Create a DataFrame with the summary statistics
    regression_summary = pd.DataFrame({
        f'{exogenous}_a': [f"{alpha*100:.2f}", f"({alpha_t:.2f})"],
        f'{exogenous}_b': [f"{beta:.2f}", f"({beta_t:.2f})"],
        }, index=pd.MultiIndex.from_product([[endogenous], ['Estimate', 'T-Value']]))            
    return regression_summary
    
def regress_umd(umd, combinations):
    """
    Performs regression analysis on a set of asset pricing portfolios using the 10 momentum sorted portfolios as the dependent variable.
    
    Parameters
    ----------
    umd : pandas DataFrame
        A DataFrame containing the UMD dataset with factor returns.
    combinations : list of tuples
        A list of tuples, where each tuple contains:
        - A label for the combination of factors / portfolios.
        - A DataFrame representing the factors / portfolios to be included in the regression.
    
    Returns
    -------
    pandas DataFrame
        A DataFrame containing regression summary statistics for each factor combination.
    """
    regress = pd.DataFrame()
    grs_tests = pd.DataFrame()   
    
    for i in umd.columns:
        temp_regress = pd.DataFrame()
        for j in range(0, len(combinations)):
            common_index = umd.merge(combinations[j][1], left_index = True, right_index = True)           
            Y = umd.reindex(common_index.index)[i]
            X =  sm.add_constant(combinations[j][1].reindex(common_index.index))
            model = sm.OLS(Y, X).fit(cov_type='HC1')
            temp_regress = pd.concat([temp_regress, model_stats(model, i, combinations[j][0])], axis=1)
        regress = pd.concat([regress, temp_regress], axis=0)
    
    alpha_abs_mean = regress.xs('Estimate', level=1).astype(float).abs().mean()
    regress.loc[('MeanAbsAlpha', ''), :] = alpha_abs_mean.map("{:.2f}".format)

    grs_stat = []
    grs_pval = []
    
    for j in range(0, len(combinations)):
        common_index = umd.merge(combinations[j][1], left_index = True, right_index = True)           

        factors = combinations[j][1].reindex(common_index.index)
        umdports = umd.reindex(common_index.index)
        
        grs_temp = pyanomaly.grs_test(umdports.drop(columns=['Winners-Losers']), factors)
       
        grs_stat.append(grs_temp[2])
        grs_stat.append('')
        
        grs_pval.append((grs_temp[3]*100))
        grs_pval.append('')       
               
        # Create a DataFrame with the summary statistics
        grs_summary = pd.DataFrame({
            f'{combinations[j][0]}': [grs_stat, grs_pval],
            '' : ['', ''],
            }, index=pd.MultiIndex.from_product([[j], ['F-value', 'P-Value']]))            
      
        grs_tests = pd.concat([grs_tests, grs_summary], axis=1)
        
    regress.loc[('GRS stat', ''), :] = grs_stat    
    regress.loc[('GRS pval', ''), :] = grs_pval
        
    return regress

#########################
### ALPHA REGRESSIONS ###
#########################

def regress_alpha(endogenous, exogenous):
    """
    Performs regression analysis to estimate alpha for each endogenous variable.
    
    Parameters
    ----------
    endogenous : pandas DataFrame
        A DataFrame containing the dependent variables.
    exogenous : list of tuples
        A list of tuples, where each tuple contains:
        - A label for the combination of independent variables.
        - A DataFrame representing the independent variables.
    
    Returns
    -------
    pandas DataFrame
    A DataFrame containing estimated alpha, t-values, p-values, and R-squared for each endogenous variable.
    """    
    regress = pd.DataFrame()

    def stats_output_alpha(model, endogenous, exogenous):
        # Extract coefficients and t-values
        alpha = model.params['const']        
        tval = model.tvalues['const']
        pval = model.pvalues['const']
        rsq = model.rsquared

        regression_summary = pd.DataFrame({
            f'{exogenous}': [f'{alpha*100}', tval, pval, f'{rsq*100}']
            }, index = pd.MultiIndex.from_product([[endogenous], ['alpha', 'tval', 'pval', 'RSQ']]))            
                  
        return regression_summary

    for i in endogenous.columns:
        temp_regress = pd.DataFrame()
        for j in range(0, len(exogenous)):
            common_index = endogenous.merge(exogenous[j][1], left_index = True, right_index = True).dropna()           
            Y = endogenous.reindex(common_index.index)[i]
            X =  sm.add_constant(exogenous[j][1].reindex(common_index.index))
            model = sm.OLS(Y, X).fit(cov_type='HC1')
            temp_regress = pd.concat([temp_regress, stats_output_alpha(model, i, exogenous[j][0])], axis=1)
        regress = pd.concat([regress, temp_regress], axis=0)
    return regress
            
###########################
### PORTFOLIO FUNCTIONS ###
###########################

def port(ret, port, port_name, window, shift, exclude_umd = 'yes'):
    """
    Computes portfolio returns of either the equal-weighted, EW, the times-series, TS, or the cross-sectional (CS) strategy.
    
    Parameters
    ----------
    ret : pandas DataFrame
        A DataFrame containing return data.
    port : str
        The type of portfolio ('EW', 'TS', or 'CS').
    port_name : str
        A label for the portfolio.
    window : int
        Rolling window size for calculating the moving average.
    shift : int
        Shift value for calculating the moving average.
    exclude_umd : str, optional
        Whether to exclude the 'UMD' anomaly (default is 'yes').
    
    Returns
    -------
    pandas DataFrame
        A DataFrame containing portfolio returns, winners' returns, and losers' returns.
    pandas DataFrame or str
        A DataFrame containing the count of winners and losers (only applicable for 'TS' and 'CS' portfolios).
    """

    df = ret.copy()
    
    df = df.stack().dropna().reset_index()
    df.columns = ['date', 'anomaly', 'ret']
    df = df.sort_values(by=['anomaly','date']).set_index('date')
    df['MA'] = df.groupby('anomaly')['ret'].rolling(window = window, min_periods = window).mean().values
    df['MA'] = df.groupby('anomaly')['MA'].shift(shift, fill_value=np.nan)
    df = df.dropna(subset = 'MA')
    
    if exclude_umd == 'yes':    
        df = df[df['anomaly'] != 'UMD']
                
    if port == 'EW':
        EW = pd.DataFrame(df.groupby('date')['ret'].mean()).rename(columns={'ret':'EW'})
        return EW, "NA"

    elif port == 'TS':
        df['flag'] = np.sign(df['MA'])
        df['ret_sign'] = df['flag'] * df['ret']
        
        portfolio = df.groupby('date')['ret_sign'].mean()       
        winners = df[df['flag'] == 1].groupby('date')['ret'].mean()
        losers = df[df['flag'] == -1].groupby('date')['ret'].mean()        
        count = df.groupby(['flag', 'date'])['ret'].count().unstack(level=0).rename(columns={-1:'Losers', 1:'Winners'})         

    elif port == 'CS':
        df = df.merge(median_ret(df, 'MA'), on = 'date', how = 'left')
        df['flag'] = np.sign(df['MA'] - df['MA_median'])
        df['ret_sign'] = df['flag'] * df['ret']

        portfolio = df.groupby('date')['ret_sign'].mean()
        winners = df[df['flag'] == 1].groupby('date')['ret'].mean()     
        losers = df[df['flag'] == -1].groupby('date')['ret'].mean()
        count = df.groupby(['flag', 'date'])['ret'].count().unstack(level=0).rename(columns={-1:'Losers', 1:'Winners'})
        
    ports = pd.concat([portfolio, winners], axis = 1)
    ports = pd.concat([ports, losers], axis = 1)
    
    ports.columns = [port_name, f'Winners_{port_name}', f'Losers_{port_name}']
        
    return ports, count

##################
### PROCESSING ###
##################

def process_portfolios(ret_data, portfolio_type, data_set_name, window, shift):
    """
    Computes portfolio statistics and performs t-tests for a given portfolio type.
    
    Parameters
    ----------
    ret_data : pandas DataFrame
        A DataFrame containing return data.
    portfolio_type : str
        The type of portfolio ('EW', 'TS', or 'CS').
    data_set_name : str
        A label for the dataset.
    window : int
        Rolling window size for calculating the portfolio.
    shift : int
        Shift value for calculating the portfolio.
    
    Returns
    -------
    pandas DataFrame
        A DataFrame containing portfolio statistics and t-test results.
    """
    _port, count = port(ret_data, portfolio_type, data_set_name,  window, shift)
    _port = _port.fillna(0)
    portfoliostats = port_stats.summary_stats(_port, 12)
    portfoliostats = pd.concat([portfoliostats, ttest(_port).T], axis=1)
    portfoliostats.index = pd.MultiIndex.from_product([[portfolio_type], portfoliostats.index])
    return portfoliostats

#########################
### PORTFOLIO SUMMARY ###
#########################

def port_summary(ret_data_WF, ret_data_NF,  window, shift):
    """
    Computes portfolio summary statistics for each of the three factor portfolios.
    
    Parameters
    ----------
    ret_data_WF : pandas DataFrame
        Returns data for factor based on the Filtered sample.
    ret_data_NF : pandas DataFrame
        Returns data for factor based on the Raw sample.
    window : int
        Rolling window size for calculating the portfolio.
    shift : int
        Shift value for calculating the portfolio.
    
    Returns
    -------
    pandas DataFrame
        A DataFrame containing portfolio summaries for each portfolio type.
    """
    portfolio_types = ['EW', 'TS', 'CS']
    data_sets = {'WF': ret_data_WF, 'NF': ret_data_NF}
    
    all_ports = []
    for portfolio_type in portfolio_types:
        for data_set_name, data_set in data_sets.items():
            all_ports.append(process_portfolios(data_set, portfolio_type, data_set_name,  window, shift))
    
    return pd.concat(all_ports)

##########################
### DECADE PERFORMANCE ###
##########################

def decade_performance(ret_data_WF, ret_data_NF, window, shift):
    """
    Computes portfolio summaries for different portfolio types based on provided return data.
    
    Parameters
    ----------
    ret_data_WF : pandas DataFrame
        Returns data for factor based on the Filtered sample.
    ret_data_NF : pandas DataFrame
        Returns data for factor based on the Raw sample.
    window : int
        Rolling window size for calculating the portfolio.
    shift : int
        Shift value for calculating the portfolio.
    
    Returns
    -------
    pandas DataFrame
        A concatenated summary of portfolio performance for each portfolio type.
    """
    output = pd.DataFrame()

    dates = [('1964-07', '1969-12'), ('1969-01', '1979-12'), ('1980-01', '1989-12'), ('1990-01', '1999-12'), ('2000-01', '2009-12'), ('2010-01', '2019-12'), ('2020-01', '2023-12')]
    
    for i in dates:
        df_decade_FF = ret_data_WF[(ret_data_WF.index >= i[0]) & (ret_data_WF.index <= i[1])]
        df_decade_NF = ret_data_NF[(ret_data_NF.index >= i[0]) & (ret_data_NF.index <= i[1])]
        
        decade_summary = port_summary(df_decade_FF, df_decade_NF, window, shift)       
        decade_summary.columns = pd.MultiIndex.from_product([[f'{i[0]} - {i[1]}'], decade_summary.columns])
        output = pd.concat([output, decade_summary], axis = 1)
        
    return output

#########################
### FORMATION PERIODS ###
#########################

def formationperiod(anomaly_data, formation_periods):
    """
    Computes portfolio performance during different formation periods based on anomaly data.
    
    Parameters
    ----------
    anomaly_data : pandas DataFrame
        A DataFrame containing anomaly data (e.g., factor returns).
    formation_periods : list of tuples
        A list of tuples, where each tuple contains:
        - A label for the formation period (e.g., '1-1', '2-1', etc.).
        - The rolling window size for calculating the portfolio.
        - The shift value for calculating the portfolio.
    
    Returns
    -------
    pandas DataFrame
    A DataFrame containing portfolio performance during different formation periods.
    """
    period_performance = port(anomaly_data, 'CS', 'CS_1-1', 1, 1, exclude_umd = 'yes')[0]['CS_1-1']
    period_performance = pd.concat([period_performance, port(anomaly_data, 'TS', 'TS_1-1', 1, 1, exclude_umd = 'yes')[0]['TS_1-1']], axis = 1)

    for i in ['CS', 'TS']:
        for j in range(0, len(formation_periods)):      
            temp_name = f'{i}_{formation_periods[j][0]}'
            temp_factor = port(anomaly_data, i, temp_name, formation_periods[j][1], formation_periods[j][2], exclude_umd = 'yes')[0][temp_name]
            period_performance = period_performance.merge(temp_factor, on = 'date', how = 'left')
    return period_performance

#######################
### EIGENPORTFOLIOS ###
#######################

def TSPC_port(df, PC_n, window, shift, beg_window = 120):
    """
    Computes the Time-Series Principal Component factor momentum portfolio (TSPC).
    
    Parameters
    ----------
    df : pandas DataFrame
        A DataFrame containing the data for factor returns.
    PC_n : int
        The index of the principal component (eigenportfolio) to use.
    window : int
        Rolling window size for calculating the moving average.
    shift : int
        Shift value for calculating the moving average.
    beg_window : int, optional
        The initial rolling window size for eigenportfolio calculation (default is 120).
    
    Returns
    -------
    pandas DataFrame
    A DataFrame containing the TSPC momentum portfolio returns.
    """
    ### CALCULATE EIGENPORTFOLIOS ###
    
    def eigenport_rolling(df, PC_n, beg_window):
            
        eigen_port_ret = pd.DataFrame(columns=['date', 'ret'])
        weights_df = pd.DataFrame(columns=['date', 'wt'])
        
        last_date = (df.index[-1] - MonthEnd(beg_window))
           
        for i in df[df.index < last_date].index:
            dt_start = df.index[0]

            dt_end = i + MonthEnd(beg_window)
            dt_name = dt_end + MonthEnd(1)
        
            # Define the active period based on the rolling window
            temp_data = df[(df.index >= dt_start) & (df.index <= dt_end)]
            
            # Calculate covariance matrix
            cov_matrix = temp_data.dropna().cov()
            
            # Perform eigen decomposition using PCA
            pca = PCA()
            pca.fit(cov_matrix)
            
            # Get the first eigenvector (eigenportfolio)
            eigenvector = pca.components_[PC_n]

            # Normalise weights            
            weights = eigenvector / eigenvector.sum()
            
            # Temp weight df
            temp_weight = pd.DataFrame({'date': [dt_name], 'wt': [eigenvector]})
           
            # Calculate port_returns (Not used)
            eigenportfolio_returns = np.dot(df.loc[dt_name].dropna(), weights)
            
            # Add the return for month t+1 using ret t+1 and weights t
            temp_eig = pd.DataFrame({'date': [dt_name], 'ret': [eigenportfolio_returns]})
            eigen_port_ret = pd.concat([eigen_port_ret, temp_eig], axis=0)
            
            # Append the weights to the main DataFrame
            weights_df = pd.concat([weights_df, temp_weight], axis=0)
            
            weights_formatted = pd.DataFrame(weights_df['wt'].tolist(), columns=[i for i in range(len(weights_df['wt'].iloc[0]))], index=weights_df['date'])
            weights_formatted.columns = df.columns
          
        eigen_port_ret = eigen_port_ret.set_index('date')    
        PC = weights_formatted * df
        return PC
    
    df_copy = df.drop(columns=('UMD')).copy()
    PC = eigenport_rolling(df_copy, PC_n, beg_window).dropna()
    
    ### RESCALE EIGEN VOL TO MATCH EX-ANTE FACTOR VOL ###

    def rescaling(df, anomalies):
        eigen = df.copy()
        data = anomalies.copy()
        
        for i in data.columns:
            for j in range(0, len(eigen.index)):
                vol_data_factor = data[data.index < eigen.index[j]]
                vol_data_pc = eigen[eigen.index <= eigen.index[j]+1]
                
                factor_vol = vol_data_factor[i].std()
                pc_vol = vol_data_pc[i].std()
                rescaled_vol = factor_vol / pc_vol
                
                eigen.loc[eigen.index[j], i] = eigen.loc[eigen.index[j], i] * rescaled_vol
        return eigen

    rescaled_TSPC = rescaling(PC, df_copy)
    
    ### CALCULATE TIME-SERIES PC MOMENTUM PORTFOLIO ###

    def PC_port(ret, window, shift):
        df = ret.copy()
            
        df = df.stack().dropna().reset_index()
        df.columns = ['date', 'anomaly', 'ret']
        df = df.sort_values(by=['anomaly','date']).set_index('date')
        df['MA'] = df.groupby('anomaly')['ret'].rolling(window = window, min_periods = window).mean().values
        df['MA'] = df.groupby('anomaly')['MA'].shift(shift, fill_value=np.nan)
        df = df.dropna(subset = 'MA')   
        
        df['flag'] = np.sign(df['MA'])
            
        winners = df[df['flag'] == 1].groupby('date')['ret'].mean()
        losers = df[df['flag'] == -1].groupby('date')['ret'].mean()
                
        portfolio = winners.sub(losers).fillna(0)
        
        return portfolio
        
    TSPC = PC_port(rescaled_TSPC, window, shift).to_frame().rename(columns={'ret': 'beta'})
    return TSPC

##############################################
### MEDIAN AND ALLOCATION FOR CS PORTFOLIO ###
##############################################

def port_tc(ret, port, port_name, window, shift, tc_data, exclude_umd = 'yes'):
    """
    Computes portfolio returns with transaction costs (TC) based on different portfolio types.
    
    Parameters
    ----------
    ret : pandas DataFrame
        Returns data as a DataFrame. The returns data can be in any time frame such as daily, monthly, etc.
    port : str
        The type of portfolio ('EW', 'TS', or 'CS').
    port_name : str
        A label for the portfolio.
    window : int
        Rolling window size for calculating the moving average.
    shift : int
        Shift value for calculating the moving average.
    tc_data : pandas DataFrame
        Transaction cost data as a DataFrame.
    exclude_umd : str, optional
        Whether to exclude the 'UMD' anomaly (default is 'yes').
    exclude_str : str, optional
        Whether to exclude the 'STR' anomaly (default is 'yes').
    
    Returns
    -------
    pandas DataFrame
        A DataFrame containing portfolio returns, winners' returns, and losers' returns.
        """
    df = ret.copy()
    
    df = df.stack().dropna().reset_index()
    df.columns = ['date', 'anomaly', 'ret']
    df = df.sort_values(by=['anomaly','date']).set_index('date')
    df['MA'] = df.groupby('anomaly')['ret'].rolling(window = window, min_periods = window).mean().values
    df['MA'] = df.groupby('anomaly')['MA'].shift(shift, fill_value=np.nan)
    df = df.dropna(subset = 'MA')
    
    # TC
    tc = tc_data.stack().dropna().reset_index()
    tc.columns = ['date', 'anomaly', 'tc']
    df = df.merge(tc, on = ['anomaly', 'date'], how = 'left')
    
    if exclude_umd == 'yes':    
        df = df[df['anomaly'] != 'UMD']
                
    if port == 'EW':
        # APPLY TC
        df['ret'] = df['ret'] - df['tc']
        EW = pd.DataFrame(df.groupby('date')['ret'].mean()).rename(columns={'ret':'EW'})
        return EW

    elif port == 'TS':
        df['flag'] = np.sign(df['MA'])
        df['ret_sign'] = df['flag'] * df['ret']
               
        # APPLY TC
        df['ret_sign'] = df['ret_sign'] - df['tc']
        df['ret'] = df['ret'] - df['tc']
        
        portfolio = df.groupby('date')['ret_sign'].mean()       
        winners = df[df['flag'] == 1].groupby('date')['ret'].mean()
        losers = df[df['flag'] == -1].groupby('date')['ret'].mean()        

    elif port == 'CS':
        df = df.merge(median_ret(df, 'MA'), on = 'date', how = 'left')
        df['flag'] = np.sign(df['MA'] - df['MA_median'])
        df['ret_sign'] = df['flag'] * df['ret']

        # APPLY TC
        df['ret_sign'] = df['ret_sign'] - df['tc']
        df['ret'] = df['ret'] - df['tc']

        portfolio = df.groupby('date')['ret_sign'].mean()
        winners = df[df['flag'] == 1].groupby('date')['ret'].mean()     
        losers = df[df['flag'] == -1].groupby('date')['ret'].mean()
        
    ports = pd.concat([portfolio, winners], axis = 1)
    ports = pd.concat([ports, losers], axis = 1)
    
    ports.columns = [port_name, f'Winners_{port_name}', f'Losers_{port_name}']
        
    return ports

##############################################
### MEDIAN AND ALLOCATION FOR CS PORTFOLIO ###
##############################################

def median_ret(df, median_val):
    """
    Computes the median value of a specified column in a DataFrame grouped by date.
    
    Parameters
    ----------
    df : pandas DataFrame
        The input DataFrame containing the data.
    median_val : str
        The name of the column for which to compute the median.
    
    Returns
    -------
    pandas DataFrame
    A DataFrame with the median value for each date.
    """
    median = (df            
              .groupby('date')[median_val]
              .median()
              .to_frame()
              .reset_index()
              .rename(columns = {f'{median_val}': f'{median_val}_median'}))
    return median

#############
### TTEST ###
#############

def ttest(port_data):
    """
    Performs a one-sample t-test on each column of the input DataFrame.
    
    Parameters
    ----------
    port_data : pandas DataFrame
        A DataFrame containing the data for which to perform the t-test.
    
    Returns
    -------
    pandas DataFrame
    A DataFrame with t-values and p-values for each column.
    """
    
    test_dict = {}
    
    for i in port_data.columns:
        data = port_data[i]
        t_statistic, p_value = stats.ttest_1samp(data, popmean=0)       
        test_dict[i] = {'T-value': t_statistic, 'P-value': p_value}

    test_df = pd.DataFrame(test_dict)
    return test_df