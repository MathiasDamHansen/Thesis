# -*- coding: utf-8 -*-
"""
Portfolio Statistics

@author: Mathias Dam Hansen
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

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
        
def CVaR_historical(ret, alpha = 5):
    """
    Computes the 1-period Conditional Value at Risk using historical data, at a specified confidence level.
   
    Parameters
    ----------
    ret : pandas DataFrame or pandas Series
        Returns data as a DataFrame or a Series. The returns data can be in any time frame such as daily, monthly, etc...
    alpha : int
        Confidence level in integer. 5% should be input as 5.
    
    Returns
    -------
    Scalar or Series
        Conditional Value at risk: Scalar for Series input, Series for DataFrame input.
    """
    
    z = ret.quantile(alpha/100)
    return -ret[ret < z].mean()
   
def VaR_normal(ret, alpha = 5):
    """
    Computes the Value at Risk assuming normal distribution in data, at a specified confidence level.
        
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
    
    mu = ret.mean()
    sigma = ret.std()
    z = stats.norm.ppf(alpha/100)
    return -(mu + z*sigma)
    
def summary_stats(ret, freq):   
    """
    Compute summary stats for each permno in returns data and display the result in a DataFrame.
    
    Parameters
    ----------
    ret : pandas DataFrame or pandas Series
        Returns data as a DataFrame or a Series. The returns data can be in any time frame such as daily, monthly, etc...
    freq : int
        Frequency of the return in ret in terms of the number of periods per year. 
    
    Returns
    -------
    pandas DataFrame
        Summary statistics DataFrame containing the following features for each ticker in the returns data:
        - Annualised return
        - Annualised volatility
        - Annualised Sharpe ratio
        - Skewness
        - Kurtosis
        - Value at risk
        - Conditional Value at risk
        - Value at risk (normal distribution)
        - Maximum drawdown
    """
    
    if isinstance(ret, pd.Series):
        ret = pd.DataFrame(ret)
    
    avg_ret = ret.mean()
    avg_std = ret.std()
    ann_ret = avg_ret * freq
    ann_vol = avg_std * np.sqrt(freq)
    sr = ann_ret / ann_vol # AS RETURNS ARE ALREADY EXCESS
    skew = ret.skew()
    kurt = ret.kurt()
    var_hist = VaR_historical(ret) 
    cvar_hist = CVaR_historical(ret)
    var_normal = VaR_normal(ret)
    maxDD, when = max_drawdown(ret) # maxDD is a Series with index being the column names in ret
    result = pd.DataFrame({'Monthly ret': avg_ret*100,
                           'Monthly vol': avg_std*100,
                           'Annualised ret': ann_ret*100,
                           'Annualised vol': ann_vol*100,
                           'Sharpe ratio': sr,
                           'Skewness': skew,
                           'Kurtosis': kurt,
                           'Hist VaR': var_hist*100,
                           'Hist CVar': cvar_hist*100,
                           'VaR_normal': var_normal*100,
                           'Max drawdown': maxDD*100
                           })
    return result

def ttest_0(df):
    """
    Perform one-sample two-sided t-test for each column in the DataFrame against a population mean of 0.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data to perform t-tests.
    
    Returns
    -------
    pandas DataFrame
        DataFrame containing t-statistic and p-value for each column.
    """

    tstat = pd.DataFrame()
    pval = pd.DataFrame()

    for column in df.columns:
        tstat_i, pval_i = stats.ttest_1samp(df[column], 0, nan_policy = 'omit')
        tstat[column] = [tstat_i]
        pval[column] = [pval_i]

    # Transpose and rename DFs
    tstat = tstat.T
    pval = pval.T

    tstat.columns = ['tstat']
    pval.columns = ['pval']

    values = tstat.join(pval)
    return values

def corr_and_p(df):
    """
    Calculate correlation coefficients and corresponding p-values for a DataFrame.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data to calculate correlations and p-values.
    
    Returns
    -------
    pandas DataFrame
        DataFrame containing correlation coefficients and p-values.
    """

    rho = df.corr()
    pval = df.corr(method=lambda x, y: stats.pearsonr(x, y)[1]) - np.eye(*rho.shape)
    p = pval.map(lambda x: ''.join(['*' for i in [.001, .01, .05] if x <= i]))
    table = rho.map("{:.2f}".format) + p
    
    rho.columns = pd.MultiIndex.from_product([['Correlation'], rho.columns])
    pval.columns = pd.MultiIndex.from_product([['P-values'], pval.columns])
    table.columns = pd.MultiIndex.from_product([['Correlation and P-values'], table.columns])
    
    output = pd.concat([rho, pval], axis = 1)
    output = pd.concat([output, table], axis = 1)
        
    return output