# -*- coding: utf-8 -*-
"""
Data Filtering functions

@author: Mathias Dam Hansen
"""

import pandas as pd
import numpy as np
from pandas.tseries.offsets import YearEnd, MonthEnd

############################
### MARKET EQUITY FILTER ###
############################

def ME_filter(crsp_data, percentage = 0.10):
    """
    Filters the DataFrame based on market capitalization (ME) and exchange code.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data to filter.
    percentage : float, optional
        Percentage for calculating market capitalization thresholds (default is 0.10).
    
    Returns
    -------
    pandas DataFrame
        DataFrame with filtered data based on market capitalization and exchange code.
    """
    df = crsp_data.copy()

    percentage_name = f'{round(percentage * 100)}%'
    
    df = df[(df['ME'] > 0) & (df['exchange_code'] == 1)]
    df = df.groupby('month_end')['ME'].describe(percentiles = [percentage]).reset_index()[['month_end', percentage_name]]
    
    df['microcap_excl'] = df[percentage_name].rolling(36).mean()
    df['microcap_incl'] = df['microcap_excl'] * 1.5
    return df

def ME_flag(group):
    """
    Sets the 'microcap_flag' column based on conditions and groups by 'permno'
    
    Parameters
    ----------
    df : group
        The group object containing ME data to process.
    
    Returns
    -------
    pandas DataFrame
        A new DataFrame with the 'ME_flag' column set according to the logic.
    """
    
    flag = 0
    for idx, row in group.iterrows():
        if row['ME'] < row['microcap_excl']:
            flag = 1

        elif row['ME'] > row['microcap_incl']:
            flag = 0

        group.loc[idx, 'microcap_flag'] = flag
    return group

##########################
### PENNY STOCK FILTER ###
##########################

def penny_filter(df, penny_threshold):
    """
    Filters penny stocks based on average price on the preeceding year.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data to filter.
    penny_threshold : float
        Threshold for determining penny stocks.
    
    Returns
    -------
    pandas DataFrame
        DataFrame with a 'penny_flag' column indicating penny stocks based on the average price.
    """

    df['avg_price'] = df.groupby('permno')['price'].rolling(window = 12).mean().reset_index(drop = True)
    df['penny_flag'] = np.where(df['avg_price'] < penny_threshold, 1, 0)
    
    return df

##################
### AMV FILTER ###
##################

def volume(df, AMV_window, AMV_threshold):
    """
    Filters stocks based on average trading volume (AMV).
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the volume data to filter.
    AMV_window : int
        Rolling window size for calculating average trading volume.
    AMV_threshold : float
        Threshold for determining if volume is too low based on AMV.
    
    Returns
    -------
    pandas DataFrame
        DataFrame with a 'volume_flag' column indicating lack of volume (volume_flag).
    """

    df['rolling_volume'] = (df
                            .groupby('permno')['volume']
                            .ffill(limit = 3)
                            .rolling(AMV_window)
                            .mean()
                            .reset_index(level = 0, drop = True))

    df['volume_flag'] = np.where((df['volume'] / df['rolling_volume']) < AMV_threshold, 1, 0)
    return df

##################################
### Extreme Returns Correction ###
##################################

# Extreme return
def extreme_return(df, max_return):
    """
    Flags extreme returns and corrects them if they exceed a maximum threshold.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the return data.
    max_return : float
        Maximum threshold for extreme returns.
    
    Returns
    -------
    pandas DataFrame
        DataFrame with an 'extreme_ret_flag' column indicating extreme returns and corrected 'ret' values.
    """

    df['extreme_ret_flag'] = np.where(df['ret'] > max_return, 1, 0)
    df['ret'] = np.where(df['ret'] > max_return, np.nan, df['ret'])
    return df

# Extreme return correction
def extreme_return_correction(group, upper, lower):
    """
    alters extreme return corrections within groups (Permnos) based on upper and lower thresholds.
    
    Parameters
    ----------
    group : pandas GroupBy object
        GroupBy object representing the grouped data.
    upper : float
        Upper threshold for extreme returns.
    lower : float
        Lower threshold for extreme returns.
    
    Returns
    -------
    pandas DataFrame
        DataFrame with corrected extreme returns flagged.
    """

    extreme_corrections = group[((group['ret'] > upper) & (group['ret'].shift(1) < lower)) |
                             ((group['ret'] < lower) & (group['ret'].shift(1) > upper))]

    count = 0
    # Replace price and ret with NaN for those months
    for idx in extreme_corrections.index:
        group.loc[idx:idx + 1, ['price', 'ret', 'retx']] = np.nan
        group.loc[idx:idx + 1, ['extreme_ret_correction_flag']] = 1       
        count += 1
        group.loc[idx, 'extreme_return_flag'] = 1
    return group

# Zero Percent end Return
def zero_pct_returns_filter(group):
    """
    Removes consecutive zeros from the ret column by iterating from the last row upwards and drops slices containing only zeros.
    
    Parameters
    ----------
    group : pandas GroupBy object
        GroupBy object representing the grouped data.
    
    Returns
    -------
    pandas DataFrame
        DataFrame with consecutive zero percent returns removed.
    """
    
    if len(group) > 2 and group['ret'].iloc[-1].sum() == 0:
        for i in range(1, len(group)):
            zero_returns = group['ret'].iloc[-i:].sum()
            if zero_returns.sum() != 0:
                return group
            else:             
                group.drop(group.iloc[-i:].index, inplace = True)
    return group

###########################
### Observations Filter ###
###########################

def min_obs(df, min_obs, grouping = 'permno'):
    """
    Filters out groups with fewer observations than a specified threshold.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data to filter.
    min_obs : int
        Minimum number of observations required for each group.
    grouping : str, optional
        Column name to group by (default is 'permno').
    
    Returns
    -------
    pandas DataFrame
        DataFrame with groups containing fewer observations than min_obs removed.
    """
    
    df = df.groupby(grouping).filter(lambda x: len(x) >= min_obs)
    return df
    
###################
### JUNE SCHEME ###
###################
    
def JuneScheme(row):
    """
    Adjusts the month_end date based on the June scheme:
        Data before April (Months 1-3, Jan-Mar) of year t is 'moved' to June of year t.
        Data after March (Months 4-12, Apr-Dec) is moved to June of year t+1.
    
    Parameters
    ----------
    row : pandas Series
        Row representing a data point.
    
    Returns
    -------
    pandas Series
        Series with the adjusted month_end date.
    """

    date = row['date']
    
    if date.month < 4:
        month_end = date + YearEnd(0) + MonthEnd(-6)
    else:
        month_end = date + YearEnd(0) + MonthEnd(6)

    return pd.Series({'month_end': month_end})