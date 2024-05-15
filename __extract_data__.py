# -*- coding: utf-8 -*-
"""
Extract data from CRSP and COMPUSTAT libraries

@author: Mathias Dam Hansen
"""

import wrds

#######################
### Connect to WRDS ###
#######################

conn = wrds.Connection(wrds_username = 'maha19bg')

##############################
### Loading COMPUSTAT Data ###
##############################
            
compustat = conn.raw_sql("""
                    select gvkey, datadate AS date, pstkl, pstkrv, pstk, fyear, 
                    at, lt, ceq, seq, mib, revt, cogs, xsga, xint, dp, ebit, 
                    txditc, txdb, itcb, txdi, csho, act, lct, che, dlc,
                    txc, xido, ib, pi, wcap
                    from comp.funda
                    where indfmt = 'INDL'
                    and datafmt = 'STD'
                    and popsrc = 'D'
                    and consol = 'C'
                    and datadate >= '12/01/1957'
                    order by gvkey, fyear, datadate
                    """, date_cols = ['datadate'])
                    
"""
Library:
compustat.funda (COMPUSTAT fundamentals file)

Variables:
gvkey:    Unique Identifier (similar to permco / permno)
datadate: date column
pstkl:    Preferred Stock; Involuntary liquidation value
pstkrv:   Preferred Stock; The higher of voluntary liquidation or redemption value
pstk:     Preferred stock; Par value (As per Balance Sheet)
fyear:    Fiscal year of the current fiscal year-end month.
at:       Total Assets 
lt:       Total Liabilities
ceq:      Book value of common equity
seq:      Stockholders’ equity
mib:      Minority Interest
revt:     Revenue
cogs:     Cost of Goods Sold
xsga:     Selling, General and Administrative Expense
xint:     Interest and Related Expense
dp:       Depreciation and Amortization
ebit:     Earnings Before Interest and Related Expense and Tax
txditc:   Deferred Taxes and Investment Tax Credit (txdb + itcb)
txdb:     Deferred Taxes (Balance Sheet)
itcb:     Investment Tax Credit (Balance Sheet)
txdi:     Deferred Income Taxes
csho:     Common shares outstanding
act:      Total Current Assets
lct:      Total Current Liabilities
che:      Cash and Short-Term Investments
dlc:      Total Debt in Current Liabilities
txc:      Income Taxes - Current
xido:     Extraordinary Items and Discontinued Operations
ib:       Income Before Extraordinary Items and Discontinued Operations)
pi:       Pre-tax Income
wcap:     Working Capital

Sourcing criteria:
indfmt: Format of company reports (Industrial, INDL, or Financial Services, FS)
popsrc: Country source (Domestic (USA, Canada and ADRs), D, otherwise, I)
consol: Financial statement reporting (Consolidated Financial Statements, D, otherwise blank)
datadate: Data date
"""

#########################
### Loading CRSP Data ###
#########################

crsp_ret = conn.raw_sql("""                                
                            select a.permno, a.permco, a.date, a.ret, a.retx, a.shrout, a.prc, a.vol, a.cfacpr, a.cfacshr,
                            b.shrcd, b.exchcd, b.comnam, b.siccd, b.ncusip
                            from crsp.msf as a
                            left join crsp.msenames as b
                            on a.permno = b.permno
                            and b.namedt <= a.date
                            and a.date <= b.nameendt
                            where a.date >= '12/01/1957'
                            and b.exchcd in (1, 2, 3)
                            """, date_cols = ['date']) 

"""
Library.a:
crsp.msf: CRSP Monthly Stock File on Securities

Variables:
permno:   Unique Identifier in CRSP file
permco:   Unique Identifier in CRSP file
date:     Date
ret:      Returns in Common Stock
retx:     Returns excl. Dividends, Ordinary dividends and certain other regularly taxable dividends
shrout:   Shares Outstanding (Publicly held shares)
vol:      Volume
cfacpr:   Cumulative Factor to Adjust Price
cfacshr:  Cumulative Factor to Adjust Shares

Library.b:
crsp.msenames: CRSP Monthly Stock Event - Name History

Variables:
shrcd:    Share Code
exchcd:   Exchange Code (-2	= Halted by NYSE or AMEX, -1 = Suspended by NYSE, AMEX, or NASDAQ, 0 = Not Trading on NYSE, AMEX, or NASDAQ, NYSE = 1, AMEX = 2, and NASDAQ = 3)
comnam:   Company Name
siccd:    SIC code
ncusip:   Unique Identifier for North American securities in CRSP file
namedt:   Names Date
nameendt: Names Ending Date
"""

######################################
### Loading CRSP Data - Delistings ###
######################################

crsp_delist = conn.raw_sql("""
                          select permno, dlret, dlstdt, dlstcd
                          from crsp.msedelist
                          """, date_cols = ['dlstdt'])

"""
Library:
crsp.msedelist: CRSP Monthly Stock Event - Delisting

Variables:
permno:  Unique Identifier in CRSP file
dlret:   Delisting Return (Post delisting)
dlstds:  Delisting Date (Last day of trading)
dlstcd:  Delisting Code (100: Active, 200: Mergers, 300: Exchanges, 400: Liquidations, 500:	Dropped, 600: Expirations, 900: Domestics that became Foreign)
"""

########################################
### Loading link data CRSP/COMPUSTAT ###
########################################

crsp_compustat = conn.raw_sql("""
                  select gvkey, lpermno as permno, linktype, linkprim, linkdt, linkenddt
                  from crsp.ccmxpf_linktable
                  where substr(linktype, 1, 1) = 'L'
                  and (linkprim = 'C' or linkprim = 'P')
                  """, date_cols=['linkdt', 'linkenddt'])

"""
Library:
crsp.ccmxpf_linktable: crsp/compustat merged - Link History

Variables: # https://www.kaichen.work/?p=138
lpermno:   Historical CRSP permno link to compustat Record
linktype:  Link Type Code (2-character code providing additional detail on the usage of the link data)
linkprim:  Primary Link Marker
linkdt:    First Effective Date of Link
linkenddt: Last Effective Date of Link
"""

#########################
### SAVE AS CSV FILES ###
#########################

import os
directory = os.getcwd()

# Naming
compustat_name = os.path.join(directory, '__import_files__', 'compustat.csv')
crsp_ret_name = os.path.join(directory, '__import_files__', 'crsp_ret.csv')
crsp_delist_name = os.path.join(directory, '__import_files__', 'crsp_delist.csv')
crsp_compustat_name = os.path.join(directory, '__import_files__', 'crsp_compustat.csv')

# output to csv
compustat.to_csv(compustat_name, date_format = '%Y-%m-%d')
crsp_ret.to_csv(crsp_ret_name, date_format = '%Y-%m-%d')
crsp_delist.to_csv(crsp_delist_name, date_format = '%Y-%m-%d')
crsp_compustat.to_csv(crsp_compustat_name, date_format = '%Y-%m-%d')
