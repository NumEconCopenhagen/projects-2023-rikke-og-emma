import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

pd.options.mode.chained_assignment = None

# user written modules
#import dataproject



def import_data():

    #loader data og sorterer ikke-receptpligtige fra
    prices = pd.read_csv(r'Data.csv', sep=',')
    ren = {0:'atc_code',1:'product_number',2:'pack_name',3:'marketing_owner',4:'form',5:'strength',6:'pack_size',7:'registration_status',8:'delivery_provision',9:'delivery_provision_change',10:'quantity_unit',11:'note',12:'subsidy_status',13:'substitution_group',14:'language'}
    product_info = pd.read_table(r'product_name_text.txt', sep=';', encoding='latin',header=None).rename(columns=ren)
    samlet_info=pd.merge(prices, product_info, on='product_number', how='left')
    rcpt = samlet_info[samlet_info['delivery_provision'].str.contains('Receptpligtigt', na=False)]
    rcpt = rcpt[rcpt['price_unit'].str.contains('ddd', na=False)]
    return rcpt