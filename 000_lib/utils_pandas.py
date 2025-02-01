
import math
import pandas as pd
import numpy as np
from scipy import stats

from utils_math import *



def standardization(df, column_name, custom_mean=None, custom_std=None):
    df_mean = df[column_name].mean()
    df_std = df[column_name].std()
    
    if custom_mean is not None:
        df_mean = custom_mean
    if custom_std is not None:
        df_std = custom_std
        
    df[column_name] = (df[column_name] - df_mean)/df_std
    return df_mean, df_std

def one_hot_encoding(df, column_name, p='none', delete=True):
    pre = column_name
    if p != 'none':
        pre = p

    df = df.join(pd.get_dummies(df[column_name], prefix=pre))
    if delete:
        return df.drop(column_name, axis=1)
    else:
        return df    
    
def remove_outliers(dataset, columns):
    dataset =  dataset[(np.abs(stats.zscore(dataset.loc[:, columns])) <= 3).all(axis=1)]
    return dataset

    
def explore(df, column_name, detect_not_numbers=True, show_info=False):
    global errors
    errors = {'nan': 0}
    if detect_not_numbers:
        result = df[~df[column_name].apply(is_real_number)]
    else:
        result = df[df[column_name].apply(is_real_number)]
        
    if show_info:
        print(result)
        
    return errors
    
def custom_replace(df, column_name: str, method: str, vars: list):
    
    replaced = df.copy()    
    
    replacement = 0
    
    if method == "mean":
        replacement = replaced[replaced[column_name].apply(is_real_number)][column_name].mean()
        # replacement = df.loc[:, df.loc[:, column_name].apply(is_real_number)][column_name].mean()
    elif method == "median":
        replacement = replaced[replaced[column_name].apply(is_real_number)][column_name].median()
        # replacement = df.loc[:, df.loc[:, column_name].apply(is_real_number)][column_name].median()
    elif method == "mode":
        replacement = replaced[replaced[column_name].notna()][column_name].mode()[0]
        # replacement = df.loc[:, df.loc[:, column_name].notna()][column_name].mode()[0]

    map = {key: replacement for key in vars}
    replaced[column_name].replace(map, inplace=True)
    return replaced
    
    
def getDeepKeyValues(tree:dict, prefix:str, column_data:dict)->None:
    """Recursive solution to convert a JSON document into play columns.
    Depending on jerarchy. IMPORTANT: Thisneeds an empty dict
    column_data, reusable for multiple calls. Should be passed by reference

    Args:
        tree (dict): the root document part
        prefix (str): prefix to append recursively
    Global:
        column_data (dict): traceback of columns generated
    """
    
    for key,value in tree.items():
        prefix_new = f"{prefix}"
        if prefix != "":
            prefix_new += "_"
        key_new = f"{prefix_new}{key}"
        
        if type(value) is dict:
            getDeepKeyValues(value, key_new, column_data)
        # elif type(value) is list:
        #    for val in value:
        #        getDeepKeyValues(val, key_new)    
        else:
            if key_new not in column_data:
                column_data[key_new] = []
            column_data[key_new].append(value)
    
def pandasLimitsDisable(disable:bool=True):
    if disable:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
    else:
        pd.reset_option('^display.', silent=True)
        
        
def typecast_column(column: pd.Series, data_type):
    if data_type == 'datetime':
        result = pd.to_datetime(column)
    elif data_type == 'timedelta':
        result = column.apply(lambda row: np.int16(pd.Timedelta(row).seconds))
    elif data_type == int:
        result = column.astype(np.int32)
    elif data_type == float:
        result = column.astype(np.float32)
    else:
        result = column.astype(data_type)
    return result