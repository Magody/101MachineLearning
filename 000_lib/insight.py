import pandas as pd
import numpy as np
from utils_math import is_discrete, is_real_number

def describeDataset(X):
    describe_numerical = X.describe()
    describe_numerical.loc['missing', :] = (1 - describe_numerical.loc['count', :]/len(X))*100
    print(describe_numerical)
    
    describe_objects = X.describe(include=['O'])
    describe_objects.loc['missing', :] = (1 - describe_objects.loc['count', :]/len(X)) * 100
    describe_objects.loc['possible_duplicates', :] = ((describe_objects.loc['count', :] - describe_objects.loc['unique', :])*100)/len(X)
    print(describe_objects)
    
def getMissingAndDuplicates(dataset: pd.DataFrame, threshold_max=0):
    X = dataset.copy()
    
    dataset_len = len(dataset)
     
    def getMissing(count):
    
        miss = []
        df_miss = (1 - count/dataset_len)
        
        for col in list(df_miss.index):
            value = df_miss[col]
            if value > threshold_max:
                miss.append((col, value))
                
        return miss
    
    miss_numerical = getMissing(X.describe().loc['count', :])
    miss_categorical = getMissing(X.describe(include=['O']).loc['count', :])
            
            
    df_duplicates_categorical = (X.describe(include=['O']).loc['count', :] - X.describe(include=['O']).loc['unique', :])/dataset_len
    duplicates = []

    for col in list(df_duplicates_categorical.index):
        value = df_duplicates_categorical[col]
        if value > 0:
            duplicates.append((col,value))

    return miss_numerical, miss_categorical, duplicates

def exploreSurface(dataset_all: pd.DataFrame):

    df_analisis:pd.DataFrame = pd.DataFrame(columns=["type1", "type2", "porc_unique", "problem", "unique_values"], index=dataset_all.columns)

    for column in list(dataset_all.columns):
        s = dataset_all[column]
        s_values = np.array(s.dropna().values)
        unique_values = np.unique(s_values)
        d = s.describe()
        # print(d)



        count = d["count"]
        if "unique" in d:
            count_unique = d["unique"]
        else:
            count_unique = len(unique_values)
            
            

        porc_unique = count_unique/count


        analisis = {"type1": "unknown", "type2": "unknown", "porc_unique": porc_unique, "problem": "", "unique_values": []}


        if porc_unique > 0.9:
            analisis["type1"] = "useless"
            analisis["problem"] = "High variability, better drop this"

        elif count_unique <= 30: # other threshold: porc_unique < 0.1:
            # is categorical
            analisis["type1"] = "categorical"
            
            
            
            if s.dtype in ["float64", "int64"]:
                analisis["type2"] = "numeric" # but with numbers. Probably we have to get bands
                are_discrete = all(map(is_discrete, unique_values))
            
                if are_discrete:
                    analisis["unique_values"] = list(map(int, unique_values))
                else:
                    analisis["unique_values"] = list(map(float, unique_values))
                
            else:
                analisis["type2"] = "text"
                analisis["unique_values"] = unique_values
            
        else:
            # numerical
            
            are_numbers = True
            if str(s.dtype) in ["object", "categorical"]:
                
                are_numbers = all(map(is_real_number, s_values))
                
            if are_numbers:
                
                analisis["type1"] = "numerical"
                
                are_discrete = all(map(is_discrete, s_values))
                
                if are_discrete:
                    analisis["type2"] = "discrete"
                else:
                    analisis["type2"] = "continuous"
            else:
                
                any_is_a_number = any(map(is_real_number, s_values))
                
                if any_is_a_number:
                
                    analisis["type1"] = "mix"
                    analisis["type2"] = "unknown"
                    analisis["problem"] = "Is numerical but has text in middle"

                else:
                    analisis["type1"] = "categorical"
                    analisis["type2"] = "text"
                    analisis["problem"] = "variability"

            

        df_analisis.loc[column, :] = pd.Series(data=analisis)

    return df_analisis


def categorizeColumns(dataset_all:pd.DataFrame, df_analisis:pd.DataFrame, target=[]):
    columns_numerical = []

    columns_categorical_numeric = []
    columns_categorical_text = []
    columns_drop = []

    columns_mix = []

    columns_categories = dict()

    for column in dataset_all.columns:
        
        if column in target:
            continue
        
        serie = df_analisis.loc[column, :]
        type1 = serie["type1"]
        
        if type1 == "useless":
            columns_drop.append(column)
        elif type1 == "numerical":
            columns_numerical.append(column)
        elif type1 == "categorical":
            
            if serie["problem"] == "variability":
                columns_drop.append(column)
            else:
                if serie["type2"] == "text":
                    columns_categorical_text.append(column)
                else:
                    columns_categorical_numeric.append(column)
                    
        elif type1 == "mix":
            columns_mix.append(column)
        else:
            print("UNKNOWN", serie)
    columns_categories["columns_numerical"] = columns_numerical
    columns_categories["columns_categorical_numeric"] = columns_categorical_numeric
    columns_categories["columns_categorical_text"] = columns_categorical_text
    columns_categories["columns_mix"] = columns_mix
    columns_categories["columns_drop"] = columns_drop
    return columns_categories