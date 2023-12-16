from feature_engineering import *

class PipelineCorrelation:
    
    
    @staticmethod
    def clean(X, y, options: dict):
        """
        mca_keep_features_min
        min_correlation
        tree_percentage_important
        """
        
        history = dict()
        
        # Multicolinearity
        multicolinear_columns = getMulticolinearColumns(np.abs(X.corr()), keep_features_min = options.get('mca_keep_features_min', 6))
        X = X.drop(multicolinear_columns, axis=1)
        history['multicolinear_columns'] = multicolinear_columns

        # Low correlation
        history['low_correlations'] = dict()
        low_correlations_labels, low_correlations_abs = getLowCorrelationsWithTarget(X, y, min_correlation=options.get('min_correlation', 0.1))
        
        history['low_correlations']['low_correlations_labels'] = low_correlations_labels
        history['low_correlations']['low_correlations_abs'] = low_correlations_abs
        
        
        X = X.drop(low_correlations_labels, axis=1)
        
        
        number_features_to_select = int(len(X.columns) * options.get('tree_percentage_important', 0.8))
        # Important features, the 80% most important
        if number_features_to_select > 0 and number_features_to_select < len(X.columns):
            features_important = getImportantFeatures(X, y, number_features_to_select, max_features = number_features_to_select)
            history['features_important'] = features_important
            X = X.loc[:,features_important]
        else:
            history['features_important'] = X.columns      
        
        return X, history
    
    
# Basic cleaning for consuming
import math

class PA:
    # Pipeline Actions
    DROP = 0
    TYPECAST = 1
    # SCALE = 2
    REPLACE = 3
    RENAME = 4
    REPLACE_WITH_MODE = 5
    REPLACE_WITH_MEDIAN = 6
    RENAME_LOWER_CASE = 7
    CREATE_DUMMIES = 8
    UDF = 9

    PIPELINE_BASE_COMMON_INT = [
        [REPLACE_WITH_MEDIAN,[np.nan]], 
        [TYPECAST,np.int16],
        [RENAME_LOWER_CASE],
    ]

    PIPELINE_BASE_COMMON_MODE_INT = [
        [REPLACE_WITH_MODE,[np.nan]],
        [TYPECAST,np.int16],
        [RENAME_LOWER_CASE],
    ]

    @staticmethod
    def is_real_number(x):
        try:
            cast_x = float(x)
            if math.isnan(cast_x):
                return False
            else:
                return True
        except:
            return False

    @staticmethod
    def exec(df: pd.DataFrame, data_scheme:dict):
        df_result: pd.DataFrame = df.copy()

        columns = df.columns
        for c in columns:
            try:
                column = c
                pipeline_actions = data_scheme.get(column,[])
                
                for pa_group in pipeline_actions:
                    pa = pa_group[0]

                    if pa == PA.DROP:
                        df_result.drop(column, axis=1, inplace=True)

                    elif pa == PA.REPLACE:
                        map_replace = {}
                        for key,value in zip(pa_group[1],pa_group[2]):
                            map_replace[key] = value

                        df_result[column].replace(map_replace, inplace=True)

                    elif pa == PA.TYPECAST:
                        df_result[column] = df_result[column].astype(pa_group[1])

                    elif pa == PA.REPLACE_WITH_MODE:
                        mode = df_result[df_result[column].notna()][column].mode()[0]
                        map_replace = {}
                        for key in pa_group[1]:
                            map_replace[key] = mode
                        df_result[column].replace(map_replace, inplace=True)

                    elif pa == PA.REPLACE_WITH_MEDIAN:
                        # works even if exist text in the column
                        median = df_result[df_result[column].apply(PA.is_real_number)][column].median()
                        map_replace = {}
                        for key in pa_group[1]:
                            map_replace[key] = median
                        df_result[column].replace(map_replace, inplace=True)

                    elif pa == PA.RENAME:
                        column_new = pa_group[1]
                        df_result.rename(columns={column: column_new}, inplace=True)
                        column = column_new

                    elif pa == PA.RENAME_LOWER_CASE:
                        column_new = column.lower()
                        df_result.rename(columns={column: column_new}, inplace=True)
                        column = column_new

                    elif pa == PA.CREATE_DUMMIES:
                        df_result = df_result.join(pd.get_dummies(df_result[column], prefix=column))
                        df_result.drop(column, axis=1, inplace=True)

                    elif pa == PA.UDF:
                        udf_pointer = pa_group[1]
                        args = pa_group[2]

                        df_result[column] = df_result[column].apply(udf_pointer, args=args)
                    else:
                        print("ERROR, unknown PA", pa)

            except Exception as e:
                print(f"Error while processing column '{column}'", e)


        return df_result
