import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas as pd
import seaborn as sns
from scipy import stats

def getVarianceLowColumns(df, threshold=0.1):
    
    #variance
    variance = df.var()

    names = []
    heights = []
    for index in variance.index:
        value = variance[index]
        if value <= threshold:
            names.append(index)
            heights.append(value)
            
    return names, heights

def getMulticolinearColumns(correlation_matrix, keep_features_min = 10, cond_index = 30):
    
    correlation_matrix_mca = correlation_matrix
    # if the values are less or equal than 
    # cond_index, then multicolinearity is not an issue
    

    """
    'Any index greater than 30 “—indicates strong collinearity.” 
    values over 30 a “serious problem”
    also suggests values greater than 15 may indicate a 
    problem that warrants a closer look. 
    """
    
    multicolinear_columns = []

    current_features = len(correlation_matrix_mca)
    while current_features > keep_features_min:
        
        
        columns = correlation_matrix_mca.keys()
        eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix_mca)
        
        
        v = max(eigenvalues)/eigenvalues
        v = list(map(lambda x: 0 if x<0 else x ** 0.5, v))
        
        if max(v) <= cond_index:
            # print("Ending early")
            break
        
        for i, val in enumerate(eigenvalues):
            
            # min value close to zero
            if val == min(eigenvalues):
                vector = eigenvectors[:, i]
                value_max = max(abs(vector))
                for j, value in enumerate(vector):
                    # var with the max weight
                    if abs(value) ==  value_max:
                        mask = np.ones(len(correlation_matrix_mca), dtype=bool)
                        
                        for n, column in enumerate(correlation_matrix_mca.keys()):
                            mask[n] = n != j
                            
                        correlation_matrix_mca = correlation_matrix_mca[mask]
                        correlation_matrix_mca.pop(columns[j])
                        
                        multicolinear_columns.append(columns[j])
                        # print(f"Pop: {columns[j]}")
                        current_features -= 1

    return multicolinear_columns

def getLowCorrelationsWithTarget(X, y, min_correlation=0.1):
    # y is the serie or slice like y['ALLOW'] in case y is dataframe
    low_correlations_labels = []
    low_correlations_abs = []
    features = list(X.columns)
    for feature in features:
        correlation = abs(y.corr(X[feature]))
        if correlation < min_correlation:
            low_correlations_labels.append(feature)
            low_correlations_abs.append(correlation)
        
    return low_correlations_labels, low_correlations_abs


def getImportantFeatures(X, y, number_features_to_select = 20, max_features = 12):
    """
    pending add
    #feature selection
dtree_rfe = feature_selection.RFECV(model_dt, step = 1, scoring = 'accuracy', cv = cv_split)
dtree_rfe.fit(X_all, y_all)
mask_importance = dtree_rfe.get_support()
    """
    # Tree, importance of features

    extra_tree_forest = ExtraTreesClassifier(
        n_estimators=1000,
        criterion ='entropy', max_features = max_features
    )
    extra_tree_forest.fit(X, y)

    importances = extra_tree_forest.feature_importances_


    feature_importance_normalized = np.std([tree.feature_importances_ for tree in 
                                            extra_tree_forest.estimators_],
                                            axis = 0)

    columns = X.columns
    

    importance_columns = list(zip(columns, feature_importance_normalized))
    importance_columns.sort(key=lambda tup: tup[1], reverse=True)


    features_tree_selection = []
    for i in range(number_features_to_select):
        features_tree_selection.append(importance_columns[i][0])
        
    return features_tree_selection


def reducePCA(X, n_components = 16):
    # PCA
    X_scaled = scale(X.copy())
    pca = PCA().fit(X_scaled)
    X_reduced_pca = PCA(n_components=n_components).fit_transform(X_scaled)
    X_reduced_pca = pd.DataFrame(X_reduced_pca, columns=[f"feature{i}" for i in range(1,n_components+1)])
    
    return X_reduced_pca, pca


def bucket_feature(df_original, column_name, target=None, custom_bins_feature=None, maximum=-1, bins=10):
    
    df = df_original.copy()
    
    bins_feature = []
    if custom_bins_feature is None:
        if maximum == -1:
            maximum = df[column_name].max()
        steps = maximum // bins
        bins_feature = [i for i in range(0,maximum+bins,steps)] # [0,20,30,35,40,45,60,80, np.inf]
    else:
        bins_feature = custom_bins_feature
        
    if bins_feature[0] == 0:
        bins_feature[0] = -1
        
    band_name = f"band_{column_name}"
    labels_feature = [i for i in range(len(bins_feature)-1)]
    
    # print(bins_feature, labels_feature)
    
    df[band_name] = pd.cut(df[column_name], bins=bins_feature, labels=labels_feature)

    if target is not None:
        # see summary
        count_band_age_churn = df[[band_name, target[0]]].groupby([band_name]).count()
        print(count_band_age_churn)
        sns.barplot(x=count_band_age_churn.index, y=count_band_age_churn[target[0]])
        
    return df


def get_any_all_outliers(df:pd.DataFrame, cols_numerical_detect_outliers:list):

    matrix_zscore_abs = np.abs(stats.zscore(df.loc[:, cols_numerical_detect_outliers]))
    nan_propagation_count = np.isnan(matrix_zscore_abs).sum()
    # If this assertion fails, check the columns to not be constant, nan, etc.
    if len(nan_propagation_count) > 1:
        assert nan_propagation_count.sum() == 0
    else:
        assert nan_propagation_count == 0
    outliers_any_col = (matrix_zscore_abs >= 3).any(axis=1)  # check all row
    outliers_all_col = (matrix_zscore_abs >= 3).all(axis=1)  # check all row
    return matrix_zscore_abs, outliers_any_col, outliers_all_col