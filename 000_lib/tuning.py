from sklearn import model_selection

def tunningCheck(X_all, y_all, model_name, model_dt, cv_split, is_before=True):
    base_results = model_selection.cross_validate(model_dt, X_all, y_all, cv =cv_split)
    out = ""
    if is_before:
        out += f"{model_name} Before: "
    else:
        out += f"{model_name} After: "
    out += f"TestScore={base_results['test_score'].mean()*100}"
    print(out)

def tunningCheckModels(model_name, tune_model):
    print(f"{model_name} After tunning: TestScore=", tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100, "\n")

def tuneModel(model_name, X_all, y_all, model_instance, param_grid, cv_split):
    tunningCheck(X_all, y_all, model_name, model_instance, cv_split, is_before=True)    
    #choose best model with grid_search: #http://scikit-learn.org/stable/modules/grid_search.html#grid-search
    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
    tune_model = model_selection.GridSearchCV(model_instance, param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)
    tune_model.fit(X_all, y_all)
    tunningCheckModels(model_name, tune_model)
    
    return tune_model.best_estimator_

    