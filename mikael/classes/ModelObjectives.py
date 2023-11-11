import optuna
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRFRegressor, XGBRegressor


def XGBRObjective(X_train, X_test, y_train, y_test, trials):
    objective_list_reg = ["reg:squarederror"]
    tree_method = ["approx", "hist"]
    metric_list = ["mse", "mae"]

    def objective(trial):
        param = {
            "objective": trial.suggest_categorical("objective", objective_list_reg),
            "eval_metric": trial.suggest_categorical("eval_metric", metric_list),
            "tree_method": trial.suggest_categorical("tree_method", tree_method),
            "max_depth": trial.suggest_int("max_depth", 3, 10),  # Adjust the range
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),  # Increase the range
            "gamma": trial.suggest_float("gamma", 0.1, 1.0),  # Increase the lower bound
            "subsample": trial.suggest_discrete_uniform("subsample", 0.6, 1.0, 0.05),  # Reduce the range
            "colsample_bytree": trial.suggest_discrete_uniform("colsample_bytree", 0.6, 1.0, 0.05),  # Reduce the range
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0),
            "random_state": trial.suggest_int("random_state", 1, 1000),
        }
        model = XGBRegressor(**param)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_absolute_error(y_test, y_pred)

    study = optuna.create_study(direction="minimize", study_name="regression")
    study.optimize(objective, n_trials=trials, n_jobs=6)

    model = XGBRegressor(**study.best_params)
    model.fit(X_train, y_train)

    selection = SelectFromModel(model, threshold=0.004002, prefit=True)
    select_X_train = selection.transform(X_train)

    selection_model = XGBRegressor(**study.best_params)
    selection_model.fit(select_X_train, y_train)
    select_X_test = selection.transform(X_test)

    # selection = RFECV(estimator=model, verbose=1, step=1, cv=KFold(5), scoring=make_scorer(mean_absolute_error))
    # selection.fit(X_train, y_train)

    # select_X_train = selection.transform(X_train)
    # select_X_test = selection.transform(X_test)

    # selection_model = xgb.XGBRegressor(**study.best_params)
    # selection_model.fit(select_X_train, y_train)

    return model, selection_model, study, selection, select_X_test


def XGBRCVObjective(X_train, X_test, y_train, y_test, trials):
    objective_list_reg = ["reg:squarederror"]
    tree_method = ["approx", "hist"]
    metric_list = ["mse", "mae"]

    def objective(trial):
        param = {
            "objective": trial.suggest_categorical("objective", objective_list_reg),
            "eval_metric": trial.suggest_categorical("eval_metric", metric_list),
            "tree_method": trial.suggest_categorical("tree_method", tree_method),
            "max_depth": trial.suggest_int("max_depth", 3, 10),  # Adjust the range
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),  # Increase the range
            "gamma": trial.suggest_float("gamma", 0.1, 1.0),  # Increase the lower bound
            "subsample": trial.suggest_discrete_uniform("subsample", 0.6, 1.0, 0.05),  # Reduce the range
            "colsample_bytree": trial.suggest_discrete_uniform("colsample_bytree", 0.6, 1.0, 0.05),  # Reduce the range
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0),
            "random_state": trial.suggest_int("random_state", 1, 1000),
        }
        model = XGBRFRegressor(**param)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_absolute_error(y_test, y_pred)

    study = optuna.create_study(direction="minimize", study_name="regression")
    study.optimize(objective, n_trials=trials, n_jobs=6)

    model = XGBRFRegressor(**study.best_params)
    model.fit(X_train, y_train)

    selection = SelectFromModel(model, threshold=0.004002, prefit=True)
    select_X_train = selection.transform(X_train)

    selection_model = XGBRFRegressor(**study.best_params)
    selection_model.fit(select_X_train, y_train)
    select_X_test = selection.transform(X_test)

    # selection = RFECV(estimator=model, verbose=1, step=1, cv=KFold(5), scoring=make_scorer(mean_absolute_error))
    # selection.fit(X_train, y_train)

    # select_X_train = selection.transform(X_train)
    # select_X_test = selection.transform(X_test)

    # selection_model = xgb.XGBRegressor(**study.best_params)
    # selection_model.fit(select_X_train, y_train)

    return model, selection_model, study, selection, select_X_test


def HGBRObjective(X_train, X_test, y_train, y_test, trials):
    def objective(trial):
        loss = "absolute_error"
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 10, 40)
        learning_rate = trial.suggest_float("learning_rate", 0, 0.5)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 40)
        max_bins = trial.suggest_int("max_bins", 50, 255)
        categorical_features = [c for c in X_train.columns if ":idx" in c]
        random_state = trial.suggest_int("random_state", 100, 1000)

        model = HistGradientBoostingRegressor(
            # task_type='GPU',
            loss=loss,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_bins=max_bins,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            categorical_features=categorical_features,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_absolute_error(y_test, y_pred)

    study = optuna.create_study(direction="minimize", study_name="regression")
    study.optimize(objective, n_trials=trials, n_jobs=6)

    model = HistGradientBoostingRegressor(**study.best_params)
    model.fit(X_train, y_train)

    # selection = SelectFromModel(model, threshold=0.004002, prefit=True)
    # select_X_train = selection.transform(X_train)

    # selection_model = HistGradientBoostingRegressor(**study.best_params)
    # selection_model.fit(select_X_train, y_train)
    # select_X_test = selection.transform(X_test)

    return model, None, study, None, None


def RFRObjective(X_train, X_test, y_train, y_test, trials):
    def objective(trial):
        criterion = "absolute_error"
        n_estimators = trial.suggest_int("n_estimators", 50, 150)
        max_depth = trial.suggest_int("max_depth", 3, 50)
        random_state = trial.suggest_int("random_state", 100, 1000)

        model = RandomForestRegressor(
            # task_type='GPU',
            criterion=criterion,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_absolute_error(y_test, y_pred)

    study = optuna.create_study(direction="minimize", study_name="regression")
    study.optimize(objective, n_trials=trials, n_jobs=6)

    model = RandomForestRegressor(**study.best_params)
    model.fit(X_train, y_train)

    # selection = SelectFromModel(model, threshold=0.004002, prefit=True)
    # select_X_train = selection.transform(X_train)

    # selection_model = HistGradientBoostingRegressor(**study.best_params)
    # selection_model.fit(select_X_train, y_train)
    # select_X_test = selection.transform(X_test)

    return model, None, study, None, None
