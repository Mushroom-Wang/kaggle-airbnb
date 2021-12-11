from kaggle_airbnb.config import (adaboost_search_space, catboost_search_space,
                                  logistic_regression_search_space,
                                  random_forest_search_space, xgb_search_space)
from kaggle_airbnb.tune import get_best_model, test, tune_algo

if __name__ == "__main__":
    algo_name = "random_forest"
    num_samples = 100
    n_cpus = 8

    search_space_dict = {
        "logistic_regression": logistic_regression_search_space,
        "random_forest": random_forest_search_space,
        "ada_boost": adaboost_search_space,
        "cat_boost": catboost_search_space,
        "xgb": xgb_search_space
    }

    analysis = tune_algo(search_space_dict[algo_name],
                         num_samples=num_samples,
                         n_cpus=n_cpus)
    best_model = get_best_model(analysis)
    test(best_model, algo_name)
