import os

import kaggle_airbnb
from kaggle_airbnb.wrapper import (AdaBoostWrapper, CatBoostWrapper,
                                   LogisticRegressionWrapper,
                                   RandomForestWrapper, XGBoostWrapper)
from ray import tune

ROOT = os.path.dirname(os.path.abspath(kaggle_airbnb.__file__))

train_data_path = f"{ROOT}/data/processed_train.csv"
test_data_path = f"{ROOT}/data/processed_test.csv"
results_path = f"{ROOT}/results"
model_path = f"{ROOT}/models"

default_irrelevant_features = [
    "id",
    "Decision"
    # "Host_is_superhost",
    # "Host_has_profile_pic",
    # "Host_identity_verified",
    # "Room_type",
    # "Bathrooms_text",
    # "Bedrooms",
    # "Beds",
    # "Essentials",
    # "Cooking",
    # "Balcony",
    # "Parking",
    # "Instant_bookable",
    # "Month"
]

default_n_jobs = 4

xgb_search_space = {
    "algo_wrapper_cls": XGBoostWrapper,
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "error"],
    "max_depth": tune.randint(1, 20),
    "min_child_weight": tune.choice([1, 2, 3, 4, 5]),
    "subsample": tune.uniform(0.1, 1.0),
    "eta": tune.loguniform(1e-4, 1e-1),
    "irrelevant_features": default_irrelevant_features
}

logistic_regression_search_space = {
    "algo_wrapper_cls": LogisticRegressionWrapper,
    # only for placehold purposes, required by the scheduler
    "n_jobs": tune.choice([1, 2]),
    "irrelevant_features": default_irrelevant_features
}

random_forest_search_space = {
    "algo_wrapper_cls": RandomForestWrapper,
    "n_estimators": tune.randint(20, 100),
    "criterion": tune.choice(["gini", "entropy"]),
    "max_depth": tune.randint(10, 25),
    "min_samples_split": tune.randint(2, 4),
    "min_samples_leaf": tune.randint(1, 20),
    "bootstrap": True,
    "max_features": tune.choice(["auto", "sqrt", "log2"]),
    "oob_score": tune.choice([False]),
    "n_jobs": default_n_jobs,
    "irrelevant_features": default_irrelevant_features
}

adaboost_search_space = {
    "algo_wrapper_cls": AdaBoostWrapper,
    "n_estimators": tune.randint(1, 200),
    "learning_rate": tune.loguniform(1e-5, 1),
    "algorithm": tune.choice(["SAMME", "SAMME.R"]),
    "irrelevant_features": default_irrelevant_features
}

catboost_search_space = {
    "algo_wrapper_cls": CatBoostWrapper,
    "num_trees": tune.randint(1, 200),
    "learning_rate": tune.loguniform(1e-5, 1),
    "depth": tune.randint(2, 16),
    "l2_leaf_reg": tune.loguniform(1e-5, 1),
    "random_strength": tune.loguniform(1e-5, 1),
    "bagging_temperature": tune.uniform(0, 1),
    "silent": True,
    "irrelevant_features": default_irrelevant_features
}
