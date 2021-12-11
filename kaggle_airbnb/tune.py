import copy
import pickle
from typing import Dict

import pandas as pd
from kaggle_airbnb.config import (default_irrelevant_features, model_path,
                                  results_path, test_data_path,
                                  train_data_path)
from kaggle_airbnb.preprocess import preprocess_feature
from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from sklearn.model_selection import train_test_split


def get_opt_algo(name: str):
    scheduler = None
    opt_algo = None

    if name == "bohb":
        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=100,
            reduction_factor=4,
            stop_last_trials=False)
        opt_algo = tune.suggest.ConcurrencyLimiter(TuneBOHB(), max_concurrent=4)

    return scheduler, opt_algo


def train(config: Dict):
    # This is a simple training function to be passed into Tune
    # Load dataset
    df = pd.read_csv(train_data_path)
    irrelevant_features = config.pop('irrelevant_features')
    data = preprocess_feature(df, irrelevant_features)
    labels = df["Decision"].to_numpy()

    algo_wrapper_cls = None
    # Split into train and test set
    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=0.1)

    # Train the classifier, using the Tune callback
    algo_wrapper_cls = config.pop("algo_wrapper_cls")
    algo = algo_wrapper_cls(config)
    algo.fit(train_x, train_y)
    test_acc = algo.evaluate(test_x, test_y)

    tune.report(acc=test_acc)
    algo.save("model.pkl")


def get_best_model(analysis):
    best_logdir = analysis.get_best_logdir()
    with open(f"{best_logdir}/model.pkl", "rb") as f:
        best_model = pickle.load(f)
    accuracy = analysis.best_result["acc"]
    print(f"Best model parameters: {analysis.best_config}")
    print(f"Best model total accuracy: {accuracy:.4f}")

    return best_model


def tune_algo(search_space: Dict,
              num_samples: int = 1000,
              n_cpus: int = 1):

    scheduler, opt_algo = get_opt_algo("bohb")

    analysis = tune.run(
        train,
        metric="acc",
        mode="max",
        resources_per_trial={"cpu": n_cpus},
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=opt_algo
    )

    return analysis


def test(best_model, algo_name):
    df = pd.read_csv(test_data_path)

    features_to_remove = copy.copy(default_irrelevant_features)
    features_to_remove.remove("Decision")
    x = preprocess_feature(df, features_to_remove)
    y = best_model.predict(x)

    res_dict = {"id": df.id, "Decision": y}
    res_df = pd.DataFrame.from_dict(res_dict)
    res_df.to_csv(f"{results_path}/{algo_name}.csv", index=False)
    best_model.save(f"{model_path}/{algo_name}.pkl")
