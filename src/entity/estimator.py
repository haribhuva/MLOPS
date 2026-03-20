"""Stateless factory that builds sklearn estimators from YAML config."""
import sys

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.exception import MyException
from src.logger import logging

_ALIASES: dict[str, str] = {
    "linear_regression":      "LinearRegression",
    "linearregression":       "LinearRegression",
    "svr":                    "SVR",
    "decision_tree":          "DecisionTree",
    "decisiontree":           "DecisionTree",
    "random_forest":          "RandomForest",
    "randomforest":           "RandomForest",
    "extra_trees":            "ExtraTrees",
    "extratrees":             "ExtraTrees",
    "gradient_boosting":      "GradientBoosting",
    "gradientboosting":       "GradientBoosting",
    "hist_gradient_boosting": "HistGradientBoosting",
    "histgradientboosting":   "HistGradientBoosting",
}

VALID_KEYS: list[str] = sorted(set(_ALIASES.keys()))

# Infrastructure-concern keys injected by Python, never from YAML model blocks
_INFRA_KEYS = frozenset({"random_state", "n_jobs"})


class ModelEstimator:
    """Builds sklearn estimators purely from YAML config — zero hardcoded hyperparameters."""

    @staticmethod
    def normalize_key(raw_key: str) -> str:
        """Map a config alias to canonical display name, or raise ValueError."""
        name = _ALIASES.get(raw_key.strip().lower())
        if name is None:
            raise ValueError(
                f"Unknown model key '{raw_key}'.\n"
                f"Valid options: {VALID_KEYS}"
            )
        return name

    @staticmethod
    def build_all(model_params: dict, random_state: int) -> dict[str, object]:
        """Build every supported estimator from the YAML `models` section."""
        try:
            logging.info("Building all model estimators from YAML config.")
            p = model_params

            for block_name, block_params in p.items():
                overlap = _INFRA_KEYS.intersection(set(block_params.keys()))
                if overlap:
                    raise ValueError(
                        f"config/model.yaml -> models.{block_name} contains "
                        f"forbidden keys: {sorted(overlap)}. "
                        "Remove them from YAML; estimator.py injects random_state and n_jobs."
                    )

            dt_params = dict(p.get("decision_tree", {}))
            dt_params["random_state"] = random_state

            rf_params = dict(p.get("random_forest", {}))
            rf_params["random_state"] = random_state
            rf_params["n_jobs"] = -1

            et_params = dict(p.get("extra_trees", {}))
            et_params["random_state"] = random_state
            et_params["n_jobs"] = -1

            gb_params = dict(p.get("gradient_boosting", {}))
            gb_params["random_state"] = random_state

            hgb_params = dict(p.get("hist_gradient_boosting", {}))
            hgb_params["random_state"] = random_state

            models = {
                "LinearRegression": make_pipeline(
                    StandardScaler(),
                    LinearRegression(**p.get("linear_regression", {})),
                ),
                "SVR": make_pipeline(
                    StandardScaler(),
                    SVR(**p.get("svr", {})),
                ),
                "DecisionTree":         DecisionTreeRegressor(**dt_params),
                "RandomForest":         RandomForestRegressor(**rf_params),
                "ExtraTrees":           ExtraTreesRegressor(**et_params),
                "GradientBoosting":     GradientBoostingRegressor(**gb_params),
                "HistGradientBoosting": HistGradientBoostingRegressor(**hgb_params),
            }
            logging.info(f"Built {len(models)} estimators.")
            return models

        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def build_single(model_key: str, model_params: dict, random_state: int) -> object:
        """Build one estimator by config key."""
        try:
            logging.info(f"Building single estimator for key: {model_key}")
            name = ModelEstimator.normalize_key(model_key)
            # TODO: optimise — currently builds all models then picks one
            return ModelEstimator.build_all(model_params, random_state)[name]
        except Exception as e:
            raise MyException(e, sys) from e