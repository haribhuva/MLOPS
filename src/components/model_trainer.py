"""Selects and trains the best regression model on transformed training data."""
import sys

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.entity.estimator import ModelEstimator
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import (
    read_csv,
    read_yaml_file,
    resolve_target_column,
    save_object,
    split_features_target,
)


class ModelTrainer:
    """Selects and trains the best model, then persists it as a pickle artifact."""

    _CFG_TARGET = "target_column"
    _CFG_RAND_STATE = "random_state"
    _CFG_SELECTED_MODEL = "selected_model"
    _CFG_VAL_SIZE = "internal_val_size"
    _CFG_MAX_SVR_SAMPLES = "max_svr_train_samples"
    _CFG_MODELS = "models"

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> None:
        try:
            self._config = model_trainer_config
            self._data_artifact = data_transformation_artifact
            self._model_config = read_yaml_file(model_trainer_config.model_config_file_path) or {}
            logging.info("ModelTrainer initialized.")
        except Exception as e:
            raise MyException(e, sys) from e

    def _select_best_model(
        self, x_train: pd.DataFrame, y_train: pd.Series
    ) -> str:
        """Train all candidates on an internal val split; return the winner's name."""
        random_state = int(self._model_config[self._CFG_RAND_STATE])
        val_size = float(self._model_config[self._CFG_VAL_SIZE])
        max_svr_samples = int(self._model_config[self._CFG_MAX_SVR_SAMPLES])

        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train, y_train,
            test_size=val_size,
            random_state=random_state,
        )

        all_models = ModelEstimator.build_all(
            model_params=self._model_config.get(self._CFG_MODELS, {}),
            random_state=random_state,
        )

        scores: dict[str, float] = {}
        for name, model in all_models.items():
            x_fit, y_fit = x_tr, y_tr

            if name == "SVR" and len(x_tr) > max_svr_samples:
                logging.warning(
                    f"SVR sample cap active: using {max_svr_samples}/{len(x_tr)} "
                    "training samples. This changes training data."
                )
                rng = np.random.RandomState(random_state)
                idx = rng.choice(len(x_tr), size=max_svr_samples, replace=False)
                x_fit, y_fit = x_tr.iloc[idx], y_tr.iloc[idx]

            model.fit(x_fit, y_fit)
            r2 = metrics.r2_score(y_val, model.predict(x_val))
            scores[name] = r2
            logging.info(f"[auto selection] {name} -> R2: {r2:.4f}")

        best = max(scores, key=scores.__getitem__)
        logging.info(f"[auto selection] Winner: {best} (R2={scores[best]:.4f})")
        return best

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """Public entry point — select, train, and persist the best model."""
        try:
            logging.info("ModelTrainer started.")

            train_df = read_csv(self._data_artifact.transformed_train_file_path)
            configured_target = str(self._model_config[self._CFG_TARGET])
            target_col = resolve_target_column(train_df, configured_target)
            x_train, y_train = split_features_target(train_df, target_col)

            random_state = int(self._model_config[self._CFG_RAND_STATE])
            selected_key = str(self._model_config[self._CFG_SELECTED_MODEL]).strip().lower()
            if not selected_key:
                raise ValueError(
                    "config/model.yaml must define non-empty 'selected_model' (or 'auto')."
                )

            if selected_key == "auto":
                logging.info("Mode: auto — running internal model selection.")
                best_model_name = self._select_best_model(x_train, y_train)
            else:
                best_model_name = ModelEstimator.normalize_key(selected_key)
                logging.info(f"Mode: manual — using configured model: {best_model_name}")

            final_model = ModelEstimator.build_single(
                model_key=best_model_name,
                model_params=self._model_config.get(self._CFG_MODELS, {}),
                random_state=random_state,
            )
            final_model.fit(x_train, y_train)
            save_object(self._config.trained_model_file_path, final_model)
            logging.info(f"Model '{best_model_name}' fitted on full training data and saved.")

            artifact = ModelTrainerArtifact(
                trained_model_file_path=self._config.trained_model_file_path,
                metric_artifact={
                    "trained_model_name": best_model_name,
                    "selection_mode": "auto" if selected_key == "auto" else "manual",
                    "target_column": target_col,
                },
            )
            logging.info("ModelTrainer completed.")
            return artifact

        except Exception as e:
            logging.error("ModelTrainer failed.")
            raise MyException(e, sys) from e