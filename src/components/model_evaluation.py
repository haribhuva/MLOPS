import os
import sys

import numpy as np
import tensorflow as tf


from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, load_numpy_array_data, write_yaml_file
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (DataTransformationArtifact,
                                         ModelTrainerArtifact,
                                         ModelEvaluationArtifact)
from src.components.model_trainer import ModelTrainer


# Suppress TensorFlow verbose logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")


class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._model_config = read_yaml_file(model_evaluation_config.model_config_file_path)
            logging.info("ModelEvaluation initialized.")
        except Exception as e:
            raise MyException(e, sys) from e

    def load_model(self, model_path: str):
        """Load model from disk. Handles .keras and .pkl formats."""
        logging.info(f"Loading model from: {model_path}")
        try:
            if model_path.endswith(".keras") or model_path.endswith(".h5"):
                return tf.keras.models.load_model(model_path)
            elif model_path.endswith(".pkl"):
                from src.utils.main_utils import load_object
                return load_object(model_path)
            else:
                raise ValueError(f"Unsupported model format: {model_path}")
        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def _get_last_window_predictions(features: np.ndarray, target: np.ndarray,
                                      model, window_size: int,
                                      unit_col_idx: int) -> tuple:
        """
        For each engine, extract the LAST sliding window and predict.
        Returns (y_pred_last, y_true_last) — one value per engine.
        """
        preds, actuals = [], []
        for uid in np.unique(features[:, unit_col_idx]):
            mask = features[:, unit_col_idx] == uid
            engine_data = np.delete(features[mask], unit_col_idx, axis=1)
            engine_target = target[mask]
            if len(engine_data) >= window_size:
                last_window = engine_data[-window_size:]
                pred = model.predict(last_window[np.newaxis, ...], verbose=0)[0, 0]
                preds.append(pred)
                actuals.append(engine_target[-1])
        return np.array(preds), np.array(actuals)

    @staticmethod
    def _nasa_cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        NASA CMAPSS asymmetric penalty scoring function.
        Late predictions (d >= 0) are penalized more harshly than early ones.

        d_i = predicted_i - actual_i
        if d_i < 0:  s_i = exp(-d_i / 13) - 1    (early)
        if d_i >= 0: s_i = exp( d_i / 10) - 1     (late)
        S = sum(s_i)
        """
        d = y_pred - y_true
        d = np.clip(d, -500, 500)  # Prevent exp overflow
        score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
        return float(np.sum(score))

    def evaluate_model(self, model) -> dict:
        """
        Load test data, recreate sequences, compute all 5 metrics:
          - eval_rmse_all_windows
          - eval_rmse_last_window
          - eval_mae
          - eval_r2
          - eval_rul_score (NASA CMAPSS asymmetric penalty)
        """
        logging.info("Starting model evaluation.")
        try:
            # Load test array
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )
            x_test = test_arr[:, :-1]
            y_test_raw = test_arr[:, -1]

            unit_col_idx = x_test.shape[1] - 1
            rul_clip = self._model_config["rul_clip"]
            window_size = self._model_config["window_size"]

            # Compute RUL (reusing ModelTrainer static method)
            y_test = ModelTrainer.compute_rul(x_test, y_test_raw, unit_col_idx, rul_clip)

            # ALL-WINDOWS evaluation
            X_test_seq, y_test_seq = ModelTrainer.create_sequences(
                x_test, y_test, window_size, unit_col_idx
            )
            y_pred_all = model.predict(X_test_seq, verbose=0).flatten()

            eval_rmse_all = float(np.sqrt(np.mean((y_test_seq - y_pred_all) ** 2)))
            eval_mae = float(np.mean(np.abs(y_test_seq - y_pred_all)))

            # LAST-WINDOW evaluation (one prediction per engine)
            y_pred_last, y_true_last = self._get_last_window_predictions(
                x_test, y_test, model, window_size, unit_col_idx
            )

            eval_rmse_last = float(np.sqrt(np.mean((y_true_last - y_pred_last) ** 2)))

            # R² score (manual to avoid sklearn dependency)
            ss_res = np.sum((y_true_last - y_pred_last) ** 2)
            ss_tot = np.sum((y_true_last - np.mean(y_true_last)) ** 2)
            eval_r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

            # NASA CMAPSS asymmetric score
            eval_rul_score = self._nasa_cmapss_score(y_true_last, y_pred_last)

            metrics = {
                "eval_rmse_all_windows": eval_rmse_all,
                "eval_rmse_last_window": eval_rmse_last,
                "eval_mae": eval_mae,
                "eval_r2": eval_r2,
                "eval_rul_score": eval_rul_score
            }

            logging.info(f"Evaluation metrics — "
                         f"RMSE_all: {eval_rmse_all:.4f}, "
                         f"RMSE_last: {eval_rmse_last:.4f}, "
                         f"MAE: {eval_mae:.4f}, "
                         f"R2: {eval_r2:.4f}, "
                         f"CMAPSS_Score: {eval_rul_score:.2f}")

            return metrics

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        logging.info("Initiating model evaluation.")
        try:
            # Load trained model
            model = self.load_model(self.model_trainer_artifact.trained_model_file_path)

            # Compute all metrics
            metrics = self.evaluate_model(model)



            # Save evaluation report as YAML
            os.makedirs(os.path.dirname(
                self.model_evaluation_config.evaluation_report_file_path), exist_ok=True)
            write_yaml_file(
                file_path=self.model_evaluation_config.evaluation_report_file_path,
                content=metrics,
                replace=True
            )
            logging.info(f"Evaluation report saved at: "
                         f"{self.model_evaluation_config.evaluation_report_file_path}")

            # Build artifact — no previous model to compare, accept all for now
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=True,
                changed_accuracy=0.0,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                best_model_path=self.model_trainer_artifact.trained_model_file_path
            )

            logging.info("Model Evaluation Completed !!!")
            return model_evaluation_artifact

        except Exception as e:
            logging.info("Model evaluation failed.")
            raise MyException(e, sys) from e
