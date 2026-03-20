"""Evaluates all models and creates plot/metrics artifacts by evaluation mode."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.estimator import ModelEstimator
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import (
    load_object,
    read_csv,
    read_yaml_file,
    resolve_target_column,
    save_object,
    split_features_target,
    write_yaml_file,
)


class ModelEvaluation:
    """Produces comparison artifacts and persists the selected best model."""

    _CFG_TARGET = "target_column"
    _CFG_RAND_STATE = "random_state"
    _CFG_VAL_SIZE = "internal_val_size"
    _CFG_MAX_SVR_SAMPLES = "max_svr_train_samples"
    _CFG_PLOT_POINTS = "evaluation_plot_points"
    _CFG_MODELS = "models"
    _CFG_EVAL_MODE = "evaluation_mode"
    _CFG_SELECTION_MODE = "model_selection_mode"
    _CFG_SPLIT_TEST_SIZE = "random_split_test_size"

    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> None:
        try:
            self._config = model_evaluation_config
            self._data_artifact = data_transformation_artifact
            self._trainer_artifact = model_trainer_artifact
            self._model_config = read_yaml_file(model_evaluation_config.model_config_file_path) or {}
            logging.info("ModelEvaluation initialized.")
        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        """Compute R2, MAE, and RMSE."""
        mse = metrics.mean_squared_error(y_true, y_pred)
        return {
            "r2": float(metrics.r2_score(y_true, y_pred)),
            "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mse)),
        }

    def _save_plot(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
        target_col: str,
        output_dir: str,
        suffix: str = "",
    ) -> str:
        """Save predicted-vs-actual line chart and return saved file path."""
        os.makedirs(output_dir, exist_ok=True)

        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        order = np.argsort(y_true_arr)

        max_points = int(self._model_config.get(self._CFG_PLOT_POINTS, 300))
        n = min(len(y_true_arr), max_points)

        title_suffix = f" ({suffix})" if suffix else ""
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(range(n), y_true_arr[order][:n], label="Actual", linewidth=2.0)
        ax.plot(range(n), y_pred_arr[order][:n], label="Predicted", linewidth=1.8)
        ax.set_title(f"Predicted vs Actual - {model_name}{title_suffix}")
        ax.set_xlabel("Sample Index (sorted by actual)")
        ax.set_ylabel(target_col)
        ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(output_dir, f"predicted_vs_actual_{model_name}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logging.info(f"Plot saved: {plot_path}")
        return plot_path

    def _evaluate_models(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_eval: pd.DataFrame,
        y_eval: pd.Series,
    ) -> tuple[dict[str, object], str, dict[str, dict[str, float]], dict[str, np.ndarray]]:
        """Fit all models on training data and evaluate on evaluation data."""
        logging.info("Evaluating model candidates.")

        random_state = int(self._model_config.get(self._CFG_RAND_STATE, 42))
        max_svr_samples = int(self._model_config.get(self._CFG_MAX_SVR_SAMPLES, 12000))

        all_models = ModelEstimator.build_all(
            model_params=self._model_config.get(self._CFG_MODELS, {}),
            random_state=random_state,
        )

        fitted_models: dict[str, object] = {}
        report: dict[str, dict[str, float]] = {}
        predictions: dict[str, np.ndarray] = {}

        for name, model in all_models.items():
            x_fit, y_fit = x_train, y_train

            # SVR can be expensive on very large training sets.
            if name == "SVR" and len(x_train) > max_svr_samples:
                logging.warning(
                    f"SVR sample cap active: {max_svr_samples}/{len(x_train)} samples."
                )
                rng = np.random.RandomState(random_state)
                idx = rng.choice(len(x_train), size=max_svr_samples, replace=False)
                x_fit, y_fit = x_train.iloc[idx], y_train.iloc[idx]

            model.fit(x_fit, y_fit)
            fitted_models[name] = model

            y_pred = model.predict(x_eval)
            predictions[name] = y_pred

            score = self._compute_metrics(y_eval, y_pred)
            report[name] = score
            logging.info(
                f"{name} -> R2: {score['r2']:.4f} | "
                f"MAE: {score['mae']:.4f} | RMSE: {score['rmse']:.4f}"
            )

        best_name = max(report, key=lambda k: report[k]["r2"])
        return fitted_models, best_name, report, predictions

    def _save_evaluation_outputs(
        self,
        evaluation_name: str,
        y_true: pd.Series,
        target_col: str,
        model_report: dict[str, dict[str, float]],
        model_predictions: dict[str, np.ndarray],
        best_model_name: str,
    ) -> dict[str, str]:
        """Save plots, metrics CSV, and summary text for one evaluation pass."""
        if evaluation_name == "random_split":
            output_dir = self._config.random_split_evaluation_dir
        elif evaluation_name == "external_holdout":
            output_dir = self._config.external_holdout_evaluation_dir
        else:
            output_dir = os.path.join(self._config.model_evaluation_dir, evaluation_name)

        os.makedirs(output_dir, exist_ok=True)

        rows: list[dict] = []
        for name, scores in model_report.items():
            rows.append({"model_name": name, **scores})
            self._save_plot(
                y_true=y_true,
                y_pred=model_predictions[name],
                model_name=name,
                target_col=target_col,
                output_dir=output_dir,
                suffix=evaluation_name,
            )

        metrics_df = pd.DataFrame(rows).sort_values("r2", ascending=False)
        metrics_csv_path = os.path.join(output_dir, self._config.metrics_file_name)
        summary_txt_path = os.path.join(output_dir, self._config.summary_file_name)

        metrics_df.to_csv(metrics_csv_path, index=False)
        logging.info(f"Metrics CSV saved: {metrics_csv_path}")

        with open(summary_txt_path, "w", encoding="utf-8") as f:
            f.write(f"Evaluation: {evaluation_name}\n")
            f.write(f"Best Model: {best_model_name}\n\n")
            f.write("Model Comparison (sorted by R2):\n")
            f.write(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
            f.write("\n")
        logging.info(f"Summary saved: {summary_txt_path}")

        return {
            "evaluation_dir": output_dir,
            "metrics_csv_path": metrics_csv_path,
            "summary_txt_path": summary_txt_path,
        }

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """Run configured evaluations, write report, and persist selected model."""
        try:
            logging.info("ModelEvaluation started.")

            configured_target = str(self._model_config.get(self._CFG_TARGET, "life_ratio"))
            train_df = read_csv(self._data_artifact.transformed_train_file_path)
            test_df = read_csv(self._data_artifact.transformed_test_file_path)

            target_col = resolve_target_column(train_df, configured_target)
            _ = resolve_target_column(test_df, configured_target)

            eval_mode = str(self._model_config.get(self._CFG_EVAL_MODE, "random_split")).strip().lower()
            selection_mode = str(self._model_config.get(self._CFG_SELECTION_MODE, "random_split")).strip().lower()
            split_test_size = float(
                self._model_config.get(
                    self._CFG_SPLIT_TEST_SIZE,
                    self._model_config.get(self._CFG_VAL_SIZE, 0.2),
                )
            )
            random_state = int(self._model_config.get(self._CFG_RAND_STATE, 42))

            random_split_report: dict[str, dict[str, float]] | None = None
            external_holdout_report: dict[str, dict[str, float]] | None = None
            random_split_outputs: dict[str, str] | None = None
            external_holdout_outputs: dict[str, str] | None = None

            random_split_models: dict[str, object] = {}
            external_holdout_models: dict[str, object] = {}
            random_split_best_name: str | None = None
            external_holdout_best_name: str | None = None

            if eval_mode in {"random_split", "both"}:
                x_all, y_all = split_features_target(train_df, target_col)
                x_train, x_val, y_train, y_val = train_test_split(
                    x_all,
                    y_all,
                    test_size=split_test_size,
                    random_state=random_state,
                )
                random_split_models, random_split_best_name, random_split_report, random_split_predictions = self._evaluate_models(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                )
                random_split_outputs = self._save_evaluation_outputs(
                    evaluation_name="random_split",
                    y_true=y_val,
                    target_col=target_col,
                    model_report=random_split_report,
                    model_predictions=random_split_predictions,
                    best_model_name=random_split_best_name,
                )

            if eval_mode in {"external_holdout", "both"}:
                x_train_ext, y_train_ext = split_features_target(train_df, target_col)
                x_test_ext, y_test_ext = split_features_target(test_df, target_col)
                external_holdout_models, external_holdout_best_name, external_holdout_report, external_holdout_predictions = self._evaluate_models(
                    x_train_ext,
                    y_train_ext,
                    x_test_ext,
                    y_test_ext,
                )
                external_holdout_outputs = self._save_evaluation_outputs(
                    evaluation_name="external_holdout",
                    y_true=y_test_ext,
                    target_col=target_col,
                    model_report=external_holdout_report,
                    model_predictions=external_holdout_predictions,
                    best_model_name=external_holdout_best_name,
                )

            if selection_mode == "external_holdout":
                selected_name = external_holdout_best_name
                selected_models = external_holdout_models
                selected_report = external_holdout_report
            else:
                selected_name = random_split_best_name
                selected_models = random_split_models
                selected_report = random_split_report

            if selected_name is None or selected_report is None or selected_name not in selected_models:
                raise ValueError("Could not determine best model from evaluation mode/settings.")

            # Refit selected best model on full transformed train data and persist it.
            x_full_train, y_full_train = split_features_target(train_df, target_col)
            best_model = selected_models[selected_name]
            best_model.fit(x_full_train, y_full_train)
            save_object(self._trainer_artifact.trained_model_file_path, best_model)

            # Log holdout metrics for the final persisted model.
            x_test, y_test = split_features_target(test_df, target_col)
            persisted_model = load_object(self._trainer_artifact.trained_model_file_path)
            y_pred = persisted_model.predict(x_test)
            holdout_score = self._compute_metrics(y_test, y_pred)
            logging.info(
                f"Holdout - R2: {holdout_score['r2']:.4f} | "
                f"MAE: {holdout_score['mae']:.4f} | RMSE: {holdout_score['rmse']:.4f}"
            )

            full_report = {
                "best_model_name": selected_name,
                "best_model_r2": float(selected_report[selected_name]["r2"]),
                "evaluation_mode": eval_mode,
                "external_holdout_outputs": external_holdout_outputs,
                "external_holdout_report": external_holdout_report,
                "model_selection_mode": selection_mode,
                "random_split_outputs": random_split_outputs,
                "random_split_report": random_split_report,
                "target_column": target_col,
            }
            write_yaml_file(
                file_path=self._config.evaluation_report_file_path,
                content=full_report,
                replace=True,
            )
            logging.info(f"Full report saved: {self._config.evaluation_report_file_path}")

            artifact = ModelEvaluationArtifact(
                is_model_accepted=True,
                changed_accuracy=float(selected_report[selected_name]["r2"]),
                trained_model_path=self._trainer_artifact.trained_model_file_path,
                best_model_path=self._trainer_artifact.trained_model_file_path,
                evaluation_report_file_path=self._config.evaluation_report_file_path,
                metric_artifact=full_report,
            )
            logging.info("ModelEvaluation completed.")
            return artifact

        except Exception as e:
            logging.error("ModelEvaluation failed.")
            raise MyException(e, sys) from e
