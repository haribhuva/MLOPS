import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, load_numpy_array_data
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")


class ModelTrainer:

    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            self.model_config = read_yaml_file(model_trainer_config.model_config_file_path)
            logging.info("ModelTrainer initialized.")
            logging.info("Model config: {self.model_config}")
        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def compute_rul(features, target, unit_col_idx, rul_clip):
        """Compute RUL = max_cycle - current_cycle per engine, clipped at rul_clip."""
        rul = np.zeros_like(target)
        for uid in np.unique(features[:, unit_col_idx]):
            mask = features[:, unit_col_idx] == uid
            max_cycle = target[mask].max()
            rul[mask] = max_cycle - target[mask]
        return np.clip(rul, 0, rul_clip)

    @staticmethod
    def create_sequences(data, target, window_size, unit_col_idx):
        """Create per-engine sliding window sequences. Returns (X, y)."""
        sequences, labels = [], []
        for uid in np.unique(data[:, unit_col_idx]):
            mask = data[:, unit_col_idx] == uid
            engine_data = np.delete(data[mask], unit_col_idx, axis=1)
            engine_target = target[mask]
            for i in range(len(engine_data) - window_size + 1):
                sequences.append(engine_data[i : i + window_size])
                labels.append(engine_target[i + window_size - 1])
        return np.array(sequences), np.array(labels)

    def build_model(self, n_features):
        """Build Multi-scale CNN + LSTM + Attention model from model.yaml config."""
        logging.info("Building model.")
        try:
            window_size = self.model_config["window_size"]
            filters = self.model_config["cnn_filters"]
            kernel_sizes = self.model_config["kernel_sizes"]
            lstm_units = self.model_config["lstm_units"]
            dropout_rate = self.model_config["dropout_rate"]
            dense_units = self.model_config["dense_units"]

            # Input
            input_layer = layers.Input(shape=(window_size, n_features))

            # Multi-scale CNN branches
            branches = []
            for ks in kernel_sizes:
                branch = layers.Conv1D(filters, ks, padding="same", activation="relu")(input_layer)
                branches.append(branch)
            merged = layers.Concatenate(axis=-1)(branches)
            merged = layers.BatchNormalization()(merged)
            merged = layers.Dropout(dropout_rate)(merged)

            # LSTM
            lstm_out = layers.LSTM(lstm_units, return_sequences=True)(merged)
            lstm_out = layers.BatchNormalization()(lstm_out)
            lstm_out = layers.Dropout(dropout_rate)(lstm_out)

            # Attention
            attention_out = layers.Attention()([lstm_out, lstm_out])
            context = layers.GlobalAveragePooling1D()(attention_out)

            # Dense
            dense_out = layers.Dense(dense_units, activation="relu")(context)
            dense_out = layers.BatchNormalization()(dense_out)
            dense_out = layers.Dropout(dropout_rate)(dense_out)
            dense_out = layers.Dense(dense_units // 2, activation="relu")(dense_out)
            output = layers.Dense(1)(dense_out)

            model = Model(inputs=input_layer, outputs=output)
            logging.info(f"Model built. Total params: {model.count_params():,}")
            return model

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Model Training Started !!!")
        try:
            # Step 1: Set seeds for reproducibility
            np.random.seed(42)
            tf.random.set_seed(42)

            # Step 2: Load train.npy and test.npy
            train_data = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_data = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            logging.info(f"Loaded train: {train_data.shape}, test: {test_data.shape}")

            # Step 3: Split into x_train, y_train, x_test, y_test (target is last column)
            x_train = train_data[:, :-1]
            y_train = train_data[:, -1]
            x_test = test_data[:, :-1]
            y_test = test_data[:, -1]

            # Step 4: Compute RUL and clip
            unit_col_idx = x_train.shape[1] - 1
            rul_clip = self.model_config["rul_clip"]
            y_train = self.compute_rul(x_train, y_train, unit_col_idx, rul_clip)
            y_test = self.compute_rul(x_test, y_test, unit_col_idx, rul_clip)
            logging.info(f"RUL computed and clipped at {rul_clip}.")

            # Step 5: Create sliding window sequences
            window_size = self.model_config["window_size"]
            x_train, y_train = self.create_sequences(x_train, y_train, window_size, unit_col_idx)
            x_test, y_test = self.create_sequences(x_test, y_test, window_size, unit_col_idx)
            n_features = x_train.shape[2]
            logging.info(f"Sequences created. x_train: {x_train.shape}, x_test: {x_test.shape}")

            # Step 6: Build and compile model
            model = self.build_model(n_features)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(self.model_config["learning_rate"]),
                loss="mse",
                metrics=["mae"]
            )

            # Step 7: Train model
            early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=self.model_config["patience"], restore_best_weights=True)
            reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=self.model_config["patience"] // 2, min_lr=1e-6, verbose=0)

            model.fit(
                x_train, y_train,
                validation_data=(x_test, y_test),
                epochs=self.model_config["epochs"],
                batch_size=self.model_config["batch_size"],
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            logging.info("Model training done.")

            # Step 8: Evaluate model
            y_pred = model.predict(x_test, verbose=0).flatten()
            rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
            mae = float(np.mean(np.abs(y_test - y_pred)))
            metric_artifact = {"rmse": rmse, "mae": mae}
            logging.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")

            # Step 9: Save model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            model.save(self.model_trainer_config.trained_model_file_path)
            logging.info(f"Model saved at: {self.model_trainer_config.trained_model_file_path}")

            # Step 10: Return artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )

            logging.info("Model Training Completed !!!")
            return model_trainer_artifact

        except Exception as e:
            logging.info("Model training failed.")
            raise MyException(e, sys) from e