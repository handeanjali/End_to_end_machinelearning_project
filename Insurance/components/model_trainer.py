from Insurance.entity import artifact_entity, config_entity
from Insurance.exception import InsuranceException
from Insurance.logger import logging
from typing import Optional
import os, sys
import xgboost as xg
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from Insurance import utils
from sklearn.metrics import r2_score

class ModelTrainer:
    """
    Model Trainer class responsible for training and evaluating models.
    """

    def __init__(self, model_trainer_config: config_entity.ModelTrainerConfig,
                 data_transformation_artifact: Optional[artifact_entity.DataTransformationArtifact] = None):
        """
        Initializes ModelTrainer with configuration and transformed data.
        """
        try:
            logging.info(f"{'>>' * 20} Model Trainer {'<<' * 20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise InsuranceException(e, sys)

    def fine_tune(self, x, y):
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        """
        try:
            param_grid = {
                'fit_intercept': [True, False]
            }
            model = LinearRegression()
            random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                               n_iter=10, scoring='r2', cv=5, verbose=1, n_jobs=-1)
            random_search.fit(x, y)
            logging.info(f"Best parameters found: {random_search.best_params_}")
            return random_search.best_estimator_
        except Exception as e:
            raise InsuranceException(e, sys)

    def train_model(self, x, y):
        """
        Train a linear regression model with hyperparameter tuning.
        """
        try:
            model = self.fine_tune(x, y)
            model.fit(x, y)
            return model
        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_model_trainer(self) -> artifact_entity.ModelTrainerArtifact:
        """
        Orchestrates model training, evaluation, and artifact preparation.
        """
        try:
            if not self.data_transformation_artifact:
                raise Exception("DataTransformationArtifact is missing.")

            # Load transformed training and testing data
            logging.info("Loading train and test array.")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            # Split features and target variable
            logging.info("Splitting input and target feature from both train and test data.")
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Train the model with hyperparameter tuning
            logging.info("Training the model with hyperparameter tuning.")
            model = self.train_model(x=x_train, y=y_train)

            # Evaluate model performance
            logging.info("Calculating R2 score for training data.")
            yhat_train = model.predict(x_train)
            r2_train_score = r2_score(y_true=y_train, y_pred=yhat_train)

            logging.info("Calculating R2 score for testing data.")
            yhat_test = model.predict(x_test)
            r2_test_score = r2_score(y_true=y_test, y_pred=yhat_test)

            logging.info(f"Train Score: {r2_train_score}, Test Score: {r2_test_score}")

            # Check for model underfitting
            logging.info("Checking for model underfitting.")
            if r2_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is underperforming. Expected: {self.model_trainer_config.expected_score}, \
                                but got: {r2_test_score}")

            # Check for model overfitting
            logging.info("Checking for model overfitting.")
            diff = abs(r2_train_score - r2_test_score)
            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Overfitting detected. Train-Test score difference ({diff}) exceeds threshold \
                                ({self.model_trainer_config.overfitting_threshold})")

            # Save the trained model
            logging.info("Saving trained model.")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            # Prepare and return the model trainer artifact
            logging.info("Preparing model trainer artifact.")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_path=self.model_trainer_config.model_path,
                r2_train_score=r2_train_score,
                r2_test_score=r2_test_score
            )
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise InsuranceException(e, sys)