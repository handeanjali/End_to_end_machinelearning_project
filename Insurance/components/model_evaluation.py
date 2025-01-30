import os
import sys
import pandas as pd
from sklearn.metrics import r2_score
from Insurance.predictor import ModelResolver
from Insurance.entity import config_entity, artifact_entity
from Insurance.exception import InsuranceException
from Insurance.logger import logging
from Insurance.utils import load_object
from Insurance.config import TARGET_COLUMN

class ModelEvaluation:
    """
    Evaluates the trained model against the previously saved model (if available) and determines whether to accept the new model.
    """
    
    def __init__(self,
                 model_eval_config: config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact: artifact_entity.ModelTrainerArtifact):
        """
        Initializes ModelEvaluation with necessary artifacts and configuration.
        """
        try:
            logging.info(f"{'>>'*20} Model Evaluation {'<<'*20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise InsuranceException(e, sys)
    
    def initiate_model_evaluation(self) -> artifact_entity.ModelEvaluationArtifact:
        """
        Compares the newly trained model with the latest saved model and determines if the new model should be accepted.
        """
        try:
            logging.info("Checking for existing saved models to compare with the newly trained model.")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            
            # If no saved model exists, accept the new model automatically
            if latest_dir_path is None:
                return artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, improved_accuracy=None)
            
            logging.info("Loading previously saved transformer, model, and target encoder.")
            transformer = load_object(self.model_resolver.get_latest_transformer_path())
            model = load_object(self.model_resolver.get_latest_model_path())
            target_encoder = load_object(self.model_resolver.get_latest_target_encoder_path())
            
            logging.info("Loading newly trained transformer, model, and target encoder.")
            current_transformer = load_object(self.data_transformation_artifact.transform_object_path)
            current_model = load_object(self.model_trainer_artifact.model_path)
            current_target_encoder = load_object(self.data_transformation_artifact.target_encoder_path)
            
            logging.info("Preparing test dataset for evaluation.")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            y_true = test_df[TARGET_COLUMN]
            
            # Encode categorical features using previous target encoder
            for col in transformer.feature_names_in_:
                if test_df[col].dtype == "object":
                    test_df[col] = target_encoder.fit_transform(test_df[col])
            
            logging.info("Evaluating previous model performance.")
            previous_input_features = transformer.transform(test_df[transformer.feature_names_in_])
            y_pred_prev = model.predict(previous_input_features)
            previous_model_score = r2_score(y_true, y_pred_prev)
            logging.info(f"Previous model accuracy: {previous_model_score}")
            
            logging.info("Evaluating current trained model performance.")
            current_input_features = current_transformer.transform(test_df[current_transformer.feature_names_in_])
            y_pred_current = current_model.predict(current_input_features)
            current_model_score = r2_score(y_true, y_pred_current)
            logging.info(f"Current model accuracy: {current_model_score}")
            
            # If the current model performs worse, reject it
            if current_model_score <= previous_model_score:
                logging.info("Current trained model is not better than the previous model.")
                raise Exception("Current trained model is not better than previous model")
            
            # Return model evaluation artifact
            return artifact_entity.ModelEvaluationArtifact(
                is_model_accepted=True,
                improved_accuracy=current_model_score - previous_model_score
            )
        except Exception as e:
            raise InsuranceException(e, sys)
