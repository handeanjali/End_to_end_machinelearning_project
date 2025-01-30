import os
import sys
from Insurance.predictor import ModelResolver
from Insurance.entity.config_entity import ModelPusherConfig
from Insurance.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ModelPusherArtifact
from Insurance.exception import InsuranceException
from Insurance.utils import load_object, save_object
from Insurance.logger import logging


class ModelPusher:
    """
    Handles pushing the trained model and its associated artifacts to a designated directory for deployment.
    """
    def __init__(self, model_pusher_config: ModelPusherConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        """
        Initializes ModelPusher with configuration and required artifacts.
        """
        try:
            logging.info("Initializing Model Pusher...")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise InsuranceException(e, sys)
        
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Loads the trained model, transformer, and target encoder, then saves them into the model pusher and saved model directories.
        """
        try:
            logging.info("Loading transformer, model, and target encoder.")
            transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            model = load_object(file_path=self.model_trainer_artifact.model_path)
            target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            logging.info("Saving model artifacts into the model pusher directory.")
            save_object(file_path= self.model_pusher_config.pusher_transformer_path, obj=transformer)
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)
            save_object(file_path=self.model_pusher_config.pusher_target_encoder_path, obj=target_encoder)

            logging.info("Saving model artifacts into the saved model directory.")
            save_object(file_path=self.model_resolver.get_latest_save_transformer_path(),obj= transformer)
            save_object(file_path=self.model_resolver.get_latest_save_model_path(),obj=model)
            save_object(file_path=self.model_resolver.get_latest_save_target_encoder_path(), obj=target_encoder)

            model_pusher_artifact = ModelPusherArtifact(
                pusher_model_dir=self.model_pusher_config.pusher_model_dir,
                saved_model_dir=self.model_pusher_config.saved_model_dir
            )
            
            logging.info(f"Model Pusher Artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise InsuranceException(e, sys)
