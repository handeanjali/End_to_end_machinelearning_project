import os
from typing import Optional
from glob import glob
from Insurance.entity.config_entity import (
    TRANSFORMER_OBJECT_FILE_NAME,
    MODEL_FILE_NAME,
    TARGET_ENCODER_OBJECT_FILE_NAME
)

class ModelResolver:
    """
    A utility class to manage model versioning and paths for transformers, models, and target encoders.
    """
    
    def __init__(self, model_registry: str = "saved_models",
                 transformer_dir_name: str = "transformer",
                 target_encoder_dir_name: str = "target_encoder",
                 model_dir_name: str = "model"):
        """
        Initializes the ModelResolver with directory paths.
        """
        self.model_registry = model_registry
        self.transformer_dir_name = transformer_dir_name
        self.target_encoder_dir_name = target_encoder_dir_name
        self.model_dir_name = model_dir_name

        # Ensure model registry directory exists
        os.makedirs(self.model_registry, exist_ok=True)

    def get_latest_dir_path(self) -> Optional[str]:
        """
        Fetches the latest model directory based on numbering.
        """
        try:
            dir_names = os.listdir(self.model_registry)
            if not dir_names:
                return None
            
            # Convert directory names to integers and find the latest one
            latest_dir_name = max(map(int, dir_names))
            return os.path.join(self.model_registry, str(latest_dir_name))
        except Exception as e:
            raise e

    def get_latest_model_path(self) -> str:
        """
        Retrieves the path of the latest saved model.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise FileNotFoundError("Model is not available")
            
            return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME)
        except Exception as e:
            raise e

    def get_latest_transformer_path(self) -> str:
        """
        Retrieves the path of the latest saved transformer.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise FileNotFoundError("Transformer is not available")
            
            return os.path.join(latest_dir, self.transformer_dir_name, TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            raise e

    def get_latest_target_encoder_path(self) -> str:
        """
        Retrieves the path of the latest saved target encoder.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise FileNotFoundError("Target encoder is not available")
            
            return os.path.join(latest_dir, self.target_encoder_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise e

    def get_latest_save_dir_path(self) -> str:
        """
        Determines the path for saving the next version of the model.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            next_dir_num = 0 if latest_dir is None else int(os.path.basename(latest_dir)) + 1
            
            return os.path.join(self.model_registry, str(next_dir_num))
        except Exception as e:
            raise e

    def get_latest_save_model_path(self) -> str:
        """
        Determines the path for saving the next model version.
        """
        try:
            latest_save_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_save_dir, self.model_dir_name, MODEL_FILE_NAME)
        except Exception as e:
            raise e

    def get_latest_save_transformer_path(self) -> str:
        """
        Determines the path for saving the next transformer version.
        """
        try:
            latest_save_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_save_dir, self.transformer_dir_name, TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            raise e

    def get_latest_save_target_encoder_path(self) -> str:
        """
        Determines the path for saving the next target encoder version.
        """
        try:
            latest_save_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_save_dir, self.target_encoder_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise e
