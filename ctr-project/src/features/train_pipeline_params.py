import yaml
import logging
import sys

from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema

from feature_params import FeatureParams
from split_params import SplittingParams
from train_params import TrainingParams


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

PATH = "../configs/train_config.yaml"


class TrainingPipelineParams:
    output_model_path: str
    output_transformer_path: str
    output_ctr_transformer_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams
    input_data_path: str = field(default="../data/raw/sampled_train_50k.csv")
    input_preprocessed_data_path: str = field(default="../data/raw/sampled_preprocessed_train_50k.csv")
    use_mlflow: bool = False
    mlflow_experiment: str = "inference_demo"


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)
