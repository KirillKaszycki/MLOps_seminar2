import json
import logging
import sys
import argparse
from datetime import datetime

import pandas as pd

from src.data.make_dataset import read_data, split_train_val_data

from src.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params
)

