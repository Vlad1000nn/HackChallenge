import numpy as np
import pandas as pd
from preprocessing_file import *

#define
def cosine_lr(iter, max_iter=1061):
    min_lr = 0.0005
    max_lr = 0.1
    lr = min_lr + .5 * (max_lr - min_lr) * (1 + np.cos((iter / max_iter) * np.pi))

    return lr

# define
def lgbm_cosine_lr_callback(env):
    """
    Callback для изменения learning rate в LightGBM по косинусному расписанию.
    """
    iteration = env.iteration
    new_lr = cosine_lr(iteration, max_iter=env.end_iteration) # env.end_iteration - общее число деревьев
    env.model.params['learning_rate'] = new_lr

# useless
"""
def get_pretrained_models(path : str):

    loaded_weights = joblib.load(path)

    model_xgb_loaded = loaded_weights['model_xgb']
    model_lgbm_loaded = loaded_weights['model_lgbm'] 
    preprocessing = loaded_weights['processing_pipeline'] 
    ratio_loaded = 0.5


    return ratio_loaded, model_xgb_loaded, model_lgbm_loaded, preprocessing
"""

