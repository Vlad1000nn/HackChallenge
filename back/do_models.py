from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as R2, root_mean_squared_error as RMSE, mean_absolute_error as MAE

from xgboost import XGBRegressor
import xgboost as xgb

from lightgbm import LGBMRegressor

import numpy as np

#define
def cosine_lr(iter, max_iter=1061):
    min_lr = 0.0005
    max_lr = 0.1
    lr = min_lr + .5 * (max_lr - min_lr) * (1 + np.cos((iter / max_iter) * np.pi))

    return lr

#define
def lgbm_cosine_lr_callback(env):
        """
        Callback для изменения learning rate в LightGBM по косинусному расписанию.
        """
        iteration = env.iteration
        new_lr = cosine_lr(iteration, max_iter=env.end_iteration) # env.end_iteration - общее число деревьев
        env.model.params['learning_rate'] = new_lr

#useless
"""
def get_models(SEED=111379):
        
    schedule = xgb.callback.LearningRateScheduler(cosine_lr)

    xgb_params = {  'n_estimators':1061,
                    'max_depth' : 7,
                    'n_jobs' : -1,
                    'reg_alpha' : 47.339101653373156,
                    'reg_lambda' : 2.014909396467783,
                    'subsample' : 0.9,
                    'colsample_bytree' : .8,
                    'callbacks' : [schedule],
                    'random_state' : SEED,
                    }
    
    lgbm_params = { 'n_estimators':723,
                    'learning_rate' : 0.03,
                    'max_depth' : 7,
                    'n_jobs' : -1,
                    'subsample' : 0.9,
                    'reg_alpha' : 47.339101653373156,
                    'random_state' : SEED,
                    'verbose':-1,
                    }
    
    model_xgb = XGBRegressor(**xgb_params)
    model_lgbm = LGBMRegressor(**lgbm_params)
    
    return model_xgb, model_lgbm
"""