import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import time
from tabpfn import TabPFNClassifier
from tabpfn import TabP

# 定义文件路径
filepath = r'E:\迅雷下载\天池\生物学年龄评价与年龄相关疾病风险预测\\'

disease_mapping = {
    'control': 0,
    "Alzheimer's disease": 1,
    "Graves' disease": 2,
    "Huntington's disease": 3,
    "Parkinson's disease": 4,
    'rheumatoid arthritis': 5,
    'schizophrenia': 6,
    "Sjogren's syndrome": 7,
    'stroke': 8,
    'type 2 diabetes': 9
}
sample_type_mapping = {'control': 0, 'disease tissue': 1}

def load_idmap(idmap_dir):
    idmap = pd.read_csv(idmap_dir, sep=',')
    age = idmap.age.to_numpy()
    age = age.astype(np.float32)
    sample_type = idmap.sample_type.replace(sample_type_mapping)
    return age, sample_type

def load_methylation_h5(prefix):
    methylation = h5py.File(filepath + prefix + '.h5', 'r')['data']  # 修改了这里
    h5py.File(filepath + prefix + '.h5', 'r').close()
    return methylation[:, :60000]

import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error



from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
import optuna
import joblib
from catboost import CatBoostRegressor

from sklearn.ensemble import VotingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor,AdaBoostRegressor
from sklearn.neural_network import MLPRegressor



#下面继续增加融合模型的数量
def train_ml(X_train, y_train):
    models = [
        ('catboost', CatBoostRegressor()),
        ('xgboost', XGBRegressor()),
        ('lightgbm', LGBMRegressor(n_jobs=-1)),
        ('histgradient', HistGradientBoostingRegressor()),
        ('mlp', MLPRegressor()),
        ('randomforest', RandomForestRegressor(n_jobs=-1))
    ]
    model = VotingRegressor(models,n_jobs=-1)
    model.fit(X_train, y_train)
    return model





from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import numpy as np
import optuna
import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout,Flatten,Attention

import tensorflow as tf

def evaluate_ml(y_true, y_pred, sample_type):
    mae_control = np.mean(
        np.abs(y_true[sample_type == 0] - y_pred[sample_type == 0]))

    case_true = y_true[sample_type == 1]
    case_pred = y_pred[sample_type == 1]
    above = np.where(case_pred >= case_true)
    below = np.where(case_pred < case_true)

    ae_above = np.sum(np.abs(case_true[above] - case_pred[above])) / 2
    ae_below = np.sum(np.abs(case_true[below] - case_pred[below]))
    mae_case = (ae_above + ae_below) / len(case_true)

    mae = np.mean([mae_control, mae_case])
    return mae, mae_control, mae_case

if __name__ == "__main__":

    idmap_train_dir = filepath + 'trainmap.csv'  # 修改
    idmap_test_dir = filepath + 'testmap.csv'  # 修改

    age, sample_type = load_idmap(idmap_train_dir)

    methylation = load_methylation_h5('train')
    print(methylation.shape)

    print(methylation)
    print(age.shape)
    print(age)
    methylation_test = load_methylation_h5('test')
    print('Load data done')
    from sklearn.model_selection import train_test_split

    # 将数据集分为训练集和验证集


    indices = np.arange(len(age))
    [indices_train, indices_valid, age_train,
     age_valid] = train_test_split(indices, age, test_size=0.3, shuffle=True)
    methylation_train, methylation_valid = methylation[
        indices_train], methylation[indices_valid]
    sample_type_train, sample_type_valid = sample_type[
        indices_train], sample_type[indices_valid]
    feature_size = methylation_train.shape[1]
    del methylation

    print('Start training...')
    start = time.time()
    #pred_model = train_ml(methylation_train, age_train)
    pred_model = train_ml(methylation_train, age_train)
    #pred_model.fit(methylation_train, age_train)


    #pred_model = train_dl(methylation_train, age_train, methylation_valid, age_valid)


    print(f'Training time: {time.time() - start}s')
    from keras.models import load_model
    #model = load_model(filepath+'ecg_model.h5')
    age_valid_pred = pred_model.predict(methylation_valid)
#    age_valid_pred = pred_model.predict(methylation_valid)
    mae = evaluate_ml(age_valid, age_valid_pred, sample_type_valid)
   # mae = evaluate_dl(age_valid, age_valid_pred, sample_type_valid)
    print(f'Validation MAE: {mae}')

    age_pred = pred_model.predict(methylation_test)

    age_pred[age_pred < 0] = 0  

    age_pred = np.around(age_pred, decimals=2)
    age_pred = ['%.2f' % i for i in age_pred]
    sample_id = pd.read_csv(idmap_test_dir, sep=',').sample_id

    submission = pd.DataFrame({'sample_id': sample_id, 'age': age_pred})
    submission_file = filepath + 'submit.txt'  # 修改
    submission.to_csv(submission_file, index=False)