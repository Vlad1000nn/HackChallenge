import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



def drop_cols(X : pd.DataFrame, cols):
    X = X.copy()
    X = X.drop(cols, axis=1)
    return X

def create_features(X : pd.DataFrame):

    X['essential_spending_ratio'] = (X['avg_by_category__amount__sum__cashflowcategory_name__supermarkety'] + X['avg_by_category__amount__sum__cashflowcategory_name__produkty'] + X['avg_by_category__amount__sum__cashflowcategory_name__odezhda']) / ((X['avg_6m_all'] + 1))

    X['luxury_spending_ratio'] = (X['avg_by_category__amount__sum__cashflowcategory_name__oteli'] + X['avg_by_category__amount__sum__cashflowcategory_name__puteshestvija']+ X['transaction_category_restaurants_percent_cnt_2m'] + X['avg_by_category__amount__sum__cashflowcategory_name__odezhda']) / (X['avg_6m_all'] + 1)

    X['cash_usage_ratio'] = X['avg_by_category__amount__sum__cashflowcategory_name__vydacha_nalichnyh_v_bankomate'] / (X['avg_6m_all'] + 1)

    X['balance_trend_3m'] = X['curr_rur_amt_3m_avg'] / (X['curr_rur_amt_cm_avg_period_days_ago_v2'] + 1)

    X['turnover_to_debt_ratio'] = X['turn_cur_cr_sum_v2'] / (X['hdb_outstand_sum'] + 1)

    X['active_products_ratio'] = X['bki_total_active_products'] / (X['bki_total_products'] + 1)

    X['hdb_bki_total_ratio'] =  X['hdb_bki_total_micro_cnt'] / (X['hdb_bki_active_pil_cnt'] + 1)

    X['hdb_bki_total_ip_ratio'] = X['hdb_bki_total_ip_max_limit'] / (X['avg_6m_all'] + 1)


    return X

def num_preprocessing(X : pd.DataFrame):
    X = X.copy()

    num_cols = X.select_dtypes(include=['float64'])
    X[num_cols] = np.log1p(X[num_cols])
    return X

def dt_preprocessing(X : pd.DataFrame):
    X = X.copy(deep=True)

    #X['period_last_act_ad'] = X['period_last_act_ad'].fillna('1900-01-01')
    X['period_last_act_ad'] = X['period_last_act_ad'].replace('1677-09-01', '1900-01-01')
    X['period_last_act_ad'] = pd.to_datetime(X['period_last_act_ad'])
    max_dt = X['period_last_act_ad'].max()

    X['last_d_count_active'] = (max_dt - X['period_last_act_ad']).dt.days
    X = X.drop('period_last_act_ad', axis=1)

    dt_map = {
        '2024-06-30' : 0,
        '2024-05-31' : 1,
        '2024-04-30' : 2,
        '2024-03-31' : 3,
        '2024-02-29' : 4,
        '2024-01-31' : 5,
    }
    X['dt'] = X['dt'].map(dt_map)
    return X


def cat_preprocessing(X : pd.DataFrame):
    X = X.copy()

    #ПОЛ
    gender_map = {
        'Женский' : -1,
        'Мужской' : 1,
        'UNKNOWN' : 0,
    }
    X['gender'] = X['gender'].map(gender_map)

    #РЕГИОН КЛИЕНТА
    adminarea_map = X['adminarea'].value_counts(normalize=True)
    X['adminarea'] = X['adminarea'].map(adminarea_map)

    #ГОРОДА
    city_map = X['city_smart_name'].value_counts()
    X['city_smart_name'] = X['city_smart_name'].map(city_map)

    #ПРОФЕССИЯ
    ep_map = X['dp_ewb_last_employment_position'].value_counts()
    X['dp_ewb_last_employment_position'] = X['dp_ewb_last_employment_position'].map(ep_map)

    #РЕГИОН ОТДЕЛЕНИЯ
    addrref_map = X['addrref'].value_counts()
    X['addrref'] = X['addrref'].map(addrref_map)

    ''' ВЫКИНУЛ
        dp_address_unique_regions
    '''
    return X


def get_preprocessor(train : pd.DataFrame, test : pd.DataFrame, SEED=111379):

    fill_bin_na = ['client_active_flag', 'accountsalary_out_flag', 'blacklist_flag', 
               'vert_has_app_ru_vtb_invest', 'vert_has_app_ru_cian_main', 'vert_has_app_ru_raiffeisennews',
               'vert_has_app_ru_tinkoff_investing'] # -- UNKNOW = -1

    fill_cat_na = ['gender', 'adminarea', 'city_smart_name', 'dp_ewb_last_employment_position']

    drop_columns = FunctionTransformer(func=drop_cols, kw_args={'cols' : ['dp_address_unique_regions']}, validate=False)
    make_fature = FunctionTransformer(create_features)
    cat_transformer= FunctionTransformer(cat_preprocessing)
    dt_transformer = FunctionTransformer(dt_preprocessing)
    num_transformer = FunctionTransformer(num_preprocessing)
    other_imputer = SimpleImputer(strategy='median')

    imputer_transformer = ColumnTransformer(transformers=[
    ('bin_imputer', SimpleImputer(strategy='constant', fill_value=-1.0), fill_bin_na),
    ('cat_imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN'), fill_cat_na),
    ('dt_imputer', SimpleImputer(strategy='constant', fill_value='1900-01-01'), ['period_last_act_ad'])
    ], remainder='passthrough',  verbose_feature_names_out=False)

    imputer_transfromer_pipeline = Pipeline(steps=[
    ('imputers', imputer_transformer)
    ])


    final_preprocessor = Pipeline(steps=[
    ('drop', drop_columns),
    ('imputers', imputer_transformer),
    ('cat_trans', cat_transformer),
    ('dt_trans', dt_transformer),
    ('make_fature', make_fature),
    #('other_fill', other_imputer),
    #('num_trans', num_transformer),
    ('scaler', StandardScaler()),
    ])

    final_preprocessor.set_output(transform='pandas')

    return final_preprocessor


def preprocess_data(train: pd.DataFrame, test: pd.DataFrame, SEED=111379):
    
    preprocessor = get_preprocessor()
    
    w = train['w'].copy()
    target = train['target'].copy()
    
    train_data = train.drop(['w', 'target'], axis=1)
    test_data = test.copy()
    
    preprocessor.fit(train_data)
    train_processed = preprocessor.transform(train_data)
    test_processed = preprocessor.transform(test_data)
    
    return train_processed, test_processed, target, w
