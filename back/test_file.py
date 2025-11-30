from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib


def my_predict(model ,data: pd.DataFrame):
    y_pred = model.predict(data)
    return y_pred

def preprocessing(data: pd.DataFrame):
    data['x1'] = data['x1'] / 1000
    return data


def main():
    generator = np.random.default_rng(seed=42)

    x1 = 1000 * generator.random(100)
    x2 = generator.random(100)
    y =  1000 * generator.random(100) * generator.random(100)


    model = LinearRegression()

    data = pd.DataFrame({'x1':x1,'x2':x2,'y':y})



    data = preprocessing(data)
    x = data.copy(deep=True)
    y = data['y']
    x = x.drop('y', axis=1)


    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)

    """
    y_pred = my_predict(x_test)
    print(x_test.iloc[0])
    print("y_pred:", y_pred[0])
    print("y_test:", y_test[0])
    """

    model_weights = {
        "model" : model
    }

    joblib.dump(model_weights, 'model.joblib')
