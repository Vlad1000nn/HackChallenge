import pandas as pd

def clear_read(path : str):

    def real_norm_number(x):
        if isinstance(x, str) and x.find('.') != -1:
            return float(x) if all([s.isdigit() for s in x.replace('-', '').split('.')]) else x
        else:
            return x


    train = pd.read_csv(path, sep=';', decimal=',')
    test = pd.read_csv(path, sep=';', decimal=',')

    for cols in train.select_dtypes(include=['object']).columns:
        train[cols] = train[cols].apply(real_norm_number)

    for cols in test.select_dtypes(include=['object']).columns:
        test[cols] = test[cols].apply(real_norm_number)


    return train, test