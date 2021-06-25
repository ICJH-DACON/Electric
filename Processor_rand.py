import numpy as np
import pandas as pd

# Learning algorithms
from sklearn.cluster import KMeans

# model validation
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')

class Preprocessor:
    def __init__(self, TRAINDIR, TESTDIR):
        self.train = pd.read_csv(TRAINDIR, encoding='euc-kr')
        self.test = pd.read_csv(TESTDIR, encoding='euc-kr')

        self.train.columns = ['num','datetime','target','temperature','windspeed','humidity','precipitation','insolation','nelec_cool_flag','solar_flag']
        self.test.columns = ['num','datetime','temperature','windspeed','humidity','precipitation','insolation','nelec_cool_flag','solar_flag']

        self.clust_to_num = self.cluster()

    def cluster(self):
        eda_df = self.train.copy()

        eda_df['datetime'] = pd.to_datetime(eda_df['datetime'])
        eda_df['hour'] = eda_df['datetime'].dt.hour
        eda_df['weekday'] = eda_df['datetime'].dt.weekday
        eda_df['date'] = eda_df['datetime'].dt.date
        eda_df['day'] = eda_df['datetime'].dt.day
        eda_df['month'] = eda_df['datetime'].dt.month
        eda_df['weekend'] = eda_df['weekday'].isin([5, 6]).astype(int)

        by_weekday = eda_df.groupby(['num', 'weekday'])['target'].median().reset_index().pivot('num', 'weekday',
                                                                                               'target').reset_index()
        by_hour = eda_df.groupby(['num', 'hour'])['target'].median().reset_index().pivot('num', 'hour',
                                                                                         'target').reset_index().drop(
            'num', axis=1)
        df = pd.concat([by_weekday, by_hour], axis=1)
        columns = ['num'] + ['day' + str(i) for i in range(7)] + ['hour' + str(i) for i in range(24)]
        df.columns = columns

        for i in range(len(df)):
            df.iloc[i, 1:8] = (df.iloc[i, 1:8] - df.iloc[i, 1:8].mean()) / df.iloc[i, 1:8].std()
            df.iloc[i, 8:] = (df.iloc[i, 8:] - df.iloc[i, 8:].mean()) / df.iloc[i, 8:].std()

        kmeans = KMeans(n_clusters=4, random_state=2)
        km_cluster = kmeans.fit_predict(df.iloc[:, 1:])

        df_clust = df.copy()
        df_clust['km_cluster'] = km_cluster

        match = df_clust[['num', 'km_cluster']]
        clust_to_num = {0: [], 1: [], 2: [], 3: []}
        for i in range(60):
            c = match.iloc[i, 1]
            clust_to_num[c].append(i + 1)
        return clust_to_num

    def train_preprocess(self):
        X_train = self.train.copy()

        X_trains = self.preprocessing(X_train)
        y_trains = [X_train['target'].values for X_train in X_trains]
        X_trains = [X_train.drop('target', axis=1) for X_train in X_trains]

        # standard scaling on numerical features
        num_features = ['temperature', 'windspeed', 'humidity', 'precipitation', 'insolation', 'min_temperature', 'THI',
                        'mean_THI', 'CDH', 'mean_CDH', 'date_num']
        means = []
        stds = []
        for i, df in enumerate(X_trains):
            means.append(df.loc[:, num_features].mean(axis=0))
            stds.append(df.loc[:, num_features].std(axis=0))
            df.loc[:, num_features] = (df.loc[:, num_features] - df.loc[:, num_features].mean(axis=0)) / df.loc[:,
                                                                                                         num_features].std(
                axis=0)
            X_trains[i] = df

        for num in self.clust_to_num[3]:
            X_trains[num - 1] = X_trains[num - 1][
                ['hour', 'THI', 'month', 'temperature', 'date_num', 'mean_THI', 'weekday',
                 'min_temperature']]

        self.means = means
        self.stds = stds

        return X_trains, y_trains

    def test_preprocess(self):
        X_test = self.test.copy()
        X_test = X_test.interpolate()

        X_tests = self.preprocessing(X_test)

        # standard scaling on numerical features
        num_features = ['temperature', 'windspeed', 'humidity', 'precipitation', 'insolation', 'min_temperature',
                        'THI', 'mean_THI', 'CDH', 'mean_CDH', 'date_num']
        for i, (df, mean, std) in enumerate(zip(X_tests, self.means, self.stds)):
            df.loc[:, num_features] = (df.loc[:, num_features] - mean) / std
            X_tests[i] = df

        for num in self.clust_to_num[3]:
            X_tests[num - 1] = X_tests[num - 1][
                ['hour', 'THI',  'month', 'temperature', 'date_num', 'mean_THI', 'weekday',
                 'min_temperature']]

        return X_tests

    def preprocessing(self, df):
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['date'] = df['datetime'].dt.date
        df['weekday'] = df['datetime'].dt.weekday

        ## daily minimum temperature
        df = df.merge(df.groupby(['num', 'date'])['temperature'].min().reset_index().rename(
            columns={'temperature': 'min_temperature'}), on=['num', 'date'], how='left')
        ## THI
        df['THI'] = 9 / 5 * df['temperature'] - 0.55 * (1 - df['humidity'] / 100) * (
                9 / 5 * df['temperature'] - 26) + 32
        ## mean_THI
        df = df.merge(
            df.groupby(['num', 'date'])['THI'].mean().reset_index().rename(columns={'THI': 'mean_THI'}),
            on=['num', 'date'], how='left')
        ## CDH
        cdhs = np.array([])
        for num in range(1, 61, 1):
            temp = df[df['num'] == num]
            cdh = self.CDH(temp['temperature'].values)
            cdhs = np.concatenate([cdhs, cdh])
        df['CDH'] = cdhs
        ## mean_CDH
        df = df.merge(
            df.groupby(['num', 'date'])['CDH'].mean().reset_index().rename(columns={'CDH': 'mean_CDH'}),
            on=['num', 'date'], how='left')
        ## date to numeric
        df['date_num'] = df['month'] + df['day'] / 31
        # split each building
        dfs = [df[df.num == num] for num in range(1, 61, 1)]

        dfs = [
            df.drop(['num', 'datetime', 'date', 'nelec_cool_flag', 'solar_flag'], axis=1).reset_index().drop('index',axis=1)
            for df in dfs]

        return dfs

    def CDH(self, xs):
        ys = []
        for i in range(len(xs)):
            if i < 11:
                ys.append(np.sum(xs[:(i + 1)] - 26))
            else:
                ys.append(np.sum(xs[(i - 11):(i + 1)] - 26))
        return np.array(ys)

    def DI(self, temp, humid):
        return 0.81 * temp + 0.01 * humid * (0.99 * temp - 14.3) + 46.3


class CV_sklearn:
    def __init__(self, models, n_folds=8):
        self.models = models
        self.n_folds = n_folds

    def check_data(self, X_trains, y_trains):
        # for each building
        for i, (X_train, y_train) in enumerate(zip(X_trains, y_trains)):
            kfold = KFold(n_splits=8, shuffle=False)
            # for each fold
            for j, (tr_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
                X_tr, X_val = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
                y_tr, y_val = y_train[tr_idx], y_train[val_idx]

                print(X_tr)
                print(X_tr.shape, X_val.shape)
                print(y_tr, y_val)

    def train(self, X_trains, y_trains,  verbose=0, perm_imp=False, feat_imp=False):
        trues = [[] for _ in range(self.n_folds)]
        preds = [[] for _ in range(self.n_folds)]
        permutation_importances = [[] for _ in range(self.n_folds)]
        feature_importances = [[] for _ in range(self.n_folds)]

        # for each building
        for i, (X_train, y_train) in enumerate(zip(X_trains, y_trains)):
            kfold = KFold(n_splits=self.n_folds, shuffle=False)
            # for each fold
            for j, (tr_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
                X_tr, X_val = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
                y_tr, y_val = y_train[tr_idx], y_train[val_idx]
                # fit model on each fold
                temp_model = self.models[j][i]
                temp_model.fit(X_tr, y_tr)
                if perm_imp:
                    r = permutation_importance(temp_model, X_val, y_val, n_repeats=10, scoring='smape', random_state=2)
                    permutation_importances[j].append(r['importances_mean'])
                    print(r.importances_mean)
                    if feat_imp:
                        feature_importances[j].append(temp_model.feature_importances_)
                self.models[j][i] = temp_model
                pred = temp_model.predict(X_val)
                true = y_val
                preds[j].append(pred)
                trues[j].append(true)

                self.models[j][i] = temp_model
            if (verbose == 1) & ((i + 1) % 5 == 0):
                print(f'{i + 1}th model complete')
        scores = []
        for true, pred in zip(trues, preds):
            true_f = np.concatenate(true)
            pred_f = np.concatenate(pred)
            scores.append(self.SMAPE(true_f, pred_f))
        self.trues = trues
        self.preds = preds
        self.permutation_importances = permutation_importances
        self.feature_importances = feature_importances
        return scores

    def SMAPE(self, true, pred):
        return np.mean((np.abs(true - pred)) / (np.abs(true) + np.abs(pred))) * 100

    def predict(self, X_tests):
        test_pred = np.array([np.array([0] * 168) for _ in range(60)]).astype(np.float64)
        for idx, test in enumerate(X_tests):
            for i in range(self.n_folds):
                test_pred[idx] += self.models[i][idx].predict(test)

        test_pred /= self.n_folds

        return np.concatenate(test_pred)