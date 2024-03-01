# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import math

df = pd.read_csv('./data.csv')

# Dropping some of the unwanted variables:
df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)

# Binarizing the target variable:
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

df_copy = df.copy()

# Downsampling
df_B = df_copy[df_copy["diagnosis"] == 0]
df_M = df_copy[df_copy["diagnosis"] == 1]

df_B = df_B.sample(frac = 1).iloc[:df_M.shape[0]]

combined_df = pd.concat([df_M, df_B]).sample(frac = 1)


# Breaking to train and test split
train_size = math.ceil(combined_df.shape[0] * 0.8)
test_size = math.floor(combined_df.shape[0] * 0.2)

y = combined_df["diagnosis"]
X = combined_df.drop(columns = "diagnosis")

X_train = X[0:train_size]
X_test = X[train_size: ]

y_train = y[0:train_size].to_numpy()
y_test = y[train_size: ].to_numpy()



# Feature Selection

from sklearn.base import BaseEstimator, TransformerMixin

class RemoveCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.correlated_features = None

    def fit(self, X, y=None):
        corr_matrix = X.corr()
        self.correlated_features = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold:
                    colname = corr_matrix.columns[i]
                    self.correlated_features.add(colname)
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.drop(labels=self.correlated_features, axis=1, inplace=True)
        return X_copy

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
# Instantiate the custom transformer
correlation_remover = RemoveCorrelatedFeatures(threshold=0.8)

# Fit and transform the training data
X_train = correlation_remover.fit_transform(X_train)

# Transform the test data using the fitted transformer
X_test = correlation_remover.transform(X_test)


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

class RFERegressionFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, num_features_to_select=10):
        self.num_features_to_select = num_features_to_select
        self.estimator = LogisticRegression()
        self.feature_selector = RFE(self.estimator, n_features_to_select=self.num_features_to_select)
    
    def fit(self, X, y=None):
        self.feature_selector.fit(X, y)
        return self
    
    def transform(self, X):
        columns_list = self.feature_selector.support_

        return X.loc[:,columns_list]
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    

feature_selector = RFERegressionFeatureSelector(num_features_to_select=10)

# 1. Fit the feature selector on the training set
feature_selector.fit(X_train, y_train)


X_train = feature_selector.transform(X_train)
X_test = feature_selector.transform(X_test)

print(X_train.columns)

# Standardization

class CustomMinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)

    def transform(self, X):
        if self.min_ is None or self.max_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.min_) / (self.max_ - self.min_)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
# 1. Instantiate the CustomStandardScaler object
scaler = CustomMinMaxScaler()

# 2. Fit the scaler
scaler.fit(X_train)

# 3. Transform the data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import pickle
pickle.dump(scaler, open("./scaling.pkl", "wb"))

import numpy as np
from collections import Counter
import math

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
                num_labels == 1 or \
                num_samples < self.min_samples_split:
            return {'prediction': Counter(y).most_common(1)[0][0]}

        # Feature selection
        if self.max_features == 'sqrt':
            feature_indices = np.random.choice(num_features, int(np.sqrt(num_features)), replace=False)
        elif self.max_features == 'log2':
            feature_indices = np.random.choice(num_features, int(np.log2(num_features)) + 1, replace=False)
        elif self.max_features is not None:
            feature_indices = np.random.choice(num_features, self.max_features, replace=False)
        else:
            feature_indices = np.arange(num_features)

        # Splitting criteria
        best_feature, best_threshold = self._best_criteria(X, y, feature_indices)

        # Splitting
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        left_tree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature_index': best_feature,
                'threshold': best_threshold,
                'left': left_tree,
                'right': right_tree}

    def _best_criteria(self, X, y, feature_indices):
        best_info_gain = -1
        best_feature, best_threshold = None, None
        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold
                info_gain = self._information_gain(y, y[left_indices], y[right_indices])
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold

    def _information_gain(self, parent, left_child, right_child):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        return entropy(parent) - (weight_left * entropy(left_child) + weight_right * entropy(right_child))

    def predict(self, X):
        return [self._predict_tree(x, self.tree) for x in X]

    def _predict_tree(self, x, tree):
        if 'prediction' in tree:
            return tree['prediction']
        feature_index, threshold = tree['feature_index'], tree['threshold']

        feature_value = float(x[feature_index])

        if feature_value <= threshold:
            return self._predict_tree(x, tree['left'])
        else:
            return self._predict_tree(x, tree['right'])




class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.estimators = []
        self.oob_scores = []

    def fit(self, X, y):
        self.estimators = []
        self.oob_scores = []
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                max_features=self.max_features)
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            oob_indices = [i for i in range(n_samples) if i not in bootstrap_indices]
            X_bootstrap, y_bootstrap = X[bootstrap_indices], y[bootstrap_indices]
            X_oob, y_oob = X[oob_indices], y[oob_indices]
            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(tree)
            self.oob_scores.append(self.oob_score(X_oob, y_oob))

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.estimators])
        return np.round(np.mean(predictions, axis=0))

    def oob_score(self, X_oob, y_oob):
        mis_label = 0
        for i in range(len(X_oob)):
            pred = self.predict_single(X_oob[i])
            if pred != y_oob[i]:
                mis_label += 1
        return mis_label / len(X_oob)

    def predict_single(self, x):
        return np.round(np.mean([tree._predict_tree(x, tree.tree) for tree in self.estimators]))


# Entropy calculation
def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


X_train_numpy = X_train.to_numpy()
X_test_numpy = X_test.to_numpy()

# Build and train the random forest model
rf = RandomForest(n_estimators=200, max_depth=10, min_samples_split=2, max_features='log2')
rf.fit(X_train_numpy, y_train)

predictions = rf.predict(X_test_numpy)


# Accuracy evaluation
accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)

pickle.dump(rf,open('./rf_model.pkl','wb'))



