import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
# model = pickle.load(open('./rf_model.pkl', 'rb'))


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


with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


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

with open('scaling.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

feature_columns = ['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
                   'symmetry_mean', 'fractal_dimension_mean', 'radius_se',
                   'concave points_se', 'symmetry_se', 'symmetry_worst']

@app.route('/')
def home():
    return render_template('index.html', feature_columns=feature_columns)

@app.route('/predict',methods=['POST'])
def predict():
    # Extract data from form
    input_features = [request.form[col] for col in feature_columns]
    input_data = np.array(input_features, dtype=float)

    scaled_data = scaler.transform(input_data)
    new_data_2d = scaled_data.values.reshape(1,-1)
   
    # Make prediction
    prediction = model.predict(new_data_2d)[0]
    print(prediction)
    # Map prediction to diagnosis
    diagnosis = 'Malignant' if prediction == 1 else 'Benign'
    
    return render_template('result.html', diagnosis=diagnosis)

if __name__ == "__main__":
    app.run(debug=True)