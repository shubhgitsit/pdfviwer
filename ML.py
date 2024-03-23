#Practicle 1

# Importing necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generating some sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Generating 100 random numbers between 0 and 2
y = 4 + 3 * X + np.random.randn(100, 1)  # Generating labels with some noise

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


b. Implement and demonstrate the FIND-S algorithm for finding the most specific
hypothesis based on a given set of training data samples. Read the training data from a .CSV
file

import pandas as pd

# Read the training data from a CSV file
def read_csv(file_path):
    return pd.read_csv(file_path)

# Implement the FIND-S algorithm
def find_s_algorithm(data):
    # Initialize the hypothesis with the first training instance
    hypothesis = data.iloc[0, :-1].tolist()  # Initialize with the first instance's attributes
    
    # Iterate over the training instances
    for i in range(1, len(data)):
        instance = data.iloc[i, :-1].tolist()  # Get the attributes of the current instance
        label = data.iloc[i, -1]  # Get the label of the current instance
        
        # If the instance is positive, refine the hypothesis
        if label == 'Yes':
            for j in range(len(hypothesis)):
                if instance[j] != hypothesis[j]:
                    hypothesis[j] = '?'
    
    return hypothesis

# Main function
def main():
    # Read the training data from CSV file
    file_path = 'training_data.csv'
    data = read_csv(file_path)
    
    # Apply the FIND-S algorithm
    hypothesis = find_s_algorithm(data)
    
    # Print the most specific hypothesis
    print("The most specific hypothesis is:", hypothesis)

# Entry point of the program
if __name__ == "__main__":
    main()

2.a.Perform Data Loading, Feature selection (Principal Component analysis) and Feature
Scoring and Ranking

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from yellowbrick.features import Rank2D

# Data Loading
def load_data(file_path):
    # Load data from CSV file
    data = pd.read_csv(file_path)
    return data

# Feature Selection using PCA
def feature_selection_pca(data, n_components=2):
    # Separate features and target variable
    X = data.drop(columns=['target_column'])  # Adjust 'target_column' to your target column name
    y = data['target_column']
    
    # Apply PCA for feature selection
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, y

# Feature Scoring and Ranking
def feature_scoring_ranking(data):
    # Separate features and target variable
    X = data.drop(columns=['target_column'])  # Adjust 'target_column' to your target column name
    y = data['target_column']
    
    # Apply SelectKBest for feature scoring
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    
    # Get scores and feature names
    scores = selector.scores_
    feature_names = X.columns.tolist()
    
    # Rank features based on scores
    feature_ranking = sorted(zip(scores, feature_names), reverse=True)
    
    return feature_ranking

# Main function
def main():
    # Load data
    file_path = 'your_data.csv'  # Provide the path to your CSV file
    data = load_data(file_path)
    
    # Perform feature selection using PCA
    X_pca, y = feature_selection_pca(data)
    
    # Perform feature scoring and ranking
    feature_ranking = feature_scoring_ranking(data)
    
    # Visualize feature ranking
    print("Feature Ranking:")
    for rank, (score, feature_name) in enumerate(feature_ranking, start=1):
        print(f"{rank}. {feature_name}: {score}")
    
    # Visualize feature relationships with Rank2D
    visualizer = Rank2D(features=data.drop(columns=['target_column']).columns, algorithm='pearson')
    visualizer.fit_transform(data.drop(columns=['target_column']), data['target_column'])
    visualizer.show()

# Entry point of the program
if __name__ == "__main__":
    main()


For a given set of training data examples stored in a .CSV file, implement and demonstrate
the Candidate-Elimination algorithm to output a description of the set of all hypotheses
consistent with the training examples

import pandas as pd

# Load training data from CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Initialize the version space to the most general and most specific hypotheses
def initialize_hypotheses(attributes):
    num_attributes = len(attributes)
    specific_hypothesis = ['0'] * num_attributes
    general_hypothesis = ['?'] * num_attributes
    return specific_hypothesis, general_hypothesis

# Candidate-Elimination algorithm
def candidate_elimination(training_data):
    # Initialize hypotheses
    attributes = training_data.columns[:-1].tolist()  # Exclude the last column which is the target
    specific_hypothesis, general_hypothesis = initialize_hypotheses(attributes)
    
    # Iterate through each training example
    for index, row in training_data.iterrows():
        instance = row[:-1].tolist()  # Extract attributes of the instance
        label = row[-1]  # Extract label of the instance
        
        if label == 'Yes':  # Positive example
            for i in range(len(attributes)):
                if specific_hypothesis[i] == '0':
                    specific_hypothesis[i] = instance[i]
                elif specific_hypothesis[i] != instance[i]:
                    specific_hypothesis[i] = '?'
            for i in range(len(attributes)):
                if specific_hypothesis[i] != general_hypothesis[i]:
                    general_hypothesis[i] = '?'
        else:  # Negative example
            for i in range(len(attributes)):
                if instance[i] != specific_hypothesis[i] and specific_hypothesis[i] != '?':
                    general_hypothesis[i] = specific_hypothesis[i]
                    specific_hypothesis[i] = '?'
    
    return specific_hypothesis, general_hypothesis

# Main function
def main():
    # Load training data
    file_path = 'training_data.csv'  # Replace with your CSV file path
    training_data = load_data(file_path)
    
    # Apply the Candidate-Elimination algorithm
    specific, general = candidate_elimination(training_data)
    
    # Output the hypotheses
    print("Specific Hypothesis:", specific)
    print("General Hypothesis:", general)

# Entry point of the program
if __name__ == "__main__":
    main()


.a Write a program to implement the naïve Bayesian classifier for a sample training data set
stored as a .CSV file. Compute the accuracy of the classifier, considering few test data sets.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load training data
def load_data(file_path):
    return pd.read_csv(file_path)

# Train Naive Bayes classifier
def train_naive_bayes(X_train, y_train):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    return clf

# Predict using the trained classifier
def predict(clf, X_test):
    return clf.predict(X_test)

# Compute accuracy
def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# Main function
def main():
    # Load training data
    train_file_path = 'training_data.csv'
    train_data = load_data(train_file_path)
    
    # Preprocess training data
    X_train = train_data.drop(columns=['label_column'])  # Features
    y_train = train_data['label_column']  # Labels
    
    # Train Naive Bayes classifier
    clf = train_naive_bayes(X_train, y_train)
    
    # Load and preprocess test data
    test_file_path = 'test_data.csv'
    test_data = load_data(test_file_path)
    X_test = test_data.drop(columns=['label_column'])  # Features
    y_test = test_data['label_column']  # True labels
    
    # Predict using the trained classifier
    y_pred = predict(clf, X_test)
    
    # Compute accuracy
    accuracy = compute_accuracy(y_test, y_pred)
    print("Accuracy:", accuracy)

# Entry point of the program
if __name__ == "__main__":
    main()

Write a program to implement Decision Tree and Random forest with Prediction, Test
Score and Confusion Matrix.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Train Decision Tree classifier
def train_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf

# Train Random Forest classifier
def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf

# Predict using the trained classifier
def predict(clf, X_test):
    return clf.predict(X_test)

# Compute accuracy
def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# Compute confusion matrix
def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

# Main function
def main():
    # Load data
    file_path = 'your_data.csv'  # Provide the path to your CSV file
    data = load_data(file_path)
    
    # Preprocess data
    X = data.drop(columns=['target_column'])  # Features
    y = data['target_column']  # Labels
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Decision Tree classifier
    dt_clf = train_decision_tree(X_train, y_train)
    
    # Train Random Forest classifier
    rf_clf = train_random_forest(X_train, y_train)
    
    # Predict using Decision Tree classifier
    dt_y_pred = predict(dt_clf, X_test)
    
    # Predict using Random Forest classifier
    rf_y_pred = predict(rf_clf, X_test)
    
    # Compute accuracy for Decision Tree classifier
    dt_accuracy = compute_accuracy(y_test, dt_y_pred)
    print("Decision Tree Accuracy:", dt_accuracy)
    
    # Compute accuracy for Random Forest classifier
    rf_accuracy = compute_accuracy(y_test, rf_y_pred)
    print("Random Forest Accuracy:", rf_accuracy)
    
    # Compute confusion matrix for Decision Tree classifier
    dt_conf_matrix = compute_confusion_matrix(y_test, dt_y_pred)
    print("Decision Tree Confusion Matrix:")
    print(dt_conf_matrix)
    
    # Compute confusion matrix for Random Forest classifier
    rf_conf_matrix = compute_confusion_matrix(y_test, rf_y_pred)
    print("Random Forest Confusion Matrix:")
    print(rf_conf_matrix)
    
    # Visualize confusion matrix for Decision Tree classifier
    plt.figure(figsize=(8, 6))
    sns.heatmap(dt_conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.title('Decision Tree Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
    
    # Visualize confusion matrix for Random Forest classifier
    plt.figure(figsize=(8, 6))
    sns.heatmap(rf_conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

# Entry point of the program
if __name__ == "__main__":
    main()


4a For a given set of training data examples stored in a .CSV file implement Least Square
Regression algorithm.

import pandas as pd
import numpy as np

# Load training data
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess data
def preprocess_data(data):
    # Drop any rows with missing values
    data.dropna(inplace=True)
    
    # Split data into features and labels
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Labels
    return X, y

# Train Least Squares Regression model
def train_least_squares_regression(X, y):
    # Add a column of ones to X for the intercept term
    X = np.c_[np.ones(X.shape[0]), X]
    
    # Compute parameters using the least squares method
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

# Make predictions
def predict(X, theta):
    # Add a column of ones to X for the intercept term
    X = np.c_[np.ones(X.shape[0]), X]
    
    # Predict using the learned parameters
    y_pred = X.dot(theta)
    return y_pred

# Main function
def main():
    # Load data
    file_path = 'training_data.csv'  # Provide the path to your CSV file
    data = load_data(file_path)
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Train Least Squares Regression model
    theta = train_least_squares_regression(X, y)
    print("Parameters (Theta):", theta)
    
    # Make predictions (Example: Using the first data point)
    example_X = X[0].reshape(1, -1)  # Reshape for compatibility
    prediction = predict(example_X, theta)
    print("Prediction for the first data point:", prediction)

# Entry point of the program
if __name__ == "__main__":
    main()

4 b For a given set of training data examples stored in a .CSV file implement Logistic
Regression algorithm

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Load training data
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess data
def preprocess_data(data):
    # Drop any rows with missing values
    data.dropna(inplace=True)
    
    # Split data into features and labels
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Labels
    return X, y

# Train Logistic Regression model
def train_logistic_regression(X_train, y_train):
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Train Logistic Regression model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf

# Make predictions
def predict(clf, X_test):
    # Feature scaling (using the same scaler from training data)
    X_test = scaler.transform(X_test)
    
    # Predict using the trained model
    y_pred = clf.predict(X_test)
    return y_pred

# Compute accuracy
def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# Compute confusion matrix
def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

# Main function
def main():
    # Load data
    file_path = 'training_data.csv'  # Provide the path to your CSV file
    data = load_data(file_path)
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression model
    clf = train_logistic_regression(X_train, y_train)
    
    # Make predictions
    y_pred = predict(clf, X_test)
    
    # Compute accuracy
    accuracy = compute_accuracy(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    # Compute confusion matrix
    conf_matrix = compute_confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

# Entry point of the program
if __name__ == "__main__":
    main()


5 a. Write a program to demonstrate the working of the decision tree based ID3
algorithm. Use an appropriate data set for building the decision tree and apply this
knowledge to classify a new sample

import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Value to be returned if this node is a leaf node

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(X, y, feature, threshold):
    left_indices = X[:, feature] < threshold
    right_indices = ~left_indices
    left_entropy = entropy(y[left_indices])
    right_entropy = entropy(y[right_indices])
    parent_entropy = entropy(y)
    num_left = np.sum(left_indices)
    num_right = len(y) - num_left
    return parent_entropy - (num_left / len(y)) * left_entropy - (num_right / len(y)) * right_entropy

def find_best_split(X, y):
    best_gain = -1
    best_feature = None
    best_threshold = None
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            gain = information_gain(X, y, feature, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold

def build_tree(X, y):
    if len(np.unique(y)) == 1:
        return Node(value=y[0])
    best_feature, best_threshold = find_best_split(X, y)
    if best_feature is None:
        return Node(value=np.argmax(np.bincount(y)))
    left_indices = X[:, best_feature] < best_threshold
    right_indices = ~left_indices
    left_subtree = build_tree(X[left_indices], y[left_indices])
    right_subtree = build_tree(X[right_indices], y[right_indices])
    return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

def predict(tree, x):
    if tree.value is not None:
        return tree.value
    if x[tree.feature] < tree.threshold:
        return predict(tree.left, x)
    else:
        return predict(tree.right, x)

# Example usage
if __name__ == "__main__":
    # Sample dataset
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])
    y = np.array([0, 1, 0, 1])
    
    # Build the tree
    tree = build_tree(X, y)
    
    # New sample
    new_sample = np.array([2, 3, 4])
    
    # Predict the class
    prediction = predict(tree, new_sample)
    print("Prediction:", prediction)


5 b Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the k-NN classifier
k = 3  # Choose the value of k
knn = KNeighborsClassifier(n_neighbors=k)

# Train the k-NN classifier
knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


6a Implement the different Distance methods (Euclidean) with Prediction, Test Score and
Confusion Matrix

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Euclidean distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# K-Nearest Neighbors classifier using Euclidean distance
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
k = 3  # Choose the value of k
knn = KNN(k=k)

# Train the KNN classifier
knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

6b Implement the classification model using clustering for the following techniques with K
means clustering with Prediction, Test Score and Confusion Matrix.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)

# Assign cluster labels to training and testing sets
train_clusters = kmeans.predict(X_train)
test_clusters = kmeans.predict(X_test)

# Use cluster labels as features for a classifier
knn = KNeighborsClassifier()
knn.fit(train_clusters.reshape(-1, 1), y_train)

# Make predictions on the testing set
y_pred = knn.predict(test_clusters.reshape(-1, 1))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


7a Implement the classification model using clustering for the following techniques with
hierarchical clustering with Prediction, Test Score and Confusion Matrix

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
train_clusters = hierarchical.fit_predict(X_train)
test_clusters = hierarchical.fit_predict(X_test)

# Use cluster labels as features for a classifier
knn = KNeighborsClassifier()
knn.fit(train_clusters.reshape(-1, 1), y_train)

# Make predictions on the testing set
y_pred = knn.predict(test_clusters.reshape(-1, 1))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


8a. Write a program to construct a Bayesian network considering medical data. Use this
model to demonstrate the diagnosis of heart patients using standard Heart Disease Data Set


import pandas as pd

# Load Heart Disease Data Set
heart_data = pd.read_csv("heart.csv")

# Define Bayesian Network structure
# We will consider the following variables: Age, Sex, Chest Pain Type, Resting Blood Pressure, Serum Cholesterol,
# Fasting Blood Sugar, Resting Electrocardiographic Results, Maximum Heart Rate Achieved, Exercise Induced Angina,
# ST Depression Induced by Exercise Relative to Rest, Slope of the Peak Exercise ST Segment, Number of Major Vessels Colored by Flourosopy,
# Thalassemia, and Diagnosis of Heart Disease

# Define the network structure
network_structure = {
    "Age": ["Chest Pain Type"],
    "Sex": ["Chest Pain Type"],
    "Chest Pain Type": ["Diagnosis"],
    "Resting Blood Pressure": ["Diagnosis"],
    "Serum Cholesterol": ["Diagnosis"],
    "Fasting Blood Sugar": ["Diagnosis"],
    "Resting Electrocardiographic Results": ["Diagnosis"],
    "Maximum Heart Rate Achieved": ["Diagnosis"],
    "Exercise Induced Angina": ["Diagnosis"],
    "ST Depression Induced by Exercise Relative to Rest": ["Diagnosis"],
    "Slope of the Peak Exercise ST Segment": ["Diagnosis"],
    "Number of Major Vessels Colored by Flourosopy": ["Diagnosis"],
    "Thalassemia": ["Diagnosis"],
}

# Display network structure
for node, parents in network_structure.items():
    print(f"{node} -> {parents}")

from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator
from pgmpy.inference import VariableElimination

# Initialize Bayesian Model
heart_model = BayesianModel(network_structure)

# Estimate CPDs from data
heart_model.fit(heart_data)

# Print CPDs
for cpd in heart_model.get_cpds():
    print(cpd)

# Perform inference
infer = VariableElimination(heart_model)

# Query for Diagnosis given observed evidence
diagnosis_probability = infer.query(variables=["Diagnosis"], evidence={"Age": 50, "Sex": "Male", "Chest Pain Type": "Typical Angina"})
print(diagnosis_probability)


b. Implement the non-parametric Locally Weighted Regression algorithm in order to fit data
points. Select appropriate data set for your experiment and draw graphs

import numpy as np
import matplotlib.pyplot as plt

class LocallyWeightedRegression:
    def __init__(self, tau=0.1):
        self.tau = tau

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            w = self.weight(x)
            theta = self.theta(x, w)
            y_pred.append(np.dot(theta, x))
        return np.array(y_pred)

    def weight(self, x):
        weights = np.exp(-np.sum((x - self.X_train) ** 2, axis=1) / (2 * self.tau ** 2))
        return np.diag(weights)

    def theta(self, x, weights):
        X_aug = np.c_[np.ones(self.X_train.shape[0]), self.X_train]
        theta = np.linalg.inv(X_aug.T.dot(weights).dot(X_aug)).dot(X_aug.T).dot(weights).dot(self.y_train)
        return theta

# Generate sample dataset
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.1, 100)

# Reshape X and y
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Initialize and fit Locally Weighted Regression model
lwr = LocallyWeightedRegression(tau=0.1)
lwr.fit(X, y)

# Generate test data
X_test = np.linspace(0, 10, 100).reshape(-1, 1)

# Make predictions
y_pred = lwr.predict(X_test)

# Plot the original data and the fitted curve
plt.scatter(X, y, color='blue', label='Original data')
plt.plot(X_test, y_pred, color='red', label='Fitted curve')
plt.title('Locally Weighted Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

9A. Build an Artificial Neural Network by implementing the Backpropagation algorithm and
test the same using appropriate data sets

import numpy as np

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(next_layer, prev_layer) for prev_layer, next_layer in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(layer, 1) for layer in layers[1:]]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward_propagation(self, X):
        a = X
        activations = [a]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.sigmoid(z)
            zs.append(z)
            activations.append(a)
        return activations, zs

    def backward_propagation(self, X, y, activations, zs):
        delta_weights = [np.zeros(w.shape) for w in self.weights]
        delta_biases = [np.zeros(b.shape) for b in self.biases]

        # Compute output layer delta
        delta = (activations[-1] - y) * self.sigmoid_derivative(zs[-1])
        delta_weights[-1] = np.dot(delta, activations[-2].T)
        delta_biases[-1] = delta

        # Backpropagate the error
        for l in range(2, len(self.layers)):
            z = zs[-l]
            sp = self.sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            delta_weights[-l] = np.dot(delta, activations[-l-1].T)
            delta_biases[-l] = delta
        return delta_weights, delta_biases

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            activations, zs = self.forward_propagation(X)
            delta_weights, delta_biases = self.backward_propagation(X, y, activations, zs)

            # Update weights and biases
            self.weights = [w - self.learning_rate * dw for w, dw in zip(self.weights, delta_weights)]
            self.biases = [b - self.learning_rate * db for b, db in zip(self.biases, delta_biases)]

            # Print the loss every 100 epochs
            if epoch % 100 == 0:
                loss = np.mean(np.square(activations[-1] - y))
                print(f"Epoch {epoch}, Loss: {loss}")

# Example usage
if __name__ == "__main__":
    # Define the dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    y = np.array([[0, 1, 1, 0]])

    # Initialize and train the neural network
    nn = NeuralNetwork(layers=[2, 3, 1], learning_rate=0.5)
    nn.train(X, y, epochs=1000)

    # Test the trained model
    predictions, _ = nn.forward_propagation(X)
    print("Predictions:", predictions[-1])

9B. ASSUMING A SET OF DOCUMENTS THAT NEED TO BE CLASSIFIED, USE THE
NAÏVE BAYESIAN CLASSIFIER MODEL TO PERFORM THIS TASK. BUILT-IN JAVA
CLASSES/API CAN BE USED TO WRITE THE PROGRAM. CALCULATE THE
ACCURACY, PRECISION, AND RECALL FOR YOUR DATA SET.


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the 20 Newsgroups dataset
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

# Convert text documents to TF-IDF feature vectors
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(newsgroups_train.data)
X_test = vectorizer.transform(newsgroups_test.data)

# Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, newsgroups_train.target)

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(newsgroups_test.target, y_pred)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(newsgroups_test.target, y_pred, average='weighted')
print("Precision:", precision)

# Calculate recall
recall = recall_score(newsgroups_test.target, y_pred, average='weighted')
print("Recall:", recall)


10A . Write a program to demonstrate the working of the decision tree based ID3
algorithm. Use an appropriate data set for building the decision tree and apply this
knowledge to classify a new sample.

import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Value to be returned if this node is a leaf node

class ID3DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return Node(value=np.argmax(np.bincount(y)))
        
        num_features = X.shape[1]
        best_feature = None
        best_threshold = None
        best_gain = -1
        
        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices
        
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth+1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth+1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _information_gain(self, X, y, feature, threshold):
        left_indices = X[:, feature] < threshold
        right_indices = ~left_indices
        
        num_left = np.sum(left_indices)
        num_right = len(y) - num_left
        
        entropy_parent = self._entropy(y)
        entropy_left = self._entropy(y[left_indices])
        entropy_right = self._entropy(y[right_indices])
        
        information_gain = entropy_parent - (num_left / len(y)) * entropy_left - (num_right / len(y)) * entropy_right
        return information_gain

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Added small epsilon to avoid log(0)
        return entropy

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

# Toy dataset for demonstration
X_train = np.array([[1, 2], [1, 3], [2, 2], [2, 3], [3, 2], [3, 3]])
y_train = np.array([0, 0, 1, 1, 0, 1])

# Create and train the ID3 Decision Tree
tree = ID3DecisionTree()
tree.fit(X_train, y_train)

# Sample test data
X_test = np.array([[1, 2], [2, 3], [3, 2]])

# Make predictions
predictions = tree.predict(X_test)
print("Predictions:", predictions)