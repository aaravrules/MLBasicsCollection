import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Function to load and split the Iris dataset
def load_and_split_data():
    """
    Loads the Iris dataset and splits it into training and testing sets.
    Returns:
        X_train, X_test, y_train, y_test: Training and testing sets.
    """
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to train a decision tree classifier
def train_decision_tree(X_train, y_train):
    """
    Trains a Decision Tree Classifier.
    Arguments:
        X_train: Training data
        y_train: Training labels
    Returns:
        classifier: Trained Decision Tree model
    """
    classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
    classifier.fit(X_train, y_train)
    return classifier

# Function to visualize the decision tree
def visualize_tree(classifier, feature_names, class_names):
    """
    Visualizes the decision tree.
    Arguments:
        classifier: Trained Decision Tree model
        feature_names: Names of the features in the dataset
        class_names: Names of the different classes in the dataset
    """
    plt.figure(figsize=(15, 10))
    plot_tree(classifier, feature_names=feature_names, class_names=class_names, filled=True)
    plt.title("Decision Tree on Iris Dataset")
    plt.show()

# Function to evaluate the model
def evaluate_model(classifier, X_test, y_test):
    """
    Evaluates the trained model on the test set.
    Arguments:
        classifier: Trained Decision Tree model
        X_test: Testing data
        y_test: Testing labels
    """
    predictions = classifier.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

# Main function to run the decision tree example
def run_decision_tree_example():
    X_train, X_test, y_train, y_test = load_and_split_data()
    classifier = train_decision_tree(X_train, y_train)
    evaluate_model(classifier, X_test, y_test)
    feature_names = load_iris().feature_names
    class_names = load_iris().target_names
    visualize_tree(classifier, feature_names, class_names)

if __name__ == "__main__":
    run_decision_tree_example()
