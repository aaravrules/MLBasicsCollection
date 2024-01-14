# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_and_split_data():
    """
    Loads the Iris dataset and splits it into training and testing sets.
    Returns:
        X_train, X_test, y_train, y_test: Training and testing sets.
    """
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def train_classifier(X_train, y_train):
    """
    Trains a K-Nearest Neighbors classifier.
    Arguments:
        X_train: Training data
        y_train: Training labels
    Returns:
        classifier: Trained KNN model
    """
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    return classifier

def evaluate_model(classifier, X_test, y_test):
    """
    Evaluates the trained model on the test set.
    Arguments:
        classifier: Trained model
        X_test: Testing data
        y_test: Testing labels
    """
    predictions = classifier.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=load_iris().target_names))

def main():
    X_train, X_test, y_train, y_test = load_and_split_data()
    classifier = train_classifier(X_train, y_train)
    evaluate_model(classifier, X_test, y_test)

if __name__ == "__main__":
    main()
