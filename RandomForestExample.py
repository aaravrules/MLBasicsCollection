from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Function to load and split the Wine dataset
def load_and_split_data():
    """
    Loads the Wine dataset and splits it into training and testing sets.
    Returns:
        X_train, X_test, y_train, y_test: Training and testing sets.
    """
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to train a Random Forest classifier
def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest Classifier.
    Arguments:
        X_train: Training data
        y_train: Training labels
    Returns:
        classifier: Trained Random Forest model
    """
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    return classifier

# Function to evaluate the model
def evaluate_model(classifier, X_test, y_test):
    """
    Evaluates the trained model on the test set.
    Arguments:
        classifier: Trained Random Forest model
        X_test: Testing data
        y_test: Testing labels
    """
    predictions = classifier.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

# Main function to run the Random Forest example
def run_random_forest_example():
    X_train, X_test, y_train, y_test = load_and_split_data()
    classifier = train_random_forest(X_train, y_train)
    evaluate_model(classifier, X_test, y_test)

if __name__ == "__main__":
    run_random_forest_example()
