from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Function to load and preprocess the dataset
def load_and_preprocess_data():
    """
    Loads the Breast Cancer dataset and applies standard scaling.
    Returns:
        X: Scaled features
        y: Target labels
    """
    data = load_breast_cancer()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.data)
    return X_scaled, data.target

# Function to train an SVM classifier
def train_svm_classifier(X_train, y_train):
    """
    Trains a Support Vector Machine (SVM) classifier.
    Arguments:
        X_train: Scaled training data
        y_train: Training labels
    Returns:
        classifier: Trained SVM model
    """
    classifier = SVC(kernel='linear', random_state=42)
    classifier.fit(X_train, y_train)
    return classifier

# Function to evaluate the model
def evaluate_model(classifier, X_test, y_test):
    """
    Evaluates the trained SVM model on the test set.
    Arguments:
        classifier: Trained SVM model
        X_test: Testing data
        y_test: Testing labels
    """
    predictions = classifier.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

# Main function to run the SVM example
def run_svm_example():
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    classifier = train_svm_classifier(X_train, y_train)
    evaluate_model(classifier, X_test, y_test)

if __name__ == "__main__":
    run_svm_example()
