import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the Titanic dataset
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

# Preprocess the dataset (handle missing values, encode categorical variables)
def preprocess_data(df):
    # Selecting relevant features for simplicity
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

    # Handling missing values - Imputing 'Age' with median value
    imputer = SimpleImputer(strategy='median')
    df['Age'] = imputer.fit_transform(df[['Age']])

    # Converting categorical 'Sex' feature to numeric
    encoder = LabelEncoder()
    df['Sex'] = encoder.fit_transform(df['Sex'])

    return df

# Main function to run the logistic regression example
def run_logistic_regression():
    # Load and preprocess data
    data = load_data()
    processed_data = preprocess_data(data)

    # Splitting the dataset into training and testing sets
    X = processed_data.drop('Survived', axis=1)
    y = processed_data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create logistic regression object
    logreg = LogisticRegression()

    # Train the model using the training sets
    logreg.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = logreg.predict(X_test)

    # Model evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_logistic_regression()
