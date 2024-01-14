import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Sample dataset
data = {
    'text': ['I love this product', 'Great product! I am very happy.', 'Poor quality, totally disappointed', 
             'Worst product ever', 'Not what I expected', 'I am so happy with my purchase', 
             'This is terrible. I hate it', 'I adore this item', 'Not good, not bad', 'This product is amazing'],
    'sentiment': [1, 1, 0, 0, 0, 1, 0, 1, 0, 1]  # 1 for positive, 0 for negative
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Text Preprocessing and Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.3, random_state=42)

# Creating a Bag of Words model
vectorizer = CountVectorizer(stop_words='english')
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Training a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectors, y_train)

# Making predictions and evaluating the model
y_pred = classifier.predict(X_test_vectors)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example Predictions
example_texts = ["This is a great product", "I am very unhappy with what I received", "Average quality"]
example_vectors = vectorizer.transform(example_texts)
predictions = classifier.predict(example_vectors)
print("Example Predictions:", predictions)
