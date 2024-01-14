import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import datasets

# Load the diabetes dataset (for simplicity, use only one feature)
diabetes = datasets.load_diabetes()
X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, diabetes.target, test_size=0.2)

# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# Plot outputs
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Linear Regression Example')
plt.show()
