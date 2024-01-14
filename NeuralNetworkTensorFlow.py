import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# Load and preprocess the Fashion MNIST dataset
def load_and_preprocess_data():
    """
    Loads the Fashion MNIST dataset and preprocesses it.
    Returns:
        X_train, X_test: Training and testing images
        y_train, y_test: Training and testing labels
    """
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    # Normalize the images to [0, 1] range
    X_train, X_test = X_train / 255.0, X_test / 255.0
    # One-hot encode the labels
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    return X_train, X_test, y_train, y_test

# Build a simple neural network model
def build_model(input_shape, num_classes):
    """
    Builds a simple neural network model.
    Arguments:
        input_shape: Shape of the input data
        num_classes: Number of classes in the dataset
    Returns:
        model: Compiled neural network model
    """
    model = tf.keras.Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Main function to run the neural network example with TensorFlow
def run_neural_network():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    model = build_model(X_train[0].shape, y_train.shape[1])
    model.fit(X_train, y_train, epochs=10, verbose=2)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Predict and print the classification report
    y_pred = model.predict(X_test)
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_true_classes = tf.argmax(y_test, axis=1)
    print(classification_report(y_true_classes, y_pred_classes))

if __name__ == "__main__":
    run_neural_network()
