import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns

# Load the Iris dataset
def load_data():
    """
    Loads the Iris dataset.
    Returns:
        X: Feature data
        y: Target labels
    """
    iris = load_iris()
    return iris.data, iris.target

# Apply PCA for dimensionality reduction
def apply_pca(X, n_components=2):
    """
    Applies Principal Component Analysis (PCA) to the dataset.
    Arguments:
        X: Feature data
        n_components: Number of principal components to keep
    Returns:
        X_pca: Transformed feature data with reduced dimensions
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca

# Visualize the results after PCA
def visualize_results(X_pca, y):
    """
    Visualizes the dataset after applying PCA.
    Arguments:
        X_pca: Transformed feature data with reduced dimensions
        y: Target labels
    """
    df_pca = pd.DataFrame(data = X_pca, columns = ['Principal Component 1', 'Principal Component 2'])
    df_pca['Target'] = y

    sns.set(style="whitegrid", palette="muted")
    sns.scatterplot(x="Principal Component 1", y="Principal Component 2", hue="Target", data=df_pca)
    plt.title('PCA of Iris Dataset')
    plt.show()

# Main function to run the PCA example
def run_pca_example():
    X, y = load_data()
    X_pca = apply_pca(X)
    visualize_results(X_pca, y)

if __name__ == "__main__":
    run_pca_example()
