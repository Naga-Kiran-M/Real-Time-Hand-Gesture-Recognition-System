import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

# Function to load gesture data files
def load_gesture_data(data_path):
    data_files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    gesture_data = {}
    for file in data_files:
        gesture_data[file] = np.load(os.path.join(data_path, file))
    return gesture_data, data_files

# Function to compute similarity (Euclidean distance) between two gestures
def compute_similarity(gesture1, gesture2):
    return euclidean(gesture1.flatten(), gesture2.flatten())

# Function to analyze the similarity between gestures
def analyze_similarity(gesture_data, data_files):
    print("Analyzing gesture similarities...")
    similarity_matrix = np.zeros((len(data_files), len(data_files)))

    for i in range(len(gesture_data)):
        for j in range(i+1, len(gesture_data)):
            gesture1 = gesture_data[data_files[i]]
            gesture2 = gesture_data[data_files[j]]
            
            # Calculate similarity between the two gestures
            similarity = compute_similarity(gesture1, gesture2)
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity

            print(f"Similarity between {data_files[i]} and {data_files[j]}: {similarity:.2f}")

    return similarity_matrix

# Function to perform PCA analysis
def perform_pca(data, data_files):
    # Flatten the data and standardize it (mean=0, variance=1)
    flattened_data = [gesture.flatten() for gesture in data.values()]
    flattened_data = np.array(flattened_data)

    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(flattened_data)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(standardized_data)

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_

    # Plot the PCA results with labels
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=np.arange(len(data_files)), cmap='Set1', s=100, edgecolors='black')

    # Add labels for each point with a small offset to avoid overlap
    for i, label in enumerate(data_files):
        x_offset = 0.05 if i % 2 == 0 else -0.05  # Offset labels to avoid overlap
        y_offset = 0.05 if i % 2 == 0 else -0.05  # Offset labels
        plt.annotate(label.split('_')[0], (pca_result[i, 0] + x_offset, pca_result[i, 1] + y_offset), fontsize=10, fontweight='bold')

    # Customize plot with title, axis labels, and grid
    plt.title('PCA of Gesture Data', fontsize=14, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({explained_variance[0] * 100:.2f}% variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({explained_variance[1] * 100:.2f}% variance)', fontsize=12)
    
    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Gesture Index', fontsize=12)

    # Display the grid and ensure better visibility
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# Main function to load data and perform PCA
def analyze_gesture_data(data_path):
    print("Loading gesture data...")
    gesture_data, data_files = load_gesture_data(data_path)
    print(f"Data files: {data_files}")
    
    # Perform similarity analysis
    similarity_matrix = analyze_similarity(gesture_data, data_files)
    print("Similarity matrix:")
    print(similarity_matrix)
    
    print("Performing PCA analysis...")
    perform_pca(gesture_data, data_files)

# Automatically detect the data path
data_path = os.path.join(os.getcwd(), 'data')

# Run the analysis
analyze_gesture_data(data_path)
