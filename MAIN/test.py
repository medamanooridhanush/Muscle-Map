import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  # Corrected import statement

# Load training data
data = pd.read_csv('knn_training_data.csv')
# Clean column names to remove leading/trailing spaces
data.columns = data.columns.str.strip()
print(data.columns)
X = data[['finger_x', 'finger_y']]

# Labels: Muscle names
y = data['muscle']

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

# Function to predict muscle based on finger position
def predict_muscle(finger_position):
    return knn.predict([finger_position])[0]  # Reshaping finger_position to a 2D array

# Example usage:
# You can now use the trained KNN model for classification
test_finger_position = [100, 200]  # Replace with actual finger positions
predicted_muscle = predict_muscle(test_finger_position)
print(f'The predicted muscle is: {predicted_muscle}')