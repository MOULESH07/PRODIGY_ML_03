import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog

# Paths
cat_dir = r"D:\Moulesh\prodigy infotech tasks\kagglecatsanddogs_5340\PetImages\Cat"
dog_dir = r"D:\Moulesh\prodigy infotech tasks\kagglecatsanddogs_5340\PetImages\Dog"

# Parameters
img_size = 64  # Resize images to 64x64
max_images = 1000  # Limit per class for speed

# Function to extract HOG features
def extract_hog(img):
    features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), block_norm='L2-Hys', channel_axis=-1, visualize=True)
    return features

# Load dataset
X, y = [], []

print("Loading cat images...")
for file in os.listdir(cat_dir)[:max_images]:
    path = os.path.join(cat_dir, file)
    img = cv2.imread(path)
    if img is None:
        continue
    img = cv2.resize(img, (img_size, img_size))
    features = extract_hog(img)
    X.append(features)
    y.append(0)  # Cat

print("Loading dog images...")
for file in os.listdir(dog_dir)[:max_images]:
    path = os.path.join(dog_dir, file)
    img = cv2.imread(path)
    if img is None:
        continue
    img = cv2.resize(img, (img_size, img_size))
    features = extract_hog(img)
    X.append(features)
    y.append(1)  # Dog

# Convert to numpy
X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
print("Training SVM...")
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Evaluate
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}")
