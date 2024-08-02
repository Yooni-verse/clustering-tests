"""
Training Data Confusion Matrix:
[[34069  8304]
 [ 7393 10834]]
Training Data Accuracy Score:
0.740973597359736
Validation Data Confusion Matrix:
[[8548 2104]
 [1832 2666]]
Validation Data Accuracy Score:
0.7401980198019802
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom dataset to add abnormal images
class Food101Abnormal(Dataset):
    def __init__(self, dataset, transform=None, abnormal_transform=None, abnormal_ratio=0.3):
        self.dataset = dataset
        self.transform = transform
        self.abnormal_transform = abnormal_transform
        self.abnormal_ratio = abnormal_ratio
        self.normal_indices, self.abnormal_indices = self._split_indices()

    def _split_indices(self):
        total_size = len(self.dataset)
        indices = list(range(total_size))
        random.shuffle(indices)
        split = int(np.floor(self.abnormal_ratio * total_size))
        return indices[split:], indices[:split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        if idx in self.abnormal_indices:
            if self.abnormal_transform:
                image = self.abnormal_transform(image)
            label = 1  # Abnormal
        else:
            label = 0  # Normal
        if self.transform:
            image = self.transform(image)
        return image, label


# Normal and abnormal transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
"""

abnormal_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
])
"""
abnormal_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.5),
    transforms.RandomRotation(degrees=90),
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), ratio=(1.0, 2.0)),
    transforms.GaussianBlur(kernel_size=(3, 15), sigma=(0.1, 10)),
    transforms.RandomPosterize(bits=2),
    transforms.CenterCrop(10),
    transforms.Pad(padding=(100, 50, 100, 200), fill=255, padding_mode='symmetric')
    #transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
])

# Load Food101 dataset
original_dataset = datasets.Food101(root='path/to/root', split='train', download=False)

# Create dataset with abnormal images
abnormal_dataset = Food101Abnormal(original_dataset, transform=transform, abnormal_transform=abnormal_transform)

# Split dataset into training and validation
train_size = int(0.8 * len(abnormal_dataset))
val_size = len(abnormal_dataset) - train_size
train_dataset, val_dataset = random_split(abnormal_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

def extract_features(data_loader):
    features = []
    labels = []
    for images, lbls in data_loader:
        images = images.to(device)
        features.append(images.view(images.size(0), -1).cpu().numpy())  # Ensure to move tensor to CPU and convert to numpy
        labels.append(lbls.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

# Extract features from training and validation datasets
train_features, train_labels = extract_features(train_loader)
val_features, val_labels = extract_features(val_loader)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)
train_features_pca = pca.fit_transform(train_features)
val_features_pca = pca.transform(val_features)

# Apply K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(train_features_pca)
train_cluster_labels = kmeans.labels_
val_cluster_labels = kmeans.predict(val_features_pca)

plt.scatter(train_features_pca[:, 0], train_features_pca[:, 1], c=train_cluster_labels, cmap='viridis', s=2)
plt.title('Kmeans Clustering of Food101 Abnormality Detection (Training Data)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig('train_clustering_kmeans.png')
plt.close()

plt.scatter(val_features_pca[:, 0], val_features_pca[:, 1], c=val_cluster_labels, cmap='viridis', s=2)
plt.title('Kmeans Clustering of Food101 Abnormality Detection (Validation Data)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig('val_clustering_kmeans.png')
plt.close()

print("Training Data Confusion Matrix:")
print(confusion_matrix(train_labels, train_cluster_labels))

print("Training Data Accuracy Score:")
print(accuracy_score(train_labels, train_cluster_labels))

# Evaluate the clustering result for validation data
print("Validation Data Confusion Matrix:")
print(confusion_matrix(val_labels, val_cluster_labels))

print("Validation Data Accuracy Score:")
print(accuracy_score(val_labels, val_cluster_labels))
