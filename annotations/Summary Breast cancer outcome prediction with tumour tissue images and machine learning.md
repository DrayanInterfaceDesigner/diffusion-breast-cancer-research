
# Methods

The authors focused more on methods for feature extraction, and the goal wasn't classification, but prediction of survival rate for patients with breast cancer.

They have used a VGG-16 CNN trained on ImageNet in generic images found in the wild, not fine-tuned for their purposes to extract features of the images, specifically the local descriptors. 
They also have used two methods to reduce the dimensionality of the extracted features using IFV(Improved Fisher Vector) and PCA (Principal Component Analysis), with L2 normalization of the dataset.


They don't provide technical details because the focus was the possibility of using CNNs for scoring and helping to diagnose survival rates.
