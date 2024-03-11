Dataset
The dataset used in this project is from the Sloan Digital Sky Survey (SDSS) DR18. It includes photometric magnitudes for various celestial objects, along with their classes. The dataset is not included in the repository due to size constraints but can be downloaded from SDSS website.

Usage
To run the classification model on your dataset, follow these steps:
1. Load your dataset in the same format as the SDSS DR18 dataset.
2. Ensure the data includes 'u', 'g', 'r', 'i', 'z' features and a 'class' target.
3. Run the classification_model.py script to train the model and evaluate its performance.

Model
The classification model is a Deep Neural Network (DNN) built with TensorFlow and Keras. It features multiple dense layers with dropout regularization, optimized through a grid search of hyperparameters.

Evaluation
The model's performance is evaluated using accuracy, precision, recall, and F1-score metrics, alongside a confusion matrix for a detailed analysis.

Deployment
Guidelines for deploying the model in a production environment are provided, including suggestions for application integration and API development.

Contributing
Contributions to the project are welcome! Please refer to the contributing guidelines for more information.

