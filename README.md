# Celestial Object Classification Using Deep Learning

The primary goal of this project is to develop a predictive model capable of classifying celestial objects based on their photometric magnitudes. Using data from the Sloan Digital Sky Survey (SDSS), we aim to accurately categorize objects into distinct classes such as stars, galaxies, and quasars. This classification is crucial for astronomical research and understanding the composition and structure of the universe.

# Dataset
The dataset utilized in this project originates from the [SDSS DR18](https://www.sdss.org) and contains observations of various celestial objects. Each object in the dataset is characterized by features representing its photometric magnitudes in five different filters: 'u', 'g', 'r', 'i', 'z'. These magnitudes provide a broad overview of the object's properties across different wavelengths of light. The target variable is the 'class' of each object, which identifies it as a star, galaxy, or quasar.

# Usage
To run the classification model on your dataset, follow these steps:
1. Load your dataset in the same format as the SDSS DR18 dataset.
2. Ensure the data includes 'u', 'g', 'r', 'i', 'z' features and a 'class' target.
3. Run the classification_model.py script to train the model and evaluate its performance.

# Model Development:
We approach the classification problem using a Deep Neural Network (DNN) constructed with the Keras API. The model architecture includes multiple dense layers with ReLU activation functions, dropout layers for regularization to combat overfitting, and a softmax activation function in the output layer to handle multi-class classification. The model is compiled with the Adam optimizer and sparse categorical crossentropy as the loss function.

# Hyperparameter Tuning:
To optimize the model's performance, we employ a Grid Search strategy to systematically explore combinations of hyperparameters, including the dropout rate and the number of neurons in the dense layers. This exhaustive search is facilitated by scikit-learn's GridSearchCV, which evaluates each combination using cross-validation to identify the configuration that yields the best validation accuracy.

# Model Evaluation:
After training, the model's performance is assessed on a separate test set. Evaluation metrics include accuracy, precision, recall, and F1-score for each class, providing a comprehensive understanding of the model's strengths and weaknesses. A confusion matrix is also generated to visualize the model's predictions in comparison to the true labels, offering insights into any systematic errors in classification.

# Deployment Considerations:
Upon satisfactory evaluation, the model is prepared for deployment in a real-world application. This could involve integrating the model into an astronomical software tool or a web application that allows users to classify celestial objects by inputting photometric data. Deployment considerations include ensuring model scalability, monitoring performance, and maintaining data privacy and security.

# Contributing
Contributions to the project are welcome! 

