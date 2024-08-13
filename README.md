## Sentiment Analysis on IMDb Reviews

This project involves building a sentiment analysis model using Recurrent Neural Networks (RNNs) to classify movie reviews from the IMDb dataset as either positive or negative. The model is built using TensorFlow and Keras and incorporates advanced RNN techniques such as Bidirectional GRU layers with dropout for regularization.

**Table of Contents**

1. Overview
2. Requirements
3. Data
4. Model Architecture
5. Training
6. Evaluation
7. Results
8. Usage
9. EDA
10. License

**Overview**

The goal of this project is to classify movie reviews from the IMDb dataset into positive or negative sentiments. The model uses an enhanced RNN architecture to achieve high performance in sentiment classification.

**Requirements**

To run this project, you need to have the following libraries installed:

- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

You can install the required libraries using pip:

```
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

**Data**

The dataset used in this project is the IMDb movie reviews dataset, which is available directly from TensorFlow/Keras. The dataset contains 50,000 reviews (25,000 for training and 25,000 for testing), each labeled as positive or negative.

**Model Architecture**

The model is a Sequential neural network with the following layers:

1. Embedding Layer: Converts word indices to dense vectors of fixed size.
2. Bidirectional GRU Layer: Processes sequences in both directions to capture dependencies.
3. Layer Normalization: Normalizes activations to stabilize training.
4. Dropout Layers: Prevents overfitting by randomly setting a fraction of input units to 0.
5. Additional Bidirectional GRU Layers: Further process the sequence data with dropout for regularization.
6. Dense Layer: Final layer with a sigmoid activation function for binary classification.

**Training**

The model is trained using the following parameters:

- Optimizer: Adam with a learning rate of 0.001
- Loss Function: Binary Crossentropy
- Epochs: 20
- Batch Size: 64
- Validation Split: 20% of training data used for validation

**Callbacks**

- Early Stopping: Stops training when the validation loss does not improve for a set number of epochs.
- Reduce Learning Rate on Plateau: Reduces the learning rate when the validation loss plateaus.

**Evaluation**

The model's performance is evaluated on the test dataset. Metrics include accuracy, precision, recall, and F1-score.

**Results**

The final model achieves the following metrics on the test dataset:

- Test Accuracy: 0.8852
- Test Loss: 0.3491

Precision, recall, and F1-scores for each class are:

- Negative:
  - Precision: 0.88
  - Recall: 0.90
  - F1-Score: 0.89

- Positive:
  - Precision: 0.90
  - Recall: 0.87
  - F1-Score: 0.88

- Accuracy: 0.89

- Macro Average:
  - Precision: 0.89
  - Recall: 0.89
  - F1-Score: 0.89

- Weighted Average:
  - Precision: 0.89
  - Recall: 0.89
  - F1-Score: 0.89

**Usage**

1. Clone this repository:
   ```
   git clone https://github.com/J3lly-Been/sentiment-analysis-imdb.git
   ```
2. Navigate to the project directory:
   ```
   cd sentiment-analysis-imdb
   ```
3. Open the Jupyter notebook for sentiment analysis:
   ```
   jupyter notebook imdb-sentiment-analysis.ipynb
   ```
   Run the cells to train and evaluate the model.

**EDA**

For exploratory data analysis (EDA) on the IMDb dataset, including insights on review lengths, most frequent words, and more, refer to the `imdb-eda.ipynb` notebook. This notebook provides detailed analysis and visualizations to understand the dataset better.

**License**

This project is licensed under the MIT License - see the LICENSE file for details.
