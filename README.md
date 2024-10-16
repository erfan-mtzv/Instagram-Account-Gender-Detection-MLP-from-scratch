# Instagram Account Gender Detection (MLP from Scratch)

This repository contains Python code for implementing a Multi-Layer Perceptron (MLP) from scratch, aimed at predicting the gender of Instagram account holders using their account data. The model is trained using preprocessed features, including both numeric and non-numeric data, such as usernames and bios.



## Overview
The purpose of this project is to predict the gender of Instagram account holders based on account information such as username, name, bio, and other available account data. The MLP model is built from scratch using Python, without relying on external deep learning libraries like TensorFlow or PyTorch.

## Preprocessing
The input data has undergone the following preprocessing steps to ensure compatibility with the MLP model:

1. **Binary Feature Transformation**:
   - To avoid gradient vanishing problems that binary values might cause during training, binary columns were replaced with bipolar values (`-1` and `1` instead of `0` and `1`).

2. **Normalization**:
   - For numerical columns, values were normalized using **Min-Max Normalization** to bring them to a standard scale, ensuring no feature dominates the model due to its range of values. This normalization was performed using the formula:
     \$
     X' = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}
     \$

3. **TF-IDF Encoding for Non-Numeric Features**:
   - Non-numeric features such as `username`, `name`, and `bio` were transformed into numerical values using **TF-IDF (Term Frequency-Inverse Document Frequency)**. TF-IDF scores help in representing text data as vectors based on the importance of each word in a sentence relative to the entire dataset.

   - **Term Frequency (TF)** is the ratio of occurrences of a word in a sentence to the total number of words in that sentence.  
   - **Inverse Document Frequency (IDF)** is the logarithmic ratio of the total number of sentences to the number of sentences that contain the word.

4. **Adjusting Vector Lengths**:
   - Using the `TfidfVectorizer` function, the length of the resulting vector for each feature was adjusted using the `max_features` parameter, which limits the number of words for which TF-IDF scores are calculated. For example:
     - **Username**: Vector length = 6981 (unique words).
     - **Name**: Vector length = 8926 (unique words).

   These vectors provide a numerical representation of non-numeric features, making them suitable for input into the MLP model.

## Model
The Multi-Layer Perceptron (MLP) model is built from scratch in Python, without using high-level libraries. It consists of:
- **Input Layer**: Takes in the preprocessed feature vectors.
- **Hidden Layer**: A layer with adjustable neurons and activation functions (e.g., ReLU, sigmoid).
- **Output Layer**: Outputs a prediction for the gender of the account holder.

## Usage
1. **Preprocess the Data**:
   Before training the model, ensure that the data is preprocessed using the steps outlined in the preprocessing section. Binary features should be transformed, numerical features normalized, and non-numeric text features converted using TF-IDF.

2. **Training the Model**:
   The MLP model can be trained on the preprocessed data by running the provided training script. Ensure that your data is in the proper format and structure before starting the training process.

3. **Evaluating the Model**:
   The model is evaluated on test data that **does not contain gender labels**. To simulate evaluation, random labels were assigned to the test data for demonstration purposes.

   > **Note**: Since the test data is randomly labeled, the evaluation results are not representative of the model's true performance.

## Results
Since the test data has been randomly labeled, the evaluation results are purely illustrative and should not be used to assess the actual performance of the model. These results should be treated as a placeholder until labeled test data is provided for proper validation.
