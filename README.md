# BreadcrumbsInstagram-Account-Gender-Detection-MLP-from-scratch
This repository contains Python code for a multi-layer perceptron (MLP) model, implemented from scratch, which predicts the gender of Instagram account holders using their account data.

## pre-processing
The training data has been preprocessed first. Since binary values often cause gradient vanishing problems in the network, the values of columns containing binary numbers have been replaced with bipolar numbers. Then, by defining a function, I've normalized the numerical column values using the max-min normalization formula.

For non-numeric features such as username, name, and bio, we use the TF-IDF method. TF-IDF stands for Term Frequency-Inverse Document Frequency, which assigns a numerical value (TF-IDF score) to each word present in a sentence. The number of elements in the resulting vector for a sentence is equal to the number of unique words in that sentence.

TF-IDF score for each word in a sentence is calculated as the product of the term frequency (TF) and inverse document frequency (IDF) of that word in the entire dataset.
The TF of a word in a sentence is the ratio of the number of times the word appears in the sentence to the total number of words in the sentence (word density).
The IDF of a word in the dataset is the logarithm of the ratio of the total number of sentences to the number of sentences containing that word (term specificity).
TF-IDF scores are used to represent non-numeric features as vectors of numerical values. These vectors can then be used in machine learning algorithms.

In addition, by using the TfidfVectorizer function, the length of the resulting vector for each column of the data frame can be changed by setting the max_feature parameter as the number of most frequent words for which TF-IDF has been calculated. The resulting vector lengths for the name and user columns are 6981 and 8926, respectively, where each number represents the count of unique words or strings (excluding the period and space characters, which are considered as separators) in each column of the data frame.
