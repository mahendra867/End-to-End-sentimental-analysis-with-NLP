# End-to-End-sentimental-analysis-with-NLP

A brief description of what this project does and who it's for


## Problem Statement
Problem Context
In today's digital age, vast amounts of textual data are generated daily through various platforms such as social media, product reviews, and customer feedback. Understanding the sentiment behind this text is crucial for businesses, researchers, and policymakers to make informed decisions. Sentiment analysis, the task of identifying and categorizing opinions expressed in text, is a powerful tool that can be applied across numerous domains, including marketing, customer service, and public relations.


## Aim :
The objective of this project is to develop a machine learning model capable of accurately classifying the sentiment of text data as either positive or negative. This classification task involves analyzing the textual content to determine the underlying emotional tone expressed by the author.

## Dataset Attributes

Text: This column contains the text data, which can be customer reviews, social media posts, or any other form of textual content. This is the primary input for the model, from which sentiment is derived.

Sentiment: This column contains the sentiment labels corresponding to each text entry. The sentiment is usually categorized as either positive or negative. It serves as the target variable for the machine learning model.

## Approach

### Project Description: Sentiment Analysis with NLP

This project focuses on developing an end-to-end pipeline for sentiment analysis using Natural Language Processing (NLP) techniques. The goal is to classify movie reviews as either positive or negative. The pipeline includes several stages: data ingestion, data validation, data transformation, model training, model evaluation, and prediction.

#### 1. **Data Ingestion**

The data ingestion phase involves downloading and preparing the raw dataset for further processing. This phase includes:
- **Downloading the Data:** The dataset is downloaded from a specified URL and saved locally as a zip file.
- **Extracting the Data:** The downloaded zip file is extracted to a directory for easy access.

#### 2. **Data Validation**

Data validation ensures the quality and integrity of the dataset before proceeding to the next steps. This phase includes:
- **Column Validation:** Checking if all required columns are present in the dataset.
- **Status Reporting:** Generating a status file indicating whether the validation passed or failed. If the data is not in the correct format, the training pipeline will not proceed, saving computational resources.

#### 3. **Data Transformation**

Data transformation prepares the data for model training by converting it into a suitable format. This phase includes:
- **Label Encoding:** Converting sentiment labels (positive, negative) into numerical format.
- **Train-Test Split:** Splitting the dataset into training and testing sets.
- **Text Cleaning:** Removing unwanted characters and normalizing text for better model performance.
- **Building Vocabulary:** Creating a vocabulary from the training data to be used in embedding.
- **Embedding Coverage:** Checking the coverage of the vocabulary in the pre-trained GloVe embeddings.
- **Padding Sequences:** Converting text data into padded sequences to feed into the neural network.
- **Saving Preprocessing Objects:** Saving the tokenizer object for use in prediction.

#### 4. **Model Training**

Model training involves building and training a deep learning model to classify the sentiments of reviews. This phase includes:
- **Model Architecture:** Defining a neural network with embedding, convolutional, and LSTM layers.
- **Training the Model:** Training the model on the training data with validation.
- **Saving the Model:** Saving the trained model for future predictions.

#### 5. **Model Evaluation**

Model evaluation involves assessing the model's performance on unseen data. This phase includes:
- **Plotting Metrics:** Plotting training and validation loss and accuracy to visualize model performance.
- **Early Stopping and Checkpoints:** Using callbacks to save the best model and prevent overfitting.

#### 6. **Prediction**

The prediction pipeline involves making predictions on new data using the trained model. This phase includes:
- **Loading the Model and Tokenizer:** Loading the saved model and tokenizer objects.
- **Data Preprocessing:** Cleaning and converting new input data into the format required by the model.
- **Making Predictions:** Using the model to predict the sentiment of new reviews.

### Main Theme

The main theme of this project is to leverage NLP techniques and deep learning to automate the process of sentiment analysis on movie reviews. By building a comprehensive pipeline that handles everything from data ingestion to model evaluation and prediction, we ensure a streamlined and efficient workflow for sentiment classification tasks. This project demonstrates the practical application of NLP in real-world scenarios, highlighting the importance of data preprocessing, model training, and evaluation in building robust predictive models.





# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/mahendra867/random_datasets/raw/main/Heart_csv.zip
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mlproj python=3.8 -y
```

```bash
conda activate mlproj
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```



