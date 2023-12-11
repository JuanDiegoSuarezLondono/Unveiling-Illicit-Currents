## Unveiling Illicit Currents: An Analytical Exploration of Electricity Consumption Patterns for Detecting Theft in the Power Grid

**Juan Diego Suárez Londoño**

### Executive summary

#### **Project overview and goals:**
The goal of this project is to develop a robust system for identifying and mitigating electricity theft within the electric grid. Using classification models, the project aims to analyze electricity consumption behavior patterns of users, employing various data mining and machine learning techniques. The ultimate objective is to enhance the efficiency of theft detection, contributing to the financial stability of utility companies, ensuring the safety and reliability of the electric grid, and aligning with legal and ethical standards.    
#### **Findings:**
The optimal model for detecting electricity theft patterns is the Gradient Boosting model, achieving a mean recall of 0.71, accuracy of 0.725, and F1 score of 0.66. This conclusion is drawn from a thorough evaluation of the finetuned models, considering metrics such as mean recall, accuracy, and F1 scores (see the summary below).

The Gradient Boosting model stands out with the highest F1 score (0.66), providing a well-balanced performance in terms of accuracy and recall. In contrast, SVM demonstrates the highest recall (0.92), while Decision Trees exhibit a balanced distribution of false negatives (FN) and false positives (FP). Neural Networks, although achieving a respectable accuracy, show a slightly higher FN count compared to FP.

The comprehensive performance summary of each model, including k-fold cross-validation results, is presented in the preceding section.

[![Summary of results in a table](https://i.postimg.cc/fTYYvn06/Summary-of-results-in-a-table.png)](https://postimg.cc/cr4Kszhm)


#### **Results and conclusion:**
Our evaluation of the best model returned a list of words associated with their feature importance, or how helpful/unhelpful they were in the classification task. A global model analysis revealing the most effective (stemmed) words to detect [+ suicide] across the entire data shows the top five features to be ["suicid", "kill", "end", "pill", "life"] (results below). 

[![Model-Feature-Weights.png](https://i.postimg.cc/XqqF6syz/Model-Feature-Weights.png)](https://postimg.cc/BPWtxcpT)

A local model analysis consisting of two example predictions (one suicide, one non-suicide) returns the most prominent features of those individual posts in the class decision. The KL-divergence for this non-suicidal class prediction is ~0.012, indicating close similarity in predictions between the black box model and the model built with LIME used for interpretation.

Image (below): Feature weights of a non-suicidal post prediction 

[![Non-Suicidal-Example.png](https://i.postimg.cc/261qyV2H/Non-Suicidal-Example.png)](https://postimg.cc/S2pQtN08)

Image (below): Feature weight of a suicidal post prediction. A dark green highlight indicates substantial importance in the prediction while dark red indicates negative importance, or the word reduces the prediction value/increases the loss. The KL-divergence for this prediction is ~0.010, indicating close similarity in predictions between the black box model and the model built with LIME used for interpretation. 

[![Suicidal-Example.png](https://i.postimg.cc/D0D81FY2/Suicidal-Example.png)](https://postimg.cc/QKQ83L3P)


This study shows that a word's presence in a document/social media post plays a significant role in the task of identifying individuals with suicide ideation -- some words more than others. Below is a word cloud, which showcases the most representative (common) words of each class. Further work can be done in crosschecking this ordering of words vs their weights assigned by not just the SVC model but also the Logistic Regression, Decision Tree, and Naive Bayes models. 

[![Suicidal-Word-Cloud.png](https://i.postimg.cc/fTdYG6x0/Suicidal-Word-Cloud.png)](https://postimg.cc/Yhp4gDDp)

[![Non-Suicidal-Word-Cloud.png](https://i.postimg.cc/66ySFLYG/Non-Suicidal-Word-Cloud.png)](https://postimg.cc/s1CwhSFV)


#### **Future research and development:**
Comparing the two feature values shows an interesting dichotomy in the feature importance assigned "feel". It has a high positive score in classifying our first post as non-suicidal and a negative importance in classifying our second post as suicidal. This gives way to a theory that the absense of certain words is a significant feature in gauging a post's non-membership of a class -- and in a binary classification problem, its membership of the second class. Because it is impossible for a person/post to be both or neither suicidal nor non-suicidal, saying that a post is not [suicidal] is synonymous with saying it is [not suicidal].

#### **Next steps and recommendations:**
We can explore this possibility by **using the pandas query function to compile two lists of words: Those that appear only in suicidal posts and those that only appear in non-suicidal posts.** One might begin by **building two vocabularies**, the first consisting of every unique word from the all the posts in the suicidal class, and the second consisting of every unique word from the non-suicidal class. It may be illuminating to **go through these two vocabularies and make a list of [words that appear in class 1 and not in class 0] and another list of [words that appear in class 0 and not in class 1]**. Additionally, one could **cross check the model permutation importance scores of words in these two lists to see if they have high absolute permutation importance values.**

Additionally, further work can be done in improving the performance of the best classification model. **The best model can be extended, further fine-tuned, or even replaced by experimenting with or incorporating other algorithms and techniques, such as ensemble methods or recurrent neural networks**. Future exploration can also be done in **crosschecking the importance of words, as indicated by the word cloud, and the words' weights, assigned by not just the SVC model but also the Logistic Regression, Decision Tree, and Naive Bayes models.**


### Rationale

This project, titled 'Unveiling Illicit Currents: An Analytical Exploration of Electricity Consumption Patterns for Detecting Theft in the Power Grid,' aims to address the issue of electricity theft in the power grid. The initiative seeks to develop a robust system to identify and mitigate electricity theft through the use of classification models and data mining and machine learning techniques. The focus is on analyzing the electricity consumption behavior patterns of users to improve efficiency in theft detection, thus contributing to the financial stability of utility companies, ensuring the safety and reliability of the power grid, and complying with legal and ethical standards.

### Research Question

The central question driving this project is: "Which classification model proves most effective in identifying individuals at risk for suicide, and what features or words contribute most significantly to this predictive task?"

### Data Sources

**Dataset:** The dataset used in this project is sourced from Kaggle and can be accessed at [SGCC Electricity Theft Detection](https://www.kaggle.com/datasets/bensalem14/sgcc-dataset/)

#### About Dataset
- **Overview**

    The State Grid Corporation of China (SGCC) dataset with 1000 records was used in the model. This is a key resource in the field of power distribution and management, with a large and varied set of data about electricity transport and grid operations. This set of data contains a lot of different kinds of information, such as history and real-time data on energy use, grid infrastructure, the integration of green energy, and grid performance.
    
- **Description**

    Electricity theft detection released by the State Grid Corporation of China (SGCC) dataset data set.csv contains 1037 columns and 42,372 rows for electric consumption from January first 2014 to 30 October 2016. SGCC data first column is consumer ID that is alphanumeric. Then from column 2 to columns 1036 daily electricity consumption is given. Last column named flag is the labels in 0 and 1 values.

- **Features**

    - ***'MM/DD/YYYY':*** The electric consumption on a given day.
    - ***CONS_NO:*** Consumer Number stands for a customer ID of string type.
    - ***FLAG:*** 0 indicating no theft and 1 for theft.

### **Exploratory data analysis:**
The initial exploration of the dataset revealed crucial insights into the challenges and characteristics of the data. With 1036 columns representing daily electricity consumption and 42,372 rows, the dataset captures a vast range of information. Notable findings include a substantial percentage (25.6%) of missing values, a skewed distribution of consumption values, and a significant class imbalance in the 'FLAG' column, where '0' (non-theft) dominates over '1' (theft) cases.

[![Proportion of Null and Non-Null Data.png](https://i.postimg.cc/rsxKpGP4/1.png)](https://postimg.cc/8fk1KvDk)

[![Number of Users by Theft Indicator.png](https://i.postimg.cc/tRx1Dy4g/2.png)](https://postimg.cc/mhT22vMx)

**Cleaning and preparation:** To prepare the data for analysis, necessary cleaning steps were taken. Columns with high missing values ('CONS_NO' and '10/3/2014') were removed, addressing issues that could hinder model performance. Additionally, the class imbalance in the 'FLAG' column was mitigated through undersampling, resulting in a balanced dataset with a shape of (7230, 1034). These steps laid the foundation for further analysis and model development.

**Preprocessing:** The preprocessing phase focused on handling missing values in the remaining dataset. Leveraging the KNNImputer algorithm, missing values were imputed, ensuring a comprehensive and complete dataset. This step was crucial for maintaining the integrity of the data and facilitating accurate model training and evaluation.

**Final Dataset:** The final dataset emerged as a well-prepared, balanced, and imputed version with no missing values. With a shape of (7230, 1034), the dataset was ready for feature selection, model development, and evaluation. The completion of this phase marked a significant milestone, setting the stage for the application of various classification models to identify patterns associated with electricity theft effectively.

The data is randomly split into train and test sets, with a test size of 0.2. 

[![Number of Users by Theft Indicator balanced.png](https://i.postimg.cc/4dJ5V5cz/Number-of-Users-by-Theft-Indicator-balanced.png)](https://postimg.cc/cKbwNfXH)

### Methodology

In this project, a thorough evaluation of various machine learning models was conducted using a similar methodology to ensure consistency and comparability of results. The following steps were undertaken:

# Machine Learning Model Evaluation

## Methodology

In this project, a thorough evaluation of various machine learning models was conducted using a similar methodology to ensure consistency and comparability of results. The following steps were undertaken:

#### Data Preparation

- The dataset was split into training and test sets to facilitate model training and evaluation.
- Text data was preprocessed and standardized using the Term Frequency-Inverse Document Frequency (TF-IDF) technique.

#### Model Selection and Training

Seven different machine learning models were considered for this task:

1. **K-Nearest Neighbors (KNN)**
   - A KNN model was built and fine-tuned using GridSearchCV to find the optimal hyperparameters, including the number of neighbors, weighting scheme, and distance metric.
   - The model was trained on the training set and evaluated on the test set.

   **Hyperparameters Evaluated:**
   - `n_neighbors`: [3, 5, 7]
   - `weights`: ['uniform', 'distance']
   - `p`: [1, 2]

2. **Decision Trees (DT)**
   - A Decision Tree model was constructed and optimized through GridSearchCV to determine the best parameters, such as maximum depth and minimum samples required for node splitting.
   - The model's performance was assessed on the test set.

   **Hyperparameters Evaluated:**
   - `max_depth`: [10, 20, 30]
   - `min_samples_split`: [2, 5, 10]

3. **Support Vector Machines (SVM)**
   - An SVM model was developed, and hyperparameters were fine-tuned using GridSearchCV, considering parameters like the regularization parameter (C), kernel type, and gamma.
   - The model's performance was evaluated on the test set.

   **Hyperparameters Evaluated:**
   - `C`: [10, 20]
   - `kernel`: ['rbf']
   - `gamma`: ['scale', 'auto']

4. **Random Forest (RF)**
   - A Random Forest model was built and optimized through GridSearchCV to find the best combination of parameters, including the number of trees, maximum depth, and bootstrap method.
   - The model's performance was assessed on the test set.

   **Hyperparameters Evaluated:**
   - `n_estimators`: [50, 200]
   - `max_depth`: [10, 20]
   - `bootstrap`: [False]

5. **Gradient Boosting**
   - A Gradient Boosting model was initialized and optimized through GridSearchCV to adjust hyperparameters such as the number of estimators, maximum depth, and minimum samples required for node splitting.
   - The model's performance was evaluated on the test set.

   **Hyperparameters Evaluated:**
   - `n_estimators`: [50]
   - `max_depth`: [3, 4, 5]
   - `min_samples_split`: [2, 5, 10]

6. **XGBoost**
   - An XGBoost model was initialized and optimized through GridSearchCV to adjust hyperparameters, including the number of estimators, learning rate, and gamma.
   - The model's performance was evaluated on the test set.

   **Hyperparameters Evaluated:**
   - `n_estimators`: [50, 100, 200]
   - `learning_rate`: [0.2]
   - `gamma`: [0, 0.1, 0.2]

7. **Neural Networks (NN)**
   - A Neural Network model was initialized using MLPClassifier, and parameters were adjusted through GridSearchCV, including hidden layer sizes, activation function, and alpha.
   - The model's performance was evaluated on the test set.

   **Hyperparameters Evaluated:**
   - `hidden_layer_sizes`: [(30, 20, 10)]
   - `activation`: ['relu', 'logistic']
   - `alpha`: [0.0001, 0.001, 0.01]

### Model Evaluation

For each model, the following steps were taken:

- **Calculate Efficiency:**
  - Obtain the best hyperparameters identified by the respective GridSearchCV.
  - Make predictions on the test set using the best-fitted model.
  - Evaluate the model's accuracy on the test set.

- **Confusion Matrix Visualization:**
  - Create a confusion matrix to visualize the model's performance in terms of true positives, true negatives, false positives, and false negatives.

### Model Evaluation and Results

In this section, we will examine the performance and results of the models applied to the electricity theft detection problem.

#### K-Nearest Neighbors (KNN)
- **Accuracy:** 64.32%
- **Best Parameters:** {'n_neighbors': 5, 'p': 2, 'weights': 'distance'}
- **Classification Report:**

              precision    recall  f1-score   support
       0       0.61      0.73      0.67       704
       1       0.69      0.56      0.62       742
       accuracy                    0.64       1446
- **Confusion Matrix:**
[![Confusion Matrix - KNN.png](https://i.postimg.cc/63m9754N/Confusion-Matrix-KNN.png)](https://postimg.cc/N9RhStxN)

#### Decision Trees
- **Accuracy:** 62.59%
- **Best Parameters:** {'max_depth': 30, 'min_samples_split': 2}
- **Classification Report:**

              precision    recall  f1-score   support
       0       0.61      0.64      0.63       704
       1       0.64      0.61      0.63       742
       accuracy                    0.63       1446
- **Confusion Matrix:**
[![Confusion Matrix - Decision Trees.png](https://i.postimg.cc/Dz7fHN3R/Confusion-Matrix-Decision-Trees.png)](https://postimg.cc/0zHqJcQC)

#### Support Vector Machines (SVM)
- **Accuracy:** 60.17%
- **Best Parameters:** {'C': 20, 'gamma': 'auto', 'kernel': 'rbf'}
- **Classification Report:**

              precision    recall  f1-score   support
       0       0.76      0.27      0.40       704
       1       0.57      0.92      0.70       742
       accuracy                    0.60       1446
- **Confusion Matrix:**
[![Confusion Matrix - SVM](https://i.postimg.cc/5jcb37m7/Confusion-Matrix-SVM.png)](https://postimg.cc/1VKkRK2w)

#### Random Forest
- **Best Parameters:** {'bootstrap': False, 'max_depth': 20, 'n_estimators': 200}
- **Accuracy:** 72.34%
- **Classification Report:**

              precision    recall  f1-score   support
       0       0.71      0.73      0.72       704
       1       0.74      0.71      0.73       742
       accuracy                    0.72       1446
- **Confusion Matrix:**
[![Confusion Matrix - Random Forest](https://i.postimg.cc/RhJvw5wJ/Confusion-Matrix-Random-Forest.png)](https://postimg.cc/K3FXyWqm)

#### Gradient Boosting
- **Best Parameters:** {'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 50}
- **Accuracy:** 71.58%
- **Classification Report:**

              precision    recall  f1-score   support
       0       0.69      0.77      0.73       704
       1       0.75      0.66      0.71       742
       accuracy                    0.72       1446
- **Confusion Matrix:**
[![Confusion Matrix - Gradient Boosting](https://i.postimg.cc/TPQ26cR1/Confusion-Matrix-Gradient-Boosting.png)](https://postimg.cc/94RHPZwH)

#### XGBoost
- **Best Parameters:** {'gamma': 0, 'learning_rate': 0.2, 'n_estimators': 100}
- **Accuracy:** 72.41%
- **Classification Report:**

              precision    recall  f1-score   support
       0       0.70      0.76      0.73       704
       1       0.75      0.69      0.72       742
       accuracy                    0.72       1446
- **Confusion Matrix:**
[![Confusion Matrix - XGBoost](https://i.postimg.cc/XNr5sr8d/Confusion-Matrix-XGBoost.png)](https://postimg.cc/ThXhwYB3)

#### Neural Networks
- **Best Parameters:** {'activation': 'logistic', 'alpha': 0.001, 'hidden_layer_sizes': (30, 20, 10)}
- **Accuracy:** 63.62%
- **Classification Report:**
  
              precision    recall  f1-score   support
       0       0.61      0.70      0.65       704
       1       0.67      0.58      0.62       742
       accuracy                    0.64       1446
- **Confusion Matrix:**
[![Confusion Matrix - NN](https://i.postimg.cc/wvvqTsvV/Confusion-Matrix-NN.png)](https://postimg.cc/LqcKx5jY)

### Summarizing Results and Choosing the Best Model

| Model             | Accuracy (%) | Recall (Flag 1) (%) |
|-------------------|--------------|---------------------|
| KNN               | 64.32        | 56                  |
| Decision Trees    | 62.59        | 61                  |
| SVM               | 60.17        | 92                  |
| Random Forest     | 72.34        | 71                  |
| Gradient Boosting | 71.58        | 66                  |
| XGBoost           | 72.41        | 69                  |
| Neural Networks   | 63.62        | 58                  |

### K-fold Cross-Validation

#### KNN
- Mean Recall: 0.5353

#### Decision Trees
- Mean Recall: 0.6081

#### Support Vector Machines (SVM)
- Mean Recall: 0.9112

#### Random Forest
- Mean Recall: 0.6895

#### Gradient Boosting
- Mean Recall: 0.6759

#### XGBoost
- Mean Recall: 0.6544

#### Neural Networks
- Mean Recall: 0.6561

### Outline of project

[Link to download data](http://localhost:8888/files/Downloads/ML%20ipynb/Capstone/SuicideDetection.csv?_xsrf=2%7C01e7821a%7C54dac168862e9a75a5e46c1d11d7822e%7C1658864864) 

[Link to notebook](http://localhost:8888/lab/tree/capstone/SuicideIdeationDetection.ipynb) 

[Link to download notebook](http://localhost:8888/files/capstone/SuicideIdeationDetection.ipynb?_xsrf=2%7Cd31ff0b1%7C93c2c365c2bec949e09c3632966e8050%7C1662161901)

[Link to evaluation](http://localhost:8888/lab/tree/capstone/CapstoneEvaluation.ipynb)

[Link to download evaluation](http://localhost:8888/files/capstone/CapstoneEvaluation.ipynb?_xsrf=2%7Cd31ff0b1%7C93c2c365c2bec949e09c3632966e8050%7C1662161901)


### Contact and Further Information

Juan Diego Suárez Londoño

Email: suarezlondonjuandiego@protonmail.com

[LinkedIn](https://www.linkedin.com/in/jdlondo%C3%B1o/)
