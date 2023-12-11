## Unveiling Illicit Currents: An Analytical Exploration of Electricity Consumption Patterns for Detecting Theft in the Power Grid

**Juan Diego Suárez Londoño**

### Executive summary

## **Project overview and goals:** The goal of this project is to develop a robust system for identifying and mitigating electricity theft within the electric grid. Using classification models, the project aims to analyze electricity consumption behavior patterns of users, employing various data mining and machine learning techniques. The ultimate objective is to enhance the efficiency of theft detection, contributing to the financial stability of utility companies, ensuring the safety and reliability of the electric grid, and aligning with legal and ethical standards.    


**Findings:** The best model for detecting suicide ideation is the Support Vector Classifier model, with an accuracy score of 0.924, a recall of 0.916, and an F-1 of 0.924. Its performance is followed by the Logistic Regression model, Naive Bayes model, and the Decision Tree model. This decision is based off comparing the finetuned models' accuracy, recall, and F1 scores (results summary below). The SVC model has the best accuracy (0.924), the Naive Bayes model has the best recall (0.955), and the SVC model has the best F1 score (0.924). As for the errors of each model, Logistic Regression has nearly twice as many FN (false negatives) as FP (false positives); Naive Bayes has nearly four times as many FP as FN; Decision Tree has similar FP and FN counts with slightly more FN, and SVC has slightly more FN than FP. 

[![Model-Scores.png](https://i.postimg.cc/tJKbRcwv/Model-Scores.png)](https://postimg.cc/30FVntK2)


**Results and conclusion:** Our evaluation of the best model returned a list of words associated with their feature importance, or how helpful/unhelpful they were in the classification task. A global model analysis revealing the most effective (stemmed) words to detect [+ suicide] across the entire data shows the top five features to be ["suicid", "kill", "end", "pill", "life"] (results below). 

[![Model-Feature-Weights.png](https://i.postimg.cc/XqqF6syz/Model-Feature-Weights.png)](https://postimg.cc/BPWtxcpT)

A local model analysis consisting of two example predictions (one suicide, one non-suicide) returns the most prominent features of those individual posts in the class decision. The KL-divergence for this non-suicidal class prediction is ~0.012, indicating close similarity in predictions between the black box model and the model built with LIME used for interpretation.

Image (below): Feature weights of a non-suicidal post prediction 

[![Non-Suicidal-Example.png](https://i.postimg.cc/261qyV2H/Non-Suicidal-Example.png)](https://postimg.cc/S2pQtN08)

Image (below): Feature weight of a suicidal post prediction. A dark green highlight indicates substantial importance in the prediction while dark red indicates negative importance, or the word reduces the prediction value/increases the loss. The KL-divergence for this prediction is ~0.010, indicating close similarity in predictions between the black box model and the model built with LIME used for interpretation. 

[![Suicidal-Example.png](https://i.postimg.cc/D0D81FY2/Suicidal-Example.png)](https://postimg.cc/QKQ83L3P)


This study shows that a word's presence in a document/social media post plays a significant role in the task of identifying individuals with suicide ideation -- some words more than others. Below is a word cloud, which showcases the most representative (common) words of each class. Further work can be done in crosschecking this ordering of words vs their weights assigned by not just the SVC model but also the Logistic Regression, Decision Tree, and Naive Bayes models. 

[![Suicidal-Word-Cloud.png](https://i.postimg.cc/fTdYG6x0/Suicidal-Word-Cloud.png)](https://postimg.cc/Yhp4gDDp)

[![Non-Suicidal-Word-Cloud.png](https://i.postimg.cc/66ySFLYG/Non-Suicidal-Word-Cloud.png)](https://postimg.cc/s1CwhSFV)


**Future research and development:** Comparing the two feature values shows an interesting dichotomy in the feature importance assigned "feel". It has a high positive score in classifying our first post as non-suicidal and a negative importance in classifying our second post as suicidal. This gives way to a theory that the absense of certain words is a significant feature in gauging a post's non-membership of a class -- and in a binary classification problem, its membership of the second class. Because it is impossible for a person/post to be both or neither suicidal nor non-suicidal, saying that a post is not [suicidal] is synonymous with saying it is [not suicidal].

**Next steps and recommendations:** We can explore this possibility by **using the pandas query function to compile two lists of words: Those that appear only in suicidal posts and those that only appear in non-suicidal posts.** One might begin by **building two vocabularies**, the first consisting of every unique word from the all the posts in the suicidal class, and the second consisting of every unique word from the non-suicidal class. It may be illuminating to **go through these two vocabularies and make a list of [words that appear in class 1 and not in class 0] and another list of [words that appear in class 0 and not in class 1]**. Additionally, one could **cross check the model permutation importance scores of words in these two lists to see if they have high absolute permutation importance values.**

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

**Exploratory data analysis:** The initial exploration of the dataset revealed crucial insights into the challenges and characteristics of the data. With 1036 columns representing daily electricity consumption and 42,372 rows, the dataset captures a vast range of information. Notable findings include a substantial percentage (25.6%) of missing values, a skewed distribution of consumption values, and a significant class imbalance in the 'FLAG' column, where '0' (non-theft) dominates over '1' (theft) cases.

[![Post-Char-Count.png](https://i.postimg.cc/8zMs46q2/Post-Char-Count.png)](https://postimg.cc/QFNXxHNm)


**Cleaning and preparation:** To prepare the data for analysis, necessary cleaning steps were taken. Columns with high missing values ('CONS_NO' and '10/3/2014') were removed, addressing issues that could hinder model performance. Additionally, the class imbalance in the 'FLAG' column was mitigated through undersampling, resulting in a balanced dataset with a shape of (7230, 1034). These steps laid the foundation for further analysis and model development.

**Preprocessing:** The preprocessing phase focused on handling missing values in the remaining dataset. Leveraging the KNNImputer algorithm, missing values were imputed, ensuring a comprehensive and complete dataset. This step was crucial for maintaining the integrity of the data and facilitating accurate model training and evaluation.

**Final Dataset:** The final dataset emerged as a well-prepared, balanced, and imputed version with no missing values. With a shape of (7230, 1034), the dataset was ready for feature selection, model development, and evaluation. The completion of this phase marked a significant milestone, setting the stage for the application of various classification models to identify patterns associated with electricity theft effectively.

The data is randomly split into train and test sets, with a test size of 0.2. 

[![Post-Class-Counts.png](https://i.postimg.cc/y6DR9k8H/Post-Class-Counts.png)](https://postimg.cc/5jMjW9z7)

The first five rows of the dataframe can be seen below. 

[![Final-Df-Head.png](https://i.postimg.cc/NMHgGqBW/Final-Df-Head.png)](https://postimg.cc/WFsRSyDn)


### Methodology

Holdout cross validation is implemented. Models were trained on the training set and validated with the test set. Additionally, RandomizedSearchCV was used to evaluate models using accuracy score, and fine tuned each model's hyperparameters to maximize this metric.  Accuracy is suitable because we have a balanced dataset and measures the proportion of correctly predicted observations out of total observations. It is calculated as (True Positive count + True Negative count) / (Total count), or

$$
\frac{TP + TN}{TP + TN + FP + FN}
$$


Four models were trained, fine-tuned, and will be later compared to find the best model for this task.

**Naive Bayes Model:** A pipeline object is created to standardize the data using TF-IDF and instantiate a Naïve Bayes model. RandomizedSearchCV is used to find the optimal alpha value (options: [1, 0.1, 0.01, 0.001, 0]), which turned out to be 0.1.

**Logistic Regression Model:** A pipeline object is created to standardize the data using TF-IDF and instantiate a Logistic Regression model. RandomizedSearchCV is used to find (1) the optimal penlaty, with the options being ['l1', 'l2', 'none'] and (2) whether an intercept term should be included (options: [True, False]). The best model has l2 penalty and an intercept term.

**Decision Tree Model:** A pipeline object is created to standardize the data using TF-IDF and instantiate a Decision Tree model. RandomizedSearchCV is used to find (1) the optimal tree criterion, with the options being ['gini', 'entropy'], (2) max depth (options: [None, 1, 2, 3, 4]), and (3) the minimum sample count required to split a node (options: [1, 2, 3, 4]). The best model has a minimum sample split of 3, no max depth, and uses the gini criterion.

**Support Vector Classifier Model:** A pipeline object is created to standardize the data using TF-IDF and instantiate a Support Vector Classifier model using a linear kernel. RandomizedSearchCV is used to (1) The optimal C value (options: [0.1, 1, 10, 100, 1000]) and (2) Probability (options: [True, False]). The best model has a probability = True and a C value of 1.

### Model evaluation and results 

Model performance will be visualized using confusion matrices, which indicate the counts of each error type a model made in the classification task. In these plots, 0 = not suicidal and 1 = suicidal.


**Support vector classifier model:** The SVC model is the best model for detecting suicide ideation, with an accuracy score of 0.924, a recall of 0.916, and an F-1 of 0.924. This decision is based off comparing the fine-tuned models' accuracy, recall, and F1 scores. Out of all the models, the SVC has the best accuracy and F1 scores, and the second best recall score. It also took the longest to train, at 2339 seconds. It predicts slightly more false negatives than false positives and its confusion matrix can be seen below. 

[![SVCMatrix.png](https://i.postimg.cc/3xzf5dJC/SVCMatrix.png)](https://postimg.cc/21F7njWV)

**Logistic regression model:**  The Logistic Regression model has an accuracy of 0.917, recall of 0.899, and F-1 of 0.916. It predicts nearly twice as many FN (false negatives) as FP (false positives), indicating that it is much more likely to classify a suicidal person as non-suicidal than vice versa. Its confusion matrix can be seen below. 

[![LGRMatrix.png](https://i.postimg.cc/MKKF1CFR/LGRMatrix.png)](https://postimg.cc/4nqPgSZN)

**Naive Bayes model:** The Naive Bayes model has an accuracy of 0.890, the best overall recall of 0.955, and an F-1 of 0.897. It predicts nearly four times as many false positives as false negatives, indicating it is much more likely to classify a non-suicidal person as suicidal than the other way around. Its confusion matrix can be seen below. 

[![NBMatrix.png](https://i.postimg.cc/xTtB7fzt/NBMatrix.png)](https://postimg.cc/crtDtNTn)
 
**Decision tree model:** The Decision Tree model has an accuracy of 0.822, a recall of 0.814, and an F-1 of 0.822. It takes the least amount of time to train at 16 seconds, and has similar false positive and false negative counts. Its confusion matrix can be seen below. 

[![DTree-Matrix.png](https://i.postimg.cc/qqQD9wzz/DTree-Matrix.png)](https://postimg.cc/9wRB9GPh)

A detailed interpretation and evaluation of the best model can be found in the results and conclusion section of the executive summary above. 

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
