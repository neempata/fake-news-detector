# Fake News Detection using Natural Language Processing

A machine learning project that classifies news articles as **REAL** or **FAKE** using Natural Language Processing (NLP). The project demonstrates the complete text classification workflow, from data preprocessing and TF-IDF feature extraction to model training and evaluation using a Passive Aggressive Classifier.

## Project Overview

This project explores how machine learning can be used to distinguish between factual and misleading news articles based solely on their textual content.

Rather than relying on manually defined rules, the model learns patterns from previously labeled news articles by transforming raw text into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**. These features are then used to train a **Passive Aggressive Classifier**, a linear algorithm designed for efficient learning on high-dimensional text data.

The project follows a complete NLP pipeline, including data cleaning, feature extraction, model training, prediction, and evaluation. The final model achieved an **accuracy of 99.25%** on the test dataset, demonstrating the effectiveness of combining TF-IDF vectorization with a Passive Aggressive Classifier for binary text classification.

## Dataset

- **Dataset:** Fake News Detection Dataset
- **Total Articles:** 3,988
- **Target Variable:** REAL / FAKE
- **Input Feature:** News article body
- **Classification Type:** Binary text classification

The dataset contains labeled news articles that allow the model to learn linguistic patterns associated with genuine and misleading information.

## Technologies Used

- Python
- 
- NumPy
- Scikit-learn
- Jupyter Notebook

## NLP Workflow

### Data Exploration

The dataset was loaded using Pandas and inspected to understand its structure, labels, and overall quality before beginning the machine learning pipeline.

### Data Preprocessing

The preprocessing stage included:

- Removing missing values
- Selecting the news article body as the input feature
- Converting labels into the human-readable classes **REAL** and **FAKE**
- Preparing the dataset for text vectorization

### Train-Test Split

The cleaned dataset was divided into training and testing sets using an **80/20 split**, allowing the model to be evaluated on articles it had never seen before.

### Text Vectorization

Since machine learning models cannot interpret raw text directly, each article was converted into a numerical feature vector using **TF-IDF Vectorization**.

The vectorizer was configured to:

- Remove English stop words
- Ignore extremely common words using `max_df = 0.7`

This creates a sparse numerical representation that emphasizes words which provide useful information while reducing the influence of overly common terms.

### Model Training

A **Passive Aggressive Classifier** was trained on the TF-IDF feature vectors extracted from the training data.

This algorithm is particularly well suited for text classification because it efficiently updates its decision boundary only when a prediction is incorrect, making it both lightweight and effective for high-dimensional datasets.

### Model Evaluation

The model was evaluated using:

- Accuracy Score
- Confusion Matrix

These metrics provide insight into both the model's overall performance and the types of prediction errors it makes.

## Results

The trained model demonstrated excellent performance on previously unseen news articles.

**Model Accuracy:** **99.25%**

### Confusion Matrix

| Actual | Predicted REAL | Predicted FAKE |
|---------|---------------:|---------------:|
| REAL | 404 | 5 |
| FAKE | 1 | 388 |

Out of 798 articles in the test set, the classifier correctly identified:

- **404 REAL** articles
- **388 FAKE** articles

Only **6 articles** were misclassified, resulting in an overall accuracy of **99.25%**. These results demonstrate the effectiveness of the NLP pipeline in distinguishing between genuine and misleading news content.

## Installation

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
fake_news_detection.ipynb
```

## What I Learned

This project gave me hands-on experience with the complete Natural Language Processing workflow. I learned how unstructured text must first be transformed into numerical representations before it can be used by machine learning models, and how TF-IDF captures the relative importance of words across a collection of documents.

It also introduced me to the Passive Aggressive Classifier and reinforced that strong model performance depends on more than simply choosing an algorithm. Careful preprocessing, thoughtful feature extraction, and evaluation on unseen data all play an important role in building reliable NLP systems.

## Future Improvements

Some improvements I'd like to explore include:

- Comparing additional classification algorithms such as Logistic Regression, Naive Bayes, and Support Vector Machines
- Hyperparameter tuning
- Lemmatization and stemming
- N-gram feature extraction
- Cross-validation
- Precision, Recall, and F1-score analysis
- Deploying the model as a web application using Flask or FastAPI

## License

This project is intended for educational and portfolio purposes.
