# Fake News Detection

A machine learning project that classifies news articles as **REAL** or **FAKE** using TF-IDF vectorization and a Passive Aggressive Classifier.

## Project Overview

This project implements a binary classification model to detect fake news articles. It uses natural language processing (NLP) techniques and machine learning to analyze news article content and predict whether it's genuine or fabricated.

## Dataset

- **Source**: `archive/data.csv`
- **Format**: CSV file containing news articles with labels
- **Labels**: 0 (REAL) and 1 (FAKE)
- **Data Storage**: Also available in HDF5 format (`archive/data.h5`)

## Methodology

### Approach
1. **Data Preprocessing**: Remove NaN values and label encoding
2. **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
3. **Train-Test Split**: 80-20 split with random state for reproducibility
4. **Classification**: Passive Aggressive Classifier with max iterations of 50
5. **Evaluation**: Accuracy score and confusion matrix analysis

### Key Components

```python
# Text Vectorization
TfidfVectorizer(stop_words='english', max_df=0.7)

# Classification Model
PassiveAggressiveClassifier(max_iter=50)
```

## Performance

The model achieves reliable accuracy metrics on the test set, evaluated using:
- **Accuracy Score**: Overall classification accuracy
- **Confusion Matrix**: True positives, true negatives, false positives, false negatives

## Project Structure

```
fake-news-detection/
├── FakeNews.ipynb              # Main Jupyter notebook with full analysis
├── archive/
│   ├── data.csv               # News articles dataset
│   └── data.h5                # HDF5 format of dataset
├── anaconda_projects/          # Conda environment configuration
│   └── db/
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn

## Installation

1. Clone the repository or download the project files
2. Set up a Python virtual environment (recommended)
3. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn jupyter
   ```

## Usage

### Running the Notebook

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook FakeNews.ipynb
   ```

2. Execute cells in sequence:
   - Data loading and exploration
   - Data preprocessing
   - TF-IDF vectorization
   - Model training
   - Prediction and evaluation

### Making Predictions

To use the trained model on new data:

```python
# After training the model
new_article = "Your news article text here"
new_vector = tfidf_vectorizer.transform([new_article])
prediction = pac.predict(new_vector)
print(prediction)  # Output: 'REAL' or 'FAKE'
```

## Model Details

### PassiveAggressiveClassifier
- **Type**: Linear classifier
- **Algorithm**: Passive-Aggressive learning
- **Advantages**: 
  - Effective for binary classification
  - Low memory footprint
  - Fast training and prediction

### TF-IDF Vectorization
- **stop_words**: English (removes common words like "the", "a", etc.)
- **max_df**: 0.7 (ignores terms appearing in more than 70% of documents)
- **Purpose**: Converts text to numerical features based on word importance

## Results

The confusion matrix shows the distribution of:
- **True Real**: Correctly classified real news
- **False Real**: Fake news misclassified as real (false negative)
- **False Fake**: Real news misclassified as fake (false positive)
- **True Fake**: Correctly classified fake news

## Future Improvements

- Experiment with other classifiers (SVM, Random Forest, Deep Learning)
- Hyperparameter tuning
- Cross-validation for more robust evaluation
- Advanced feature engineering
- Handling class imbalance
- Model persistence for production deployment

## Notes

- The dataset is stored in the `archive/` folder to keep it separate from code
- This is a basic implementation for educational purposes
- For production use, consider additional validation and error handling

## Author

Machine Learning Project

## License

Open source - feel free to use and modify for learning purposes.
