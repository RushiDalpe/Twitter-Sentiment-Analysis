This project focuses on performing sentiment analysis on a Twitter dataset consisting of 1.6 million tweets. The goal is to classify tweets into positive or negative sentiment using various machine learning models, including:

Logistic Regression
Naive Bayes
Random Forest
DistilBERT (Transformers)
Each model is trained, tested, and evaluated to determine the most accurate and efficient algorithm for sentiment analysis.

Dataset
The dataset consists of 1.6 million tweets, with sentiment labels:

0 for negative sentiment
1 for positive sentiment
Technologies and Libraries Used
Languages: Python
Libraries:
pandas for data manipulation
scikit-learn for machine learning algorithms
transformers for DistilBERT
nltk for natural language processing
matplotlib and seaborn for data visualization
wordcloud for text visualization
torch for working with PyTorch in DistilBERT
Setup Instructions
Clone the Repository:

bash
git clone https://github.com/RushiDalpe/Twitter-Sentiment-Analysis/new/main?filename=README.md
Install Dependencies: Install all required Python packages using pip:

bash
pip install -r requirements.txt
Running the Models
1. Logistic Regression
Description: This model uses TfidfVectorizer for feature extraction and Logistic Regression to classify sentiments.

Training Size: 100,000 tweets

Key Files: logistic_regression.py
bash
python logistic_regression.py
2. Naive Bayes
Description: This model applies Multinomial Naive Bayes with TF-IDF feature extraction to classify the sentiment of tweets.

Training Size: 100,000 tweets

Key Files: naive_bayes.py

bash
python naive_bayes.py
3. Random Forest
Description: Uses Random Forest with TfidfVectorizer to predict tweet sentiments.

Training Size: 200,000 tweets

Key Files: random_forest.py

bash
python random_forest.py
4. DistilBERT (Transformer Model)
Description: Applies DistilBERT from Hugging Face's transformers library for sequence classification.

Training Size: 50,000 tweets (due to memory and computational constraints)

Key Files: distilbert.py

bash
python distilbert.py
Model Evaluation
After running the models, the results are evaluated based on:

Accuracy: How well the model predicts the correct sentiment.
Precision, Recall, and F1-Score: To measure the balance between sensitivity and specificity.
Confusion Matrix: A matrix depicting the correct vs. incorrect classifications.
ROC Curve: A graphical representation of a model’s performance.
Example of Model Output:
plaintext
Copy code
Accuracy: 85.67%
Precision: 0.86
Recall: 0.85
F1 Score: 0.85
Visualizations:
Confusion Matrix: Displays correct vs. incorrect classifications.
ROC Curve: Shows the model’s trade-off between true positive and false positive rates.
Word Clouds: Displays the most frequent words in positive and negative tweets.
Results Summary
Logistic Regression: Provided solid baseline accuracy and fast training time.
Naive Bayes: Efficient with text data but slightly underperformed compared to other models.
Random Forest: Achieved high accuracy but required more computational resources.
DistilBERT: Outperformed the other models in accuracy but required the most computational power.
Future Work
Fine-tune the DistilBERT model for better performance.
Explore additional pre-processing steps for improving classification accuracy.
Scale the project using distributed computing for larger datasets.
Conclusion
This project demonstrates the effectiveness of various machine learning and deep learning techniques in sentiment analysis. Each model has its strengths and trade-offs, and the choice of model will depend on the accuracy and resource constraints for future applications.
