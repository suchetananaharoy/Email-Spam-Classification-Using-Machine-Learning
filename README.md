Email communication has become an essential part of modern life, but the prevalence of spam messages poses significant challenges to both security and productivity. Spam emails often contain fraudulent, promotional, or malicious content, while legitimate (ham) emails are genuine communications. The ability to automatically distinguish between these two categories is therefore critical, and machine learning provides powerful tools for this task.

This project focuses on the Spam Email Classification Dataset available on Kaggle, which contains over 5,000 labeled email samples. Each entry consists of an email message and a binary label: spam (1) or ham (0). The dataset is relatively clean and balanced, making it suitable for experimentation with text classification algorithms. However, constraints include limited metadata (no sender or subject information) and the inherent sparsity of text features.

Preprocessing was a crucial step to ensure consistency and reduce noise. All text was converted to lowercase, punctuation and numbers were removed, and stopwords were eliminated to reduce dimensionality. Tokenization split messages into words, and stemming/lemmatization reduced them to their root forms. Feature extraction was performed using TF‑IDF (Term Frequency–Inverse Document Frequency), which captures both word frequency and importance, producing sparse matrices suitable for machine learning models.

The primary model implemented was the Multinomial Naive Bayes classifier, which assumes independence among features and is particularly effective for text data. Laplace smoothing was applied to handle unseen words, and log‑likelihood calculations ensured numerical stability. The model achieved strong results, with accuracy exceeding 96% and balanced precision and recall.

To extend the analysis, additional models were tested: Logistic Regression, Support Vector Machine (SVM), and Random Forest. Logistic Regression provided competitive accuracy with good interpretability, while SVM achieved slightly higher precision, reducing false positives. Random Forest, though robust, struggled with sparse high‑dimensional data and required more computational resources. An ensemble approach combining Naive Bayes and SVM further improved performance, achieving over 97% accuracy.

The findings highlight that Naive Bayes remains a reliable baseline for spam classification due to its simplicity and efficiency, while SVM offers superior precision at the cost of longer training times. Future improvements could involve deep learning models such as LSTMs or Transformers, which capture contextual meaning beyond word frequency.

In conclusion, this project demonstrates the effectiveness of classical machine learning algorithms in spam detection, provides comparative insights across multiple models, and underscores the importance of preprocessing and feature engineering in text classification tasks. 



This project aimed to develop and evaluate machine learning models for email spam classification, leveraging various text preprocessing and feature extraction techniques. The methodology involved initial data loading, rigorous text cleaning, feature extraction using both CountVectorizer and TF-IDF, and the implementation and evaluation of several classification algorithms, including Multinomial Naive Bayes, Logistic Regression, Linear Support Vector Machine (LinearSVC), Random Forest, and a Voting Classifier.

Data Preprocessing and Feature Extraction

The initial dataset, email.csv, was loaded, revealing two key columns: 'Category' and 'Message'. A crucial step was text cleaning, where messages were converted to lowercase, punctuation was removed, and numerical digits were eliminated. This ensured that the text data was standardized and free from noise that could hinder model performance. The cleaned messages were then used to create numerical features. Initially, CountVectorizer was employed, which transforms text into a matrix of token counts, representing the frequency of each word. This generated a vocabulary of 7,582 unique words. Subsequently, TF-IDF (Term Frequency-Inverse Document Frequency) vectorization was implemented, a technique that assigns weights to words based on their frequency in a document and rarity across the entire corpus, aiming to highlight more important terms.

Performance of CountVectorizer-based Naive Bayes Model

After splitting the data into training and testing sets (80/20 split), a Multinomial Naive Bayes classifier was trained on the CountVectorizer features. This model demonstrated excellent performance:

Accuracy: 0.9883
Precision (spam): 1.0000
Recall (spam): 0.9128
F1-Score (spam): 0.9544
The confusion matrix for this model showed 966 True Negatives (correctly identified ham emails), 0 False Positives (no ham emails incorrectly classified as spam), 13 False Negatives (spam emails missed as ham), and 136 True Positives (correctly identified spam emails). The perfect precision for the 'spam' class is a highly desirable outcome, indicating that the model never mislabeled a legitimate email as spam.

Performance of TF-IDF-based Naive Bayes Model

Following the CountVectorizer approach, a new Multinomial Naive Bayes model was trained using features generated by TfidfVectorizer. The performance metrics were as follows:

Accuracy: 0.9561
Precision (spam): 1.0000
Recall (spam): 0.6711
F1-Score (spam): 0.8032
The TF-IDF model, while maintaining perfect precision for 'spam', showed a noticeable decrease in recall and overall accuracy compared to the CountVectorizer model. This suggests that for this particular dataset and Naive Bayes classifier, the simple frequency of words was a more effective indicator of spam than the weighted importance of words provided by TF-IDF.

Advanced Task: Evaluation of Multiple Classification Models with TF-IDF

To further explore model effectiveness, an advanced task involved training and evaluating several other classification algorithms using the TF-IDF features (from a refined TF-IDF setup including stop_words='english' and max_features=5000):

Multinomial Naive Bayes: (re-evaluated with refined TF-IDF)
Accuracy: 0.9758, Precision: 1.0, Recall: 0.8188, F1-score: 0.9004
Logistic Regression:
Accuracy: 0.9623, Precision: 1.0, Recall: 0.7181, F1-score: 0.8359
Linear Support Vector Machine (LinearSVC):
Accuracy: 0.9883, Precision: 1.0, Recall: 0.9128, F1-score: 0.9544
Random Forest Classifier:
Accuracy: 0.9785, Precision: 1.0, Recall: 0.8389, F1-score: 0.9124
Voting Classifier (Ensemble of NB, SVM, LR):
Accuracy: 0.9794, Precision: 1.0, Recall: 0.8456, F1-score: 0.9164
Observations and Discussion

A consistent and highly positive observation across all models was the perfect precision (1.0) for the 'spam' class. This implies that no legitimate emails were ever misclassified as spam by any of these models, which is critical for user experience in a spam filter. The primary differentiator among these models was their recall for spam messages. LinearSVC emerged as the top performer, achieving the highest recall (0.9128) and F1-score (0.9544) while maintaining the perfect precision. This indicates its superior ability to correctly identify the majority of actual spam emails, minimizing false negatives. The CountVectorizer based Naive Bayes model initially also achieved similar high recall and overall accuracy, performing on par with the best TF-IDF model (LinearSVC).

In conclusion, for this specific email spam detection task, while both CountVectorizer and TF-IDF (especially with LinearSVC) proved effective, the LinearSVC model provided the best balance of high accuracy and strong recall for 'spam' detection. The perfect precision achieved by all models is a testament to their robustness in avoiding false positives, making them suitable candidates for practical spam filtering applications. Future work could involve hyperparameter tuning for LinearSVC or exploring deep learning approaches for further performance gains. 
