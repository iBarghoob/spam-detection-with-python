from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)
global Classifier 
global Vectorizer

# v1 = label, v2 = message
data = pd.read_csv('spam.csv', encoding='latin-1')

# 80/20 train/test split
train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['v1'])

Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
Vectorizer = TfidfVectorizer()

# transform text to numerical vectors for classifier and train SVM on vectorized data + labels
vectorize_text = Vectorizer.fit_transform(train_data.v2)
Classifier.fit(vectorize_text, train_data.v1)

@app.route('/', methods=['GET', 'POST'])
def index():
  message = ''
  prediction = ''
  confidence = ''
  error = ''

  if request.method == 'POST':
    message = request.form.get('message', '')
    try:
      if message:
        vectorize_message = Vectorizer.transform([message])
        prediction = Classifier.predict(vectorize_message)[0]
        confidence = max(Classifier.predict_proba(vectorize_message)[0])
    except Exception as e:
      error = str(e)

  return render_template('index.html', message=message, prediction=prediction, confidence=confidence, error=error)

if __name__ == '__main__':
  app.run(debug=True)

