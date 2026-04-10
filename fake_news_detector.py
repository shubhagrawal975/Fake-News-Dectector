#Importing necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import re

# Loading the dataset into our program
data = pd.read_csv("fake_news_large_dataset.csv")
data=data[["text"],["label"]]
data=data.dropna()
data['label'] = data['label'].map({'fake': 0, 'real': 1})


x=data['text']
y=data['label']

#Removing unwanted texts
model = MultinomialNB()
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text
data['text'] = data['text'].apply(clean_text)

#TF-IDF vectorising
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2)
)
X_vectorized=vectorizer.fit_transform(x)

#Train-test split
x_train,x_test,y_train,y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state = 42)

#Model training
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

#Evaluation Model
y_pred = model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)

print("\nModel Training Complete")
print("Accuracy: ",round(accuracy*100,2), "%")

#Prediction Function
def predict_news(text):
    text_vec = vectorizer.transform([text])
    probs = model.predict_proba(text_vec)[0]
    
    fake_prob = probs[0]
    real_prob = probs[1]

    if real_prob > fake_prob:
        return "REAL", real_prob
    else:
        return "FAKE", fake_prob
      
# User input 
print("\nEnter your own news text to test the model: ")
while True:
  user_input=input("\nEnter news text (or type 'exit'): ")
  if user_input.lower() == 'exit':
    print("Exiting...")
    break
  
  label, confidence = predict_news(user_input)
  if confidence < 0.6:
      print("Prediction: UNCERTAIN")
  else:
      print("Prediction:", label)
      print("Confidence:", round(confidence * 100, 2), "%")

