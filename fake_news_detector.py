#Importing necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset into our program
data=pd.read_csv("fake_news_dataset.csv")
data=data[["text"],["label"]]
data=data.dropna()

x=data['text']
y=data['label']

#TF-IDF vectorising
vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)
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
  text_vec=vectorizer.transform([text])
  prediction=model.predict(text_vec)
  return prediction[0]

# User input 
print("\nEnter your own news text to test the model: ")
while True:
  user_input=input("\nEnter news text (or type 'exit'): ")
  if user_input.lower() == 'exit':
    print("Exiting...")
    break
  
  result=predict_news(user_input)
  print("Prediction: ", result)
  


