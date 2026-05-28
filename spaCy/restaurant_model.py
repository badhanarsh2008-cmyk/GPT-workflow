import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import joblib
from sklearn.metrics import accuracy_score

data = pd.read_csv("Restaurant_Reviews.tsv", sep="\t")

corpus = []
ps = PorterStemmer()
all_stopwords = stopwords.words("english")
all_stopwords.remove('not')

for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]',' ',data['Review'][i])
    
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    
    review = " ".join(review)
    corpus.append(review)

print("Cleaning done!!")

cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = data.iloc[:, -1].values #last column 0 or 1

model = LogisticRegression()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(f"model accuracy : {accuracy_score(y_test,y_pred) * 100}%")

joblib.dump(model,"restaurant_t.joblib")
joblib.dump(cv,"vectorizer.joblib")
print("Model saved!!!")
print("Now check by yourselve ---------!")

new_review = input("Enter your Review : ")
new_corpus = [ps.stem(word) for word in new_review.lower().split() if not word in set(all_stopwords)]
new_review_transformed = cv.transform([' '.join(new_corpus)]).toarray()

prediction = model.predict(new_review_transformed)
print("positive" if prediction == 1 else "negative")
