from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

train_texts = [
    "hello",
    "hi",
    "good morning",
    "bye",
    "goodbye",
    "see you",
    "what is your name",
    "who are you",
    "tell me your name",
]

train_labels = [
    "greeting",
    "greeting",
    "greeting",
    "goodbye",
    "goodbye",
    "goodbye",
    "ask_name",
    "ask_name",
    "ask_name",
]

vectorizer=CountVectorizer()

x_train=vectorizer.fit_transform(train_texts)
model=LogisticRegression()
model.fit(x_train,train_labels)

word=input("Enter word:")
xword=vectorizer.transform([word])
prediction=model.predict(xword)[0]

print("Prediction:",prediction)