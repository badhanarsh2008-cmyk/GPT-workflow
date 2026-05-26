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


vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)


model = LogisticRegression()
model.fit(X_train, train_labels)


test_sentence = input("Enter word :")
X_test = vectorizer.transform([test_sentence])
prediction = model.predict(X_test)[0]

print("Input:", test_sentence)
print("Predicted intent:", prediction)