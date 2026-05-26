import spacy

nlp = spacy.load("en_core_web_sm")

text = "My name is Arsh and I know Python, Machine Learning and Web Development."

doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)