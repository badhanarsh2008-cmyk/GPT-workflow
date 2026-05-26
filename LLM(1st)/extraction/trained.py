import spacy

# Saved model load karein
nlp = spacy.load("./my_tech_model")

test_text = "I am Elon and I am an expert in Rocket Science"
doc = nlp(test_text)

print("Entities found:")
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")

# Test 2: Thoda alag pattern
test_text2 = "Meet Sarah, she is a specialist in Cyber Security"
doc2 = nlp(test_text2)

for ent in doc2.ents:
    print(f"{ent.text} -> {ent.label_}")