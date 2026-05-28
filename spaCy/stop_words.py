import spacy


nlp = spacy.load("en_core_web_sm")

s1 = "My name is arshdeep singh"

stop__words = nlp.Defaults.stop_words
print(len(stop__words))
#it checks whether the text is stopword or not 
nlp.vocab['the'].is_stop
nlp.vocab['good'].is_stop

doc = nlp(s1)

#stopwords expelled
tokens_from_sentence_without_stopwords = [token.text for token in doc if token.text.lower() in nlp.Defaults.stop_words]

print(tokens_from_sentence_without_stopwords)

#if you want to add or remove any new word from stopword list ( providedd by spacy sm)
nlp.Defaults.stop_words.add('example')

nlp.Defaults.stop_words.remove('example')