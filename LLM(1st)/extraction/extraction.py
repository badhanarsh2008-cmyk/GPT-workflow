import spacy
from spacy.training.example import Example
import random


TRAIN_DATA = [
("My name is Arsh and I know Python", {"entities": [(11, 15, "NAME"), (27, 33, "SKILL")]}),
("My name is Rahul and I know Java", {"entities": [(11, 16, "NAME"), (27, 31, "SKILL")]}),
("My name is Priya and I know Machine Learning", {"entities": [(11, 16, "NAME"), (27, 43, "SKILL")]}),
("My name is Aman and I know Deep Learning", {"entities": [(11, 15, "NAME"), (27, 40, "SKILL")]}),
("My name is Neha and I know C++", {"entities": [(11, 15, "NAME"), (27, 30, "SKILL")]}),

("Arsh knows Python and Machine Learning", {"entities": [(0, 4, "NAME"), (11, 17, "SKILL"), (22, 38, "SKILL")]}),
("Rahul knows Java and Spring Boot", {"entities": [(0, 5, "NAME"), (11, 15, "SKILL"), (20, 31, "SKILL")]}),
("Priya knows Data Science and Python", {"entities": [(0, 5, "NAME"), (11, 23, "SKILL"), (28, 34, "SKILL")]}),
("Aman knows Deep Learning and NLP", {"entities": [(0, 4, "NAME"), (11, 24, "SKILL"), (29, 32, "SKILL")]}),
("Neha knows SQL and Database Management", {"entities": [(0, 4, "NAME"), (11, 14, "SKILL"), (19, 38, "SKILL")]}),

("I am Arsh skilled in Python and AI", {"entities": [(5, 9, "NAME"), (20, 26, "SKILL"), (31, 33, "SKILL")]}),
("I am Rahul skilled in Java and Backend Development", {"entities": [(5, 10, "NAME"), (20, 24, "SKILL"), (29, 48, "SKILL")]}),
("I am Priya skilled in Data Analysis and Pandas", {"entities": [(5, 10, "NAME"), (20, 33, "SKILL"), (38, 44, "SKILL")]}),
("I am Aman skilled in Deep Learning and Computer Vision", {"entities": [(5, 9, "NAME"), (20, 33, "SKILL"), (38, 54, "SKILL")]}),
("I am Neha skilled in SQL and Data Warehousing", {"entities": [(5, 9, "NAME"), (20, 23, "SKILL"), (28, 45, "SKILL")]}),

("Arsh is experienced in Python, ML and AI", {"entities": [(0, 4, "NAME"), (23, 29, "SKILL"), (31, 33, "SKILL"), (38, 40, "SKILL")]}),
("Rahul is experienced in Java, Spring and Microservices", {"entities": [(0, 5, "NAME"), (23, 27, "SKILL"), (29, 35, "SKILL"), (40, 53, "SKILL")]}),
("Priya is experienced in Data Science, Pandas and NumPy", {"entities": [(0, 5, "NAME"), (23, 35, "SKILL"), (37, 43, "SKILL"), (48, 53, "SKILL")]}),

("Arsh has skills in Python and Deep Learning", {"entities": [(0, 4, "NAME"), (20, 26, "SKILL"), (31, 44, "SKILL")]}),
("Rahul has skills in Java and Backend Systems", {"entities": [(0, 5, "NAME"), (20, 24, "SKILL"), (29, 44, "SKILL")]}),
("Priya has skills in Data Analysis and Visualization", {"entities": [(0, 5, "NAME"), (20, 33, "SKILL"), (38, 51, "SKILL")]}),

("Arsh works with Python and Machine Learning", {"entities": [(0, 4, "NAME"), (17, 23, "SKILL"), (28, 44, "SKILL")]}),
("Rahul works with Java and Spring Boot", {"entities": [(0, 5, "NAME"), (17, 21, "SKILL"), (26, 37, "SKILL")]}),
("Priya works with Data Science and AI", {"entities": [(0, 5, "NAME"), (17, 29, "SKILL"), (34, 36, "SKILL")]}),

("Myself Arsh, skilled in Python and NLP", {"entities": [(7, 11, "NAME"), (24, 30, "SKILL"), (35, 38, "SKILL")]}),
("Myself Rahul, skilled in Java and SQL", {"entities": [(7, 12, "NAME"), (24, 28, "SKILL"), (33, 36, "SKILL")]}),
("Myself Priya, skilled in Data Science and ML", {"entities": [(7, 12, "NAME"), (24, 36, "SKILL"), (41, 43, "SKILL")]}),

("Arsh is a Python developer", {"entities": [(0, 4, "NAME"), (10, 16, "SKILL")]}),
("Rahul is a Java developer", {"entities": [(0, 5, "NAME"), (10, 14, "SKILL")]}),
("Priya is a Data Scientist skilled in Python", {"entities": [(0, 5, "NAME"), (10, 24, "SKILL"), (36, 42, "SKILL")]}),

("Aman builds models using Deep Learning", {"entities": [(0, 4, "NAME"), (27, 40, "SKILL")]}),
("Neha builds dashboards using Power BI", {"entities": [(0, 4, "NAME"), (30, 38, "SKILL")]}),

("Arsh is proficient in Python and AI", {"entities": [(0, 4, "NAME"), (20, 26, "SKILL"), (31, 33, "SKILL")]}),
("Rahul is proficient in Java and Microservices", {"entities": [(0, 5, "NAME"), (20, 24, "SKILL"), (29, 42, "SKILL")]}),
("Priya is proficient in Data Science and Visualization", {"entities": [(0, 5, "NAME"), (20, 32, "SKILL"), (37, 50, "SKILL")]}),

("Arsh specializes in Python and Machine Learning", {"entities": [(0, 4, "NAME"), (22, 28, "SKILL"), (33, 49, "SKILL")]}),
("Rahul specializes in Java and Backend Development", {"entities": [(0, 5, "NAME"), (22, 26, "SKILL"), (31, 50, "SKILL")]}),
("Priya specializes in Data Analysis and Pandas", {"entities": [(0, 5, "NAME"), (22, 35, "SKILL"), (40, 46, "SKILL")]}),
]

nlp = spacy.blank("en")  # empty model
ner = nlp.add_pipe("ner")

ner.add_label("NAME")
ner.add_label("SKILL")

optimizer = nlp.begin_training()

for epoch in range(20):
    random.shuffle(TRAIN_DATA)
    
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], losses={})
        
doc = nlp("My name is Arsh and I know Deep Learning")

for ent in doc.ents:
    print(ent.text, ent.label_)