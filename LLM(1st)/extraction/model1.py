import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import random



train_texts=[]


names = [

    "Aarav", "Ananya", "Abhinav", "Aditi", "Aditya", "Akansha", "Alok", "Amrita", "Aniket", "Anjali",
 
    "Ansh", "Anushka", "Arjun", "Arpita", "Aryan", "Avni", "Ayush", "Bhavna", "Chaitanya", "Chetan",

    "Daksh", "Deepika", "Dev", "Divya", "Eshan", "Ekta", "Faizan", "Farhan", "Gaurav", "Gayatri",

    "Gautam", "Geeta", "Hardik", "Harini", "Himanshu", "Hina", "Inder", "Ishani", "Ishaan", "Isha",

    "Jatin", "Jiya", "Kabir", "Kajal", "Karan", "Kavya", "Kunal", "Kiran", "Laksh", "Lata",

    "Manish", "Mehak", "Mayank", "Meera", "Mukul", "Mona", "Naman", "Nandini", "Naveen", "Neha",

    "Nikhil", "Nidhi", "Nitin", "Nisha", "Om", "Ojaswi", "Pankaj", "Pari", "Parth", "Payal",

    "Pranav", "Prerna", "Rahul", "Riya", "Rohan", "Rashmi", "Rohit", "Roshni", "Sahil", "Sakshi",

    "Sameer", "Sana", "Sanjay", "Sapna", "Sarthak", "Seema", "Shivam", "Shreya", "Siddharth", "Sneha",

    "Tushar", "Tanya", "Uday", "Upasana", "Varun", "Vanya", "Vikram", "Vidya", "Yash", "Zoya"

]



skills = [

    "Python", "JavaScript", "Java", "C++", "C#", "Rust", "Go", "Swift", "Kotlin", "TypeScript",

    "HTML5", "CSS3", "React.js", "Angular", "Vue.js", "Svelte", "Next.js", "Node.js", "Express.js", "Django",

    "Flask", "FastAPI", "Ruby on Rails", "PHP", "Laravel", "SQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra",

    "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "GitHub Actions", "AWS", "Azure", "Google Cloud", "Firebase",

    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-Learn", "Natural Language Processing", "Computer Vision", "Data Analysis", "Pandas", "NumPy",

    "Matplotlib", "Power BI", "Tableau", "Apache Spark", "Hadoop", "Kafka", "Data Engineering", "ETL Pipelines", "Big Query", "Snowflake",

    "Cybersecurity", "Ethical Hacking", "Penetration Testing", "Cryptography", "Network Security", "Wireshark", "Linux Administration", "Bash Scripting", "PowerShell", "Unity",

    "Unreal Engine", "Blender 3D", "UI Design", "UX Research", "Figma", "Adobe XD", "GraphQL", "REST API", "gRPC", "WebSockets",

    "Microservices", "Serverless Computing", "CI/CD", "Site Reliability Engineering", "Agile Methodology", "Scrum", "Git", "Bitbucket", "Jira", "Selenium",

    "Cypress", "Appium", "Flutter", "React Native", "SwiftUI", "Dart", "Solidity", "Blockchain", "Smart Contracts", "Web3"

]

templates = [
    "This is {name}, a certified {skill} professional.",
    "Profile of {name}, specializing in {skill} development.",
    "Introducing {name}, whose expertise lies in {skill}.",
    "{name} here, with a strong background in {skill}.",
    "The candidate {name} is proficient in {skill}.",
    "{name} has been working on {skill} projects for years.",
    "Building scalable solutions in {skill} is what {name} does best.",
    "Currently, {name} is mastering advanced {skill} techniques.",
    "{name} recently completed a major project using {skill}.",
    "Expert level knowledge of {skill} is demonstrated by {name}.",
    "Hi! I'm {name} and I love coding in {skill}.",
    "Meet {name}, our go-to person for anything related to {skill}.",
    "Yo, it's {name}. I'm all about {skill} these days.",
    "Just a dev named {name} who is obsessed with {skill}.",
    "Hey there, {name} this side. I'm a {skill} enthusiast.",
    "Name: {name} | Expertise: {skill}",
    "{name} - {skill} Specialist",
    "Featured Developer: {name} ({skill} expert)",
    "Top Skills: {skill}. Contact: {name}.",
    "I highly recommend {name} for his {skill} skills.",
    "Have you seen {name}'s work with {skill}? It's amazing.",
    "{name} is the one you need for {skill} optimization.",
    "The technical interview of {name} for {skill} went well.",
    "Although {name} knows many tools, {skill} remains the favorite.",
    "In the world of {skill}, {name} is a well-known name.",
    "Primary focus of {name}'s career has been {skill}.",
    "Is {name} the best at {skill} in the team?",
    "Whether it is {skill} or backend, {name} handles it all.",
    "Check out {name}'s portfolio in {skill} today.",
    "Professional profile: {name}, Expertise: {skill}."
]

def fix_entities(text, raw_entities):
    entities = []
    for word, label in raw_entities:
        start = text.find(word)
        if start != -1:
            entities.append((start, start + len(word), label.upper())) # Label upper case
    return (text, {"entities": entities})



train_data = []

for _ in range(20000):
    name = random.choice(names)
    skill = random.choice(skills)
    template = random.choice(templates)
    
    # Sentence taiyar karein
    text = template.format(name=name, skill=skill)
    if random.random() > 0.7:
        noise = random.choice(["|", "[", "—", "_", "*"])
        text = f"{noise} {text} {noise}"
    # Entities fix karke list mein daalein
    train_data.append(
        fix_entities(text, [(name, "NAME"), (skill, "SKILL")])
    )
print(f"Total training examples generated: {len(train_data)}")
# Model Setup
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")
ner.add_label("NAME")
ner.add_label("SKILL")

# 3. Training with Minibatch (FAST)
optimizer = nlp.begin_training()

print("Training started on CPU...")
for epoch in range(20):
    random.shuffle(train_data)
    losses = {}
    

    batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
    
    for batch in batches:
        examples = []
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            examples.append(Example.from_dict(doc, annotations))
        
        nlp.update(examples, drop=0.2, losses=losses)
    
    print(f"Epoch {epoch+1} - Loss: {losses['ner']:.4f}")

# 4. Save
nlp.to_disk("./my_tech_model")
print("Model Saved!")