import random
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding, fix_random_seed
from sklearn.model_selection import train_test_split


# -----------------------------
# 1. Reproducibility
# -----------------------------
# Fixing seeds makes training more repeatable.
SEED = 42
random.seed(SEED)
fix_random_seed(SEED)


# -----------------------------
# 2. Source data
# -----------------------------
names = [
    "Aarav", "Ananya", "Abhinav", "Aditi", "Aditya", "Akansha", "Alok", "Amrita",
    "Aniket", "Anjali", "Ansh", "Anushka", "Arjun", "Arpita", "Aryan", "Avni",
    "Ayush", "Bhavna", "Chaitanya", "Chetan", "Daksh", "Deepika", "Dev", "Divya",
    "Eshan", "Ekta", "Faizan", "Farhan", "Gaurav", "Gayatri", "Gautam", "Geeta",
    "Hardik", "Harini", "Himanshu", "Hina", "Inder", "Ishani", "Ishaan", "Isha",
    "Jatin", "Jiya", "Kabir", "Kajal", "Karan", "Kavya", "Kunal", "Kiran",
    "Laksh", "Lata", "Manish", "Mehak", "Mayank", "Meera", "Mukul", "Mona",
    "Naman", "Nandini", "Naveen", "Neha", "Nikhil", "Nidhi", "Nitin", "Nisha",
    "Om", "Ojaswi", "Pankaj", "Pari", "Parth", "Payal", "Pranav", "Prerna",
    "Rahul", "Riya", "Rohan", "Rashmi", "Rohit", "Roshni", "Sahil", "Sakshi",
    "Sameer", "Sana", "Sanjay", "Sapna", "Sarthak", "Seema", "Shivam", "Shreya",
    "Siddharth", "Sneha", "Tushar", "Tanya", "Uday", "Upasana", "Varun", "Vanya",
    "Vikram", "Vidya", "Yash", "Zoya"
]

skills = [
    "Python", "JavaScript", "Java", "C++", "C#", "Rust", "Go", "Swift", "Kotlin",
    "TypeScript", "HTML5", "CSS3", "React.js", "Angular", "Vue.js", "Svelte",
    "Next.js", "Node.js", "Express.js", "Django", "Flask", "FastAPI",
    "Ruby on Rails", "PHP", "Laravel", "SQL", "PostgreSQL", "MongoDB", "Redis",
    "Cassandra", "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins",
    "GitHub Actions", "AWS", "Azure", "Google Cloud", "Firebase",
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-Learn",
    "Natural Language Processing", "Computer Vision", "Data Analysis", "Pandas",
    "NumPy", "Matplotlib", "Power BI", "Tableau", "Apache Spark", "Hadoop",
    "Kafka", "Data Engineering", "ETL Pipelines", "Big Query", "Snowflake",
    "Cybersecurity", "Ethical Hacking", "Penetration Testing", "Cryptography",
    "Network Security", "Wireshark", "Linux Administration", "Bash Scripting",
    "PowerShell", "Unity", "Unreal Engine", "Blender 3D", "UI Design",
    "UX Research", "Figma", "Adobe XD", "GraphQL", "REST API", "gRPC",
    "WebSockets", "Microservices", "Serverless Computing", "CI/CD",
    "Site Reliability Engineering", "Agile Methodology", "Scrum", "Git",
    "Bitbucket", "Jira", "Selenium", "Cypress", "Appium", "Flutter",
    "React Native", "SwiftUI", "Dart", "Solidity", "Blockchain",
    "Smart Contracts", "Web3"
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
    "Hey there, {name} this side. I'm a {skill} enthusiast.",
    "Name: {name} | Expertise: {skill}",
    "{name} - {skill} Specialist",
    "Featured Developer: {name} ({skill} expert)",
    "Top Skills: {skill}. Contact: {name}.",
    "The technical interview of {name} for {skill} went well.",
    "Although {name} knows many tools, {skill} remains the favorite.",
    "Professional profile: {name}, Expertise: {skill}.",
    # A few more realistic resume/profile styles
    "Resume snippet: {name} worked extensively with {skill}.",
    "{name}\nPrimary Skill: {skill}",
    "Candidate Name: {name}\nTechnical Skill: {skill}",
    "{name} has hands-on experience in {skill} and related tools."
]


# -----------------------------
# 3. Offset helper
# -----------------------------
def create_training_example(text, entities_info):
    """
    Convert entity text into spaCy offset format:
    (start_char, end_char, label)

    Why this is better than plain text.find():
    - easy to read
    - validates that entity text exists
    - keeps entity creation in one place
    """
    entities = []

    for value, label in entities_info:
        start = text.find(value)

        if start == -1:
            # If the entity text is missing, skip this example.
            # In production, you may want to log it instead.
            return None

        end = start + len(value)
        entities.append((start, end, label))

    return (text, {"entities": entities})


# -----------------------------
# 4. Synthetic data generation
# -----------------------------
def generate_dataset(num_examples=20000):
    dataset = []

    noise_tokens = ["|", "[", "_", "*", "-"]

    for _ in range(num_examples):
        name = random.choice(names)
        skill = random.choice(skills)
        template = random.choice(templates)

        text = template.format(name=name, skill=skill)

        # Add light noise sometimes so the model sees slightly messy text too.
        if random.random() < 0.30:
            noise = random.choice(noise_tokens)
            text = f"{noise} {text} {noise}"

        example = create_training_example(
            text,
            [
                (name, "NAME"),
                (skill, "SKILL"),
            ],
        )

        if example is not None:
            dataset.append(example)

    return dataset


# -----------------------------
# 5. Build train / validation split
# -----------------------------
data = generate_dataset(num_examples=20000)

train_data, val_data = train_test_split(
    data,
    test_size=0.2,
    random_state=SEED,
    shuffle=True
)

print(f"Total examples: {len(data)}")
print(f"Train examples: {len(train_data)}")
print(f"Validation examples: {len(val_data)}")


# -----------------------------
# 6. Build spaCy pipeline
# -----------------------------
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

# Register labels before training.
ner.add_label("NAME")
ner.add_label("SKILL")

optimizer = nlp.begin_training()


# -----------------------------
# 7. Simple evaluation helper
# -----------------------------
def evaluate_model(nlp, dataset):
    """
    Measures exact-match entity accuracy at example level.

    This is not as rich as precision/recall/F1,
    but it's easy to understand while learning.
    """
    correct = 0
    total = 0

    for text, annotations in dataset:
        doc = nlp(text)

        predicted = {(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents}
        expected = set(annotations["entities"])

        if predicted == expected:
            correct += 1

        total += 1

    return correct / total if total > 0 else 0.0


# -----------------------------
# 8. Training loop
# -----------------------------
epochs = 20

for epoch in range(epochs):
    random.shuffle(train_data)
    losses = {}

    batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))

    for batch in batches:
        examples = []

        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)

        nlp.update(examples, drop=0.2, losses=losses)

    val_acc = evaluate_model(nlp, val_data)

    print(
        f"Epoch {epoch + 1:02d} | "
        f"NER Loss: {losses.get('ner', 0.0):.4f} | "
        f"Validation Exact Match: {val_acc:.4f}"
    )


# -----------------------------
# 9. Save model
# -----------------------------
output_dir = "./my_tech_model_best"
nlp.to_disk(output_dir)
print(f"Model saved to: {output_dir}")


# -----------------------------
# 10. Quick test
# -----------------------------
test_texts = [
    "Rahul is skilled in Python and Django.",
    "Candidate Name: Ananya | Technical Skill: React.js",
    "Meet Zoya, our Web3 expert.",
]

print("\nSample predictions:")
for text in test_texts:
    doc = nlp(text)
    print(f"\nText: {text}")
    for ent in doc.ents:
        print(f"  {ent.text} -> {ent.label_}")
