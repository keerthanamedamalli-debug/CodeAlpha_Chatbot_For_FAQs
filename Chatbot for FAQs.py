import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK tokenizer
nltk.download('punkt')

# FAQ Dataset
faq_questions = [
    "What is Artificial Intelligence?",
    "What is Machine Learning?",
    "What is Python?",
    "What is Deep Learning?",
    "What is NLP?"
]

faq_answers = [
    "Artificial Intelligence is the simulation of human intelligence in machines.",
    "Machine Learning is a subset of AI that enables computers to learn from data.",
    "Python is a popular programming language used for AI, ML, and web development.",
    "Deep Learning is a branch of Machine Learning based on neural networks.",
    "NLP stands for Natural Language Processing, which helps computers understand human language."
]

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq_questions)

print("===== FAQ Chatbot =====")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    # Convert user query to vector
    user_vector = vectorizer.transform([user_input])

    # Calculate similarity
    similarity_scores = cosine_similarity(user_vector, faq_vectors)

    # Find best match
    best_match_index = similarity_scores.argmax()

    # Response
    if similarity_scores[0][best_match_index] > 0.2:
        print("Chatbot:", faq_answers[best_match_index])
    else:
        print("Chatbot: Sorry, I don't understand your question.")