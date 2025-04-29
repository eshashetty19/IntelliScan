from flask import Flask, request, render_template
import re
import math
import string
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.corpus import stopwords
import os  # <-- Add this import to fix the port binding

app = Flask(__name__)

import nltk
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load GPT-2 model and tokenizer (one-time loading)
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define thresholds for AI and plagiarism detection
AI_GENERATED_THRESHOLD = 45   # AI text detection threshold
AI_DETECTION_THRESHOLD = 10   # Soft AI threshold for partial detection
PLAGIARISM_THRESHOLD = 15     # Plagiarism detection threshold

def calculate_perplexity(text):
    """Calculate perplexity of a given text using GPT-2."""
    if not text or len(text.split()) < 10:
        return 100  # High perplexity for short texts (assumed human)

    # Tokenize text
    tokens = gpt2_tokenizer.encode(text, return_tensors="pt")

    with torch.no_grad():
        outputs = gpt2_model(tokens, labels=tokens)
        loss = outputs.loss
        perplexity = torch.exp(loss)  # Compute perplexity

    return round(perplexity.item(), 2)


def calculate_ai_percentage(text):
    """Improved AI detection using perplexity, word variety, and sentence patterns."""
    if not text:
        return 0.0  # Avoid division by zero

    # Normalize text (lowercase & remove punctuation)
    clean_text = re.sub(rf"[{re.escape(string.punctuation)}0-9]", "", text.lower())
    words = clean_text.split()
    sentences = text.split(".")  # Rough sentence count

    if len(words) < 10:
        return 0.0  # Ignore very short texts

    # 1️⃣ Unique Word Ratio (AI text often has a lower ratio)
    unique_words = set(words)
    unique_ratio = len(unique_words) / len(words)

    # 2️⃣ Average Word Length (AI text has longer words)
    avg_word_length = sum(len(word) for word in words) / len(words)

    # 3️⃣ Sentence Repetition Score (AI text often follows patterns)
    sentence_repetition = len(sentences) / (len(words) / 10 + 1)  # Normalize for text length

    # 4️⃣ Common AI Words Indicator
    ai_indicative_words = {"thus", "therefore", "moreover", "additionally", "furthermore"}
    ai_word_usage = sum(1 for word in words if word in ai_indicative_words) / len(words)

    # 5️⃣ Perplexity Score (Lower = More Likely AI)
    perplexity = calculate_perplexity(text)
    perplexity_score = (50 - perplexity) * 2  # Scale it to fit AI probability range

    # 🏆 Final AI Probability Calculation
    ai_score = ((1 - unique_ratio) * 30) + ((avg_word_length - 4) * 20) + (sentence_repetition * 20) + (ai_word_usage * 10) + perplexity_score
    ai_percentage = min(100, max(0, ai_score))  # Keep percentage in 0-100 range

    return round(ai_percentage, 2)

@app.route("/")
def load_page():
    return render_template("index.html", query="", plagiarized_texts="[]")

@app.route("/", methods=["POST"])
def detect_plagiarism_and_ai_text():
    try:
        input_query = request.form.get("query", "").strip()
        if not input_query:
            return "Error: No input query provided."

        lowercase_query = input_query.lower()
        query_word_list = re.sub(r"[^\w\s]", " ", lowercase_query).split()

        # Read external database file safely
        try:
            with open("database1.txt", "r", encoding="utf-8") as file:
                database_text = file.read().lower()
        except FileNotFoundError:
            return "Error: database1.txt not found."

        database_word_list = re.sub(r"[^\w\s]", " ", database_text).split()

        # ✅ Apply Stopword Removal Here
        query_word_list = [word for word in query_word_list if word.lower() not in stop_words]
        database_word_list = [word for word in database_word_list if word.lower() not in stop_words]

        universal_word_set = list(set(query_word_list) | set(database_word_list))

        query_tf = [query_word_list.count(word) for word in universal_word_set]
        database_tf = [database_word_list.count(word) for word in universal_word_set]

        # Cosine Similarity Calculation for Plagiarism Detection
        dot_product = sum(q * d for q, d in zip(query_tf, database_tf))
        query_vector_magnitude = math.sqrt(sum(tf ** 2 for tf in query_tf))
        database_vector_magnitude = math.sqrt(sum(tf ** 2 for tf in database_tf))

        match_percentage = (
            (dot_product / (query_vector_magnitude * database_vector_magnitude) * 100)
            if query_vector_magnitude > 0 and database_vector_magnitude > 0
            else 0
        )

        plagiarized_texts = list(set(query_word_list) & set(database_word_list)) if query_word_list and database_word_list else []
        plagiarism_status = "Plagiarism Detected" if match_percentage >= PLAGIARISM_THRESHOLD else "Limited Plagiarism" if match_percentage > 0 else "No Plagiarism"

        ai_percentage = calculate_ai_percentage(input_query)
        is_ai_text = ai_percentage >= AI_GENERATED_THRESHOLD
        ai_text_detected = ai_percentage > AI_DETECTION_THRESHOLD

        return render_template(
            "index.html",
            query=input_query,
            percentage=match_percentage,
            plagiarized_texts=json.dumps(plagiarized_texts),
            ai_text=input_query,
            ai_percentage=ai_percentage,
            is_ai_text=is_ai_text,
            ai_text_detected=ai_text_detected,
            plagiarism_status=plagiarism_status,
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
