# spam_detection_interactive.py
# SMS Spam Detection with user input

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Step 1: Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.rename(columns={'v1':'label', 'v2':'text'}, inplace=True)

# Step 2: Text preprocessing
nltk.download("stopwords")
stop = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# Step 3: TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['clean_text'])
y = df['label']

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Step 6: Show model performance
y_pred = nb.predict(X_test)
print("\nNaive Bayes Performance:")
print(classification_report(y_test, y_pred))

# Step 7: Function to predict messages
def predict_message(msg):
    msg_clean = clean_text(msg)
    msg_vec = tfidf.transform([msg_clean])
    return nb.predict(msg_vec)[0]

# Step 8: Interactive loop for user input
print("\n--- SMS Spam Detector ---")
print("Type a message and press Enter to check if it's spam or ham.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Your message: ")
    if user_input.lower() == "exit":
        print("Exiting SMS Spam Detector. Goodbye!")
        break
    prediction = predict_message(user_input)
    print("Prediction:", prediction)
