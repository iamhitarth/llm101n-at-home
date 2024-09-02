from PerplexityWordBigramModel import PerplexityWordBigramModel
import pandas as pd
import re, os

# Load the dataset
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the CSV file
csv_path = os.path.join(current_dir, 'chat_dataset.csv')
df = pd.read_csv(csv_path, encoding='ISO-8859-1')

# Keep only the relevant columns
df = df[['message']]
df.columns = ['text']

# Print the first 5 messages
print("First 5 messages before pre-processing:")
for i, message in enumerate(df['text'][:5]):
    print(f"{i+1}. {message}")
print()  # Add an empty line for better readability


# Define a simple function to clean and tokenize the text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.lower()  # Convert to lowercase
    # tokens = text.split(' ')  # Split by spaces to tokenize
    # return tokens
    return text

df['processed_text'] = df['text'].apply(preprocess_text)

# Use the processed messages for training
messages = df['processed_text'].tolist()

# Print the first 5 messages
print("First 5 messages after pre-processing:")
for i, msg in enumerate(messages[:5]):
    print(f"{i+1}. {msg}")
print()  # Add an empty line for better readability

# Train the model
model = PerplexityWordBigramModel()
for msg in messages:
    model.train(msg)

current_text = "I am"
predictions = model.predict_next(current_text)
print(f"Current text: '{current_text}'")
print("Top 3 predictions:", predictions)

"""
This exercise demonstrates the implementation and usage of a predictive text system using a bigram language model. Here's what we're doing:

1. Data Loading and Preprocessing:
   - We load a chat dataset from a CSV file (downloaded from https://www.kaggle.com/datasets/nursyahrina/chat-sentiment-dataset).
   - We preprocess the text data by removing non-word characters, converting to lowercase, and standardizing spaces.

2. Model Training:
   - We use the PerplexityWordBigramModel, which is a bigram model that can calculate perplexity.
   - The model is trained on the preprocessed messages from the dataset.

3. Prediction:
   - We demonstrate the model's predictive capabilities by providing a partial text input ("I am").
   - The model then predicts the top 3 most likely next words based on the trained bigram probabilities.

This demonstration showcases how a simple bigram model can be used for predictive text applications, such as auto-complete or next-word prediction in messaging apps. 

Note that in this case, we're splitting the text dataset into training tokens simply by removing spaces; however, in production, we would use a separate tokenizer because it can handle punctuation, contractions, and multi-word expressions more effectively. Preprocessing steps like normalizing text, removing or standardizing punctuation, and handling language-specific nuances are crucial to optimize tokenization for specific domains or datasets. The choice of tokenizer not only impacts the accuracy and consistency of the bigram model but also influences training time and model complexity. While a simple tokenizer might suffice for straightforward text, a sophisticated tokenizer becomes necessary in scenarios involving complex, technical, or punctuation-heavy text, ensuring that the model generates more meaningful and reliable predictions.
"""
