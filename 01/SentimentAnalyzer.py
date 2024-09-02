from PerplexityWordBigramModel import PerplexityWordBigramModel
from training_data import reviews_texts

class SentimentAnalyzer(PerplexityWordBigramModel):
    def __init__(self):
        super().__init__()
        self.positive_model = PerplexityWordBigramModel()
        self.negative_model = PerplexityWordBigramModel()

    def train_sentiment(self, reviews):
        for review, rating in reviews:
            if rating > 3:
                self.positive_model.train(review)
            elif rating < 3:
                self.negative_model.train(review)
            # Neutral reviews (rating == 3) are not used for training

    def analyze_sentiment(self, text):
        words = text.lower().split()
        positive_prob = self.calculate_probability(words, self.positive_model)
        negative_prob = self.calculate_probability(words, self.negative_model)
        
        if positive_prob > negative_prob:
            return "Positive"
        elif negative_prob > positive_prob:
            return "Negative"
        else:
            return "Neutral"

    def calculate_probability(self, words, model):
        probability = 1
        for i in range(len(words) - 1):
            probability *= model.probability(words[i], words[i+1])
        return probability

# Train the model
sentiment_analyzer = SentimentAnalyzer()
sentiment_analyzer.train_sentiment(reviews_texts)

# Example usage
test_reviews = [
    ("This product exceeded my expectations. Great value for money!", "Positive"),
    ("Terrible experience. The item broke after a week.", "Negative"),
    ("It's okay, but nothing special. Does the job I guess.", "Neutral"),
    ("Absolutely love it! Best purchase I've made this year.", "Positive"),
    ("Disappointed with the quality. Not worth the price.", "Negative")
]

correct_predictions = 0
total_predictions = len(test_reviews)

for i, (review, expected_sentiment) in enumerate(test_reviews, 1):
    predicted_sentiment = sentiment_analyzer.analyze_sentiment(review)
    print(f"Review {i}: {review}")
    print(f"Expected sentiment: {expected_sentiment}")
    print(f"Predicted sentiment: {predicted_sentiment}")
    match = expected_sentiment == predicted_sentiment
    print(f"Match: {'Yes' if match else 'No'}")
    print()  # Add a blank line for readability
    if match:
        correct_predictions += 1

accuracy = (correct_predictions / total_predictions) * 100
print(f"Test Results Summary:")
print(f"Total reviews: {total_predictions}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")

"""
Lesson: Sentiment Analysis with Bigram Models

1. Model Structure:
   - SentimentAnalyzer extends PerplexityWordBigramModel
   - Uses separate bigram models for positive and negative sentiment
   - Each model maintains its own vocabulary and bigram counts

2. Training Process:
   - Initializes two PerplexityWordBigramModel instances: positive_model and negative_model
   - Iterates through the training data (reviews_texts)
   - For reviews with ratings > 3:
     * Preprocesses the review text
     * Trains the positive_model using the preprocessed text
   - For reviews with ratings < 3:
     * Preprocesses the review text
     * Trains the negative_model using the preprocessed text
   - Neutral reviews (rating == 3) are not used in training

3. Sentiment Analysis:
   - Takes a new piece of text as input
   - Preprocesses the input text (tokenization, lowercasing, etc.)
   - Calculates the probability of the preprocessed text using the positive_model
   - Calculates the probability of the preprocessed text using the negative_model
   - Compares the two probabilities:
     * If positive probability > negative probability, classifies as "Positive"
     * If negative probability > positive probability, classifies as "Negative"
     * If probabilities are equal, classifies as "Neutral"
   - Returns the determined sentiment

4. Probability Calculation:
   - For each model (positive and negative):
     * Initializes probability to 1
     * Iterates through the words in the input text
     * For each pair of consecutive words (bigram):
       - Multiplies the current probability by the bigram probability
     * Returns the final probability

This approach leverages the idea that positive reviews are more likely to contain word patterns common in other positive reviews, and similarly for negative reviews. The model with the higher probability for a given input text indicates the more likely sentiment.

4. Limitations:
   - Lacks context (only considers word pairs)
   - Struggles with sarcasm and complex sentiment
   - Limited vocabulary
   - No explicit training for neutral sentiment
   - Potential bias from imbalanced data
   - Ignores word importance

5. Performance Example:
   - 3 out of 5 sentiments correctly identified (60% accuracy)
   - Correctly classified strong positive and negative reviews
   - Struggled with neutral and subtle negative reviews

6. Improvement Suggestions:
   - Include neutral reviews in training
   - Use larger n-gram models (e.g., trigrams)
   - Incorporate sentiment lexicons or word embeddings
   - Expand training and testing datasets
   - Adjust thresholds for neutral classification
   - Use perplexity instead of probability for sentiment comparison:
     * Perplexity better captures model uncertainty
     * Lower perplexity indicates better fit to the model
     * Can provide more nuanced sentiment scores

In summary, Bigram models offer a simple starting point for sentiment analysis but have significant limitations. More advanced techniques like deep learning models often perform better, especially for nuanced sentiments. 

As an exercise, try using perplexity as a metric to potentially improve the model's performance and provide more meaningful sentiment scores.
"""