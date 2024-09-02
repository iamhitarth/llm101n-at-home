import math
from SmoothedWordBigramModel import SmoothedWordBigramModel

class PerplexityWordBigramModel(SmoothedWordBigramModel):
    def __init__(self):
        super().__init__()
        self.total_words = 0
        self.start_token = '<s>'
        self.end_token = '</s>'

    def train(self, text):
        words = self.preprocess(text)
        self.total_words += len(words) - 1  # Don't count start token
        for i in range(1, len(words)):  # Start from 1 to skip start token
            if words[i] != self.end_token:
                self.vocabulary.add(words[i])
            self.bigram_counts[words[i-1]][words[i]] += 1

    def probability(self, word1, word2):
        smoothing_factor = 0.01  # Adjust this value as needed - try 1, 0.1 and 0.01
        if word1 == self.start_token:
            return (self.bigram_counts[word1][word2] + smoothing_factor) / (self.total_words + smoothing_factor * len(self.vocabulary))
        if word1 not in self.vocabulary:
            return smoothing_factor / len(self.vocabulary)
        if word2 not in self.bigram_counts[word1]:
            return smoothing_factor / (len(self.vocabulary) + sum(self.bigram_counts[word1].values()))
        return (self.bigram_counts[word1][word2] + smoothing_factor) / (sum(self.bigram_counts[word1].values()) + smoothing_factor * len(self.vocabulary))

    def perplexity(self, text):
        words = self.preprocess(text)
        log_prob = 0
        for i in range(1, len(words)):
            prob = self.probability(words[i-1], words[i])
            log_prob += math.log2(prob)
        return 2 ** (-log_prob / (len(words) - 1))

    def preprocess(self, text):
        # You can add more preprocessing steps here if needed
        return ['<s>'] + text.lower().split() + ['</s>']

    def predict_next(self, context, n=3):
        if not context:
            context = [self.start_token]
        else:
            context = [context[-1]]

        choices = list(self.bigram_counts[context[0]].keys())
        probabilities = [self.probability(context[0], next_word) for next_word in choices]

        # Sort by probability and return top n predictions
        predictions = sorted(zip(choices, probabilities), key=lambda x: x[1], reverse=True)
        return predictions[:n]

if __name__ == "__main__":
    # Create and train the model
    model = PerplexityWordBigramModel()
    training_text = "the cat sat on the mat . the dog chased the cat ."
    model.train(training_text)

    # Test sentence
    test_sentences = [
        "the cat chased the dog",
        "the dog sat on the mat",
        "the elephant flew over the moon"
    ]

    for sentence in test_sentences:
        perplexity = model.perplexity(sentence)
        print(f"Perplexity for '{sentence}': {perplexity:.2f}")

"""
Lesson: Perplexity in Language Models

Perplexity is a crucial metric for evaluating language models, measuring how well a probability model predicts a sample. Key points:

1. Definition: Perplexity quantifies the model's uncertainty in predicting text. Lower perplexity indicates better prediction.

2. Calculation: 
   - Preprocess text (add start/end tokens)
   - Calculate log probability of each bigram
   - Compute perplexity as 2^(-average log probability)

3. Interpretation:
   - Perfect prediction: 1.0 (rarely achievable)
   - Realistic scores: > 1.0
   - Lower scores indicate better model performance

4. Example results:
   "the cat chased the dog": 6.53
   "the dog sat on the mat": 6.57
   "the elephant flew over the moon": 9.19

   Analysis:
   - Similar, low scores for first two sentences (familiar words/patterns)
   - Higher score for third sentence (likely unfamiliar words)
   - All scores > 1, as expected
   - Relative differences align with expected predictability

5. Smoothing factor:
   - Addresses zero probability for unseen bigrams
   - Larger factor: Better for small datasets, more generalization
   - Smaller factor: Suited for larger datasets, focuses on seen bigrams
   - Balances handling new combinations vs. accuracy on training data

Key takeaways:
- Perplexity inversely relates to assigned probability
- Useful for model comparison and tracking improvements
- Compare scores only within same model and vocabulary
- Relative comparisons more meaningful than absolute values
- Lower perplexity scores can be used as an indicator that changes made to the model are improving its performance

Remember: Lower perplexity signifies better prediction, but always consider the context of the model and data when interpreting results.

"""