import random
from WordBigramModel import WordBigramModel
from training_data import training_texts

class SmoothedWordBigramModel(WordBigramModel):
    def __init__(self):
        super().__init__()
        self.vocabulary = set()

    def train(self, text):
        super().train(text)
        # Only include actual words and the end token in the vocabulary - otherwise we might end up with start token in the generated text
        self.vocabulary.update(set(text.split() + ['</s>']))

    def probability(self, word, next_word):
        numerator = self.bigram_counts[word][next_word] + 1
        denominator = self.context_counts[word] + len(self.vocabulary)
        return numerator / denominator
if __name__ == "__main__":
    # Compare the output of the smoothed and unsmoothed models
    unsmoothed_model = WordBigramModel()
    smoothed_model = SmoothedWordBigramModel()

    # Train both models with the same training data
    for text in training_texts:
        unsmoothed_model.train(text)
        smoothed_model.train(text)

    # Scenario for completion of a set of unseen bigrams
    unseen_start = "The zephyr"
    unseen_bigrams = unseen_start.split() + ['</s>']

    # Generate text from the unseen bigrams for both models
    def generate_completion(model, start_bigram):
        current_word = start_bigram.split()[-1]
        generated_text = [start_bigram]

        while True:
            if isinstance(model, SmoothedWordBigramModel):
                choices = list(model.vocabulary)
            elif isinstance(model, WordBigramModel):
                choices = list(model.bigram_counts[current_word].keys())
            
            if not choices:
                break
            
            probabilities = [model.probability(current_word, next_word) for next_word in choices]
            next_word = random.choices(choices, probabilities)[0]

            if next_word == '</s>':
                break

            generated_text.append(next_word)
            current_word = next_word

        return ' '.join(generated_text)

    unsmoothed_completion = generate_completion(unsmoothed_model, unseen_start)
    smoothed_completion = generate_completion(smoothed_model, unseen_start)

    # Print the completions
    print("Unsmoothed Model Completion for Unseen Bigrams:")
    print(unsmoothed_completion)
    print("\nSmoothed Model Completion for Unseen Bigrams:")
    print(smoothed_completion)


"""
Lesson: Add-One (Laplace) Smoothing for Bigram Models

Add-one smoothing, also known as Laplace smoothing, is a technique used to handle unseen bigrams in language models. This method is crucial for several reasons:

1. It allows the model to calculate probabilities for sequences not present in the training data.
2. It enables text generation starting from words not seen during training.
3. It improves model evaluation on test sets containing novel word combinations.

Implementation:
The SmoothedWordBigramModel class extends the WordBigramModel with the following key modifications:

1. Vocabulary Creation: During training, a set of all unique words (including the end token) is created.
2. Probability Calculation: The probability method is adjusted to implement add-one smoothing:
   - Add 1 to the count of the specific bigram (numerator)
   - Add the vocabulary size to the count of the context word (denominator)

Formula:
P(word2 | word1) = (count(word1, word2) + 1) / (count(word1) + |V|)
Where |V| is the vocabulary size.

Benefits:
- Ensures non-zero probabilities for all bigrams, including unseen ones.
- Allows for more diverse text generation.
- Particularly useful with small datasets.

Example:
Consider a small corpus: "The cat sat" and "The dog barked"

1. Seen bigram "The cat":
   Without smoothing: P(cat | The) = 1 / 2 = 0.5
   With smoothing: P(cat | The) = (1 + 1) / (2 + 5) ≈ 0.286
   (Assuming vocabulary size = 5)

2. Unseen bigram "The mouse":
   Without smoothing: P(mouse | The) = 0 / 2 = 0
   With smoothing: P(mouse | The) = (0 + 1) / (2 + 5) ≈ 0.143

This example demonstrates how smoothing assigns a small but non-zero probability to unseen bigrams, allowing the model to generate more varied and natural-sounding text.
"""