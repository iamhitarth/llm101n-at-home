from collections import defaultdict
from training_data import ngram_training_texts
import random


class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)

    def train(self, text):
        words = ['<s>'] * (self.n - 1) + text.lower().split() + ['</s>']
        for i in range(len(words) - self.n + 1):
            ngram = tuple(words[i:i+self.n])
            self.ngram_counts[ngram] += 1
            self.context_counts[ngram[:-1]] += 1

    def probability(self, ngram):
        ngram = tuple(word.lower() for word in ngram)
        context = ngram[:-1]
        # print(f"Checking probability for ngram: {ngram}")
        # print(f"Context: {context}")
        # print(f"Ngram count: {self.ngram_counts[ngram]}")
        # print(f"Context count: {self.context_counts[context]}")
        if self.context_counts[context] == 0:
            return 0
        return self.ngram_counts[ngram] / self.context_counts[context]


    def generate(self, num_words):
        context = ('<s>',) * (self.n - 1)
        generated_words = []

        for _ in range(num_words):
            possible_ngrams = [ngram for ngram in self.ngram_counts.keys() if ngram[:-1] == context]
            if not possible_ngrams:
                break

            ngram = random.choices(possible_ngrams, 
                                   weights=[self.probability(ng) for ng in possible_ngrams])[0]
            next_word = ngram[-1]
            generated_words.append(next_word)

            if next_word == '</s>':
                break

            context = context[1:] + (next_word,)

        return ' '.join(generated_words)
    
if __name__ == "__main__":
    model = NGramModel(n=3)  # Create a trigram model
    for training_text in ngram_training_texts:
        model.train(training_text)

    # Calculate probabilities
    print(model.probability(("The", "sun", "set")))
    print(model.probability(("The", "old", "man")))
    print(model.probability(("The", "spaceship", "was"))) # This should return a 0 probability

    # Generate text
    generated_text = model.generate(num_words=10)
    print("Generated text:", generated_text)

"""
This exercise extends the bigram model to handle arbitrary n-grams, offering a more flexible approach to language modeling:

1. Model Initialization:
   - The __init__ method takes an 'n' parameter to specify the n-gram size.
   - It uses a single dictionary for n-grams and another for (n-1)-grams (contexts), replacing separate bigram and unigram dictionaries.

2. Training Process:
   - The train method creates n-grams by adding (n-1) start tokens and one end token.
   - It counts occurrences of both n-grams and their contexts.

3. Probability Calculation:
   - The probability(ngram) method calculates the likelihood of an n-gram by dividing its count by its context count.
   - It returns 0 if the context is unseen.

4. Text Generation:
   - The generate(num_words) method produces text based on the trained model:
     a. It begins with (n-1) start tokens as the initial context.
     b. For each new word, it:
        - Finds all n-grams matching the current context.
        - Selects the next word probabilistically based on n-gram probabilities.
        - Adds the chosen word to the output and updates the context.
     c. Generation stops when it reaches the specified word count, encounters the end token, or finds no matching n-grams.

This generalization captures longer-range dependencies in text. Higher 'n' values can produce more coherent text but require more training data and computational resources. These methods form the core functionality for probability calculation and text generation using the n-gram model.

"""