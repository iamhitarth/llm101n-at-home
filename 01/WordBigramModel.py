import random
from collections import defaultdict
from training_data import training_texts
    
class WordBigramModel:
    def __init__(self):
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)

    def print_model_stats(self):
      print("Bigram Counts:")
      for context, next_tokens in self.bigram_counts.items():
          for next_token, count in next_tokens.items():
              print(f"({context}, {next_token}): {count}")
      print("\nContext Counts:")
      for context, count in self.context_counts.items():
          print(f"{context}: {count}")

    def train(self, text):
        # Here instead of splitting the text into characters, we split it into words
        words = ['<s>'] + text.split() + ['</s>']
        
        for i in range(len(words) - 1):
            word, next_word = words[i], words[i+1]
            self.bigram_counts[word][next_word] += 1
            self.context_counts[word] += 1

        # To see what the model learned
        # self.print_model_stats()

    def probability(self, word, next_word):
        if self.context_counts[word] == 0:
            return 0
        return self.bigram_counts[word][next_word] / self.context_counts[word]

    def generate(self, num_words=20):
        current_word = '<s>'
        generated_text = []

        for _ in range(num_words):
            choices = list(self.bigram_counts[current_word].keys())
            probabilities = [self.probability(current_word, next_word) for next_word in choices]
            next_word = random.choices(choices, probabilities)[0]

            if next_word == '</s>':
                break

            generated_text.append(next_word)
            current_word = next_word

        # Join the words to form a sentence
        return ' '.join(generated_text)

if __name__ == "__main__":
    # Usage example
    model = WordBigramModel()
    
    for text in training_texts:
        model.train(text)

    generated_text = model.generate()
    print(generated_text)

"""
This word-level model follows the same structure as the character-level model but operates on words instead of characters.
"""