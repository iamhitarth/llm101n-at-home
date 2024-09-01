import random
from collections import defaultdict
from training_data import training_texts
    
class CharacterBigramModel:
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
        # Add start and end tokens
      tokens = ['<s>'] + list(text) + ['</s>']
      
      for i in range(len(tokens) - 1):
          token, next_token = tokens[i], tokens[i+1]
          self.bigram_counts[token][next_token] += 1
          self.context_counts[token] += 1

      # To see what the model learned  
      # self.print_model_stats()
        
    def probability(self, char, next_char):
        if self.context_counts[char] == 0:
            return 0
        return self.bigram_counts[char][next_char] / self.context_counts[char]

    def generate(self, max_chars=100):
      current_token = '<s>'
      generated_text = ''

      while len(generated_text) < max_chars:
          choices = list(self.bigram_counts[current_token].keys())
          if not choices:
              break
          probabilities = [self.probability(current_token, next_token) for next_token in choices]
          next_token = random.choices(choices, probabilities)[0]

          if next_token == '</s>':
              break

          if current_token != '<s>':
              generated_text += current_token
          current_token = next_token

      return generated_text

if __name__ == "__main__":
    # Using the CharacterBigramModel
    model = CharacterBigramModel()

    for text in training_texts:
        model.train(text)

    generated_text = model.generate()
    print(generated_text)

"""
This implementation creates a CharacterBigramModel class with methods for training on text data, calculating probabilities, and generating new text.
"""