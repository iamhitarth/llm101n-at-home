from PerplexityWordBigramModel import PerplexityWordBigramModel
from training_data import sci_fi_texts, fantasy_texts, romance_texts

def train_genre_model(texts, genre):
    model = PerplexityWordBigramModel()
    for text in texts:
        model.train(text)
    
    # Debug output
    print(f"Trained {genre} model:")
    print(f"Vocabulary size: {len(model.vocabulary)}")
    print(f"Number of bigrams: {sum(len(bigrams) for bigrams in model.bigram_counts.values())}")
    print(f"Top 10 most common words: {sorted(model.vocabulary, key=lambda w: sum(model.bigram_counts[w].values()), reverse=True)[:10]}")
    
    return model

def classify_text(text, genre_models):
    perplexities = {}
    for genre, model in genre_models.items():
        perplexity = model.perplexity(text)
        normalized_perplexity = perplexity * (len(model.vocabulary) / 100) * (model.total_words / 1000)
        perplexities[genre] = normalized_perplexity

    # Debug output
    print("Perplexities:", perplexities)
    
    return min(perplexities, key=perplexities.get)

if __name__ == "__main__":
    # Train the genre models
    genre_models = {
        "sci-fi": train_genre_model(sci_fi_texts, "sci-fi"),
        "fantasy": train_genre_model(fantasy_texts, "fantasy"),
        "romance": train_genre_model(romance_texts, "romance")
    }

    # Classify a new text
    example_texts = [
        "The dragon swooped down from the sky and breathed fire on the village.",
        "She looked into his eyes and felt her heart skip a beat.",
        "The spaceship hovered above Earth, preparing to make first contact.",
        "He drew his sword, the enchanted blade glimmering in the twilight.",
        "Her robotic companion beeped cheerfully as it handed her the tools."
    ]

    for text in example_texts:
        genre = classify_text(text, genre_models)
        print(f"The text '{text}' is classified as: {genre}")
        print("Word-by-word probabilities:")
        words = text.split()
        for i in range(1, len(words)):
            for g, model in genre_models.items():
                prob = model.probability(words[i-1], words[i])
                print(f"  {g}: P({words[i]} | {words[i-1]}) = {prob:.4f}")
        print()

"""
Lesson: Text Classification using Perplexity and Word Bigram Models

This lesson covers a text classification system using perplexity-based word bigram models for genre identification. Key points include:

1. Model Training:
   - Sample texts from different genres (sci-fi, fantasy, romance) are used.
   - A PerplexityWordBigramModel is trained for each genre using the train_genre_model function.

2. Classification Method:
   - The classify_text function determines the genre by finding the model that assigns the lowest perplexity to a given text.
   - Perplexity measures how well a probability model predicts a sample, indicating how closely a text matches a genre's language patterns.
   - Lower perplexity suggests a better match to the genre.

3. Perplexity Normalization:
   - Perplexity is normalized by vocabulary size to account for models with larger vocabularies assigning lower probabilities to individual words.
   - Normalization ensures fair comparison across models with different vocabulary sizes.

4. Underflow Prevention:
   - Underflow occurs when multiplying many small probabilities results in a number too small for the computer to represent accurately.
   - Log probabilities are used instead of raw probabilities to avoid underflow issues in perplexity calculations.
   - The sum of log probabilities replaces the product of probabilities:
     log P(w1, w2, ..., wn) = log P(w1) + log P(w2|w1) + log P(w3|w2) + ... + log P(wn|wn-1)
   - Perplexity is calculated as: 2 ** (- (1/N) * sum(log P(wi|w(i-1))))

5. Output and Analysis:
   - The script outputs genre classification for example texts and word-by-word probabilities for each genre model.
   - This helps understand the classification reasoning.

6. Key Observations:
   - Sparsity: Many probabilities are low (0.0001) due to limited training data.
   - Genre-specific words: Some words have higher probabilities in certain genres (e.g., "enchanted" in fantasy, "heart" in romance).
   - Common words: Some bigrams have high probabilities across genres.
   - Limited differentiation: Many words have low probabilities in all models.
   - Effective classification: Despite limitations, cumulative small probability differences lead to meaningful genre classifications.

7. Potential Improvements:
   - Increase training data volume.
   - Use more advanced models.
   - Add features beyond bigrams for enhanced accuracy.

This approach demonstrates both the potential and limitations of bigram-based genre classification, highlighting why more complex tasks often require advanced techniques like neural networks.
"""