# Chapter 1: Bigram Language Model

## Introduction
Welcome to the world of language modeling! As a junior software engineer, you're about to embark on an exciting journey into the foundations of natural language processing. In this chapter, we'll explore the bigram language model, a simple yet powerful tool that forms the basis for more complex language models.

## 1. Basics of Language Modeling

Language modeling is the task of predicting the next word or character in a sequence of text. It's a fundamental concept in natural language processing (NLP) and has numerous applications, from predictive text in messaging apps to more advanced tasks like machine translation and speech recognition.

The goal of a language model is to learn the probability distribution of words or characters in a language. This allows us to:
- Predict the next word in a sentence
- Generate new text that sounds natural
- Evaluate how likely a given sentence is in a particular language

## 2. Probability Distributions

At the heart of language modeling lies the concept of probability distributions. In the context of language, a probability distribution tells us how likely each possible word (or character) is to appear next in a sequence.

For example, consider the partial sentence: "The cat sat on the ___"

A good language model would assign higher probabilities to words like "mat", "floor", or "roof", and lower probabilities to words like "spaghetti" or "democracy".

Mathematically, we can represent this as:

P(word | "The cat sat on the")

This reads as "the probability of a word given the preceding context 'The cat sat on the'".

## 3. Maximum Likelihood Estimation

To build our language model, we need to estimate these probabilities. One common method is Maximum Likelihood Estimation (MLE). The idea behind MLE is simple: we count how many times each word sequence appears in our training data and use these counts to estimate probabilities.

For a bigram model (which we'll discuss in more detail shortly), the maximum likelihood estimate is:

P(w_n | w_(n-1)) = count(w_(n-1), w_n) / count(w_(n-1))

Where:
- w_n is the current word
- w_(n-1) is the previous word
- count(w_(n-1), w_n) is the number of times the sequence (w_(n-1), w_n) appears in the training data
- count(w_(n-1)) is the number of times w_(n-1) appears in the training data

## 4. Markov Assumption

The bigram model relies on a simplifying assumption known as the Markov assumption. In the context of language modeling, the Markov assumption states that the probability of a word depends only on the previous word, not on any earlier words.

Mathematically, we can express this as:

P(w_n | w_1, w_2, ..., w_(n-1)) ≈ P(w_n | w_(n-1))

This assumption greatly simplifies our model and makes it computationally tractable, but it's also a limitation. In reality, language often depends on longer-range context, which is why more advanced models consider longer sequences.

## 5. Bigram Language Model

Now that we've covered the foundational concepts, let's put it all together to understand the bigram language model.

A bigram is a sequence of two adjacent elements (in our case, words or characters) in a string of tokens. The bigram model predicts the next word based solely on the previous word.

The probability of a sequence of words W = (w_1, w_2, ..., w_n) under the bigram model is:

P(W) = P(w_1) * P(w_2|w_1) * P(w_3|w_2) * ... * P(w_n|w_(n-1))

Where P(w_1) is the probability of the first word, and each subsequent term is the conditional probability of a word given the previous word.

To build a bigram model:

1. Tokenize your training text into words or characters.
2. Count all bigrams in the text.
3. Calculate probabilities using the MLE formula mentioned earlier.

When using the model to generate text or calculate probabilities, you simply look up the appropriate conditional probabilities based on the previous word.

## Conclusion

The bigram language model, while simple, introduces key concepts that form the foundation of more advanced language models. As you progress in your NLP journey, you'll encounter models that consider longer contexts and more complex relationships between words. However, the fundamental ideas of probability distributions, maximum likelihood estimation, and the Markov assumption will continue to be relevant.

In the next chapters, we'll build upon these concepts to create more sophisticated models that can capture nuanced language patterns and generate more coherent text.

## Exercises

1. [CharacterBigramModel.py](./CharacterBigramModel.py) - Implement a character-level bigram model in Python. Use it to generate random text and observe the results.
2. [WordBigramModel.py](./WordBigramModel.py) - Extend your implementation to a word-level bigram model. How does the generated text differ from the character-level model?
3. Experiment with different text corpora (e.g., news articles, novels, tweets). How does the choice of training data affect the model's output?
4. [SmoothedWordBigramModel.py](./SmoothedWordBigramModel.py) - Implement a simple smoothing technique (e.g., add-one smoothing) to handle unseen bigrams. How does this affect the model's performance?
5. [PerplexityWordBigramModel.py](./PerplexityWordBigramModel.py) - Demonstrate the concept of perplexity and its uses.
6. [TextClassification.py](./TextClassification.py) - Implement text classification for multiple genres.
7. [PredictiveText.py](./PredictiveText.py) - Use bigram model for generating next word for a partial text message.

These exercises will give you hands-on experience with implementing and working with bigram (and n-gram) language models. They cover various aspects including smoothing, evaluation metrics (perplexity), and practical applications like text generation and classification. As you work through these exercises, you'll gain a deeper understanding of the strengths and limitations of these models, setting a strong foundation for exploring more advanced language modeling techniques.

----

For the Character Bigram model had to fix the code generated by Sonnet as it was parsing the starting and ending sequences `<s>` and `</s>` as individual characters (ie. '<', 's', '>') instead of whole tokens.
I also updated the code for the Character Bigram model to include a bit more training data (15 sentences vs 1 sentence) just to see if it makes a difference - it didn't.
The generated text from Bigram model seems like gibberish but it makes it very easy to understand the concepts of training and generation (in the context of LLMs) and how that works at a very basic level.

Very interesting with Word Bigram model as to how quickly the text generated by the model starts to sound human. `Thesng ingamesthed mehe t s jurs flee pug` vs `The sun sets behind the lazy dog.`
Summary of the main differences between CharacterBigramModel.py and WordBigramModel.py:
1. Tokenization: 
CharacterBigramModel: Splits text into individual characters.
WordBigramModel: Splits text into words.
2. Generation:
CharacterBigramModel: Generates text character by character, up to a maximum number of characters.
WordBigramModel: Generates text word by word, up to a specified number of words.
3. Granularity:
CharacterBigramModel: Captures character-level patterns and transitions.
WordBigramModel: Captures word-level patterns and transitions.

Both models use similar underlying logic for training and probability calculation, but they operate at different levels of text granularity.

---

Suggested Datasets for Experimentation

For general text processing and generation:

Project Gutenberg (https://www.gutenberg.org/): A large collection of free eBooks.
Wikipedia dumps (https://dumps.wikimedia.org/): Contains the full text of Wikipedia in various languages.


For the Predictive Text for Messaging Apps project:

SMS Spam Collection Dataset (https://www.kaggle.com/uciml/sms-spam-collection-dataset): Contains real SMS messages.
Twitter Sentiment Analysis Dataset (https://www.kaggle.com/kazanova/sentiment140): While intended for sentiment analysis, it contains real tweets that can be used for message prediction.


For the Email Subject Line Generator project:

Enron Email Dataset (https://www.cs.cmu.edu/~enron/): A large dataset of real emails, including subjects and content.
Clinton Email Dataset (https://www.kaggle.com/kaggle/hillary-clinton-emails): Another real-world email dataset.


For the Sentiment Analysis for Product Reviews project:

Amazon Product Reviews (https://nijianmo.github.io/amazon/index.html): A large dataset of Amazon product reviews with ratings.
Yelp Dataset (https://www.yelp.com/dataset): Contains business reviews and ratings.