import random
import re
from collections import defaultdict, Counter

class BigramModel:
    def __init__(self, corpus: list[str], frequency_threshold: int = None):
        # Combine the list of sentences from the corpus into a single string
        full_text = " ".join(corpus)
        self.vocab, self.bigram_probs = self.analyze_bigrams(full_text, frequency_threshold)

    def simple_tokenizer(self, text, frequency_threshold=5):
        """Simple tokenizer that splits text into words."""
        # Convert to lowercase and extract words using regex
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not frequency_threshold:
            return tokens
        # Count word frequencies
        word_counts = Counter(tokens)
        # Define a threshold for less frequent words (e.g., words appearing fewer than 5 times)
        filtered_tokens = [
            token for token in tokens if word_counts[token] >= frequency_threshold
        ]
        return filtered_tokens

    def analyze_bigrams(self, text, frequency_threshold=None):
        """Analyze text to compute bigram probabilities."""
        words = self.simple_tokenizer(text, frequency_threshold)
        bigrams = list(zip(words[:-1], words[1:]))

        # Count bigram and unigram frequencies
        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        # Compute bigram probabilities
        bigram_probs = defaultdict(dict)
        for (word1, word2), count in bigram_counts.items():
            bigram_probs[word1][word2] = count / unigram_counts[word1]

        return list(unigram_counts.keys()), bigram_probs

    def generate_text(self, start_word: str, num_words: int = 20) -> str:
        """Generate text based on bigram probabilities."""
        current_word = start_word.lower()
        if current_word not in self.vocab:
            return f"Error: '{start_word}' is not in the model's vocabulary. Please try a different word."

        generated_words = [current_word]

        for _ in range(num_words - 1):
            next_words = self.bigram_probs.get(current_word)
            if not next_words:  # If no bigrams for the current word, stop generating
                break

            # Choose the next word based on probabilities
            next_word = random.choices(
                list(next_words.keys()), weights=next_words.values())[0]
            generated_words.append(next_word)
            current_word = next_word  # Move to the next word

        return " ".join(generated_words)