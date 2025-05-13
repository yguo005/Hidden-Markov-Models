import re
from collections import defaultdict
import difflib

class SpellingFixer:
    def __init__(self):
        self.emission_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.correct_words = set()

    def train(self, training_data):
        for line in training_data.split('\n'):
            if ':' in line:
                correct_word, misspellings = line.split(':')
                correct_word = correct_word.strip()
                misspellings = misspellings.strip().split()
                self.correct_words.add(correct_word)
            
                # Calculate emission probabilities
                for misspelled in misspellings:
                    for position, char_pair in enumerate(zip(correct_word, misspelled)):
                        correct_char, typed_char = char_pair
                        self.emission_probs[position][typed_char][correct_char] += 1

                # Calculate transition probabilities 
                self.calculate_transition_probabilities(correct_word)

        # Normalize probabilities
        self.normalize_probabilities()

    #Only correct words are used to build the transition matrix
    def calculate_transition_probabilities(self, word):
        # Start transition
        self.transition_probs['start'][word[0]] += 1
        
        # Transitions between characters
        for i in range(len(word) - 1):
            self.transition_probs[word[i]][word[i+1]] += 1
        
        # End transition
        self.transition_probs[word[-1]]['end'] += 1

    def normalize_probabilities(self):
        # Normalize emission probabilities to get probabilities
        for pos in self.emission_probs:
            for typed_char in self.emission_probs[pos]:
                #self.emission_probs[pos][typed_char]:Keys are correct characters, sum how many times typed_char should have been correct_char at position pos
                total = sum(self.emission_probs[pos][typed_char].values())
                for correct_char in self.emission_probs[pos][typed_char]:
                    self.emission_probs[pos][typed_char][correct_char] /= total

        # Normalize transition probabilities to get probabilities
        for state in self.transition_probs:
            total = sum(self.transition_probs[state].values())
            for next_state in self.transition_probs[state]:
                self.transition_probs[state][next_state] /= total

    def compare_words(self, word1, word2):
        """
        Compare two words and return a similarity score.
        The score is based on the number of matching characters in the same position.
        """
        min_length = min(len(word1), len(word2))
        matches = sum(1 for i in range(min_length) if word1[i] == word2[i])
        max_length = max(len(word1), len(word2))
        return matches / max_length

    def viterbi_decode(self, typed_word):
        candidates = []
        for correct_word in self.correct_words:
            similarity = self.compare_words(typed_word, correct_word)
            if similarity > 0.6:  # Only consider words with similarity above 60%
                prob = 1.0
                for t, char in enumerate(typed_word):
                    if t < len(correct_word):
                        prob *= self.emission_probs[t].get(char, {}).get(correct_word[t], 1e-10)
                    else:
                        prob *= 1e-10
                candidates.append((prob * similarity, correct_word))
        
        if candidates:
            return max(candidates, key=lambda x: x[0])[1]
        else:
            return typed_word

    def correct(self, input_text):
        words = input_text.split()
        corrected_words = []

        for word in words:
            if word.lower() in self.correct_words:
                corrected_words.append(word)
            else:
                corrected_word = self.viterbi_decode(word)
                corrected_words.append(corrected_word)
                print(f"Decoded word: {corrected_word}")

        return ' '.join(corrected_words)


with open('spell-testset1.txt', 'r') as file:
    training_data = file.read()

fixer = SpellingFixer()
fixer.train(training_data)

# Example correction
input_text = input("Enter text to correct: ")
corrected_text = fixer.correct(input_text)
