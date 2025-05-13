import re
from collections import defaultdict

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
                        #key: type_char, It models the probability of observing typed_char given correct_char
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

    def viterbi_decode(self, typed_word):
        V = [{}]
        path = {}

        # Initialize base cases (t == 0)
        for state in self.correct_words:
            #If the probability doesn't exist, it uses 1e-10 as a small default value to avoid zero probabilities.
            #self.emission_probs[0].get(typed_word[0], {}).get(state[0], 1e-10):gets the probability of observing the first typed character (typed_word[0]) given that the correct character should be the first letter of the current state (state[0]).
            V[0][state] = self.transition_probs['start'].get(state[0], 1e-10) * self.emission_probs[0].get(typed_word[0], {}).get(state[0], 1e-10)
            #initializes the path for each state, starting with the state itself.
            path[state] = [state]

        # Run Viterbi for t > 0
        for t in range(1, len(typed_word)):
            V.append({})
            newpath = {}

            for state in self.correct_words:
                if len(state) > t:
                    #max(prob, prev_state):prob is the maximum probability found, prev_state is the state that led to this maximum probability
                    (prob, prev_state) = max((V[t-1][prev_state] * 
                                              self.transition_probs[prev_state[t-1]].get(state[t], 1e-10) * 
                                              self.emission_probs[t].get(typed_word[t], {}).get(state[t], 1e-10), 
                                              prev_state) 
                                              #ensures that we only consider previous states (words) that are long enough to have a character at position t-1.
                                             for prev_state in V[t-1] if len(prev_state) > t-1)
                    V[t][state] = prob #example: Store the maximum probability for "hint" at t = 2: V[2]["hint"] = 0.05
                    newpath[state] = path[prev_state] + [state]

            path = newpath

        # Find the best path
        if V[len(typed_word) - 1]: #check if there are any calculated probabilities for the last character in typed_word
            # V[len(typed_word) - 1][state]: gives the probability of being in state at the last position.
            (prob, state) = max((V[len(typed_word) - 1][state], state) for state in V[len(typed_word) - 1])
            return path[state][-1] #[-1] gets the last item in this list, which is the final correct word.
        else:
            return typed_word  # Return original word if no valid path found
    
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
