
## Assignment Overview

This assignment involves creating a spelling fixer using a Hidden Markov Model (HMM). The program takes user input (a potentially misspelled word) and uses the Viterbi algorithm to determine the most likely correct spelling based on a trained model. The model consists of transition probabilities (between characters of correct words) and emission probabilities (likelihood of typing a character given the intended correct character).

## Questions Specific to This Assignment

The Viterbi algorithm implementation detailed in the document (second version, using `compare_words` and a similarity threshold) sometimes failed to correct words or corrected them to unexpected words. A more standard Viterbi implementation is also shown later.

### 1. Correctly Spelled Word by User, Incorrectly "Corrected" by Algorithm

*   **Example:**
    *   User input: `write`
    *   Algorithm output: `wrote`
*   **Reason:** The algorithm's dictionary of "correct words" contained "wrote" but did not contain "write". Since "write" was not a known correct word, the HMM, guided by its learned probabilities and available dictionary, found "wrote" as a more probable sequence.

### 2. Incorrectly Spelled Word by User, Still Incorrectly "Corrected" by Algorithm

*   **Example:**
    *   User input: `vot` (intended: "vote")
    *   Algorithm output: `voting`
*   **Reason:** The dictionary of "correct words" only contained "voting". The algorithm could not find a path to "vote" because "vote" was not in its list of valid hidden states (correct words). It found "voting" as a more probable (or the only viable long enough) candidate from its known correct words.

### 3. Incorrectly Spelled Word by User, Correctly Corrected by Algorithm

*   **Example:**
    *   User input: `contende`
    *   Algorithm output: `contented`
*   **Reason for Success (and contrast to previous failures):**
    *   Both the misspelled word "contende" (as an observed sequence) and the correct word "contented" (as a possible hidden state) were implicitly or explicitly handled by the model. "contented" was present in the `correct_words` list.
    *   The Viterbi implementation described here (first version discussed, not the full Viterbi shown later) includes a `compare_words` function that calculates a similarity score. This score is multiplied by the emission probabilities.
        ```python
        def compare_words(self, word1, word2):
            # ... calculates similarity based on matching characters in the same position ...
            min_length = min(len(word1), len(word2))
            matches = sum(1 for i in range(min_length) if word1[i] == word2[i])
            max_length = max(len(word1), len(word2))
            return matches / max_length

        def viterbi_decode(self, typed_word): # Simplified version from document
            candidates = []
            for correct_word in self.correct_words:
                similarity = self.compare_words(typed_word, correct_word)
                if similarity > 0.6: # Similarity threshold
                    prob = 1.0
                    for t, char in enumerate(typed_word):
                        if t < len(correct_word):
                            prob *= self.emission_probs[t].get(char, {}).get(correct_word[t], 1e-10)
                        else:
                            prob *= 1e-10 # Penalty for length mismatch
                    candidates.append((prob * similarity, correct_word))
            if candidates:
                return max(candidates, key=lambda x: x[0])[1]
            else:
                return typed_word
        ```
    *   This approach selects the best correction by considering words with >60% similarity and then choosing the one with the highest product of (cumulative emission probability * similarity score). "contented" likely had a high similarity and a favorable probability path.

### 4. Viterbi Algorithm Not Correcting / Miscorrecting

*   **Example (different Viterbi context):**
    *   User input: `contenpted`
    *   Algorithm output: `transferred`
*   **Reason:** This example likely refers to a more standard Viterbi implementation (shown later in the document's code snippets). The Viterbi algorithm finds the most probable sequence of hidden states (correct words/characters) given the observed sequence (typed word). If "transferred" had a higher overall probability (based on start probabilities, transition probabilities between its characters, and emission probabilities of typing `contenpted` given `transferred`) than other candidates in the dictionary, it would be chosen, even if it seems semantically or orthographically distant. This can happen due to:
    *   The specific probabilities learned during training (from `spell-testset1.txt`).
    *   Limited vocabulary of correct words.
    *   The nature of the HMM, which optimizes for probability, not necessarily edit distance or perceived similarity in all cases.

## Viterbi Algorithm Implementation (More Standard Version)

The document also shows a more typical Viterbi implementation:

```python
def viterbi_decode(self, typed_word):
    V = [{}]  # Stores probabilities of best paths
    path = {} # Stores backpointers for best paths

    # Initialize base cases (t == 0)
    for state in self.correct_words: # 'state' here is a full correct word
        # This assumes transition_probs['start'] gives P(state[0])
        # and emission_probs[0] gives P(typed_word[0] | state[0])
        V[0][state] = self.transition_probs['start'].get(state[0], 1e-10) * \
                      self.emission_probs[0].get(typed_word[0], {}).get(state[0], 1e-10)
        path[state] = [state] # Path starts with the state itself

    # Run Viterbi for t > 0
    for t in range(1, len(typed_word)):
        V.append({})
        newpath = {}
        for state in self.correct_words: # Current correct word being considered
            if len(state) > t: # Ensure current word is long enough
                # Find the best previous state that transitions to the current character of 'state'
                (prob, prev_st) = max(
                    (V[t-1][prev_state] * \
                     self.transition_probs[prev_state[t-1]].get(state[t], 1e-10) * \
                     self.emission_probs[t].get(typed_word[t], {}).get(state[t], 1e-10),
                     prev_state)
                    for prev_state in V[t-1] if len(prev_state) > t-1 # Ensure prev_state is long enough
                )
                V[t][state] = prob
                newpath[state] = path[prev_st] + [state] # This path reconstruction seems to append full words
        path = newpath

    # Find the best path
    # This part needs to correctly identify the most probable *final word* at the last time step.
    # The current 'path' structure might store lists of full words, which needs careful handling for final selection.
    if not V[len(typed_word) - 1]: # If no path found for the full length
        return typed_word # Return original word

    (prob, final_state_word) = max((V[len(typed_word) - 1][state], state) for state in V[len(typed_word) - 1])
    # The path[final_state_word][-1] would return the last word in the most probable sequence of words.
    return path[final_state_word][-1] # This assumes the path stores the sequence correctly.
