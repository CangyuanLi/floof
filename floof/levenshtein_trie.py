import jellyfish

from trie import Trie, TrieNode

word_trie = Trie()
for word in ("danger", "dangerous", "blah", "blooping"):
    word_trie.insert(word)

def recursive_search(node: TrieNode, letter: str, word: str, previous_row, results):

    columns = len(word) + 1
    current_row = [previous_row[0] + 1]

    # Build one row for the letter, with a column for each letter in the target
    # word, plus one for the empty string at column 0
    for column in range(1, columns):

        insert_cost = current_row[column - 1] + 1
        delete_cost = previous_row[column] + 1

        if word[column - 1] != letter:
            replace_cost = previous_row[column - 1] + 1
        else:                
            replace_cost = previous_row[column - 1]

        current_row.append(min(insert_cost, delete_cost, replace_cost))

    # if the last entry in the row indicates the optimal cost is less than the
    # maximum cost, and there is a word in this trie node, then add it.
    if node.word != None:
        results.append((node.word, current_row[-1]))

    # if any entries in the row are less than the maximum cost, then 
    # recursively search each branch of the trie
    for letter, val in node.children.items():
        recursive_search(
            val, 
            letter, 
            word, 
            current_row, 
            results 
        )


# maximum distance from the target word
def search(trie: Trie, word: str):

    # build first row
    current_row = range(len(word) + 1)

    results = []

    # recursively search each branch of the trie
    for letter, child in trie.root.children.items():
        recursive_search(
            child, 
            letter, 
            word, 
            current_row, 
            results 
        )

    return results

# This recursive helper is used by the search function above. It assumes that
# the previous_row has been filled in already.

res = search(word_trie, "blah")
print(res)

print(jellyfish.damerau_levenshtein_distance("blah", "dangerous"))

