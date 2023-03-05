import time

# The Trie data structure keeps a set of words, organized with one node for
# each letter. Each node has a branch for each letter that may follow it in the
# set of words.
# read dictionary file into a trie

class TrieNode:

    def __init__(self):
        self._word = None
        self._is_end = False
        self._children = {}

class Trie:

    def __init__(self):
        self._root = TrieNode()

    def insert(self, word: str):
        node: TrieNode = self._root
        for letter in word:
            if letter not in node._children: 
                node._children[letter] = TrieNode()

            node = node._children[letter]

        node._word = word
        node._is_end = True

    def __repr__(self):
        def recur(node, indent):
            return "".join(indent + key + ("$" if child._is_end else "") 
                                  + recur(child, indent + "  ") 
                for key, child in node._children.items())

        return recur(self._root, "\n")

trie = Trie()
for word in ["A", "to", "tea", "ted", "ten", "i", "in", "inn"]:
    trie.insert(word)

print(trie)


# The search function returns a list of all words that are less than the given
# maximum distance from the target word
def search(word, max_cost, trie):
    # build first row
    current_row = range(len(word) + 1)

    results = []

    # recursively search each branch of the trie
    for letter in trie._children:
        search_recursive(
            trie._children[letter], 
            letter, 
            word, 
            current_row, 
            results, 
            max_cost
        )

    return results

# This recursive helper is used by the search function above. It assumes that
# the previous_row has been filled in already.
def search_recursive(node, letter, word, previous_row, results, max_cost):
    columns = len(word) + 1
    current_row = [previous_row[0] + 1]

    # Build one row for the letter, with a column for each letter in the target
    # word, plus one for the empty string at column 0
    for column in range(1, columns):
        insert_cost = current_row[column - 1] + 1
        delete_cost = previous_row[column] + 1

        if word[column - 1] != letter:
            replace_cost = previous_row[ column - 1 ] + 1
        else:                
            replace_cost = previous_row[ column - 1 ]

        current_row.append(min(insert_cost, delete_cost, replace_cost))

    # if the last entry in the row indicates the optimal cost is less than the
    # maximum cost, and there is a word in this trie node, then add it.
    if current_row[-1] <= max_cost and node._word != None:
        results.append((node._word, current_row[-1]))

    # if any entries in the row are less than the maximum cost, then 
    # recursively search each branch of the trie
    if min(current_row) <= max_cost:
        for letter in node._children:
            search_recursive(
                node._children[letter], letter, word, current_row, 
                results, max_cost 
            )

start = time.time()
results = search("apple", 100, trie)
end = time.time()

for result in results: 
    print(result)      
