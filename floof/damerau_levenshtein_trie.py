import heapq

import pandas as pd
import tqdm

from .trie import Trie, TrieNode

def recursive_search(node: TrieNode, letter: str, word: str, previous_row, results):
    word_len = len(word)
    columns = word_len + 1
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
        max_len = max(word_len, len(node.word))
        pct = (max_len - current_row[-1]) / max_len
        results.append((pct, node.word))

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

def build_trie(lookup: list[str]):
    word_trie = Trie()
    for l in lookup:
        word_trie.insert(l)

    return word_trie

def _match(trie: Trie, word: str):

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

def match(original: list[str], lookup: list[str], k_matches: int=5, ncpus: int=1):
    _trie = build_trie(lookup)
    og_colname = "a"
    lu_colname = "b"
    merged = {og_colname: [], lu_colname: [], "score": []}

    with tqdm.tqdm(total=len(original)) as pbar:
        if ncpus == 1: # this is just single-threading
            for o_str in original:
                matches = _match(_trie, o_str)
                matches = heapq.nlargest(k_matches, matches)
                for score, lu_str in matches:
                    merged[og_colname].append(o_str)
                    merged[lu_colname].append(lu_str)
                    merged["score"].append(score)

                pbar.update(1)

    merged = pd.DataFrame(merged)

    return merged

# original = ["apple", "appl", "bannan"]
# lookup = ["apple", "ascit", "one dane", "lickitysplicity"]
# match(original, lookup, 5, 1)