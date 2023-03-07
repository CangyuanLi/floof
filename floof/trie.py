class TrieNode:

    def __init__(self):
        self.is_end: bool = False
        self.children: dict = {}

    def is_empty(self):
        return bool(self.children)

class Trie:
    """
    The Trie data structure basically holds common prefixes. For example, if we insert "apple" and
    "appli" into the Trie, it would look like:
                a
                p
                p
                l
            e       i
    This will be useful for our fuzzymatching because we can avoid doing some work. We don't need
    to recalculate edit distance for each common substring. This simple Trie class only implements
    insert, a helper function to turn the trie into a dict of dicts, and an iterator for convenience.
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node: TrieNode = self.root
        for char in word:
            if char in node.children: 
                node = node.children[char]
            else:
                # If we don't find the character, create a new node
                child = TrieNode()
                node.children[char] = child
                node = child

        node.is_end = True

    def __iter__(self):
        def recur(node: TrieNode):
            for k, v in node.children.items():
                if v.is_end and not v.is_empty():
                    yield k, v
                else:
                    yield k, v
                    yield from recur(v)

        return recur(self.root)

    def asdict(self) -> dict:
        def recur(node: TrieNode):
            dct = {}
            for k, v in node.children.items():
                if v.is_end and not v.is_empty():
                    dct[k] = v.children
                else:
                    dct[k] = recur(v)

            return dct

        return recur(self.root)
