# floof: simple fuzzymatching / comparison library
![PyPI - Downloads](https://img.shields.io/pypi/dm/floof)

## What is it?

**floof** is a pure Python package that makes fuzzymatching easy. Fuzzymatching is a common data task
required whenever two strings don't quite exactly match. There are many algorithms to calculate
string similarity, with dozens of disparate implementations. **floof** aims to collect all of these
in an easy-to-use package, and reduce the boilerplate needed to apply these algorithms to your
data.

# Usage:

## Dependencies

- [jarowinkler - Fast implementation of Jaro and Jaro-Winkler algorithms]
- [jellyfish - Implements many edit distance algorithms, including Damerau-Levenshtein]
- [Levenshtein - Fast C implementation of Levenshtein distance]
- [sklearn - USed to implement TFIDF]

## Installing

The easiest way is to install **floof** is from PyPI using pip:

```sh
pip install floof
```

## Running

First, import the library.

```python
import floof
```

Floof provides two classes: Comparer and Matcher. Both are instantiated the same way, taking as
arguments two Pandas Series, an "original" and a "lookup", although in practice the order doesn't
madder.

```python
matcher = floof.Matcher(original, lookup)
comparer = floof.Comparer(original, lookup)
```

All functions in the Matcher class return a crosswalk of the original strings and the best ``k`` matches
from the lookup strings. The primary convenience function is floof.Matcher().match(), which
applies several different similarity algorithms and produces a composite score. Given an example
input of:

```python
original_names = ["apple", "pear"]
lookup_names = ["appl", "apil", "prear"]
```

A matcher function would return something like:

| original_name 	| lookup_name 	| levenshtein_score 	| tfidf_score 	| final_score 	|
|---------------	|-------------	|-------------------	|-------------	|-------------	|
| apple         	| appl        	| 90                	| 80          	| 85          	|
| apple         	| apil        	| 70                	| 85          	| 77.5        	|
| pear          	| prear       	| 95                	| 90          	| 92.5        	|

The Comparer class is meant to compare strings one-to-one. That is to say, given an input of:

```python
original_names = ["apple", "pear"]
lookup_names = ["appl", "apil"]
```

A comparer function would return something like:

| levensthein_score |
|-------------------|
| 90                |
| 95                |

## Performance

Fuzzymatching can be very intense, as many algorithms are by nature quadratic. For each original string,
you must compare against all lookup strings. Therefore, **floof** is by default concurrent. It also
can perform common-sense speedups, like first removing exact matches from the pool, and using a non-quadratic
algorithm (TFIDF) to filter the pool.

# TODO:

* Allow custom scorers