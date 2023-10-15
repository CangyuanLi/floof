# floof: simple fuzzymatching library
![PyPI - Downloads](https://img.shields.io/pypi/dm/floof)

## What is it?

**floof** is a Python package that makes fuzzymatching / record linkage / entity resolution fast and easy. 
Fuzzymatching is a common data task required whenever two strings don't quite exactly match. However, it easy to run
into performance problems, especially when datasets are large. **floof** aims to provide two things. The first is
performance at scale. Most of the core logic, from the string similarity implementations to threading, is implemented
in Rust. This makes **floof** very fast and memory efficient, even compared to other libraries that implement edit
distance algorithms in lower-level languages, since they usually do not provide threading support. The second is a
simple, high-level Python API that provides a suite of different algorithms and makes matching as easy as calling a one-liner.

# Usage:

## Dependencies

- [pandas - Output ]
- [scikit-learn - Used to implement TFIDF]
- [sparse_dot_topn - Fast sparse matrix multiplication]

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

The main workhorses of **floof** are its three classes: `Comparer`, `Matcher`, and `Linker`.
In addition, the base string similarity algorithms are also exposed. It is recommended that
you only use these for prototyping and testing, as the classes are able take advantage of many
optimizations (such as Rust threading) and provide a much more ergonomic interface for
most common tasks. 

`Comparer` and `Matcher`. Both are instantiated the same way, taking as
arguments two Pandas Series, an "original" and a "lookup", although in practice the order doesn't
matter. A `Linker` class that implements Record Linkage is coming soon!

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