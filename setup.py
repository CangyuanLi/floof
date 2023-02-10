import setuptools

setuptools.setup(
    name="floof",
    version="0.0.5",
    author="Cangyuan Li",
    author_email="everest229@gmail.com",
    description="A library for fuzzymatching",
    url="https://github.com/CangyuanLi/floof",
    packages=["floof"],
    install_requires=[
        "jarowinkler",
        "jellyfish",
        "Levenshtein",
        "numpy",
        "pandas",
        "sklearn",
        "thefuzz[speedup]",
        "tqdm"
    ],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
)