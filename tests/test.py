import floof
import pandas as pd

data = {
    "name1": ["janice", "apple", "bordula"],
    "name2": ["janic", "appil", "bordula"]
}

df = pd.DataFrame(data)
lu = df["name2"]

matcher = floof.Matcher(df["name1"], df["name2"])

a = matcher.tfidf(threshold=60)
valid = set(a["name2"])

# print(lu.loc[lu.isin(valid)])

a = matcher.match(threshold=80, drop_intermediate=False)
a = matcher.match(scorers=["match_rating_codex",
    "metaphone",
    "nysiis",
    "soundex"], threshold=80, drop_intermediate=False)

print(a)