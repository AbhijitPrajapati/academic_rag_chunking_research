import re
from wordsegment import load, segment

load()


def get_all_artifacts(text):
    n = 14
    pattern = rf"\b\w{{{n + 1},}}\b"
    return re.findall(pattern, text)


with open("drop.txt", "r", errors="ignore") as f:
    text = f.read()

text = text.replace("\n", " ")

art = get_all_artifacts(text)
if len(art) > 0:
    for a in art:
        text = text.replace(a, " ".join(segment(a)))

pattern = re.compile(r"\([^)]*\d\)")

matches = [(m.start(), m.end()) for m in pattern.finditer(text)]

for start, end in matches[::-1]:
    text = text[:start] + text[end + 1 :]

with open("drop.txt", "w") as f:
    f.write(text)
