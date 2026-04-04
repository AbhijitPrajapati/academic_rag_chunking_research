import re

with open("drop.txt", "r", errors="ignore") as f:
    text = f.read()

text = text.replace("\n", " ")

pattern = re.compile(r"\([^)]*\d\)")

matches = [(m.start(), m.end()) for m in pattern.finditer(text)]

for start, end in matches[::-1]:
    text = text[:start] + text[end + 1 :]

with open("drop.txt", "w") as f:
    f.write(text)
