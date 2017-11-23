import re

filePath = '/home/prachi/Downloads/texttpms.txt'
with open(filePath, "r") as f:
    data = f.readlines()
for i, x in enumerate(data): print i, x
