import numpy as np
import os

with open("document.txt", "r") as f:
    text = f.read()

print(f"Document loaded: {len(text)} characters")