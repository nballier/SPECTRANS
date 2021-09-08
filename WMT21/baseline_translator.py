"""
Translate the test set (in `blind_test.eng`) using HuggingFace standard `translation pipeline'

The script relies on the HuggingFace library
"""
from tqdm import tqdm
from transformers import pipeline


translator = pipeline("translation_en_to_fr")

data = open("blind_test.eng")

sentences = [s.strip() for s in data.readlines()]

with open("baseline.pred.fra", "wt") as ofile:
    for sent in tqdm(sentences):
        res = translator(sent, max_length=400)
        ofile.write(res[0]['translation_text'])
        ofile.write("\n")

