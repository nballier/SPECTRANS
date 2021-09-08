"""
Translate the test set (in `blind_test.eng`) using Facebook MBART model.

The script relies on the HuggingFace library
"""

import spacy
from tqdm import tqdm

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX")

sentences = [s.strip() for s in open("blind_test.eng")]

with open("mbart.pred.fra", "wt") as ofile:
    for sentence in tqdm(sentences):
        model_inputs = tokenizer(sentence, return_tensors="pt")
        generated_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"])
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True) 
        ofile.write(translation[0])
        ofile.write("\n")


