#!/usr/local/bin/python3
## SCRIPT FOR MAUD BENARD DESIGNED BY JEAN-BAPTISTE YUNES (and Nicolas Ballier): complex NP extractor


# Usage:
# script-name text-file [...]
# Output : output.csv

# inspired from Advanced Natural Language Processing
#from https://raw.githubusercontent.com/Apress/natural-language-processing-recipes/master/Chapter%204.ipynb
#Akshay Kulkarni, Adarsha Shivananda (2019) Natural Language Processing Recipes: Unlocking Text Data with Machine Learning and Deep Learning using Python. Aap Press.

# display_as_table(['1gram', '2gram', '3gram', '4gram', '5gram', '6gram', '7gram', '8gram'], np)


#Import libraries
import spacy
import os.path
import sys
import csv

# For new infix tokenization (hyphens are part of a single word)
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

nlp = spacy.load("en_core_web_sm")

# Tokenizer patch (from spaCy docs)
infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        # ✅ Commented out regex that splits on hyphens between letters:
        # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS), ### JBY expression that takes (roughly) word-hyphen-word and tag each...
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)
# Install the patch into the engine
infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer

def display_as_table(row_headers,dict):
    if not row_headers:
        row_headers = sorted(dict.keys())
    for header in row_headers:
        print('--------------{}-gram'.format(header))
        if header in dict:
            for ngram in dict[header]:
                print(ngram)
    return

def save_as_csv(filename,result):
#    if os.path.exists(filename):
#        raise FileExistsError(filename)
#        return
    print("Generating csv file {}".format(filename))
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['n','ngram','tags','sentence','file','left grams','right grams']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row_header in sorted(result.keys()):
            data = result[row_header]
            for d in data:
                left = ""
                for w in d[5]:
                    left = left+"("+w+")"
                right = ""
                for w in d[6]:
                    right = right+"("+w+")"
                writer.writerow({'n':d[0], 'ngram':d[1], 'tags': d[2], 'sentence':d[3] ,'file':d[4], 'left grams':left ,'right grams':right})
    return

def get_noun_phrases_old(blob):
    return blob.noun_phrases

def get_noun_phrases(blob):
    noun_phrases = []
    state = 0
    current_noun_phrase = []
    print(blob.tags)
    for tagged_word in blob.tags:
        if tagged_word[1].startswith('NN'):
            if state == 0:
                state = 1
            current_noun_phrase.append(tagged_word[0])
        else:
            if state == 1:
                state = 0
                noun_phrases.append(' '.join(current_noun_phrase))
                current_noun_phrase = []
    if state == 1:
        noun_phrases.append(' '.join(current_noun_phrase))
    return noun_phrases

def get_tagged_noun_phrases(blob):
    """Get a text from spaCy processor and extract sequences of NN* (words and tags)
    Returned value is a couple of lists. First member is a list of noun-phrases and second a list of corresponding noun-tags.
    Ex. : "This how-to paper guides the computationally literate linguist from the data modelling to the actual web-interface for the deliverables of her linguistic project."
    returns "['how-to paper', 'linguist', 'data modelling', 'web-interface', 'deliverables', 'project'] ['NNP NN', 'NN', 'NNS NN', 'NN', 'NNS', 'NN']"

    blob -- spaCy processed text.
    """
    noun_phrases = []
    tag_phrases = []
    state = 0
    current_noun_phrase = []
    current_tag_phrase = []
#    print(">>>>>>>>>>>>>", blob)
#    print("-------------", [(token.text,token.tag_) for token in blob])
    for tagged_word in blob:
        if tagged_word.tag_.startswith('NN'):
            if state == 0:
                state = 1
            current_noun_phrase.append(tagged_word.text)
            current_tag_phrase.append(tagged_word.tag_)
        else:
            if state == 1:
                state = 0
                noun_phrases.append(' '.join(current_noun_phrase))
                tag_phrases.append(' '.join(current_tag_phrase))
                current_noun_phrase = []
                current_tag_phrase = []
    if state == 1:
        noun_phrases.append(' '.join(current_noun_phrase))
        tag_phrases.append(' '.join(current_tag_phrase))
    print(noun_phrases,tag_phrases)
    print("+++++++++++++++++++++")
    return (noun_phrases,tag_phrases)


print (sys.argv)
if len(sys.argv) == 1:
    raise Exception("Il manque des noms de fichiers texte en argument")
result = {}
for filename in sys.argv[1:]:
    with open(filename,"r") as file:
        print("Analyzing file {}...".format(filename))
        for line in file:
            sentences = line.split('.')             # cut a line into sentences (rough approx.).
            for sentence in sentences:
                blob = nlp(sentence+'.');
                tagged_noun_phrases = get_tagged_noun_phrases(blob) # construct a coupe of list of noun-phrases and list of corresponding POS-tags
                for (np,tp) in zip(tagged_noun_phrases[0],tagged_noun_phrases[1]): # iterate over both lists at the same time
                    n_in_ngram = len(np.split())    # compute n in ngram
                    if n_in_ngram == 1:             # ignore 1-grams
                        continue
                    if n_in_ngram not in result:    # if n is a first occurence
                        result[n_in_ngram] = []     # add a list to contain the values
                    left = []
                    right = []
                    for l in range(n_in_ngram-1,1,-1): # construct prefix and suffix noun-phrases from the original one
                        for i in range(n_in_ngram-l):
                            left_element = ' '.join(np.split()[i:i+l])
                            left.append(left_element)
                        right_element = ' '.join(np.split()[n_in_ngram-l:n_in_ngram])
                        right.append(right_element)
                    result[n_in_ngram].append((n_in_ngram,np,tp,sentence,filename,left,right))
                
#display_as_table([1,6],result)             # to print explicits ngram values
display_as_table([],result)                 # DEBUG prints ngrams
save_as_csv('output.csv',result)
