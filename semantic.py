import spacy

# Current code is using the advanced language model 'md'. 
# To test it with the simpler language model 'sm' please swap just the last to letters from 'md' to 'sm'.
# Ex: 'en_core_web_sm'
nlp = spacy.load('en_core_web_md')

# Will Access the meta attribute to display the spacy mode type used
char = "*********"
spacy_model_type = nlp.meta['name']
print(f"{char} You are currently using the spacy model type:{spacy_model_type} {char}")

# first code extract from the PDF file

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(f"{char} Similarities between {word1, word2, word3}:")

print(f"{word1} and {word2} :{word1.similarity(word2)}")
print(f"{word3} and {word2} :{word3.similarity(word2)}")
print(f"{word3} and {word1} :{word3.similarity(word1)}")

# Second code extract from the PDF file
print(f"{char} Similarities between short texts:")
tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# Third code extract from the PDF file
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

'''Cat and monkey words have over 50% in similarities. I can only assume this is because they are both animals, have fur and have life within.
The similarities between banana and monkey are about 40% if we use mathematical rounding, and there is a similarity of only 22% after rounding between banana and cat.
I believe the difference between banana and the two species relate to the fact that monkey has a strong association
with eating bananas and living near banana trees than cats, which have a stronger association with a meat die.'''

# My own example of nlp word similarities

word4 = nlp("pangolin")
word5 = nlp("komodo dragon")
word6 = nlp("red panda")
print(f"{char} Similarities between {word4, word5, word6}:")

print(f"{word4} and {word5} :{word1.similarity(word5)}")
print(f"{word6} and {word5} :{word6.similarity(word5)}")
print(f"{word6} and {word4} :{word6.similarity(word4)}")


# Running the file with the 'sm' module, note below.

'''When using the 'sm' module to run the similarity code for words, an advisory message appears stating that 
the module used "may not give useful similarity judgments" because it does not come with word vectors and only uses context-sensitive tensors. 
It is recommended to use the 'md' module or create our own word vectors for better results. However, when comparing similarities in the word string, 
the code runs and is returned without any advisory notes. '''


# References:
# 1 - Files provided with the task
# 2 - https://stackoverflow.com/questions/43071775/spacy-how-to-get-the-spacy-model-name
