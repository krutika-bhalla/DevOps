import nltk
from nltk.tokenize import word_tokenize
text = "I am Vengeance, I am Batman, With great power comes great responsibility , I am the best avenger and I am limited by the technology of my time."

print("----Stemming----")

from nltk.stem import PorterStemmer
ps = PorterStemmer()
tokenized_text = word_tokenize(text)
for token in tokenized_text:
	print("Stemming of ",token, "is: ", ps.stem(token))
print()

print("----Lemmatization----")
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
for tokenwnl in tokenized_text:
	print("Lemma of ", tokenwnl, "is: ", wnl.lemmatize(tokenwnl))
print()

print("----Stopwords----")
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
print(text)
print(tokenized_text)
filtered_text = []
for tokensw in tokenized_text:
	if tokensw not in stop_words:
		print(tokensw)
		filtered_text.append(tokensw)
print(filtered_text)
print()

print("----Bigrams_Probab----")
bigrams = list(nltk.bigrams(tokenized_text))
no_of_bg = len(bigrams)
print("Total Bigrams: ", no_of_bg)
i_count = text.count("I")
print("Count of I: ", i_count)
i_am_count = bigrams.count(('I', 'am'))
print("Probab of I given AM: ", i_count/i_am_count)
print("Probab of bigram I-AM: ", i_am_count/no_of_bg)
print()

print("----POS_Tagging----")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
postagg = nltk.pos_tag(tokenized_text)
for pos in postagg:
	print("POS of pos is: ", postagg)
print()

print("----WSD----")

from nltk.wsd import lesk
sentence = "I depositted money in the bank."
toks = word_tokenize(sentence)
ambig = "bank"
print(lesk(toks, ambig).definition())











