import nltk

from util import *
import string
# Add your import statements here
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer

class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""
		stop_words = set(stopwords.words('english'))
		for x in string.punctuation:
			stop_words.add(x)
		stopwordRemovedText = [[word for word in sentence if word.lower() not in stop_words] for sentence in text]

		#Fill in code here

		return stopwordRemovedText




	