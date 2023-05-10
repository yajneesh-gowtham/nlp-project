import nltk

from util import *
import string
# Add your import statements here
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
import enchant
class StopwordRemoval():
	def isNumber(self,word):
		try:
			float(word)
			return True
		except ValueError:
			return False
	
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
		stopwordRemovedText = [[word.lower() for word in sentence if word.lower() not in stop_words and self.isNumber(word.lower())==False and any(map(str.isdigit,word))==False] for sentence in text]
		# stopwordRemovedText = [[word for word in sentence ] for sentence in text]
		
		# stopwordRemovedText = [[d.suggest(word)[0] for word in l ]for l in stopwordRemovedText]
		#Fill in code here

		return stopwordRemovedText




	