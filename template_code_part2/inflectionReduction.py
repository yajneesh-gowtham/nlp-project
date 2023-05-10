from util import *

# Add your import statements here
import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer


class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""
		lemmatizer = WordNetLemmatizer()
		reducedText = [[lemmatizer.lemmatize(word) for word in string]for string in text]
		# reducedText = [[word for word in string]for string in text]

		#Fill in code here
		
		return reducedText


