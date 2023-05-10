from util import *

# Add your import statements here
import nltk.data
import re


class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		_RE_COMBINE_WHITESPACE = re.compile(r"\s+")
		text=_RE_COMBINE_WHITESPACE.sub(" ",text).strip()
		segmentedText =re.split("\. |\? |\! ",text)
		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		# print("punkt")
		sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		result=""
		for x in text:
			if x==" ":
				result+=x
			elif x=='.' or x=='?' or x=='!':
				result+=" "
			elif x.isalnum():
				result+=x
			else:
				result+=" "
		result = " ".join(result.split())

		segmentedText = sent_detector.tokenize(result.strip())
		#Fill in code here
		# segmentedText = sent_detector.tokenize(text)
		
		
		return segmentedText