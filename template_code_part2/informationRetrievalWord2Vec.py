import math
import numpy
from util import *
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
# Add your import statements here


class InformationRetrievalWord2Vec():

    def __init__(self):
        self.index = None
        self.docIDs = None
        self.numDocs = 0

    def logarithm2(self, frequency):
        return math.log2(1 + frequency)

    def invLogarithm2(self, frequency, total):
        return math.log2(total / frequency)

    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """
        self.docIDs = docIDs
        self.numDocs = len(docIDs)
        index = []
        for idx in range(self.numDocs):
            document = docs[idx]
            documentId = docIDs[idx]
            for sentence in document:
                for word in sentence:
                    index.append(word)
        # for idx in range(self.numDocs):
        # 	document = docs[idx]
        # 	documentId = docIDs[idx]
        # 	for sentence in document:
        # 		for word in sentence:
        # 			index[word][documentId] = self.logarithm2(index[word][documentId])

        self.index = index

    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query


        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """

        doc_IDs_ordered = []
        queries_Ids = [max(self.docIDs) + idx + 1 for idx in range(len(queries))]

        for idx in range(len(queries)):
            query = queries[idx]
            # queryId = queries_Ids[idx]
            for sentence in query:
                for word in sentence:
                        self.index.append(word)
        # for idx in range(len(queries)):
        # 	query = queries[idx]
        # 	queryId = queries_Ids[idx]
        # 	for sentence in query:
        # 		for word in sentence:
        # 			self.index[word][queryId] = self.logarithm2(self.index[word][queryId])

        # Create word_vectors using TF*IDF Values

        model1 = gensim.models.Word2Vec([self.index], min_count=1, vector_size=100, window=5, sg=0)
        model2 = gensim.models.Word2Vec([self.index], min_count=1, vector_size=100, window=5, sg=1)
        # print(self.index)
        print(model1.wv.most_similar("wing",topn = 10))
        return doc_IDs_ordered


