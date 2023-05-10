import math
import numpy
from util import *
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import wordnet as wn
# Add your import statements here
from nltk.corpus import wordnet_ic
from nltk.corpus import genesis
class InformationRetrievalGVSM():

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
        index = {}
        for idx in range(self.numDocs):
            document = docs[idx]
            documentId = docIDs[idx]
            for sentence in document:
                for word in sentence:
                    if word not in index.keys():
                        index[word] = {}
                    if documentId not in index[word].keys():
                        index[word][documentId] = 0
                    index[word][documentId] += 1

        # for idx in range(self.numDocs):
        # 	document = docs[idx]
        # 	documentId = docIDs[idx]
        # 	for sentence in document:
        # 		for word in sentence:
        # 			index[word][documentId] = self.logarithm2(index[word][documentId])

        self.index = index

    def rank(self, queries):
        print("ranking started")
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
            queryId = queries_Ids[idx]
            for sentence in query:
                for word in sentence:
                    if word not in self.index.keys():
                        continue
                    if queryId not in self.index[word].keys():
                        self.index[word][queryId] = 0
                    self.index[word][queryId] += 1

        # for idx in range(len(queries)):
        # 	query = queries[idx]
        # 	queryId = queries_Ids[idx]
        # 	for sentence in query:
        # 		for word in sentence:
        # 			self.index[word][queryId] = self.logarithm2(self.index[word][queryId])

        # Create word_vectors using TF*IDF Values

        word_vectors = {key: [] for key in queries_Ids + self.docIDs}

        N = self.numDocs
        for word in self.index.keys():
            frequency = len([key for key in self.index[word].keys() if key <= max(self.docIDs)])
            inverse_document_frequency = self.invLogarithm2(frequency, N)
            for id in word_vectors.keys():
                if id in self.index[word].keys():
                    word_vectors[id].append(self.index[word][id] * inverse_document_frequency)
                else:
                    word_vectors[id].append(0)

        # Normalize word vectors
        for id in self.docIDs + queries_Ids:
            res = math.sqrt(sum(map(lambda i: i * i, word_vectors[id])))
            if res != 0:
                word_vectors[id] = (numpy.array(word_vectors[id]) / res).tolist()

        genesis_ic = wn.ic(genesis, False, 0.0)
        Sim = [[0 for j in range(len(self.index.keys()))]for i in range(len(self.index.keys()))]
        for i,x in enumerate(self.index.keys()):
            for j,y in enumerate(self.index.keys()):
                print(i,j)
                try:
                    syn1 = wn.synsets(x)[0]
                    syn2 = wn.synsets(y)[0]
                    Sim[i][j] = syn1.lin_similarity(syn2,genesis_ic)
                    
                except:
                    Sim[i][j] = 0
        return 

        for queryId in queries_Ids:
            cosine_product = []
            for documentId in self.docIDs:
                product = sum([x * y for x, y in zip(word_vectors[queryId], word_vectors[documentId])])
                cosine_product.append(product)
            sorted_order = numpy.argsort(-numpy.array(cosine_product))
            docsOrder = (numpy.array(self.docIDs)[sorted_order]).tolist()
            doc_IDs_ordered.append(docsOrder)
        # print(doc_IDs_ordered[0])
        return doc_IDs_ordered




