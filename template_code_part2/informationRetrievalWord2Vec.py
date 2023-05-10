import math
import numpy
from util import *
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
# Add your import statements here
from wikipedia2vec import Wikipedia2Vec
from sklearn.model_selection import GridSearchCV
import time
class InformationRetrievalWord2Vec():

    def __init__(self):
        self.index = None
        self.docIDs = None
        self.numDocs = 0
        self.docs = None

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
        self.docs = docs
        self.docIDs = docIDs
        self.numDocs = len(docIDs)
        index = []
        for idx in range(self.numDocs):
            document = self.docs[idx]
            documentId = self.docIDs[idx]
            y = ""
            for sentence in document:
                x = " ".join(sentence)
                # for word in sentence:
                if y=="":
                    y=x
                else:
                    y = y + " " + x
            index.append(y.split(" "))

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
            queryId = queries_Ids[idx]
            y = ""
            for sentence in query:
                x = " ".join(sentence)
                # for word in sentence:
                if y=="":
                    y=x
                else:
                    y = y + " " + x
            self.index.append(y.split(" "))
        # Create word_vectors using TF*IDF Values
        # for x in self.index:
            # print(x)
        # return
        start = time.time()
        vec_size = 500
        model = gensim.models.Word2Vec(min_count = 1,window=6,vector_size = vec_size,sg=1)
        model.build_vocab(self.index)
        model.train(self.index, total_examples=model.corpus_count, epochs=500, report_delay=1)
        end = time.time()
        print("time taken to train the model ",(end-start)/60,'mins')
        # model1 = gensim.models.Word2Vec(self.index, min_count=1, vector_size=vec_size, window=3, sg=0)
        # model2 = gensim.models.Word2Vec(self.index, min_count=1, vector_size=vec_size, window=3, sg=1)
        
        # print(model.wv["approximate"])
        # return

        word_vectors = {key : [0 for i in range(vec_size)] for key in queries_Ids+self.docIDs}
        for idx in range(len(queries)):
            query = queries[idx]
            queryId = queries_Ids[idx]
            for sentence in query:
                for word in sentence:
                    for i in range(vec_size):
                        word_vectors[queryId][i]+=model.wv[word][i]
        for idx in range(self.numDocs):
            document = self.docs[idx]
            documentId = self.docIDs[idx]
            for sentence in document:
                for word in sentence:
                    for i in range(vec_size):
                        word_vectors[documentId][i]+=model.wv[word][i]
        for id in self.docIDs + queries_Ids:
            res = math.sqrt(sum(map(lambda i: i * i, word_vectors[id])))
            if res != 0:
                word_vectors[id] = (numpy.array(word_vectors[id]) / res).tolist()
        for queryId in queries_Ids:
            cosine_product = []
            for documentId in self.docIDs:
                product = sum([x * y for x, y in zip(word_vectors[queryId], word_vectors[documentId])])
                cosine_product.append(product)
            sorted_order = numpy.argsort(-numpy.array(cosine_product))
            docsOrder = (numpy.array(self.docIDs)[sorted_order]).tolist()
            doc_IDs_ordered.append(docsOrder)
        return doc_IDs_ordered



