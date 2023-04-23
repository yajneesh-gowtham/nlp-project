import math

from util import *

# Add your import statements here

from operator import eq
import pandas

class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		#precision = retrieved ^ relevant / retrieved

		precision = -1
		retrieved_relevant = 0
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				retrieved_relevant+=1
		retrieved = k
		precision = retrieved_relevant / retrieved
		#Fill in code here
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = 0
		for idx in range(len(query_ids)):
			trueDocIds = []
			for i in range(len(qrels)):
				if int(qrels[i]["query_num"])==query_ids[idx]:
					trueDocIds.append(int(qrels[i]["id"]))
			meanPrecision+=self.queryPrecision(doc_IDs_ordered[idx],query_ids[idx],trueDocIds,k)
		#change qrels
		meanPrecision = meanPrecision/len(query_ids)
		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1
		retrieved_relevant = 0
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				retrieved_relevant += 1
		relevant = len(true_doc_IDs)
		recall = retrieved_relevant / relevant
		#Fill in code here

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""


		meanRecall = 0
		for idx in range(len(query_ids)):
			trueDocIds = []
			for i in range(len(qrels)):
				if int(qrels[i]["query_num"]) == query_ids[idx]:
					trueDocIds.append(int(qrels[i]["id"]))
			meanRecall += self.queryRecall(doc_IDs_ordered[idx], query_ids[idx], trueDocIds, k)
		# change qrels
		meanRecall = meanRecall / len(query_ids)
		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1
		beta = 1.0
		betaSquare = beta**2
		precision = self.queryPrecision(query_doc_IDs_ordered,query_id,true_doc_IDs,k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		if precision==0 or recall==0:
			fscore = 0
		else:
			fscore = (1+betaSquare)*(precision)*(recall)/(betaSquare*precision+recall)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = 0
		for idx in range(len(query_ids)):
			trueDocIds = []
			for i in range(len(qrels)):
				if int(qrels[i]["query_num"]) == query_ids[idx]:
					trueDocIds.append(int(qrels[i]["id"]))
			meanFscore += self.queryFscore(doc_IDs_ordered[idx], query_ids[idx], trueDocIds, k)
		meanFscore = meanFscore / len(query_ids)

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = 0
		DCG =0
		iDCG =0

		true_doc_IDs = sorted(true_doc_IDs,key=lambda x: x[1])
		k = min(k,len(true_doc_IDs))
		for i in range(k):
			score = 0
			for idx in range(len(true_doc_IDs)):
				if query_doc_IDs_ordered[i]==true_doc_IDs[idx][0]:
					score = 5-true_doc_IDs[idx][1]
					break
			DCG+=(1/math.log2(i+2))*(score)
		for i in range(k):
			iDCG+=(1/math.log2(i+2))*(5-true_doc_IDs[i][1])
		nDCG = DCG/iDCG

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = 0
		for idx in range(len(query_ids)):
			trueDocIds = []
			for i in range(len(qrels)):
				if int(qrels[i]["query_num"]) == query_ids[idx]:
					trueDocIds.append([int(qrels[i]["id"]),int(qrels[i]["position"])])
			meanNDCG+=self.queryNDCG(doc_IDs_ordered[idx],query_ids[idx],trueDocIds,k)
		meanNDCG = meanNDCG/len(query_ids)
		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = 0
		length=0
		retrieved_relevant = 0
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				retrieved_relevant+=1
				avgPrecision+=(retrieved_relevant)/(i+1)
				length+=1
		if length!=0:
			avgPrecision = avgPrecision/length
		#Fill in code here

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = 0

		for idx in range(len(query_ids)):
			trueDocIds = []
			for i in range(len(q_rels)):
				if int(q_rels[i]["query_num"]) == query_ids[idx]:
					trueDocIds.append(int(q_rels[i]["id"]))
			meanAveragePrecision+=self.queryAveragePrecision(doc_IDs_ordered[idx],query_ids[idx],trueDocIds,k)
		meanAveragePrecision = meanAveragePrecision/len(query_ids)
		return meanAveragePrecision

