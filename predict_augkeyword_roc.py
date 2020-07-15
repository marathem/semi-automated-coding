# -*- coding: utf-8 -*-

"""This script applies the augmented keyword technique a text (or set of texts)
	at score thresholds ranging from [0.0--0.5] with a 0.01 increment. For
	each threshold value, it outputs the precision, recall, F-score, and 
	Scott's Pi values. This produces a tabulated ROC curve per text that
    helps in choosing a score threshold as follows.

   The score threshold dictates the selectivity of the augmented keyword 
   technique's predictions: predicted annotations with scores below 
   the threshold are ignored for improved precision. Typically, this
   improved precision comes at the cost of a decrease in recall. The 
   tabulated ROC curve helps us choose a threshold that achieves a balance
   between precision, recall, and Scott's pi. This threshold can then
   be used as the command-line input for predict_augkeyword.py.

   This script does not generate any textual output other than the ROC curve.

   Modify the values of these variables to customize input and output:
   	- base_dir: path to the folder containing the raw and gold standard data.
   		The raw data should be in a folder named 'uncoded' within base_dir.
   		Raw data should contain turn-of-dialogue annotations in the BRAT format.
   	- coder: name of the researcher whose coding we treat as gold standard.
   		The gold standard data should be in a folder named 'coded_{coder}' 
   		within base_dir. Gold standard annotations should follow the BRAT
   		format. The gold standard folder is also expected to contain four 
   		configuration files as follows.
   			- annotation.conf (codebook): this is the list of all codes
   			- visual.conf (codebook visuals): human-readable versions of codes
   			- code_ignore.conf: the list of codes to ignore
   			- code_keywords.conf: the list of augmented keywords
   	- transcripts: a list of file names, one per text file to be coded. The
   		appropriate un-coded and coded versions of each file must respectively 
   		be present within base_dir/uncoded and base_dir/coded_{coder}.

   	The repository provides sample input and output as exemplars.
   	The sample input is in data/uncoded (raw data) and data/coded_jane (gold 
   	standard data coded by Jane). Configuration file examples
   	are included too; see data/coded_jane for formatting instructions.

   	The most important configuration file for the augmented keyword technique
   	is code_keywords.conf with the keywords used to augment the codebook!

   	See here for information about BRAT: https://brat.nlplab.org/index.html

"""

import codecs
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import scipy
import operator
import os
import sys

from utils.indexer_utils import read_codes_stopstem, read_annotations2, read_ignored_codes2
from utils.indexer_utils import read_annotations_special2, scotts_pi
from utils.entities import Annotation

base_dir = 'data'
coder = 'jane'
transcripts = ['1_token']

encoding = 'utf-8'

INVALID_NUMBER = -123.0

input_dir = os.path.join(base_dir, 'uncoded')
gold_dir = os.path.join(base_dir, 'coded_{0}'.format(coder))

gold_annconf_file = os.path.join(gold_dir, 'annotation.conf')
gold_visconf_file = os.path.join(gold_dir, 'visual.conf')
ignored_codes_file = os.path.join(gold_dir, 'code_ignore.conf')
code_keywords_file = os.path.join(gold_dir, 'code_keywords.conf')

# What we're trying to do here is to build a precision-recall ROC-type curve.
# So by varying the value of the threshold from say 0-1, what are the different
# P-R values we get. We ideally want one P-R curve per code.

# Returns precision, recall, and f-value given the 
# number of correct predictions, the total number predicted, and 
# the number of gold-standard annotations 
def calc_prf(num_correct, num_pred, num_gold):
	precision = 0.0
	recall = 0.0
	fscore = 0.0
	
	if num_gold == 0:
		recall, fscore = INVALID_NUMBER, INVALID_NUMBER
	elif num_pred != 0:
		precision, recall = num_correct/(1.0*num_pred), num_correct/(1.0*num_gold)
		if precision + recall == 0.0:
			fscore = INVALID_NUMBER
		else:
			fscore = 2*precision*recall/(precision+recall)	
	
	return precision, recall, fscore

# We won't create any output files by default, just because waste of time.
def score_roc(predicted_anns, gold_anns, threshold=0.0):
	total_exmatch = 0
	exmatches_by_code = dict()
	pred_by_code = dict()
	gold_by_code = dict()
	bm_scottspi_dict = dict()
	num_gold = len(gold_anns)

	predicted_anns = [ann for ann in predicted_anns if ann.match_score >= threshold]
	num_pred = len(predicted_anns)

	# build dictionary of number of gold annotations by code
	for gold_ann in gold_anns:
		code = gold_ann.code.encode(encoding)
		gold_by_code[code] = gold_by_code.get(code, 0) + 1

	# count total number of exact matches & build dictionaries of number of predicted annotations by code, 
	# and number of exact matches by code
	for pred_ann in predicted_anns:
		pred_code, pred_start, pred_end = pred_ann.code.encode(encoding), pred_ann.start, pred_ann.end

		exmatches_by_code[pred_code] = exmatches_by_code.get(pred_code, 0)
		pred_by_code[pred_code] = pred_by_code.get(pred_code, 0) + 1

		for gold_ann in gold_anns:
			if pred_code != gold_ann.code.encode(encoding):
				continue

			gold_start, gold_end = gold_ann.start, gold_ann.end
			if (pred_start == gold_start and pred_end == gold_end) or (pred_start <= gold_start and pred_end >= gold_end):
				total_exmatch += 1
				exmatches_by_code[pred_code] += 1

	macro_precision, macro_recall, macro_fscore = calc_prf(total_exmatch, num_pred, num_gold)

	all_codes = set(gold_by_code.keys()).union(pred_by_code.keys())
	micro_prf = dict()
	coded_dict = dict()
	for code in all_codes:
		code_pred = pred_by_code.get(code, 0)
		code_gold = gold_by_code.get(code, 0)
		code_correct = exmatches_by_code.get(code, 0)
		micro_prf[code] = (code_pred, code_correct, calc_prf(code_correct, code_pred, code_gold))

		if code_gold > 0 or code_pred > 0:
			coded_dict[code] = (code_gold, code_pred)
			bm_scottspi_dict[code] = [code_gold, code_pred]

	scottspi = scotts_pi(coded_dict)
	bm_scottspi = scotts_pi(bm_scottspi_dict)

	return num_pred, total_exmatch, macro_precision, macro_recall, macro_fscore, micro_prf, scottspi, bm_scottspi


def start_roc(input_text_file, input_ann_file, gold_ann_file, gold_annconf_file, gold_visconf_file, code_keywords_file, ignored_codes_file,
	filter_stopwords=True, stem=False, tf=False, idf=False, cosine=True, write_output_files=False):		
	
	english_stopwords = set(stopwords.words('english'))
	stemmer = PorterStemmer()

	codes, filtered_codes, stemmed_codes, codes_plus, orig_codes = read_codes_stopstem(gold_annconf_file, 
		english_stopwords, stemmer, ignored_codes_file, code_keywords_file)

	codes_to_ignore = read_ignored_codes2(ignored_codes_file)

	input_anns = read_annotations_special2(input_ann_file, english_stopwords, stemmer)
	corpus = None
	if stem:
		corpus = [c.stemmed_ftext for c in input_anns]
	elif filter_stopwords:
		corpus = [c.filtered_text for c in input_anns]
	else:
		corpus = [c.text for c in input_anns]

	#print len(corpus)

	vectorizer = None
	if filter_stopwords:
		vectorizer = CountVectorizer(stop_words='english')
	else:
		vectorizer = CountVectorizer()
	corpus_matrix = vectorizer.fit_transform(corpus)

	tfidfer = None
	if tf or idf:
		tfidfer = TfidfTransformer(use_idf=idf)
		corpus_matrix = tfidfer.fit_transform(corpus_matrix)

	inverted_vocab = {v:k for k, v in vectorizer.vocabulary_.items()}

	codes_in_use = codes
	if stem:
		codes_in_use = stemmed_codes
	elif filter_stopwords:
		codes_in_use = filtered_codes

	#print len(codes_in_use)

	code_matrix = vectorizer.transform(codes_in_use)
	if tf or idf:
		code_matrix = tfidfer.transform(code_matrix)

	print 'Vectorized the corpus and codes'

	similarity = None
	if cosine:
		similarity = cosine_similarity(code_matrix, corpus_matrix)
	else:
		similarity = np.multiply(code_matrix, np.transpose(corpus_matrix))

	code_chunk_dict = dict()
	nonzero_rows, nonzero_cols = similarity.nonzero()
	for i, j in zip(nonzero_rows, nonzero_cols):
		v = similarity[i, j]
		code = orig_codes[i]
		if code not in code_chunk_dict:
			code_chunk_dict[code] = list()
		code_chunk_dict[code].append((j, v))

	print 'Predicted {0} unique codes'.format(len(code_chunk_dict))

	algo_anns = list()
	for _iter1, (code, results) in enumerate(code_chunk_dict.items()):
		for _iter2, result in enumerate(sorted(results, key=operator.itemgetter(1), reverse=True)):
			chunk = input_anns[result[0]]
			score = result[1]
			ann = Annotation('T{0}-{1}'.format(_iter1, _iter2) ,chunk.start, chunk.end, chunk.text, chunk.filtered_text, chunk.stemmed_ftext, 
									code=str(code).lower().replace('-', ' '), match_score=score, orig_code=str(code))
			algo_anns.append(ann)

	#algo_anns = [a for a in algo_anns if not codes_to_ignore or a.code.encode(encoding) not in codes_to_ignore]
	gold_anns = read_annotations2(gold_ann_file)
	gold_anns = [g for g in gold_anns if not codes_to_ignore or g.code.encode(encoding) not in codes_to_ignore]

	print 'Threshold\tNum Pred\tNum Correct\tPrecision\tRecall\t\tF-score\t\tScott\'s pi'
	for t in np.arange(0, 0.5, 0.01):
		num_gold, num_pred, macro_precision, macro_recall, macro_fscore, micro_prf, scottspi, bm_scottspi = score_roc(algo_anns, gold_anns, threshold=t)
		if bm_scottspi is None:
			bm_scottspi = INVALID_NUMBER
		if scottspi is None:
			scottspi = INVALID_NUMBER
		print '{0:.2f}\t\t{1}\t\t{2}\t\t{3:.2f}\t\t{4:.2f}\t\t{5:.2f}\t\t{6:.2f}'.format(t, num_gold, num_pred, 100*macro_precision, 100*macro_recall, 100*macro_fscore, bm_scottspi)


	#if write_output_files:
	#	create_output_folder(output_path, output_folder, input_text_file, gold_annconf_file, gold_visconf_file)
	#	write_annotations(algo_annotations, output_path, output_folder, output_ann_file, no_text=True)
	#	write_spreadsheet(input_anns, algo_annotations, output_path, output_folder, output_xlsx_file)


for transcript in transcripts:	
	print transcript
	input_ann_file = os.path.join(input_dir, '{0}.ann'.format(transcript))
	input_text_file = os.path.join(input_dir, '{0}.txt'.format(transcript))
	gold_ann_file = os.path.join(gold_dir, '{0}.ann'.format(transcript))
	start_roc(input_text_file, input_ann_file, gold_ann_file, gold_annconf_file, gold_visconf_file, 
		code_keywords_file, ignored_codes_file)



