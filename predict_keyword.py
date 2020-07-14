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
from utils.indexer_utils import read_annotations_special2, score_advanced, score, write_spreadsheet_special
from utils.entities import Annotation

class AnnotationResult:
	annotation = None
	algo_codes = set()
	correct_codes = set()
	wrong_codes = set()
	missed_codes = set()

encoding = 'utf-8'

INVALID_NUMBER = -123.0

coder = 'jane'
base_dir = 'data'
input_dir = os.path.join(base_dir, 'uncoded')
gold_dir = os.path.join(base_dir, 'coded_{0}'.format(coder))
result_dir = 'results'
output_dir = os.path.join(result_dir, '{0}_{1}'.format('simplek', coder))

transcripts = ['1_token']

score_threshold = float(str(raw_input('Threshold: ')).strip())

gold_annconf_file = os.path.join(gold_dir, 'annotation.conf')
gold_visconf_file = os.path.join(gold_dir, 'visual.conf')
ignored_codes_file = os.path.join(gold_dir, 'code_ignore.conf')
output_xlsx_pattern = '{0}_result.xlsx'
stem = False
filter_stopwords = True
tf = idf = False
cosine=True

avg_precision, avg_recall, avg_fscore = 0.0, 0.0, 0.0
avg_bm_scottspi = 0.0
all_results = list()
global_bm_scottspi_dict = dict()

for transcript in transcripts:	
	input_ann_file = os.path.join(input_dir, '{0}.ann'.format(transcript))
	input_text_file = os.path.join(input_dir, '{0}.txt'.format(transcript))
	gold_ann_file = os.path.join(gold_dir, '{0}.ann'.format(transcript))
	output_xlsx_file = os.path.join(output_dir, output_xlsx_pattern.format(transcript))

	english_stopwords = set(stopwords.words('english'))
	stemmer = PorterStemmer()

	codes, filtered_codes, stemmed_codes, codes_plus, orig_codes = read_codes_stopstem(gold_annconf_file, 
		english_stopwords, stemmer, ignored_codes_file, None)

	codes_to_ignore = read_ignored_codes2(ignored_codes_file)

	gold_anns = read_annotations2(gold_ann_file)
	gold_anns = [g for g in gold_anns if not codes_to_ignore or g.code.encode(encoding) not in codes_to_ignore]
	gold_codes_to_anns = dict()
	for gold_ann in gold_anns:
		code = gold_ann.code
		if code not in gold_codes_to_anns:
			gold_codes_to_anns[code] = list()
		gold_codes_to_anns[code].append(gold_ann)

	input_anns = read_annotations_special2(input_ann_file, english_stopwords, stemmer)
	corpus = None
	if stem:
		corpus = [c.stemmed_ftext for c in input_anns]
	elif filter_stopwords:
		corpus = [c.filtered_text for c in input_anns]
	else:
		corpus = [c.text for c in input_anns]

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

	bm_scottspi_dict = dict()
	for code in codes:
		code_gold = len(gold_codes_to_anns.get(code, []))
		if code_gold > 0:
			bm_scottspi_dict[code] = [code_gold, 0]

	algo_anns = list()
	inputannid_to_results = dict()
	for _iter1, (code, results) in enumerate(code_chunk_dict.items()):
		for _iter2, result in enumerate(sorted(results, key=operator.itemgetter(1), reverse=True)):
			chunk = input_anns[result[0]]
			score = result[1]
			if score >= score_threshold:
				input_ann_id = chunk._id

				# Create annotation and add to list
				ann = Annotation('T{0}-{1}'.format(_iter1, _iter2), chunk.start, chunk.end, chunk.text, chunk.filtered_text, chunk.stemmed_ftext, 
									code=str(code).lower().replace('-', ' '), match_score=score, orig_code=str(code))
				ann._inputannid = input_ann_id
				algo_anns.append(ann)

	for ann in algo_anns:
		# Create annotation result and add to dict
		input_ann_id = ann._inputannid
		algoresult = inputannid_to_results.get(input_ann_id, None)
		if algoresult is None:
			algoresult = AnnotationResult()
			algoresult.correct_codes = set()
			algoresult.wrong_codes = set()
			algoresult.annotation = chunk
			inputannid_to_results[input_ann_id] = algoresult

		pred_code = ann.code
		pred_start, pred_end = ann.start, ann.end
		bm_entry = bm_scottspi_dict.get(pred_code, [0, 0])
		bm_entry[1] += 1
		bm_scottspi_dict[pred_code] = bm_entry

		m_gold_anns = gold_codes_to_anns.get(pred_code, None)
		correct = False
		if m_gold_anns:
			for m_gold_ann in m_gold_anns:
				gold_start, gold_end = m_gold_ann.start, m_gold_ann.end
				if (pred_start == gold_start and pred_end == gold_end) or (pred_start <= gold_start and pred_end >= gold_end):
					# Exact match!
					correct = True
					algoresult.correct_codes.add(pred_code)
							
					# increment second value of bm_scottspi_dict entry
					#bm_entry = bm_scottspi_dict.get(pred_code, [0, 0])
					#bm_entry[1] += 1
					#bm_scottspi_dict[pred_code] = bm_entry
					break

		if not correct:
			algoresult.wrong_codes.add(pred_code)

	for key, val in bm_scottspi_dict.items():
		print key, val
	
	input_anns = read_annotations2(input_ann_file) # re-read input annotations without question text

	results = list()
	for input_ann in input_anns:
		result = AnnotationResult()
		result.annotation = input_ann
		result.missed_codes = set()
		result.correct_codes = set()
		result.wrong_codes = set()
		results.append(result)
		_id = input_ann._id
		start, end = input_ann.start, input_ann.end

		for gold_ann in gold_anns:
			gold_start, gold_end = gold_ann.start, gold_ann.end
			if (start == gold_start and end == gold_end) or (start <= gold_start and end >= gold_end):
				result.missed_codes.add(gold_ann.code)

		pred_result = inputannid_to_results.get(_id, None)
		if pred_result:
			result.correct_codes = pred_result.correct_codes
			result.wrong_codes = pred_result.wrong_codes
			result.missed_codes = result.missed_codes - pred_result.correct_codes


	write_spreadsheet_special(results, output_xlsx_file)
	print '{1} Wrote {0}'.format(output_xlsx_file, transcript)

	all_results.extend(results)

	for code, entry in bm_scottspi_dict.items():
		if code not in global_bm_scottspi_dict:
			global_bm_scottspi_dict[code] = [0, 0]
		
		global_bm_scottspi_dict[code][0] += entry[0]
		global_bm_scottspi_dict[code][1] += entry[1]

	precision, recall, fscore, scottspi, bm_scottspi = score_advanced(results, None, bm_scottspi_dict)

	if precision is not None:
		avg_precision += precision
		print '{1} Precision:\t{0:.2f}%'.format(precision, transcript)
	else:
		print '{0} Precision:\tn/a'.format(transcript)
	if recall is not None:
		avg_recall += recall
		print '{1} Recall:\t\t{0:.2f}%'.format(recall, transcript)
	else:
		print '{0} Recall:\t\tn/a'.format(transcript)
	if precision is not None and recall is not None and fscore is not None:
		avg_fscore += fscore
		print '{1} F-score:\t{0:.2f}'.format(fscore, transcript)
	else:
		print '{0} F-score:\tn/a'.format(transcript)

	if bm_scottspi is not None:
		avg_bm_scottspi += bm_scottspi
		print '{1} BM Scott\'s pi:\t{0:.3f}'.format(bm_scottspi, transcript)
	else:
		print '{0} BM Scott\'s pi:\tn/a'.format(transcript)

	print '-'*40


print 'Weighted average'
precision, recall, fscore, blah, bm_scottspi = score_advanced(all_results, None, global_bm_scottspi_dict)
if precision is not None:
	print 'Avg Precision:\t\t{0:.2f}%'.format(precision)
else:
	print 'Avg Precision:\t\tn/a'
if recall is not None:
	print 'Avg Recall:\t\t{0:.2f}%'.format(recall)
else:
	print 'Avg Recall:\tn/a'
if fscore is not None:
	print 'Avg F-score:\t\t{0:.2f}'.format(fscore)
else:
	print 'Avg F-score:\t\tn/a'
if bm_scottspi is not None:
	print 'BM Scott\'s pi:\t\t{0:.3f}'.format(bm_scottspi)
else:
	print 'BM Scott\'s pi:\t\tn/a'

