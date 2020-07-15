# -*- coding: utf-8 -*-

"""This script applies the search-style querying technique to a set of texts
   and produces two kinds of output:
   		1) precision, recall, F-score, and Scott's Pi values for 
   			each text individually and averaged overall.
   		2) a spreadsheet containing the code annotations made by the 
   			search-style querying technique. This also classifies predicted 
   			codes into three categories: correct codes (true positives), 
   			wrong codes (false positives), and missed codes (false negatives).

   BEFORE RUNNING THIS SCRIPT be sure to install the whoosh package (use pip), 
   then customize and run the indexer script (indexer.py) to generate a 
   Whoosh index.

   The script does not accept command-line arguments. Instead, modify the 
   values of these variables to customize input and output:
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
   			- code_query.conf: one search-style query per code

	- index_dir: path to the folder containing the Whoosh search index.
   	- result_dir: path to the output folder. The coded spreadsheets will 
   		be saved in a folder named 'query_[coder]' within result_dir.
   	- transcripts: a list of file names, one per text file to be coded. The
   		appropriate un-coded and coded versions of each file must respectively 
   		be present within base_dir/uncoded and base_dir/coded_{coder}.

   The repository provides sample input and output as exemplars.
   The sample input is in data/uncoded (raw data) and data/coded_jane (gold 
   standard data coded by Jane). The sample output is in 
   results/query_jane (an output spreadsheet). Configuration file examples
   are included too; see data/coded_jane for formatting instructions. 

   The most important configuration file for search-style querying is
   code_query.conf, the one with search-style queries for the codebook!

   See here for information about Whoosh: https://pypi.org/project/Whoosh/
   And here for information about BRAT: https://brat.nlplab.org/index.html

"""

import os, os.path
import codecs
from whoosh import index
from whoosh.query import Term
from whoosh.qparser import QueryParser

from utils.indexer_utils import encoding, read_codes, read_annotations, write_spreadsheet_special, score_advanced, score
from utils.entities import *

base_dir = 'data'
coder = 'jane'
index_dir = 'index'
result_dir = 'results'
transcripts = ['1_token']

class AnnotationResult:
	annotation = None
	correct_codes = set()
	wrong_codes = set()
	missed_codes = set()

input_dir = os.path.join(base_dir, 'uncoded')
gold_dir = os.path.join(base_dir, 'coded_{0}'.format(coder))
output_dir = os.path.join(result_dir, '{0}_{1}'.format('query', coder))

limit = None

gold_annconf_file = os.path.join(gold_dir, 'annotation.conf')
gold_visconf_file = os.path.join(gold_dir, 'visual.conf')
ignored_codes_file = os.path.join(gold_dir, 'code_ignore.conf')
code_queries_file = os.path.join(gold_dir, 'code_query.conf')
output_xlsx_pattern = '{0}_result.xlsx'

codes, code_queries = read_codes(gold_annconf_file, ignored_codes_file, code_queries_file)
#print 'Read codes'

ix = index.open_dir(index_dir)
qparser = QueryParser("content", schema=ix.schema)

avg_precision, avg_recall, avg_fscore = 0.0, 0.0, 0.0
avg_bm_scottspi = 0.0

all_results = list()
global_bm_scottspi_dict = dict()

for transcript in transcripts:	
	input_ann_file = os.path.join(input_dir, '{0}.ann'.format(transcript))
	input_text_file = os.path.join(input_dir, '{0}.txt'.format(transcript))
	gold_ann_file = os.path.join(gold_dir, '{0}.ann'.format(transcript))
	output_xlsx_file = os.path.join(output_dir, output_xlsx_pattern.format(transcript))

	annid_to_codes = dict()
	pred_by_code = dict()

	print '{0} Predicted codes:'.format(transcript)
	for code, query in code_queries.items():
		qry = qparser.parse(query)
		transcript_filter = Term(u'transcriptID', transcript)

		with ix.searcher() as srchr:
			matched_anns = srchr.search(qry, filter=transcript_filter, limit=limit)
			print '{2} {0}: {1} results'.format(code, len(matched_anns), transcript)
			for ann in matched_anns:
				ann_id = ann['annotationID']
				if ann_id not in annid_to_codes:
					annid_to_codes[ann_id] = list()
				annid_to_codes[ann_id].append(code)
				if code not in pred_by_code:
					pred_by_code[code] = list()
				pred_by_code[code].append(ann_id)

	input_anns = read_annotations(input_ann_file)
	#print '{0} Read input annotations'.format(transcript)

	#print annid_to_codes
	pred_anns = []
	for input_ann in input_anns:
		pred_codes = annid_to_codes.get(input_ann._id, None)
		if pred_codes:
			pred_anns.append((input_ann, pred_codes))

	print '{1} Total {0} annotations coded'.format(len(pred_anns), transcript)

	gold_anns = read_annotations(gold_ann_file, codes)
	gold_codes_to_anns = dict()
	for gold_ann in gold_anns:
		code = gold_ann.code
		if code not in gold_codes_to_anns:
			gold_codes_to_anns[code] = list()
		gold_codes_to_anns[code].append(gold_ann)

	scottspi_dict = dict()
	bm_scottspi_dict = dict() # boundary-matching scott's pi. A more reliable measure I think.
	for code in codes:
		code_pred = len(pred_by_code.get(code, []))
		code_gold = len(gold_codes_to_anns.get(code, []))
		if code_pred > 0 or code_gold > 0:
			scottspi_dict[code] = (code_gold, code_pred)
			bm_scottspi_dict[code] = [code_gold, 0]

	annid_to_results = dict()
	for pred_ann, pred_codes in pred_anns:
		result = AnnotationResult()
		result.annotation = pred_ann
		result.correct_codes = set()
		result.wrong_codes = set()
		annid_to_results[pred_ann._id] = result

		pred_start, pred_end = pred_ann.start, pred_ann.end
		for pred_code in pred_codes:
			m_gold_anns = gold_codes_to_anns.get(pred_code, None)
			correct = False
			if m_gold_anns:
				for m_gold_ann in m_gold_anns:
					gold_start, gold_end = m_gold_ann.start, m_gold_ann.end
					if (pred_start == gold_start and pred_end == gold_end) or (pred_start <= gold_start and pred_end >= gold_end):
						correct = True
						result.correct_codes.add(pred_code)
						# increment second value of bm_scottspi_dict entry
						bm_entry = bm_scottspi_dict.get(pred_code, [0, 0])
						bm_entry[1] += 1
						bm_scottspi_dict[pred_code] = bm_entry
						break

			if not correct:
				result.wrong_codes.add(pred_code)

		#print '{0}: {1} correct\t{2} wrong'.format(pred_ann._id, len(result.correct_codes), len(result.wrong_codes))

	#for k, v in bm_scottspi_dict.items():
	#	print '{0}: {1}'.format(k, v)

	results = list()
	#print 'Ann ID\t\tCorrect\t\tWrong\t\tMissed'
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

		pred_result = annid_to_results.get(_id, None)
		if pred_result:
			result.correct_codes = pred_result.correct_codes
			result.wrong_codes = pred_result.wrong_codes
			result.missed_codes = result.missed_codes - pred_result.correct_codes

		#print '{0}\t{1}\t{2}\t{3}'.format(_id, list(result.correct_codes), list(result.wrong_codes), list(result.missed_codes))


	write_spreadsheet_special(results, output_xlsx_file)
	print '{1} Wrote {0}'.format(output_xlsx_file, transcript)

	all_results.extend(results)

	for code, entry in bm_scottspi_dict.items():
		if code not in global_bm_scottspi_dict:
				global_bm_scottspi_dict[code] = [0, 0]
		
		global_bm_scottspi_dict[code][0] += entry[0]
		global_bm_scottspi_dict[code][1] += entry[1]


	precision, recall, fscore, scottspi, bm_scottspi = score_advanced(results, scottspi_dict, bm_scottspi_dict)
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

	if scottspi is not None:
		print '{1} Scott\'s pi:\t{0:.3f}'.format(scottspi, transcript)
	else:
		print '{0} Scott\'s pi:\tn/a'.format(transcript)

	if bm_scottspi is not None:
		avg_bm_scottspi += bm_scottspi
		print '{1} BM Scott\'s pi:\t{0:.3f}'.format(bm_scottspi, transcript)
	else:
		print '{0} BM Scott\'s pi:\tn/a'.format(transcript)

	print '-'*40

print 'Simple average'
print 'Avg Precision:\t\t{0:.2f}%'.format(avg_precision/len(transcripts))
print 'Avg Recall:\t\t{0:.2f}%'.format(avg_recall/len(transcripts))
print 'Avg F-score:\t\t{0:.2f}'.format(avg_fscore/len(transcripts))
print 'Avg BM Scott\'s pi:\t{0:.2f}'.format(avg_bm_scottspi/len(transcripts))
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

