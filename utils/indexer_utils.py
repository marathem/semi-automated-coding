import codecs
import os
import shutil
import xlsxwriter
import operator

from entities import *

encoding = 'utf-8'

# Reads annotations from a .ann file, returns a list of Annotation objects
# Stemming will ONLY BE DONE if stopwords are also supplied
# Special: this attaches the question to every subsequent paragraph until the next question.
# So basically, paragraphs of the form: Q1 A11 A12 A13 Q2 A21
# will be converted to:
# Q1, Q1-A11, Q1-A12, Q1-A13, Q2, Q2-A21.
def read_annotations_special(annotations_file, codes_to_fetch=None):
	annotations = list()
	question_text = None
	is_question = False
	for line in codecs.open(annotations_file, 'r', encoding):
		clean_line = line.lstrip().rstrip()
		parts = clean_line.split('\t')
		if len(parts) > 1:
			_id = parts[0]
			
			spanparts = parts[1].split()
			orig_code = spanparts[0]
			code = orig_code #BIG CHANGE: orig_code.lower().replace('-', ' ')
			start = int(spanparts[1])
			end = int(spanparts[-1])

			if codes_to_fetch and code not in codes_to_fetch:
				continue
			
			text = None
			if len(parts) > 2:
				text = parts[2].lstrip().rstrip()

				if _id.endswith('-0'):
					is_question = True
					question_text = text
				elif question_text:
					is_question = False
					text = question_text + ' ' + text

			
			ann = Annotation(_id, start, end, text, code=code, orig_code=orig_code)
			annotations.append(ann)

	return annotations

# Reads annotations from a .ann file, returns a list of Annotation objects
# Stemming will ONLY BE DONE if stopwords are also supplied
def read_annotations(annotations_file, codes_to_fetch=None):
	annotations = list()
	for line in codecs.open(annotations_file, 'r', encoding):
		clean_line = line.lstrip().rstrip()
		parts = clean_line.split('\t')
		if len(parts) > 1:
			_id = parts[0]
			
			spanparts = parts[1].split()
			orig_code = spanparts[0]
			code = orig_code #BIG CHANGE: orig_code.lower().replace('-', ' ')
			start = int(spanparts[1])
			end = int(spanparts[-1])

			if codes_to_fetch and code not in codes_to_fetch:
				continue
			
			text = None
			if len(parts) > 2:
				text = parts[2].lstrip().rstrip()
			
			ann = Annotation(_id, start, end, text, code=code, orig_code=orig_code)
			annotations.append(ann)

	return annotations

def read_annotations_special2(annotations_file, stopwords=None, stemmer=None):
	annotations = list()
	question_text = None
	is_question = False
	for line in codecs.open(annotations_file, 'r', encoding):
		clean_line = line.lstrip().rstrip()
		parts = clean_line.split('\t')
		if len(parts) > 1:
			_id = parts[0]
			
			spanparts = parts[1].split()
			orig_code = spanparts[0]
			code = orig_code.lower().replace('-', ' ')
			start = int(spanparts[1])
			end = int(spanparts[-1])
			
			text, filtered_text, stemmed_ftext = None, None, None
			if len(parts) > 2:
				text = parts[2].lstrip().rstrip()

				if _id.endswith('-0'):
					is_question = True
					question_text = text
				elif question_text:
					is_question = False
					text = question_text + ' ' + text

				filtered_text = text.lower()
				if stopwords:
					filtered_text = ' '.join([w.lower() for w in text.split() if w not in stopwords])

				if stemmer:
					stems = set()
					for w in filtered_text.split():
						stems.add(stemmer.stem(w))
					stemmed_ftext = ' '.join(stems)
			
			ann = Annotation(_id, start, end, text, filtered_text, stemmed_ftext, code=code, orig_code=orig_code)
			annotations.append(ann)

	return annotations

# Reads annotations from a .ann file, returns a list of Annotation objects
# Stemming will ONLY BE DONE if stopwords are also supplied
def read_annotations2(annotations_file, codes_to_fetch=None):
	annotations = list()
	for line in codecs.open(annotations_file, 'r', encoding):
		clean_line = line.lstrip().rstrip()
		parts = clean_line.split('\t')
		if len(parts) > 1:
			_id = parts[0]
			
			spanparts = parts[1].split()
			orig_code = spanparts[0]
			code = orig_code.lower().replace('-', ' ')
			start = int(spanparts[1])
			end = int(spanparts[-1])

			if codes_to_fetch and code not in codes_to_fetch:
				continue
			
			text = None
			if len(parts) > 2:
				text = parts[2].lstrip().rstrip()
			
			ann = Annotation(_id, start, end, text, code=code, orig_code=orig_code)
			annotations.append(ann)

	return annotations

def read_ignored_codes(code_ignore_file):
	codes = set()
	if not os.path.exists(code_ignore_file):
		return None

	with codecs.open(code_ignore_file, 'r', encoding) as ignore_file:
		for line in ignore_file:
			word = line.lstrip().rstrip()
			if len(word) == 0 or word.startswith('#'):
				continue
			codes.add(word)

	if len(codes) == 0:
		return None
	return codes

# Reads codes (or, as brat likes to call them, entities) from an annotation.conf file
# codes, code_queries = read_codes(gold_annconf_file, ignored_codes_file, code_queries_file)
def read_codes(code_config_file, code_ignore_file, code_queries_file):
	found_section = False
	codes = list()
	code_queries = dict()

	ignore_codes = read_ignored_codes(code_ignore_file)

	with codecs.open(code_config_file, 'r', encoding) as code_file:
		for line in code_file:
			clean_line = line.lstrip().rstrip()
			if len(clean_line) == 0:
				continue
			if not found_section and clean_line == '[entities]':
				found_section = True
				continue
			if clean_line.startswith('#'):
				continue
			if found_section:
				if clean_line.startswith('['):
					found_section = False
					break

				code = clean_line
				if ignore_codes is not None and code in ignore_codes:
					continue

				codes.append(code)


	with codecs.open(code_queries_file, 'r', encoding) as queries_file:
		for line in queries_file:
			clean_line = line.lstrip().rstrip()
			if len(clean_line) == 0 or clean_line.startswith('#'):
				continue

			code, query = clean_line.split(u'=')
			code = code.lstrip().rstrip()
			if code in codes:
				query = query.lstrip().rstrip()
				code_queries[code] = query

	return codes, code_queries


def read_ignored_codes2(code_ignore_file):
	codes = set()
	if not os.path.exists(code_ignore_file):
		return None

	with codecs.open(code_ignore_file, 'r', encoding) as ignore_file:
		for line in ignore_file:
			word = line.lstrip().rstrip()
			if len(word) == 0 or word.startswith('#'):
				continue
			word = word.replace(u'-', u' ').lower()
			codes.add(word)

	if len(codes) == 0:
		return None
	return codes

def read_keywords_file(code_keywords_file):
	if code_keywords_file is None:
		return None
	keywords = dict()
	for line in codecs.open(code_keywords_file, 'r', encoding):
		clean_line = line.lstrip().rstrip()
		if len(clean_line) == 0 or ':' not in clean_line:
			continue
		word, wordlist = clean_line.split(':')
		keywords[word] = wordlist.split()

	if len(keywords) == 0: 
		return None
	return keywords

# This is for the stemming/stopwords readcodes that is needed for the
# simple/augmented keywords methods.
def read_codes_stopstem(code_config_file, stopwords, stemmer, 
	ignored_codes_file, code_keywords_file):

	found_section = False
	codes = list()
	filtered_codes = list()
	stemmed_codes = list()
	codes_plus = list()
	orig_codes = list()

	ignore_codes = read_ignored_codes2(ignored_codes_file)
	keywords = read_keywords_file(code_keywords_file)

	with codecs.open(code_config_file, 'r', encoding) as code_file:
		for line in code_file:
			clean_line = line.lstrip().rstrip()
			if len(clean_line) == 0:
				continue
			if not found_section and clean_line == '[entities]':
				found_section = True
				continue
			if clean_line.startswith('#'):
				continue
			if found_section:
				if clean_line.startswith('['):
					found_section = False
					break

				orig_code = clean_line
				clean_line = clean_line.replace(u'-', u' ').lower()
				if ignore_codes is not None and clean_line in ignore_codes:
					continue

				orig_codes.append(orig_code)
				codes.append(clean_line)

				if keywords is not None:
					new_code = list()
					for word in clean_line.split():
						new_code.append(word)
						for syn in keywords.get(word, list()):
							new_code.append(syn)
					clean_line = ' '.join(new_code)
					codes_plus.append(clean_line)

				if stopwords:
					filtered_code = list()
					stemmed_code = set()
					for p in clean_line.split():
						if p not in stopwords:
							filtered_code.append(p)
							if stemmer:
								stemmed_code.add(stemmer.stem(p))
					if len(filtered_code) > 0:
						filtered_codes.append(' '.join(filtered_code))
					if stemmer and len(stemmed_code) > 0:
						stemmed_codes.append(' '.join(stemmed_code))

	return codes, filtered_codes, stemmed_codes, codes_plus, orig_codes


# This method is like write_spreadsheet, but it will also attempt to correctly assign codes that are marked for a specific
# span whose start/end don't match exactly with the chunks, but lie within a chunk.
# That is, if a rater has marked only a sentence or two instead of the whole paragraph.
def write_spreadsheet_special(text_results, output_file, include_score=True):
	if text_results is None or output_file is None:
		return

	output_dir = os.path.dirname(output_file)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	workbook = xlsxwriter.Workbook(output_file)
	worksheet = workbook.add_worksheet()
	bold = workbook.add_format({'bold': True})
	text_wrap = workbook.add_format({'text_wrap': True})
	green = workbook.add_format({'color': 'green', 'bold': True, 'text_wrap': True})
	red = workbook.add_format({'color': 'red', 'bold': True, 'text_wrap': True})
	orange = workbook.add_format({'color': 'orange', 'bold': True, 'text_wrap': True})
	worksheet.set_column(0, 0, 50, text_wrap)
	worksheet.set_column(1, 10, 25, text_wrap)
	row = 0
	col = 0

	#heading
	worksheet.write_string(row, col, 'Text', bold)
	worksheet.write_string(row, col + 1, 'Correct Codes', bold)
	worksheet.write_string(row, col + 2, 'Wrong Codes', bold)
	worksheet.write_string(row, col + 3, 'Missed Codes', bold)
	row += 1

	for idx, result in enumerate(text_results):
		col = 0
		ann = result.annotation
		worksheet.write_string(row, col, ann.text)
		col += 1
		if result.correct_codes and len(result.correct_codes) > 0:
			worksheet.write_string(row, col, ', '.join([code.replace(u'-', u' ') for code in result.correct_codes]), green)
		col += 1
		if result.wrong_codes and len(result.wrong_codes) > 0:
			worksheet.write_string(row, col, ', '.join([code.replace(u'-', u' ') for code in result.wrong_codes]), red)
		col += 1
		if result.missed_codes and len(result.missed_codes) > 0:
			worksheet.write_string(row, col, ', '.join([code.replace(u'-', u' ') for code in result.missed_codes]), orange)
		row += 2

	workbook.close()

# Calculates each step in Rebecca Frank's IRR spreadsheet and
# returns the Scott's pi and Holsti's coefficient.
#
# Expected input is a dictionary: code -> (num_rater1, num_rater2)
# that maps each code to a tuple containing the number of times rater 1 and rater 2
# marked that code respectively. Order does not matter.
def scotts_pi(coded_dict):
	scottspi = None
	sum_dict = dict()
	twom_dict = dict()
	agr_dict = dict()
	for code, vals in coded_dict.items():
		sum_dict[code] = sum(vals)
		twom_dict[code] = 2 * min(vals)
		if sum_dict[code] > 0:
			agr_dict[code] = round(twom_dict[code]/float(sum_dict[code]), 3)
		
	total = sum(sum_dict.values())
	if total == 0:
		return scottspi

	twom_total = sum(twom_dict.values())
	agr_total = round(twom_total/float(total), 3)
	
	propsq_dict = dict()
	propsq_sum = 0.0
	for code, val in sum_dict.items():
		propsq = pow(val/float(total), 2)
		propsq_dict[code] = propsq
		propsq_sum += propsq

	#print 'Scottspi: propsq_sum', propsq_sum
	if propsq_sum != 1:
		scottspi = 100*(agr_total - propsq_sum)/(1 - propsq_sum)
		#holsti = twom_total/float(total)

	return scottspi


def score(text_results, coded_dict):
	num_correct, num_predicted, num_actually_correct = 0, 0, 0
	for result in text_results:
		if result.correct_codes:
			num_correct +=  len(result.correct_codes)
			num_actually_correct += len(result.correct_codes)
			num_predicted += len(result.correct_codes)
			
		if result.wrong_codes:
			num_predicted += len(result.wrong_codes)

		if result.missed_codes:
			num_actually_correct += len(result.missed_codes)

	precision, recall, fscore = None, None, None
	if num_predicted > 0: 
		precision = 100*num_correct/(1.0*num_predicted)

	if num_actually_correct > 0: 
		recall = 100*num_correct/(1.0*num_actually_correct)

	if precision and recall:
		fscore = 2*precision*recall/(precision + recall)
		
	scottspi = None
	if coded_dict is not None:
		scottspi = scotts_pi(coded_dict)

	return precision, recall, fscore, scottspi

def score_advanced(text_results, coded_dict, bm_coded_dict):
	precision, recall, fscore, scottspi = score(text_results, coded_dict)
	bm_scottspi = None
	if bm_coded_dict is not None:
		print 'Calculating bm scotts pi'
		bm_scottspi = scotts_pi(bm_coded_dict)
	return precision, recall, fscore, scottspi, bm_scottspi

