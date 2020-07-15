# -*- coding: utf-8 -*-

"""This script uses the Whoosh library to create a search index. This
   is a prerequisite for the search-style querying technique.

   The script does not accept command-line arguments. Instead, modify the 
   values of these variables to customize input and output:
   	- index_dir: path to the folder where the search index should be created.
   	- uncoded_dir: path to the folder containing raw/uncoded data. Note that
   		data are expected to be in the format used by BRAT.

   BEFORE RUNNING THIS SCRIPT be sure to install the whoosh package 
   (pip install whoosh).

   AFTER SUCCESSFULLY RUNNING THIS SCRIPT, you will have created a search
   index at the location that index_dir points to. Don't forget to modify 
   the value of the index_dir variable in predict_search.py accordingly.

   See here for information about Whoosh: https://pypi.org/project/Whoosh/
   And here for information about BRAT: https://brat.nlplab.org/index.html

"""

import os, os.path
from whoosh import index

from utils.indexer_utils import encoding, read_annotations_special
from utils.msqr_schema import SpanSchema

index_dir = 'index'
uncoded_dir = 'data/uncoded'

if not os.path.exists(index_dir):
	os.mkdir(index_dir)

ix = index.create_in(index_dir, SpanSchema)
writer = ix.writer()

for filename in os.listdir(uncoded_dir):
	if filename.endswith('ann'):
		filepath = os.path.join(uncoded_dir, filename)
		transcript_id = filename.split('.')[0].decode(encoding)
		anns = read_annotations_special(filepath)
		print '{0}: read {1} annotations'.format(transcript_id, len(anns))
		for ann in anns:
			unique_id = u'{0}_{1}'.format(transcript_id, ann._id)
			writer.add_document(uniqueID=unique_id, annotationID=ann._id, 
				transcriptID=transcript_id, content=ann.text, code=ann.code, start=ann.start, end=ann.end)
			print '{0}: inserted ({1}, {2}, {3}, {4})'.format(transcript_id, ann._id, ann.code, ann.start, ann.end)
		print ' '

writer.commit()
