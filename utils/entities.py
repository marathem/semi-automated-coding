encoding = 'utf-8'

class Annotation:
	_id = None
	start = -1
	end = -1
	code = None
	text = None
	filtered_text = None
	stemmed_ftext = None
	match_score = -1
	_inputannid = None

	# Default constructor
	def __init__(self):
		return

	# This is useful for creating a per-paragraph or per-conv.-turn annotation
	def __init__(self, _id, start, end, text, filtered_text, stemmed_ftext):
		self._id = _id
		self.start = start
		self.end = end
		if text:
			self.text = text.encode(encoding)
		if filtered_text:
			self.filtered_text = filtered_text.encode(encoding)
		if stemmed_ftext:
			self.stemmed_ftext = stemmed_ftext.encode(encoding)

	# This is useful for creating both gold-standard and predicted annotations
	def __init__(self, _id, start, end, text, filtered_text=None, stemmed_ftext=None, code=None, match_score=-1, orig_code=None):
		self._id = _id
		self.start = start
		self.end = end
		self.code = code
		if text:
			self.text = text #.encode(encoding)
		if filtered_text:
			self.filtered_text = filtered_text #.encode(encoding)
		if stemmed_ftext:
			self.stemmed_ftext = stemmed_ftext #.encode(encoding)
		self.match_score = match_score
		self.orig_code = code
		if orig_code:
			self.orig_code = orig_code

	def __str__(self):
		return u'{0}: {1}-{2} {3}'.format(self._id, self.start, self.end, self.code)

	# Return brat format of this notation. Like the following:
	# [_id]\t[code] [start] [end]\t[text] 
	def to_brat_format(self, no_text=False):
		code = self.code.replace(' ', '-')
		if no_text:
			return u'{0}\t{1} {2} {3}'.format(self._id, code, self.start, self.end)
		return u'{0}\t{1} {2} {3}\t{4}'.format(self._id, code, self.start, self.end, self.text)

		



