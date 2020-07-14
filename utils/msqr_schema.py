from whoosh.fields import SchemaClass, TEXT, ID, STORED, NUMERIC

class SpanSchema(SchemaClass):
	uniqueID = ID(unique=True, stored=True)
	annotationID = ID(stored=True)
	transcriptID = ID(stored=True)
	content = TEXT
	code = STORED
	start = NUMERIC
	end = NUMERIC

