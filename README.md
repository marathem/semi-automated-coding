# Semi-Automated Coding for Qualitative Research: Initial Prototypes

This repository contains the code used in the research article "Semi-Automated Coding for Qualitative Research: A User-Centered Inquiry and Initial Prototypes" by Megh Marathe and Kentaro Toyama published in CHI 2018.

Qualitative researchers perform an important and painstaking data annotation process known as coding. However, much of the process can be tedious and repetitive, becoming prohibitive for large datasets. Could coding be partially automated, and should it be? To answer this question, we interviewed researchers and observed them code interview transcripts. We found that across disciplines, researchers follow several coding practices well-suited to automation. Further, researchers desire automation after having developed a codebook and coded a subset of data, particularly in extending their coding to unseen data. Researchers also require any assistive tool to be transparent about its recommendations. Based on our findings, we built prototypes to partially automate coding using simple natural language processing (NLP) techniques. Our top-performing system generates coding that matches human coders on inter-rater reliability measures.

This repository provides the Python code for the three NLP techniques mentioned in our paper. Use of this code and/or data must be acknowledged by citing the paper as follows:
Megh Marathe and Kentaro Toyama. 2018. Semi-Automated Coding for Qualitative Research: A User-Centered Inquiry and Initial Prototypes. In Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems (CHI '18). ACM, New York, NY, USA, Paper 348, 12 pages. DOI: https://doi.org/10.1145/3173574.3173922

The three NLP techniques and their associated scripts are:
- simple keyword matching (predict_keyword.py and predict_keyword_roc.py)
- augmented keyword matching (predict_augkeyword.py and predict_augkeyword_roc.py)
- search-style query matching (predict_search.py and indexer.py)

The input data (both raw and gold standard) is expected to follow the format used by the BRAT annotation tool (https://brat.nlplab.org/index.html). The output takes the form of precision, recall, F-score, and Scott's Pi values on the command-line and spreadsheets containing text that has been annotated with correct, wrong, and missing codes. See the provided input (raw data at data/uncoded and gold-standard data at data/coded_jane) and output (simplek_jane, augmentk_jane, and query_jane) for an illustration.

These Python modules must be installed for the scripts to run: nltk, numpy, scipy, sklearn, whoosh.

Please get in touch with Megh Marathe (marathem@umich.edu) for any questions or concerns!
