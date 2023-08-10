# ResearchLink

Code needed to compute the features used by ResearchLink, condensed into a single file for convenience.

The hypotheses must be in the file `data/hypotheses.csv` with the following format: `subject,predicate,object,label,scicheck_score`. Then, execute `run.py` to generate a processed file with all features computed for each hypothesis, which can then be passed on to a classifier.

A number of precomputations are needed for different features. In the `data/` folder, we provide all of them for the CSKG-600 dataset.
