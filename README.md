# Stance Context Representations

This project includes the code for the paper:
> Jai Aggarwal, Brian Diep, Julia Watson, and Suzanne Stevenson. 2023. Investigating Online Community Engagement through Stancetaking. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 5814–5830, Singapore. Association for Computational Linguistics.

## Main Analyses
The code for the main analyses can be found in src/main.py. The smaller data files can be found in the data_files folder, and the larger data files can be found at the [following OSF link](https://osf.io/z9rxw/).
To generate the data required for the main analyses from scratch, you can refer to the Data Extraction section below.


## Data Extraction
### Preliminary steps
To extract the data from the raw data dumps, you can first run the command `python data_extraction.py YEAR`. This code will extract all Reddit comments from a particular year that use the stance markers in our dataset. This code is set up to use multiprocessing, so please specific the number of cores to use. As a default, it uses 6 cores.

The next step is to apply split_large_files.sh. This breaks up the large counts.json files created in the previous steps into more manageable files of 500K comments each. 


### Wang2Vec Usage
From each of the files created by split_large_files.sh, we can sample *n* comments using ./get_wang2vec_sample.sh. To join these files together, we can run the command
`cat ./* > wang2vec_sample.json`
Once we create a sample of roughly 25M comments, we can then run preprocess_text_for_wang2vec.py which uses the 
sampled comments and writes them in a format amenable to the wang2vec algorithm. We run the Wang2Vec algorithm
with the following parameters. The number of threads can be altered as needed.

./word2vec -train DATA_DIR/reddit_dataset.txt -output OUTPUT_DIR/embedding_file -type 3 -size 100 -window 5 -negative 10 -nce 0 -hs 0 -sample 1e-4 -threads 12 -binary 1 -iter 5 -cap 0


### Main Data Pipeline
Our data extraction pipeline runs from the main.py file. This script relies on three files within the data_extraction folder.

1. Corpus_statistics.py: this file is used to extract our posts from the Reddit data dumps, and also filters the data according to the preprocessing steps mentioned in our paper.
2. Extract_embedding_utils.py: this file contains utility functions that create SBERT embeddings for each sentence in our dataset
3. Feature_extraction.py: this file is used to extract our linguistic features using saved regression models


