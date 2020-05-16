### Classification of online reviews using NLP techniques

### The project implements:
* n-gram for features extraction
* SVM classifier with 10-fold cross-validation

### Datasets
* amazon_cells_labelled.txt
* imdb_labelled.txt
* yelp_labelled)   

### The project Calculates 7 metrics for performance evaluation: 
* False Positive Rate
* False Negative Rate
* True Positive Rate
* True Negative Rate
* accuracy
* precision
* recall

### N-gram representations used:
The classifier runs with:
* Uni-gram(F1)
* bi-gram(F2)
* tri-gram(F3)
* 4-gram(F4)
* 5-gram(F5) representation of the reviews.

### The project also uses two or more n-gram representations together such as:
* F1F2
* F1F2F3
* F1F2F3F4
* F1F2F3F4F5 

These representations are computed and used as an input to SVM classifier for reviews classification.
