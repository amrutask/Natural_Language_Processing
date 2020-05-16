### Implementation of word2vec from scratch using co-occurrence matrix

### Steps Involved:
1. Read the filename to be processed as a command line argument along with the window size.
2. Remove all the punctuations, change the words to lower case and get the list of all the unique words.
3. Find the co-occurrence matrix for those unique words considering each word at the center of the window of a given size.
   Number of occurrences of all the neighbouring words are counted and inserted into the co-occurence matrix for each center word.
4. Each row (for each unique word) of occurrences is normalized with the total count of that row.
5. The co-occurrence matrix is then passed to the SVD function for singular Value Decomposition to get the U matrix.
6. The U matrix from SVD is then stored as float values in the txt file.


### To run this file use:

`python Word2VecModel.py [text_file] [window_size]`
