import sys
import numpy as np


class word2vec():
    def __init__ (self, corpus, unique_w, w_size):
        self.Corpus=corpus
        self.Window_Size=w_size
        
        # append # key at the beginning
        unique_w.insert(0,'#')
        self.Uniques=unique_w
        
        self.C_Matrix=np.zeros([len(self.Uniques), len(self.Uniques)], dtype=float)
        
    
    def create_co_occurrence(self):
        
        for word in self.Uniques:
            if word=='#':
                continue
            
            occ=[index for index, value in enumerate(self.Corpus) if value==word]
            
            center_word_index=self.Uniques.index(word)
    
            for occurrence in occ:
                i=occurrence-int(self.Window_Size)
    
                while i != (occurrence+int(self.Window_Size)+1):
                    if (i < 0) or (i >= len(self.Corpus)):
                        y=self.Uniques.index('#')
                        self.C_Matrix[center_word_index][y]= self.C_Matrix[center_word_index][y] + 1                     
                    elif self.Corpus[i] != word:
                        y=self.Uniques.index(self.Corpus[i])
                        self.C_Matrix[center_word_index][y]= self.C_Matrix[center_word_index][y] + 1
                        
                    i=i+1    
            no_targets= len(occ) * (self.Window_Size *2)
            self.C_Matrix[center_word_index:]=self.C_Matrix[center_word_index:]/no_targets
        return(self.C_Matrix)

    
    def SVD_Calculation(self):
        la= np.linalg
        u, s, vh = la.svd(self.C_Matrix, full_matrices=False)
        w= u[0:len(self.Uniques), 0:int(self.Window_Size) * 2]
        np.savetxt("out.txt", w, fmt='%.5f')
        

def main():
    #print("Number of Arguments: ", len(sys.argv))
    
    arguments=sys.argv
    if len(arguments) !=3:
        print("Python python_file [textfile] [window_size]")
        exit(0)
        
    filename=str(arguments[1])
    w_size=int(arguments[2])

    with open(filename, "r") as fp:
        corpus = []
        unique_w=[]
        line = fp.readline()
        while line:
            x = line.split(" ")
        
            for each_w in x:
                each_w=str(each_w).lower()
                if "\n" in each_w:
                    each_w=each_w.rstrip('\n')
                if each_w == '\n' or each_w=='':
                    continue
                corpus.append(each_w)
                if each_w not in unique_w:
                    unique_w.append(each_w)
            line = fp.readline()
        
        fp.close()

    #print("Corpus: {}\n".format(corpus))
    print("Number of Unique words: {}".format(len(unique_w)))

    w2v= word2vec(corpus, unique_w, w_size)
    Occ_Matrix=w2v.create_co_occurrence()
    print(" Size of Occurrence Matrix :\n {}\n".format(Occ_Matrix.shape))
    w2v.SVD_Calculation()
    

if __name__== "__main__":
  main()
    