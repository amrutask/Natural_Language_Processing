import sys
import string
import numpy as np
from sklearn import svm
from numpy import array


def confusion_matrix(predY, testY):
        TP=0; TN=0; FP=0; FN=0
        for i in range(0, len(testY)):
            if predY[i]==1 and testY[i]==1:
                TP=TP+1
            elif predY[i]==0 and testY[i]==0:
                TN=TN+1
            elif predY[i]==1 and testY[i]==0:
                FP=FP+1
            else:
                FN=FN+1
        
        if(FP+TN) !=0:
            fpr=FP/(FP+TN)
            tnr=TN/(TN+FP)
        else:
            fpr=0
            tnr=0
            
        if(FN+TP) !=0:    
            fnr=FN/(FN+TP)
            tpr=TP/(TP+FN)
            recall=TP/(TP+FN)
        else:
            fnr=0
            tpr=0
            recall=0
        
        acc=(TP+TN)/(TP+FP+TN+FN)
        
        if(TP+FP)!=0:
            precision=TP/(TP+FP)
        else:
            precision=0
        return fpr,fnr,tpr,tnr,acc, precision, recall
                    


class n_gram():
    
    def __init__(self, filename):
        
        self.x_inputs_without_pun=[]
        self.unique_words=[]
        self.x_inputs=[]
        self.y_labels=[]
        
        with open(filename, "r") as f:
            line = f.readline()
            while line:
                
                temp = line.split("\t")
                self.x_inputs.append(temp[0])
                self.y_labels.append(int(temp[1]))
                       
                
                line = f.readline()
        #print(len(self.x_inputs)) 
        #print(len(self.y_labels))
        f.close()
        
    def remove_punctuation(self):

        tr=str.maketrans("","", string.punctuation)
        for sentence in self.x_inputs:
            sentence=sentence.translate(tr)
            sentence=str(sentence).lower()
            self.x_inputs_without_pun.append(sentence)
            
            temp = sentence.split(" ")
            for word in temp:
                if word not in self.unique_words:
                    self.unique_words.append(word)
            
        self.unique_words=sorted(self.unique_words)
        #print("Unique words len",len(self.unique_words))
        return self.unique_words
    
    def one_gram(self):
        gram1=np.zeros([len(self.x_inputs_without_pun), len(self.unique_words)], dtype=int)
        
        for i in range(0, len(self.x_inputs_without_pun)):
            #print(self.x_inputs_without_pun[i])
            temp=self.x_inputs_without_pun[i].split(" ")
            for word in temp:
                occ=[index for index, value in enumerate(temp) if value==word]
                if len(occ)>0:
                    idx=self.unique_words.index(word)
                    gram1[i][idx]=len(occ)
                    
        return gram1
    
    def find_unique_strings(self, n):
        fk=[]
        for i in range(0, len(self.x_inputs_without_pun)):
            
            temp=self.x_inputs_without_pun[i].split(" ")
            
            if len(temp)<n:
                continue
            if len(temp)==n:
                if self.x_inputs_without_pun[i] not in fk:
                    
                    fk.append(self.x_inputs_without_pun[i])
                    continue
            for k in range(0, len(temp)+1):
                if k+n <= len(temp):
                    temp_string=temp[k:k+n]
                    temp_string="".join(temp_string)
                    
                    if temp_string not in fk:
                        
                        fk.append(temp_string)
            fk=sorted(fk)            
        return fk
    
    def find_freq_matrix(self, fk):
        
        freq_matrix=np.zeros([len(self.x_inputs_without_pun), len(fk)], dtype=int)
        for i in range(0, len(self.x_inputs_without_pun)):
            
            for j in range(len(fk)):
                    if fk[j] in self.x_inputs_without_pun[i]:
                        freq_matrix[i][j]+=1
                
        return freq_matrix  
    
    """def cross_validation(self, X):
        model = svm.SVC(kernel='linear')
        scores = cross_val_score(model, X, self.y_labels, cv=10)
        avg_accuracy=sum(scores)/len(scores)
        print("Accuracy :",avg_accuracy)"""
    
    
        
    def cross_validation_And_SVM(self, X, no_folds):
        
        Y=array(self.y_labels)
        div=int(X.shape[0]/no_folds)
        accuracy=[]
        fpr=[]
        fnr=[]
        tpr=[]
        tnr=[]
        recall=[]
        precision=[]
        
        model = svm.SVC(kernel='linear')
        start_index=0
        end_index=div-1      
        for i in range(0, no_folds):
            length=list(range(X.shape[0]))
            testX=X[start_index:end_index, :]
            testY=Y[start_index:end_index]
            del length[start_index:end_index]
            trainX=X[length, :]
            trainY=Y[length]
            
            model = svm.SVC(kernel='linear')
            model.fit(trainX, trainY)
            predY = model.predict(testX)
            
            start_index=start_index+div
            end_index=end_index+div
            fp,fn,tp,tn,acc, prsn, rcl=confusion_matrix(predY, testY)
            
            fpr.append(fp)
            fnr.append(fn)
            tpr.append(tp)
            tnr.append(tn)
            accuracy.append(acc)
            precision.append(prsn)
            recall.append(rcl)
            
        avg_fpr=sum(fpr)/len(fpr)
        avg_fnr=sum(fnr)/len(fnr)
        avg_tpr=sum(tpr)/len(tpr)
        avg_tnr=sum(tnr)/len(tnr)
        avg_accuracy=sum(accuracy)/len(accuracy)
        avg_precision=sum(precision)/len(precision)
        avg_recall=sum(recall)/len(recall)
        
        print("FPR: {}\nFNR: {}\nTPR: {}\nTNR: {}\nAccuracy: {}\nPrecision: {}\nRecall: {}\n".format(avg_fpr, avg_fnr, avg_tpr, avg_tnr, avg_accuracy, avg_precision, avg_recall))

    
       
def main():
        
    print("Number of Arguments: ", len(sys.argv))
    if len(sys.argv)!=2:
        print("USE: python [pyfile] [filename]\n")
        exit(0)
    
    arguments=sys.argv
    filename=str(arguments[1])
    print("Entered Text file is:", filename)

    ng=n_gram(filename)
    unique_words=ng.remove_punctuation()
    print("Number of Unique Words:",len(unique_words))
    gram1=ng.one_gram()

    f2=ng.find_unique_strings(2)
    f3=ng.find_unique_strings(3)
    f4=ng.find_unique_strings(4)
    f5=ng.find_unique_strings(5)

    gram2=ng.find_freq_matrix(f2)
    gram3=ng.find_freq_matrix(f3)
    gram4=ng.find_freq_matrix(f4)
    gram5=ng.find_freq_matrix(f5)


    f1f2= np.concatenate((gram1, gram2), axis=1)
    f1f2f3=np.concatenate((f1f2, gram3), axis=1)
    f1f2f3f4=np.concatenate((f1f2f3, gram4), axis=1)
    f1f2f3f4f5=np.concatenate((f1f2f3f4, gram5), axis=1)

    print("Processing F1\n")
    ng.cross_validation_And_SVM(gram1, 10)
    print("Processing F2\n")
    ng.cross_validation_And_SVM(gram2, 10)
    print("Processing F3\n")
    ng.cross_validation_And_SVM(gram3, 10)
    print("Processing F4\n")
    ng.cross_validation_And_SVM(gram4, 10)
    print("Processing F5\n")
    ng.cross_validation_And_SVM(gram5, 10)
    print("Processing F1F2\n")
    ng.cross_validation_And_SVM(f1f2, 10)
    print("Processing F1F2F3\n")
    ng.cross_validation_And_SVM(f1f2f3, 10)
    print("Processing F1F2F3F4\n")
    ng.cross_validation_And_SVM(f1f2f3f4, 10)
    print("Processing F1F2F3F4F5\n")
    ng.cross_validation_And_SVM(f1f2f3f4f5, 10)

if __name__== "__main__":
  main()
    