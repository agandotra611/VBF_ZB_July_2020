import tensorflow  as tf  
import numpy as np
from tensorflow import keras   
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split




inputFilename_VBF = "/afs/cern.ch/work/a/addropul/public/forAditya/l1TNtuple-VBF_restrictedeta_060420_recotest.root"
inputFilename_ZB =  "/afs/cern.ch/work/a/addropul/public/forAditya/l1TNtuple-ZeroBias-060520_restrictedeta.root"

inputFile_VBF = root.TFile(inputFilename_VBF)
inputFile_ZB = root.TFile(inputFilename_ZB)
outputFilename = "Keras_output_ZB_newsig.root"
outFile = root.TFile(outputFilename, "RECREATE")


inputFileVBF  =root.TFile.Open( '/afs/cern.ch/work/a/addropul/public/forAditya/l1TNtuple-VBF_restrictedeta_060420_recotest.root')
inputTreeVBF= inputFileVBF.Get("l1NtupleProducer/Stage3Regions/efficiencyTree")

inputFileZB  =root.TFile.Open( '/afs/cern.ch/work/a/addropul/public/forAditya/l1TNtuple-ZeroBias-060520_restrictedeta.root')
inputTreeZB = inputFileZB.Get("l1NtupleProducer/Stage3Regions/efficiencyTree")

                                                                                                                                                                                     
fVBF = root.TFile.Open("l1TNtuple-VBF_restrictedeta_060420_recotest.root","UPDATE")
#fVBF.Print()                                                                                                                                                                                                                               
fZB = root.TFile.Open("l1TNtuple-ZeroBias-060520_restrictedeta.root", "UPDATE")
#fZB.Print()                                                                                                                                                                                                                                
fVBFTree = fVBF.Get("l1NtupleProducer/Stage3Regions/efficiencyTree")
fZBTree = fZB.Get("l1NtupleProducer/Stage3Regions/efficiencyTree")
#print(type(fVBFTree))                                                                                                                                                                                                                      
#print(type(fZBTree))                                                                                                                                                                                                                       

fVBFDataDeltaEta = rootnp.tree2array(fVBFTree, branches = 'l1DeltaEta')
fVBFDataDeltaPhi = rootnp.tree2array(fVBFTree, branches = 'l1DeltaPhi')
fVBFDataMass = rootnp.tree2array(fVBFTree, branches = 'l1Mass')
fVBFDataPt1 = rootnp.tree2array(fVBFTree, branches = 'l1Pt_1')
fVBFDataPt2 = rootnp.tree2array(fVBFTree, branches = 'l1Pt_2')


fVBFDataDeltaEta = fVBFDataDeltaEta.reshape(len(fVBFDataDeltaEta), 1)
fVBFDataDeltaPhi = fVBFDataDeltaPhi.reshape(len(fVBFDataDeltaPhi), 1)
fVBFDataMass = fVBFDataMass.reshape(len(fVBFDataMass), 1)
fVBFDataPt1 = fVBFDataPt1.reshape(len(fVBFDataPt1), 1)
fVBFDataPt2 = fVBFDataPt2.reshape(len(fVBFDataPt2), 1)



fZBDataDeltaEta = rootnp.tree2array(fZBTree, branches = 'l1DeltaEta')
fZBDataDeltaPhi = rootnp.tree2array(fZBTree, branches = 'l1DeltaPhi')
fZBDataMass = rootnp.tree2array(fZBTree, branches = 'l1Mass')
fZBDataPt1 = rootnp.tree2array(fZBTree, branches = 'l1Pt_1')
fZBDataPt2 = rootnp.tree2array(fZBTree, branches = 'l1Pt_2')

fZBDataDeltaEta = fZBDataDeltaEta.reshape(len(fZBDataDeltaEta), 1)
fZBDataDeltaPhi = fZBDataDeltaPhi.reshape(len(fZBDataDeltaPhi), 1)
fZBDataMass = fZBDataMass.reshape(len(fZBDataMass), 1)
fZBDataPt1 = fZBDataPt1.reshape(len(fZBDataPt1), 1)
fZBDataPt2 = fZBDataPt2.reshape(len(fZBDataPt2), 1)




np.savetxt('VBFDeltaEta.csv', fVBFDataDeltaEta, delimiter = ' ')
np.savetxt('VBFDeltaPhi.csv', fVBFDataDeltaPhi, delimiter = ' ')
np.savetxt('VBFPt1.csv', fVBFDataPt1, delimiter = ' ')
np.savetxt('VBFPt2.csv', fVBFDataPt2, delimiter = ' ')
np.savetxt('VBFMass.csv', fVBFDataMass, delimiter = ' ')

np.savetxt('ZBDeltaEta.csv', fZBDataDeltaEta, delimiter = ' ')
np.savetxt('ZBDeltaPhi.csv', fZBDataDeltaPhi, delimiter = ' ')
np.savetxt('ZBPt1.csv', fZBDataPt1, delimiter = ' ')
np.savetxt('ZBPt2.csv', fZBDataPt2, delimiter = ' ')
np.savetxt('ZBMass.csv', fZBDataMass, delimiter = ' ')




#Importing data and applying cut 

VBFDeltaEta = np.loadtxt('VBFDeltaEta.csv', delimiter = ' ')
VBFDeltaEta = VBFDeltaEta[(VBFDeltaEta >= -6)]

ZBDeltaEta = np.loadtxt('ZBDeltaEta.csv', delimiter = ' ')
ZBDeltaEta = ZBDeltaEta[(ZBDeltaEta >= -6)]

VBFDeltaPhi = np.loadtxt('VBFDeltaPhi.csv', delimiter = ' ')
VBFDeltaPhi = VBFDeltaPhi[(VBFDeltaPhi >= -6)]

ZBDeltaPhi = np.loadtxt('ZBDeltaPhi.csv', delimiter = ' ')
ZBDeltaPhi = ZBDeltaPhi[(ZBDeltaPhi >= -6)]

VBFPt1 = np.loadtxt('VBFPt1.csv', delimiter = ' ')
VBFPt1 = VBFPt1[(VBFPt1 >= 0)]

ZBPt1 = np.loadtxt('ZBPt1.csv', delimiter = ' ')
ZBPt1 = ZBPt1[(ZBPt1 >= 0)]

VBFPt2 = np.loadtxt('VBFPt2.csv', delimiter = ' ')
VBFPt2 = VBFPt2[(VBFPt2 >= -6)]

ZBPt2 = np.loadtxt('ZBPt2.csv', delimiter = ' ')
ZBPt2 = ZBPt2[(ZBPt2 >= -6)]

VBFMass = np.loadtxt('VBFMass.csv', delimiter = ' ')
VBFMass = VBFMass[(VBFMass >= -6)]
    
ZBMass = np.loadtxt('ZBMass.csv', delimiter = ' ')
ZBMass = ZBMass[(ZBMass >= -6)]



#Preparing testing and training data 

extraZero = np.zeros(3362) #Padding 0s on the variables other than Pt1 since it has extra events to train on

extraZeroZB = np.zeros(9303) 

VBFDeltaEta = np.concatenate((VBFDeltaEta, extraZero))
VBFDeltaPhi = np.concatenate((VBFDeltaPhi, extraZero))
VBFMass = np.concatenate((VBFMass, extraZero))
VBFPt2 = np.concatenate((VBFPt2, extraZero))

VBFDeltaEta = VBFDeltaEta.reshape(len(VBFPt1), 1)
VBFDeltaPhi = VBFDeltaPhi.reshape(len(VBFPt1), 1)
VBFMass = VBFMass.reshape(len(VBFPt1), 1)
VBFPt1 = VBFPt1.reshape(len(VBFPt1), 1)
VBFPt2 = VBFPt2.reshape(len(VBFPt2), 1)


ZBDeltaEta = np.concatenate((ZBDeltaEta, extraZeroZB))
ZBDeltaPhi = np.concatenate((ZBDeltaPhi, extraZeroZB))
ZBMass = np.concatenate((ZBMass, extraZeroZB))
ZBPt2 = np.concatenate((ZBPt2, extraZeroZB))



ZBDeltaEta = ZBDeltaEta.reshape(len(ZBDeltaEta), 1)
ZBDeltaPhi = ZBDeltaPhi.reshape(len(ZBDeltaPhi), 1)
ZBMass = ZBMass.reshape(len(ZBMass), 1)
ZBPt1 = ZBPt1.reshape(len(ZBPt1), 1)
ZBPt2 = ZBPt2.reshape(len(ZBPt2), 1)


fVBFData = np.hstack((VBFDeltaEta, VBFDeltaPhi, VBFPt1, VBFPt2, VBFMass))
fVBFData = fVBFData.reshape(6395, 5)
print(fVBFData)
fZBData = np.hstack((ZBDeltaEta, ZBDeltaPhi, ZBPt1, ZBPt2, ZBMass))
fZBData = fZBData.reshape(10677, 5)

x = np.concatenate((fVBFData, fZBData), axis = 0)


len1 = len(fVBFData)
len2 = len(fZBData)

ones = np.ones(len1, np.int8)
zeros = np.zeros(len2, np.int8)

ones = ones.transpose()
zeros = zeros.transpose()

y = np.concatenate((ones, zeros), axis = 0)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


#--------------
# BDT 
#--------------

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME.R", n_estimators=200)
       

#bdt = RandomForestClassifier(n_estimators=200)
   

#bdt = DecisionTreeClassifier(max_depth = 1)

bdt.fit(x_train,y_train)

y_pred_kerasBDT = bdt.predict_proba(x_test)

y_pred_kerasBDT = np.delete(y_pred_kerasBDT, 0, 1)
fpr_kerasBDT, tpr_kerasBDT, thresholds_kerasBDT = roc_curve(y_test, y_pred_kerasBDT)

auc_kerasBDT = auc(1-fpr_kerasBDT, tpr_kerasBDT)


plt.plot(1 - fpr_kerasBDT, tpr_kerasBDT,label='BDT AUC (area = %0.2f)' % auc_kerasBDT)
#plt.scatter(1 - fpr_keras, tpr_keras)
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.title('BDT ROC')
plt.legend(loc="lower right")


