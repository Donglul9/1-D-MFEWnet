Data_importing.py is used for importing the experimental dataset; 
Train_model.py is used for the construction and training of the 1-D MFCNN network model, and the dataset used here is train_3.csv; 
TrainSet_Validation. py is used to validate the known class test set, i.e., test_3.csv dataset; 
Fit_weibull.py is used to construct the Weibull model and get the MAV; 
Classifier.py is used to classify the data to get the event types of the test data; 
TestSet_Validation.py is used to test the full event set, using the dataset of test_5.csv. set is test_5.py.


In order to save the time, we wrote the relevant pseudo-code for elaboration. The first stage of the EW method provides N MAVs and N different Weibull models for our N-class classification problem.The Weibull model computation stage is shown in Algorithm .1, starting from the output of the last FC layer of the network, i.e., the AV. The AV is first recomputed by applying an activation function to map the  to different domains as described in line 1 of Algorithm .1. Then a respective mean activation vector (MAV) is formed for each sample of each class as described in line 3 of Algorithm .1. The Euclidean distance between  and MAV is then computed for each sample data for each class, as described in Algorithm .1 line 4. Finally, a Weibull distribution was fitted to the η maximal distances for each class to obtain a Weibull model for each class. as described in line 5 of Algorithm .1.

![image](https://github.com/Donglul9/1-D-MFEWnet/assets/154125395/a925c739-3180-4a3c-9df4-51bdd8eab35f)


Given a test data, the second stage of the EW method is described in Alg.2. First the AV is recalculated by applying an activation function to map the  to different domains as described in line 1 of Algorithm .2. Secondly the scalar CD is obtained by calculating the Euclidean distance between the AV of the test data and the MAV of each class, as described in line 3 of Algorithm .2. Subsequently the Wscore is obtained from CD and the Weibull model as described in Algorithm .2 line 4. This is the first factor that changes (k). The other factor is α, which specifies the number of largest elements in the  to be modified. Assign 1 to the largest element in the ,  to the second largest value, and so on. (k) is then modified to ) by using a recalculation of the Wscore, as described in line 5 of Algorithm .2. The difference between (k) and ) is then calculated and added together as the score for the unknown class, as described in Algorithm .2 line 6. Finally, the  is input to the SoftMax function to obtain the probability of each classification and threshold, as described in lines 7-8 of Algorithm .2.

![image](https://github.com/Donglul9/1-D-MFEWnet/assets/154125395/0571f651-1c3a-4df9-bc0c-8b92e8963674)
