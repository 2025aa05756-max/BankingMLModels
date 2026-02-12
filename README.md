a. **Problem Statement**


b. **Dataset Description:**

The data is related with direct marketing campaigns of a Portuguese banking institution. 
The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 
The datasets consists of  bank.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010).
The classification goal is to predict if the client will subscribe (yes/no) a term deposit.

c. **Models Used and comparison of metrics**

Model Name	            Model Accuracy  AUC Score   Precision	  Recall  	Score  	MCC Score  
XGBoost  	              0.907995        0.929058    0.634845    0.502836	0.561181	0.514894  
Random Forest	          0.905341	      0.926016	  0.659306	  0.395085	0.494090 	0.463289  
Logistic Regression     0.901360        0.905499	  0.645105	  0.348771	0.452761	0.426959  
Naive Bayes (Gaussian)	0.864315 	      0.823126	  0.433096	  0.517013	0.471349 	0.396248  
KNN	                    0.893619 	      0.808369	  0.586022 	  0.309074	0.404703	0.374213  
Decision Tree 	        0.872719	      0.701213 	  0.457842	  0.477316	0.467376	0.395246  

Observations:

**Model Name & Observations  **

XGBoost  	XGBoost is the most effective at identifying customers who are genuinely interested in term deposits.
          Highest AUC (0.929) mean best at ranking customers that would subscribe
          Highest MCC (0.514) would mean most reliable predictions overall
          Has strong balance of precision and recall
Random Forest	
          High accuracy and AUC
          High precision (0.659)
          Low recall (0.395). This would mean this model might miss many potential subscribers
Logistic Regression	
          High accuracy, good AUC score
          Due to interpretable co-efficients, this is a much better explainable model for transparency
          Low recall and hence would miss many potential subscribers
Naive Bayes (Gaussian)	
          Best recall and captures potential customers. 
          But low precision would mean, there could be false positives
KNN	
          Low recall and AUC. KNN is not suitable for this type of dataset.
          It fails to identify many potential subscribers and there is no interpretability
Decision Tree	
          Lowest AUC and lower accuracy too. 
          Decision Trees are easy to visualize but not reliable enough for real world marketing decisions. 
          They overfit and do not generalize well

