# Stat-154-Project-2
Cloud Data
## EDA:
We started our project with exploratory data analysis. We first use some basic pandas methods to calculate the % of pixels for the different classes. Then we used Seaborn package to plot the expert label according to x, y coordinates. We also use “corr = image2.corr()” and “sns.heatmap” to plot the correlation of each variable. After that, we use “sns.boxplot” to plot the distribution of each variables hued by different expert labels. (All of the Problem 1 Section)
## Preprocess Data: 
To preprocess the data, we first drop all the unlabeled data since it doens't matter much for our prediction. You can choose not to run this part if you want to keep the unlabeled data for future usage. 
Then, we used two different ways of splitting the data. (Two ways of splitting the data are seperated in two different sections.)
### Method 1 Split
For the whole merged data, we first divided the ranges of x-coordinate and y-coordinate into 20 groups and split the data into 400 small data chunks according to their x-coordinate and y-coordinate. Then, using np.random, 70%, 10%, and 10% of the 400 data chunks were randomly selected as training set, validation set, and test set respectively.
(stored as features: X_train, X_val, X_test, labels: y_train, y_val, y_test)
(the overall data is store as: dat_cor as lists of pandas.DataFrame which contains 400 data chunks)
### Method 2 Split
We split the data for each of image1 and image3 into 130 groups by 129 lines that have the same slope as the downward diagonal line with different intercepts. For image 2, we split data for image2 vertically into 130 groups according to their x-coordinate. Eventually, we combined all the groups for three images together and using np.random randomly select 70%, 10%, and 10% of the 390 data chunks as training set, validation set, and test set.
(stored as features: X_train2, X_val2, X_test2, labels: y_train2, y_val2, y_test2)
(the overall data is store as: dat_list as lists of pandas.DataFrame which contains 390 data chunks)

### [function] CVgeneric: 
It's a generic cross validation function for Problem 3 to train and test the performance of different classifiers. It takes in:
     1. clf: generic classiffer,
     2. dat: training data chunks, 
     3. k: number of folds, 
     4. loss_func: loss function,
     5. train_val_idx: list of available index to select data chunks 
          from data chunks list if only consider data within training 
          set and validation set
     6. test_dat: list of test data in form of [features, labels]
     7. features list to train the model on: features (Default is all 8 features
          excluding x,y coordinates)
     8. verbose: if True, it will print out all loss, accuracy, and 
         test accuracy for all K-fold CV (Default is True)
         
   Output:
     1. K-fold CV loss on the training set(dictionary) in the form of 
          {"CV1":loss} : results
     2. K-fold CV accuracy on the val set(dictionary): accuracy
     3. K-fold CV test accuracy (dictionary): test_accuracy dic
     4. K-fold CV models(list): model_list 
     
### Normalize:
For better accuracy rate, we first perform normalization on both data results from different ways of splitting. We minus all the data chunks by the columnwise mean of X_train_val (merged dataframes of training and validation set), and divide all data by the columnwise std of X_train_val.

## Modeling (Problem 2)
#### For this part, both Method 1 Data and Method 2 Data are performed similarly and for each classifier, we seperate them into Method 1, Method 2 chunk
### Logistic Regression: 
To fit in logistic regression model, we used “linear_model.LogisticRegression(solver = 'lbfgs', multi_class='multinomial’)” Then, we used “log_loss” and “CVgeneric” to calculate the loss, test accuracy, and validation accuracy. Then we use “np.mean” to calculate average test accuracy and validation accuracy. Then, we used the scatter plot and line plot in matplotlib to plot a 5-Fold CV Test Set & Validation Set accuracy, 5-Fold CV Loss (Logistic Regression) for both Method 1 and Method 2.
### QDA:
Firstly, we checked the normality of features by using “sns.distplot”. Then, we used “discriminant_analysis.QuadraticDiscriminantAnalysis(priors=prior)” to fit the QDA model. Then, we used “log_loss” and “CVgeneric” to calculate the loss, test accuracy, and validation accuracy. Then we use “np.mean” to calculate average test accuracy and validation accuracy.
### kNN:
Firstly, we did hyperperameter tuning with numbers 15, 30, 35, 40, 45. After we fit kNN with “neighbors.KNeighborsClassifier(n_neighbors=n)”, we found that 35 is the best number of neighbors.Then, we used “log_loss” and “CVgeneric” to calculate the loss, test accuracy, and validation accuracy. Then we use “np.mean” to calculate average test accuracy and validation accuracy. After that, we used the scatter plot and line plot in matplotlib to plot “Number of Neighbors for kNN vs Average test accuracy” for both Method 1 and Method 2 data.
### Random forest:
Firstly, we used “ensemble.RandomForestClassifier(n_estimators=1000,  criterion="entropy", max_depth=5, min_samples_split=3, max_features='log2’)” to fit the random forest model. Then, we used “log_loss” and “CVgeneric” to calculate the loss, test accuracy, and validation accuracy. Then we use “np.mean” to calculate average test accuracy and validation accuracy. After that, we used the scatter plot and line plot in matplotlib to plot a 5-Fold CV Test Set & Validation Set accuracy (Random forest), 5-Fold CV Loss (Random forest) for both Method 1 and Method 2.

## Calc ROC & Plot ROC (Problem 3b):
For each of the classifier above, on the Calc Roc part at the end of each classifier, we first pick out the model with best test accuracy during the cross validation, then we calculate the respective fpr,tpr,thre,roc_auc using sklearn.metrics. Next, we find the optimal cutoff point on the graph by finding the intersection of the line connecting the left-upper corner and the right-lower corner of the unit square (the line TP = FP), and the ROC curve for each classifier. We use function line(p1, p2), and function intersection(L1, L2) to claculate the intersections of two lines, and plotting it.
### [function] line(p1, p2): 
Input:
  p1: point 1, a list in the form of [x_coordinate, y_coordinate]
  p2: point 2, a list in the form of [x_coordinate, y_coordinate]
  
Output:
  suppose ax + by = c
  A: Coefficient of x in the equation (a)
  B: Coefficient of y in the equation (b)
  C: the intercept the line (c)
### [function] intersection(L1, L2): 
Function for calculating the intersection of two line using Cramer's Rule
Input:
      suppose ax + by = c
      L1: line 1 in the form of [coeff of x1, coeff of y1, intercept]
      L1: line 1 in the form of [coeff of x2, coeff of y2, intercept] 
Output: 
      X coordinate of the intersection between L1 and L2
      Y coordinate of the intersection between L1 and L2
      or 
      False if L1 and L2 don't have intersection

## Confusion Matrix (Problem 3c):
### [function] plot_confusion_matrix:
This function will help you plot the confusion matrix. It takes in the true label of y_test(y_true), and the predicted labels from your classifier(y_pred), and you can choose to normalize the data or not(normalize, default is False), you can also input the title (title, default is None) and color (cmap, defualt is Blues) for the plot.  

## Convergence plot of coefficients (Problem 4a.):
We did a bootstrapping for 1000 times for logistic regression on a randomly selected, 25% of the X_train_val, and Y_train_val data. Then we fit our logistics regression model on that data, and record the coefficients for 8 features. Next, we used line plot to plot the data to observe the stability of the coefficients of the features.

## Box Plot for Random Forest (Problem 4b.):
For our best classifier: Random Forest, we first record all the pixel that are misclassified, all the cloud pixel which are misclassied as cloudless pixel, and all the cloudless pixel which are misclassified as cloud pixel for each way of splitting the data. Then to visualize the pattern on the misclassification error for Random Forest, we plot out the box plot of all misclassified data, of misclassified cloud pixel, misclassified cloudless pixel, and the X_test set. 

## Features V.S. Test Accuracy Rate(Problem 4c):
We are curious about the relationship between features and test accuracy rate. Also, KNN performed pretty well for both ways of splitting the data. So, we loop through all different combanitions of n-features (3<= n <= 8), and perform the knn model with 35 n_neighbors (the param we chose after hyperparameter tuning in Modeling part) on data set, and record its accuracy rate. Then we plot the test accuracy rate for each combanition of features.
