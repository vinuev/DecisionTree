# Predicting Forest Covers via Decision Trees/Forests

Here, we consider the [CoverType dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/) 
to systematically learn the hyper-parameters of a *Random Forest* classifier. 

We utilize the CrossValidator Scala API of Spark 2.2.3 to implement a fully automatic hyper-parameter tuning step. Specifically,
we use the 90% split of the data to initialize a 5-fold cross-validation loop over a fixed hyper-parameter setting. We use the classification accuracy over all of the 7 tree types 
(obtained from the MulticlassMetrics package of Spark) to measure the performance of your model for the given hyper-parameters.

We investigate the following range of parameters to also find the best amount of trees (using the “automatic” training mode) in your random forest:
```bash
– impurity <- Array("gini", "entropy"); – depth <- Array(10, 25, 50);
– bins <- Array(10, 100, 300));
– numTrees <- Array(5, 10, 20);
```
We then identify the best choice of all hyper-parameters obtain from the cross-validations as described above, and use the 
10% testing set to measure the accuracy of the final model over the 7 classes of trees given by the CoverType dataset.
