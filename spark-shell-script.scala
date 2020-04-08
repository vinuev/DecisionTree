
//A decision tree is a greedy algorithm that performs a recursive binary partitioning of the feature space.

import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._

import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._

import org.apache.spark.rdd._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.util.MLUtils // spark.mllib supports decision trees for binary and multiclass classification and for regression, using both continuous and categorical features


val sampleSize = 0.01 // use 1 percent sample size for debugging!
val rawData = sc.textFile("./data/covtype.data").sample(false, sampleSize)

val data0 = rawData.map { line =>
  val values = line.split(',').map(_.toDouble)
  val featureVector = Vectors.dense(values.init)
  val label = values.last - 1
  LabeledPoint(label, featureVector)
}

// --- improve the feature vectors by replacing the "1-hot" encoding --- 

val data = rawData.map { line =>
  val values = line.split(',').map(_.toDouble)
  val wilderness = values.slice(10, 14).indexOf(1.0).toDouble
  val soil = values.slice(14, 54).indexOf(1.0).toDouble
  val featureVector =
    Vectors.dense(values.slice(0, 10) :+ wilderness :+ soil)
  val label = values.last - 1
  LabeledPoint(label, featureVector)
}

val Array(trainData, valData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))

trainData.cache() // subset of dataset used for training
valData.cache() // subset of dataset used for optimization of hyperparameters
testData.cache() // subset of dataset used for final evaluation ("testing")

trainData.count()
valData.count()
testData.count()

// --- train a first model ---

val model = DecisionTree.trainClassifier(trainData, 7, Map[Int,Int](), "gini", 4, 100)

println(model.toDebugString)

def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionsAndLabels = data.map(example =>(model.predict(example.features), example.label))
    new MulticlassMetrics(predictionsAndLabels)
  }// returns metrics containing the label (0 to 7) and prediction

val metrics = getMetrics(model, valData)

(0 until 7).map(label => (metrics.precision(label), metrics.recall(label))).foreach(println)


// --- train a random forest based on arbitrary hyperparameters --- 

val forest = RandomForest.trainClassifier(data, 7, Map(10 -> 4, 11 -> 40), 20, "auto", "entropy", 10, 100)
    
val input = "2709,125,28,67,23,3224,253,207,61,6094,0,29"
val vector = Vectors.dense(input.split(',').map(_.toDouble))

forest.predict(vector)



// --- optimize the hyperparameters of the random forest model using cross-validation--- 

var nFold = 0;

val cvData = MLUtils.kFold(data, 5, 0); // kFold function
// return an array of 5 elements. Each element is a pair of RDD in which the first element is the training set (4/5 of the corresponding fold size)
// and the second element is the validation set (1/4 of the corresponding fold size)
var m = scala.collection.mutable.Map[(String, Int, Int, Int), List[Double]]() // Map structure that keeps the accuracy values for each parameter setting over the 5 folds.
 
cvData.foreach { couple =>
      nFold = nFold + 1
      print("Fold #" + nFold + "\n")
      val (foldTrainingSet, foldValidationSet) = couple
      println("Training set size: " + foldTrainingSet.count)
      println("Validation set size: " + foldValidationSet.count)
         
      println("hyperparameters optimization started now...")
      val evaluations =
        for (
          impurity <- Array("gini", "entropy");
          depth <- Array(10, 20, 30);
          bins <- Array(100, 200, 300);
          numTrees <- Array(10, 20, 30)
        ) yield {
        	  val forest = RandomForest.trainClassifier(foldTrainingSet, 7, Map(10 -> 4, 11 -> 40), numTrees, "auto", impurity, depth, bins)
      	          val predictionsAndLabels = foldValidationSet.map(example => (forest.predict(example.features), example.label) // RDD of couples (predicted_label, actual_label)
                     )
                  val accuracy = new MulticlassMetrics(predictionsAndLabels).accuracy // measures the accuracy of the predictions
                     ((impurity, depth, bins, numTrees), accuracy)
                } // returns a couple made of a quadruple (corresponding to the hyperparameters) and the accuracy obtained with these parameters

	println("hyperparameters optimization finished now.")

 	evaluations.foreach { t => 
		m(t._1) = if (m.contains(t._1)) m(t._1) :+ t._2 else List(t._2)
	}
    }



// --- finally train a random forest based on the improved vectors --- 

val forest = RandomForest.trainClassifier(
  trainData2.union(valData2), 7, Map(10 -> 4, 11 -> 40), 20,
    "auto", "entropy", 30, 300)
    
val input = "2709,125,28,67,23,3224,253,207,61,6094,0,29"
val vector = Vectors.dense(input.split(',').map(_.toDouble))

forest.predict(vector)
