# -- coding: utf-8 --
import sys

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import input_file_name
from pyspark.sql.functions import lit
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import NaiveBayes
from pyspark.sql.functions import regexp_replace, col
from pyspark.sql.functions import col
from pyspark.sql.functions import length
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
## Constants
APP_NAME = "LAB3"

# Configure Spark
conf = SparkConf().setAppName(APP_NAME)
conf = conf.setMaster("local[*]")
sc   = SparkContext(conf=conf)
spark = SparkSession(sc)
sqlContext = SQLContext(sc)

# LABELS -description
# Business - 0
# Entertainment - 1
# Politics - 2
# Sports - 3

df = spark.read.text('./lab3/data/Business/*')
df = df.withColumn("filename", input_file_name())
df = df.withColumn("Category", lit("Business"))

dfS = spark.read.text('./lab3/data/Sports/*')
dfS = dfS.withColumn("filename", input_file_name())
dfS = dfS.withColumn("Category", lit("Sports"))
dfT = df.union(dfS)

df = spark.read.text('./lab3/data/Entertainment/*')
df = df.withColumn("filename", input_file_name())
df = df.withColumn("Category", lit("Entertainment"))
dfT = dfT.union(df)

dfS = spark.read.text('./lab3/data/Politics/*')
dfS = dfS.withColumn("filename", input_file_name())
dfS = dfS.withColumn("Category", lit("Politics"))
dfT = dfT.union(dfS)

dfT = dfT[dfT.value != '']
dfT = dfT.where(length(col("value")) > 20)

#---------------------- PIPELINE -----------------
# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="value", outputCol="words", pattern="\\W")
# stop words
add_stopwords = ["real","here's","isn't","similar","far","end","putting","continued","led","recently","makes","remains","high","sought","2015","able","brought","based","print","thousands","whether","s.&p","thought","increasingly","appeared","previously","starting","whose","example","affect","side","ads","tried","needed","however","responded","them.","don't","received","asked","things","adding","got","help","possible","january","here.","monday.","heard","name","take","ms.","nearly","across","job.","among","gone","certain","reached","city","consider","joined","deeply","tuesday.","takes","pushed","huge","held","number","thing","april","4","seen","around","using","often","back","become","paper | subscribe","8","x","2014","greater","went","given","comes","specifically","already","coming","sometimes","asking","became","made","still","years","still","though","recent","time","collapse","screens","make","everything","supported","might","seem","keep","bunt","contributor","op","ed","difference","longtime","california","father","follow","twitter","near","took","ones","could","left","want","long","either","moment","saw","never","willing","facing","nyt","big","raises","try","car","expected","event","yesterday","would","pay","one","many","ms","mr","elsewhere","hold","first","three","rounds","add","stop","come","open","ran","another","going","big","met","start","soon","today","last","leaving","next","two","love","please","email","thoughts","suggestions","us","home","inbox","morning","sign","best","saying","likely","maybe","third","need","red","cannot","required","york","times","context","include","critic","corner","look","says","verify","robot","clicking","box","invalid","address","enter","select","newsletter","subscribe","view","newsletters","behind","scenes","called","run","com","interested","think","home","good","watching","soon","clear","saying","five","via","anonymous","wrote","sources","citing","months","knowing","less","columnist","trouble","find","need","although","especially","playlist","wsj","low","updates","live","find","bizday","options","information","view","\n","http","https","amp","rt","t","c","the","i","me","my","myself","we","our","ours","ourselves","you","you're","you've","you'll","you'd","your","yours","yourself","yourselves","he","him","his","himself","she","she's","her","hers","also","herself","it","it's","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","that'll","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","don't","should","should've","now","d","ll","m","o","re","ve","y","ain","aren","aren't","couldn","couldn't","didn","didn't","doesn","doesn't","hadn","hadn't","hasn","hasn't","haven","haven't","isn","isn't","ma","mightn","mightn't","must","mustn","mustn't","needn","needn't","shan","shan't","shouldn","shouldn't","wasn","wasn't","weren","weren't","won","won't","wouldn","wouldn't","feedback","monday","tuesday","wednesday","thursday","friday","saturday","sunday","much","all","news","days","day","advertisement","-","--","”","»",")","&","—","0","1","$","rt","abc","said","yes","something","ago","basically","suggest","since","anything","nc","used","say","see","da","h","well","put","k","w","wh","&amp","c","u","co","isnt","doj","i'm","yet","fo","let","le","sure","get","we're","=","ph","sgt","told","tell","see","b","like","dont","'he","rt_com:","ied","j","sa","he","I'm","n","lot","we'll","can't","let","didnt","f","wont","e","page","i","go","put","new","de","done","later","without","may","It","p","gave","came","within","cut","use","way","al","four","others","set","know","year","found"]
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
# bag of words count
countVectors = CountVectorizer(inputCol="filtered", outputCol="rawFeatures", vocabSize=20000, minDF=5)
# Give labels to categories
label_stringIdx = StringIndexer(inputCol = "Category", outputCol = "label")
# Document frequency
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5) 
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, idf, label_stringIdx])
# Fit the pipeline to training documents.
pipelineFit = pipeline.fit(dfT)
dataset = pipelineFit.transform(dfT)

# #----------------- CLASSIFICATION -------------------
(trainingData, testData) = dataset.randomSplit([0.8, 0.2])
lr = LogisticRegression(maxIter=20, regParam=0.1, elasticNetParam=0)
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("Category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
hLR = evaluator.evaluate(predictions)

nb = NaiveBayes(smoothing=2)
model = nb.fit(trainingData)
predictions = model.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
   .select("Category","probability","label","prediction") \
   .orderBy("probability", ascending=False) \
   .show(n = 10, truncate = 30)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
hNB = evaluator.evaluate(predictions)


# #----------------- UNKNOWN TEST DATA -------------------
(trainingData, testData, unknownDataSet) = dataset.randomSplit([0.6, 0.3, 0.1])
predictionsLR = lrModel.transform(unknownDataSet)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
unknownTestLRAccuracy = evaluator.evaluate(predictionsLR)

nb = NaiveBayes(smoothing=2)
predictionsNB = model.transform(unknownDataSet)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
unknownDataNBAccuracy = evaluator.evaluate(predictionsNB)

# #--------------- output -----------------------------------
print('Test Data:')
print('LR Accuracy')
print(hLR)
print('NB Accuracy')
print(hNB)
print('Unknown Test Data:')
print('LR Accuracy')
print(unknownTestLRAccuracy)

print('NB Accuracy') 
print(unknownDataNBAccuracy)
spark.stop()