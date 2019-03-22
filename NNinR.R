#     Our prime reason for trying to run Neural Networks in R was that we wanted to compare how the model works 
#     in different programming languages and if the result are similiar.
#     The try was not a big success but we have learnt something so that is why we include the scipt in the project.
#     
#     One of the problems we faces in Python was that the UDP and TCP tended to be misclassified 
#     What is interesting is that the classification belowe performs much better when it comes to distingusing between TCP and UDP.
#
#     The disadvanatge of running the program in R is that the computation power is much lower than in Python. 
#     I run the NN only on the doorbell data, in fact using only a subset of the doorbell data. 
#
#     I tried to run the NN using tensorflow and keras, however It did not work. 
#     I tried to follow the example on https://keras.rstudio.com/


#     First set your working directory ( where you downloaded all the files with data)
#     CHANGE THE BELOW TO YOUR WORKING DIRECTORY
setwd("/home/katarzyna/Downloads")


#      Some packages that are useful to download. 
#      [Some of them I did not use in the end, so if an error appear it might be fine.]
install.packages("neuralnet")
library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(ggplot2)
library(lattice)
library(caret)
library(C50)
library(kernlab)
library(mlbench)
library(randomForest)
library(caretEnsemble)
library(MASS)
library(klaR)
library(nnet)
library(neurlnet)


#      Below I run the NN on the doorbell data. 
#      The reason I could not include all 5 devices is that the size of the "DATA.csv" is too large to upload to R.


#     Download the files with all kinds of attacks on the doorbell
doorbell <- read.csv("danmini_doorbell.csv")
db_tcp <- read.csv("tcp.csv")
db_combo <- read.csv("combo.csv")
db_junk <- read.csv("junk.csv")
db_scan <- read.csv("scan.csv")
db_udp <- read.csv("udp.csv")

##     Add additional column containing all kinds of an attack
db_tcp["Type"] <- "tcp"
db_combo["Type"] <- "combo"
db_junk["Type"] <- "junk"
db_scan["Type"] <- "scan"
db_udp["Type"] <- "udp"

#     Combine all the types of attacks into one data frame
alldoorbell <- rbind(doorbell, db_tcp, db_combo, db_junk, db_scan, db_udp) 

#     I split the data into training and testing data. I pick the value of p to be small, otherwise the R crush.

TrainingDataIndex <- createDataPartition(alldoorbell$Type, p=0.000001, list = FALSE)
trainingData <- alldoorbell[TrainingDataIndex,]

#     We could increase the number of repeats to 10, I put 2 to speed up
TrainingParameters <- trainControl(method = "repeatedcv", number = 10, repeats=2)

#     I chose the training data to be small as well. The percentage can be changed by chaning p.
TestingDataIndex2 <- createDataPartition(alldoorbell$Type, p=0.00005, list = FALSE)
testData <- alldoorbell[TestingDataIndex2,]


#     Below the train the model. 
NNModel <- train(trainingData, trainingData$Type,
                 method = "nnet",
                 trControl= TrainingParameters,
                 preProcess=c("scale","center"),
                 na.action = na.omit
)

#      We run ht emodel on the testing dataset,get the predicted values and compare them to the real ones.   
NNPredictions <-predict(NNModel, testData)

#
cmNN <-confusionMatrix(NNPredictions, testData$Type)
print(cmNN)

#    It is hard to read much from the plot below, because there are 115 input values 

library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

plot.nnet(NNModel)





#    Field try of using tensorflow and keras to run the NN
#    First, download the librares( some might require closing and opening R again)
install.packages("devtools")
library(devtools)
devtools::install_github("rstudio/tensorflow")
library(tensorflow)
install_tensorflow()
devtools::install_github("rstudio/keras")
library(keras)


#    Below is an example of creating such a model
model <- keras_model_sequential()

model %>%
  layer_dense(units = 32, input_shape = c(115)) %>%
  layer_activation('relu') %>%
  layer_dense(units = 10) %>%
  layer_activation('softmax')

model %>% compile(
  optimizer = 'rmsprop',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)
summary(model)

#    Having specified the model, we prepare the data. 
#    We split the training data into the parameters corresponding to all the features of the data apart from the attack type, set it to be x_train
#    y_train is the column with the types of the attacks.

x_train=trainingData[,1:115]
y_train=trainingData[,116]
x_train=data.matrix(x_train)

#    Here is where the model crushes. I did not manage to debug it.
history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)


#https://tensorflow.rstudio.com/tfestimators/articles/examples/iris_custom_decay_dnn.html


#Below i just tried to make a plot of neuron network

plot(NNModel)

plot(NNModel, nid = TRUE, all.out = TRUE, all.in = TRUE, bias = TRUE, 
     wts.only = FALSE, rel.rsc = 5, circle.cex = 5, node.labs = TRUE, 
     var.labs = TRUE, x.lab = NULL, y.lab = NULL, line.stag = NULL, 
     struct = NULL, cex.val = 1, alpha.val = 1, circle.col = lightblue,
     pos.col = black, neg.col = grey, max.sp = FALSE, ...)





#Below i put the sources I used for programming. 
#https://www.datacamp.com/community/tutorials/neural-network-models-r
#https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/
#https://www.kaggle.com/tianxinl0106/classification-with-nnet
#https://beckmw.wordpress.com/tag/nnet/


