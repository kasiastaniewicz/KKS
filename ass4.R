#https://www.kaggle.com/tianxinl0106/classification-with-nnet
#https://beckmw.wordpress.com/tag/nnet/

#First set your working directory ( where you downloaded all the files with data)
setwd("/home/katarzyna/Downloads")

#some packages,() not entirely sure which one I need)
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

#Download the files
doorbell <- read.csv("danmini_doorbell.csv")
db_tcp <- read.csv("tcp.csv")
db_combo <- read.csv("combo.csv")
db_junk <- read.csv("junk.csv")
db_scan <- read.csv("scan.csv")
db_udp <- read.csv("udp.csv")

##Add additional column containing all kinds of an attack
db_tcp["Type"] <- "tcp"
db_combo["Type"] <- "combo.csv"
db_junk["Type"] <- "junk"
db_scan["Type"] <- "scan"
db_udp["Type"] <- "udp"

alldoorbell <- rbind(doorbell, db_tcp, db_combo, db_junk, db_scan, db_udp) 

#library(neuralnet)
library(caret)
library(nnet)

TrainingDataIndex <- createDataPartition(alldoorbell$Type, p=0.00001, list = FALSE)
trainingData <- alldoorbell[TrainingDataIndex,]
#trainingData
#testData <- alldoorbell[-TrainingDataIndex,]

#change later back repeats to 10
TrainingParameters <- trainControl(method = "repeatedcv", number = 10, repeats=1)

TrainingDataIndex2 <- createDataPartition(alldoorbell$Type, p=0.1, list = FALSE)
testData <- alldoorbell[TrainingDataIndex2,]



NNModel <- train(trainingData, trainingData$Type,
                 method = "nnet",
                 trControl= TrainingParameters,
                 preProcess=c("scale","center"),
                 na.action = na.omit
)


NNPredictions <-predict(NNModel, testData)

cmNN <-confusionMatrix(NNPredictions, testData$Type)
print(cmNN)






#Below i just tried to make a plot of neuron network

plot(NNModel)
plot(NNModel, nid = TRUE, all.out = TRUE, all.in = TRUE, bias = TRUE, 
       wts.only = FALSE, rel.rsc = 5, circle.cex = 5, node.labs = TRUE, 
       var.labs = TRUE, x.lab = NULL, y.lab = NULL, line.stag = NULL, 
       struct = NULL, cex.val = 1, alpha.val = 1, circle.col = lightblue,
       pos.col = black, neg.col = grey, max.sp = FALSE, ...)

library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

#plot each model
plot.nnet(NNModel)

#nn=neuralnet(Type~.,data=alldoorbell, hidden=3, act.fct = "logistic",
 #            linear.output = FALSE)
#plot(nn)


