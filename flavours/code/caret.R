library(caret)
library(Metrics)
library(doMC)
registerDoMC(cores=1)

train <- read.csv('../input/training.csv', header=T)
test <- read.csv('../input/test.csv', header=T)

train$production <- NULL
train$mass <- NULL
train$min_ANNmuon <- NULL
train$SPDhits <- NULL

train$id <- NULL
test.ids <- test$id
test$ids <- NULL
test$SPDhits <- NULL

train$signal[train$signal==1] <- 'Positive'
train$signal[train$signal==0] <- 'Negative'

ctrl <- trainControl(method="cv", number=10, classProbs=T, savePred=T, summaryFunction=twoClassSummary)

tuneGrid <- data.frame(.mtry=2)
mod <- train(as.factor(signal) ~., data=train,method="rf",trControl=ctrl, ntree=51, maximize=F,metric='ROC',tuneGrid=tuneGrid)
mod
mod <- train(as.factor(signal) ~., preProcess="BoxCox", data=train,method="rf",trControl=ctrl, ntree=51, maximize=F,metric='ROC',tuneGrid=tuneGrid)
mod

predictions<-as.data.frame(predict(mod,newdata=as.matrix(test), type="prob"))

submit<-as.data.frame(cbind(test.ids,predictions[,2]))
colnames(submit)<-c("id","prediction")
submit$id <- as.integer(submit$id)
write.csv(submit,format(Sys.time(), "%m-%d_%H:%M_submission.csv"),row.names=FALSE,quote=FALSE)

