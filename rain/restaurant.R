#library('e1071')
library('lubridate')
#library('randomForest')
#library('cvTools')
library('caret')
library(caretEnsemble)
train <- read.csv("train_2013.csv",header=TRUE)
test <- read.csv("test_2014.csv",header=TRUE)


train_cols<-train[,c(1:19)]
labels<-as.matrix(train[,20])
testdata<-test[,1:19]

train_cols <- data.frame(lapply(train_cols,as.numeric))
testdata<-data.frame(lapply(testdata,as.numeric))


#fit.rf <- randomForest(revenue ~ ., data=train)
#print(fit.rf)
#importance(fit.rf)
#plot(fit.rf)
#plot( importance(fit.rf), lty=2, pch=16)
#lines(importance(fit.rf))
#imp = importance(fit.rf)

ctrl <- trainControl(method="cv", savePred=T, index=createResample(labels, 25))
#set.seed(1500)
#tuneGrid <- data.frame(ntree=50000,mtry=3)
mod <- train(x=as.matrix(train_cols),y=as.vector(labels),method="rf",trControl=ctrl, ntree=501)#,tuneGrid=tuneGrid)
#mod
#predictions<-as.data.frame(predict(mod,newdata=testdata))

#Used when Px are numeric
#method_list = c('gbm','bdk','blackboost','bstLs','gaussprPoly','gcvEarth','kernelpls','kknn','knn','pls','rpart','rpart2','simpls','spls','svmPoly','svmRadial','svmRadialCost','treebag','widekernelpls','xyf','rf')
#tuneList = list(rf1=caretModelSpec(method='rf', ntree=1000),
#		bag1=caretModelSpec(method='treebag', nbagg=500),
#		bag2=caretModelSpec(method='treebag', nbagg=1000),
#		#spls1=caretModelSpec(method='spls', select='simpls'),
#		svm1=caretModelSpec(method='svmRadialCost',tuneGrid=expand.grid(C=c(2.4))),
#		rf1=caretModelSpec(method='rf', ntree=1000, preprocess='pca'),
#                bag1=caretModelSpec(method='treebag', nbagg=500, preprocess='pca'),
#                bag2=caretModelSpec(method='treebag', nbagg=1000, preprocess='pca'),
#                #spls1=caretModelSpec(method='spls', select='simpls', preprocess='pca'),
#                svm1=caretModelSpec(method='svmRadialCost', preprocess='ica'),
#		rf1=caretModelSpec(method='rf', ntree=1000, preprocess='ica'),
#                bag1=caretModelSpec(method='treebag', nbagg=500, preprocess='ica'),
#                bag2=caretModelSpec(method='treebag', nbagg=1000, preprocess='ica'),
#                #spls1=caretModelSpec(method='spls', select='simpls', preprocess='ica'),
#                svm1=caretModelSpec(method='svmRadialCost', preprocess='ica'))
#method_list = c('gbm','bstLs','gcvEarth','kernelpls','kknn','knn','pls','rpart','rpart2','simpls','treebag','widekernelpls','rf')
#mod <- caretList(x=train_cols,y=as.vector(labels),methodList=method_list,
#                 trControl=ctrl, tuneList=tuneList)
greedy_ensemble <- caretEnsemble(mod)
summary(greedy_ensemble)
predictions<-as.data.frame(predict(greedy_ensemble,newdata=testdata))

submit<-as.data.frame(cbind(test[,1],predictions))
colnames(submit)<-c("Id","Prediction")

write.csv(submit,format(Sys.time(), "%m-%d_%H:%M_submission.csv"),row.names=FALSE,quote=FALSE)
