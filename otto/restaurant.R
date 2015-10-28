#library('e1071')
library('lubridate')
#library('randomForest')
#library('cvTools')
library('caret')
library(caretEnsemble)
library(Metrics)
library(doMC)
registerDoMC(cores=1)

is.nan.data.frame <- function(x)
do.call(cbind, lapply(x, is.nan))


logloss <- function (data, lev = NULL, model = NULL) {
    LogLos <- function(actual, pred, eps = 1e-15) {
        stopifnot(all(dim(actual) == dim(pred)))
        pred[pred < eps] <- eps
        pred[pred > 1 - eps] <- 1 - eps
        -sum(actual * log(pred)) / nrow(pred)
    }
    if (is.character(data$obs)) data$obs <- factor(data$obs, levels = lev)
    pred <- data[, "pred"]
    obs <- data[, "obs"]
    isNA <- is.na(pred)
    pred <- pred[!isNA]
    obs <- obs[!isNA]
    data <- data[!isNA, ]
    cls <- levels(obs)

    if (length(obs) + length(pred) == 0) {
        out <- rep(NA, 2)
    } else {
        pred <- factor(pred, levels = levels(obs))
        require("e1071")
        out <- unlist(e1071::classAgreement(table(obs, pred)))[c("diag", "kappa")]
        probs <- data[, cls]
        actual <- model.matrix(~ obs - 1)
        out2 <- LogLos(actual = actual, pred = probs)
    }
    out <- c(out, out2)
    names(out) <- c("Accuracy", "Kappa", "LogLoss")

    if (any(is.nan(out))) out[is.nan(out)] <- NA
    out
}
rfFuncs$summary = logloss


train <- read.csv("train.csv",header=TRUE)
test <- read.csv("test.csv",header=TRUE)


train_cols<-train[,c(2:94)]
labels<-train[,95]
testdata<-test[,2:94]

train_cols <- data.frame(lapply(train_cols,as.numeric))
testdata<-data.frame(lapply(testdata,as.numeric))


#fit.rf <- randomForest(revenue ~ ., data=train)
#print(fit.rf)
#importance(fit.rf)
#plot(fit.rf)
#plot( importance(fit.rf), lty=2, pch=16)
#lines(importance(fit.rf))
#imp = importance(fit.rf)

ctrl <- trainControl(method="cv", number=10, classProbs=T, savePred=T, summaryFunction = logloss,index=createResample(labels, 25))
#ctrl <- trainControl(method="cv", savePred=T, index=createResample(labels, 25))
#set.seed(1500)
tuneGrid <- data.frame(.mtry=2)
mod <- train(x=as.matrix(train_cols),y=labels,method="rf",trControl=ctrl, ntree=501, maximize=F,metric='LogLoss',tuneGrid=tuneGrid)
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
#mod <- caretList(x=train_cols,y=labels,methodList=method_list,
#                 trControl=ctrl, tuneList=tuneList)
#greedy_ensemble <- caretEnsemble(mod)
#summary(greedy_ensemble)
predictions<-as.data.frame(predict(mod,newdata=testdata, type="prob"))

submit<-as.data.frame(cbind(test[,1],predictions))
colnames(submit)<-c("Id","Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")
write.csv(submit,format(Sys.time(), "%m-%d_%H:%M_submission.csv"),row.names=FALSE,quote=FALSE)
