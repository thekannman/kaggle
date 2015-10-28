require(xgboost)
require(caret)
require(qdapTools)
require(ggplot2)

factorToNumeric <- function(train, test, response, variables, metrics){
  temp <- data.frame(c(rep(0,nrow(test))), row.names = NULL)

  for (variable in variables){
    for (metric in metrics) {
      x <- tapply(train[, response], train[,variable], metric)
      x <- data.frame(row.names(x),x, row.names = NULL)
      temp <- data.frame(temp,round(lookup(test[,variable], x),2))
      colnames(temp)[ncol(temp)] <- paste(metric,variable, sep = "_")
    }
  }
  return (temp[,-1])
}

SumModelGini <- function(solution, submission) {
  df = data.frame(solution = solution, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE),]
  df
  df$random = (1:nrow(df))/nrow(df)
  df
  totalPos <- sum(df$solution)
  df$cumPosFound <- cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
  df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
  df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
  return(sum(df$Gini))
}

NormalizedGini <- function(solution, submission) {
  SumModelGini(solution, submission) / SumModelGini(solution, solution)
}

evalgini <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- NormalizedGini(exp(as.numeric(labels)),exp(as.numeric(preds)))
  return(list(metric = "Gini", value = err))
}

# Load data
train <- read.csv('../input/train.csv', header=T, stringsAsFactors=T)
test <- read.csv('../input/test.csv', header=T, stringsAsFactors=T)

train$Hazard <- log(train$Hazard)
y = train$Hazard
test.ids <- test$Id

# Remove unnecessary columns
train$Id <- NULL
test$Id <- NULL

train.factor.cols <- c(5:10,12:13,16:18,21,23,29:31)
test.factor.cols <- c(4:9,11:12,15:17,20,22,28:30)
#train[,-c(1,train.factor.cols)] <- apply(train[,-c(1,train.factor.cols)],2,function(x) (x-min(x))/(max(x)-min(x)))
#test[,-test.factor.cols] <- apply(test[,-test.factor.cols],2,function(x) (x-min(x))/(max(x)-min(x)))

train.factors <- names(train)[train.factor.cols]
train[,train.factor.cols] <- lapply(train[,train.factor.cols], function(x) as.factor(x))
test.factors <- names(test)[test.factor.cols]
test[,test.factor.cols] <- lapply(test[,test.factor.cols], function(x) as.factor(x))

train.factor.stats <- factorToNumeric(train, train, "Hazard", train.factors, c("mean","median","max","min","sd"))
train <- cbind(train,train.factor.stats)
test.factor.stats <- factorToNumeric(train, test, "Hazard", test.factors, c("mean","median","max","min","sd"))
test <- cbind(test,test.factor.stats)

# Convert factors to 0/1 indicators
train <- cbind(train,model.matrix(as.formula(paste("~ ", paste(train.factors, collapse=" + "),"- 1")), data=train))
test <- cbind(test,model.matrix(as.formula(paste("~ ", paste(test.factors, collapse=" + "),"- 1")), data=test))
train <- train[,-train.factor.cols]
test <- test[,-test.factor.cols]
train <- train[,-1]

train$T1_sum <- apply(train[,1:6],1,sum)
train$T1_min <- apply(train[,1:6],1,min)
train$T1_max <- apply(train[,1:6],1,max)
train$T1_mean <- apply(train[,1:6],1,mean)
train$T1_med <- apply(train[,1:6],1,median)
train$T1_sd <- apply(train[,1:6],1,sd)
train$T2_sum <- apply(train[,7:16],1,sum)
train$T2_min <- apply(train[,7:16],1,min)
train$T2_max <- apply(train[,7:16],1,max)
train$T2_mean <- apply(train[,7:16],1,mean)
train$T2_med <- apply(train[,7:16],1,median)
train$T2_sd <- apply(train[,7:16],1,sd)
train$T_sum <- apply(train[,1:16],1,sum)
train$T_min <- apply(train[,1:16],1,min)
train$T_max <- apply(train[,1:16],1,max)
train$T_mean <- apply(train[,1:16],1,mean)
train$T_med <- apply(train[,1:16],1,median)
train$T_sd <- apply(train[,1:16],1,sd)

test$T1_sum <- apply(test[,1:6],1,sum)
test$T1_min <- apply(test[,1:6],1,min)
test$T1_max <- apply(test[,1:6],1,max)
test$T1_mean <- apply(test[,1:6],1,mean)
test$T1_med <- apply(test[,1:6],1,median)
test$T1_sd <- apply(test[,1:6],1,sd)
test$T2_sum <- apply(test[,7:16],1,sum)
test$T2_min <- apply(test[,7:16],1,min)
test$T2_max <- apply(test[,7:16],1,max)
test$T2_mean <- apply(test[,7:16],1,mean)
test$T2_med <- apply(test[,7:16],1,median)
test$T2_sd <- apply(test[,7:16],1,sd)
test$T_sum <- apply(test[,1:16],1,sum)
test$T_min <- apply(test[,1:16],1,min)
test$T_max <- apply(test[,1:16],1,max)
test$T_mean <- apply(test[,1:16],1,mean)
test$T_med <- apply(test[,1:16],1,median)
test$T_sd <- apply(test[,1:16],1,sd)

# Drop columns with no variance
no.var <- apply(train,2,var) == 0
train <- train[,!no.var]
test <- test[,!no.var]

trainMatrix = as.matrix(train)
testMatrix = as.matrix(test)

comboInfo <- findLinearCombos(trainMatrix)
train <- train[,-comboInfo$remove]
test <- test[,-comboInfo$remove]

descrCor <-  cor(trainMatrix)
highlyCorDescr <- findCorrelation(descrCor, cutoff = .9)
trainMatrix2 <- trainMatrix[,-highlyCorDescr]
testMatrix2 <- testMatrix[,-highlyCorDescr]

dtrain =  xgb.DMatrix(trainMatrix2,label=y)
dtest = xgb.DMatrix(testMatrix2)

param <- list("objective" = "reg:linear")
cv.nround <- 500
cv.nfold <- 3
#Better
bst.cv = xgb.cv(param=param, data = dtrain, label = y, nfold = cv.nfold, nrounds=cv.nround, max.depth=5,nthread=4,eta=0.02, min_child_weight=1, subsample=0.7, gamma = 1, colsample_bytree=1, feval = evalgini)
#Old
bst.cv = xgb.cv(param=param, data = dtrain, label = y, nfold = cv.nfold, nrounds=cv.nround, max.depth=5,nthread=4,eta=0.02, min_child_weight=1, subsample=0.7, gamma = 1, colsample_bytree=0.5, feval = evalgini)

#nround = 500
#bst = xgboost(param=param, data = dtrain, label = y, nrounds=nround, max.depth=7,nthread=4,eta=0.01, min_child_weight=1, subsample=0.5, gamma = 0.1, colsample_bytree=1, feval = evalgini)
bst = xgb.train(param=param, data = dtrain, label = y, nrounds=nround, max.depth=5,nthread=4,eta=0.02, min_child_weight=1, subsample=0.7, gamma = 1, colsample_bytree=1, feval=evalgini, verbose=1, watchlist=list(train=dtrain))
pred = exp(predict(bst,testMatrix2))

importances <- xgb.importance(colnames(as.data.table(trainMatrix2)), model=bst)
xgb.plot.importance(importances)

predictions = data.frame(test.ids,pred)
names(predictions) = c("Id","Hazard")
write.csv(predictions,file=format(Sys.time(), "%m-%d_%H:%M_boost_submission.csv"), quote=FALSE,row.names=FALSE)
