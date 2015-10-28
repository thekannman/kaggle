require(xgboost)
require(lubridate)
require(methods)
require(data.table)
require(magrittr)
require(zoo)
require(plyr)


train <- read.csv('X.csv')
test <- read.csv('X_test.csv')
y <- read.csv('y.csv')
test_id <- read.csv('test_id.csv')

#Convert to matrix from datatable
trainMatrix <- as.matrix(train)
testMatrix <- as.matrix(test)
y <- as.vector(y$y)
test_id <- as.vector(test_id$Id)

param <- list("objective" = "binary:logistic",
              "eval_metric" = "auc")

cv.nround <- 100
cv.nfold <- 3
#bst.cv = xgb.cv(param=param, data = trainMatrix, label = y, nfold = cv.nfold, nrounds = cv.nround, max.depth=20,nthread=2,eta=0.1, min_child_weight=4, subsample=0.8, gamma = 0.8, colsample_bytree=0.5)
#bst.cv = xgb.cv(param=param, data = trainMatrix, label = y, nfold = cv.nfold, nrounds=cv.nround, max.depth=10,nthread=4,eta=0.05, min_child_weight=4, subsample=0.8, gamma = 1, colsample_bytree=0.5)

nround = 100
#bst = xgboost(param=param, data = trainMatrix, label = y, nrounds=nround, max.depth=100,nthread=4,eta=0.1, min_child_weight=4, subsample=0.8, gamma = 1, colsample_bytree=0.8)
bst = xgboost(param=param, data = trainMatrix, label = y, nrounds=nround, max.depth=10,nthread=2,eta=0.05, min_child_weight=4, subsample=0.8, gamma = 1, colsample_bytree=0.5)
#bst = xgboost(param=param, data = trainMatrix, label = y, nrounds=nround, max.depth=20,nthread=2,eta=0.1, min_child_weight=4, subsample=0.8, gamma = 0.8, colsample_bytree=0.5)



pred = predict(bst,testMatrix)
for (i_pred in 1:length(pred)) {
  if (pred[i_pred] < 0) {
    pred[i_pred] =0 
  }
}
pred = data.frame(test_id,pred)
names(pred) = c('Id', 'WnvPresent')
write.csv(pred,file='submission_100.csv', quote=FALSE,row.names=FALSE)

names <- dimnames(trainMatrix)[[2]]
importance_matrix <- xgb.importance(names, model = bst)
xgb.plot.importance(importance_matrix[1:10,])
