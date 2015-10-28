require(xgboost)
require(methods)
require(data.table)
require(magrittr)

train <- fread('train.csv', header = T, stringsAsFactors = F)
test <- fread('test.csv', header=TRUE, stringsAsFactors = F)

train_id = train[,id]
test_id = test[,id]
train_target = train[,target]

train[, id := NULL]
test[, id := NULL]

nameLastCol <- names(train)[ncol(train)]

y <- train[, nameLastCol, with = F][[1]] %>% gsub('Class_','',.) %>% {as.integer(.) -1}

train[, nameLastCol:=NULL, with = F]

#Extra features:
train = transform(train[,1:93, with=F], sum=rowSums(train))
train[,c('min')] <-  t(apply(train[,1:93, with=F], 1, FUN=function(x) min(x)))
train[,c('max')] <-  t(apply(train[,1:93, with=F], 1, FUN=function(x) max(x)))
train[,c('sd')] <-  t(apply(train[,1:93, with=F], 1, FUN=function(x) sd(x)))
train[,c('mean')] <-  t(apply(train[,1:93, with=F], 1, FUN=function(x) mean(x)))
train[,c('zeros')] <-  t(apply(train[,1:93, with=F]==0, 1, FUN=function(x) sum(x)))
train[,c('11_34_60_sum')] <-  t(apply(train[,c(11,34,60), with=F], 1, FUN=function(x) sum(x)))
train[,c('11_34_60_min')] <-  t(apply(train[,c(11,34,60), with=F], 1, FUN=function(x) min(x)))
train[,c('11_34_60_max')] <-  t(apply(train[,c(11,34,60), with=F], 1, FUN=function(x) max(x)))
train[,c('11_34_60_sd')] <-  t(apply(train[,c(11,34,60), with=F], 1, FUN=function(x) sd(x)))
train[,c('11_34_60_mean')] <-  t(apply(train[,c(11,34,60), with=F], 1, FUN=function(x) mean(x)))

test = transform(test[,1:93, with=F], sum=rowSums(test))
test[,c('min')] <-  t(apply(test[,1:93, with=F], 1, FUN=function(x) min(x)))
test[,c('max')] <-  t(apply(test[,1:93, with=F], 1, FUN=function(x) max(x)))
test[,c('mean')] <-  t(apply(test[,1:93, with=F], 1, FUN=function(x) mean(x)))
test[,c('sd')] <-  t(apply(test[,1:93, with=F], 1, FUN=function(x) sd(x)))
test[,c('zeros')] <-  t(apply(test[,1:93, with=F]==0, 1, FUN=function(x) sum(x)))
test[,c('11_34_60_sum')] <-  t(apply(test[,c(11,34,60), with=F], 1, FUN=function(x) sum(x)))
test[,c('11_34_60_min')] <-  t(apply(test[,c(11,34,60), with=F], 1, FUN=function(x) min(x)))
test[,c('11_34_60_max')] <-  t(apply(test[,c(11,34,60), with=F], 1, FUN=function(x) max(x)))
test[,c('11_34_60_sd')] <-  t(apply(test[,c(11,34,60), with=F], 1, FUN=function(x) sd(x)))
test[,c('11_34_60_mean')] <-  t(apply(test[,c(11,34,60), with=F], 1, FUN=function(x) mean(x)))

train_augment = cbind(train_id,train,train_target)
test_augment = cbind(test_id,test)

write.csv(train_augment,file='train_augment.csv', quote=FALSE,row.names=FALSE)
write.csv(test_augment,file='test_augment.csv', quote=FALSE,row.names=FALSE)

#Convert to matrix from datatable
trainMatrix <- train[,lapply(.SD,as.numeric)] %>% as.matrix
testMatrix <- test[,lapply(.SD,as.numeric)] %>% as.matrix

numberOfClasses <- max(y) + 1

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = numberOfClasses)

cv.nround <- 500
cv.nfold <- 3

bst.cv = xgb.cv(param=param, data = trainMatrix, label = y, nfold = cv.nfold, nrounds = cv.nround, max.depth=10,nthread=4,eta=0.1, min_child_weight=4, subsample=0.8, gamma = 1, colsample_bytree=0.8)

nround = 500
bst = xgboost(param=param, data = trainMatrix, label = y, nrounds=nround, max.depth=10,nthread=4,eta=0.1, min_child_weight=4, subsample=0.8, gamma = 1, colsample_bytree=0.8)

pred = predict(bst,testMatrix)
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

pred = format(pred, digits=2,scientific=F)
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='submission_match_chrisdubois.csv', quote=FALSE,row.names=FALSE)

