require(xgboost)
require(methods)
require(data.table)
require(magrittr)
require(zoo)
require(plyr)

bids <- fread('bids.csv', header=T, stringsAsFactors=T)

train <- fread('train.csv', header=T, stringsAsFactors=F)
test <- fread('test.csv', header=T, stringsAsFactors=F)

train$num_bids <- 0
train$unique_auction <- 0
train$unique_merch <- 0
train$unique_device <- 0
train$unique_country <- 0
train$unique_ip <- 0
train$unique_url <- 0
for (i in 1:length(train$bidder_id)) {
  bid.subset <- bids[bidder_id==train$bidder_id[i]]
  train[i,'num_bids'] <- dim(bids[bidder_id==train$bidder_id[i]])[1]
  train[i,'unique_auction'] <- length(unique(bid.subset$auction))
  train[i,'unique_merch'] <- length(unique(bid.subset$merchandise))
  train[i,'unique_device'] <- length(unique(bid.subset$device))
  train[i,'unique_country'] <- length(unique(bid.subset$country))
  train[i,'unique_ip'] <- length(unique(bid.subset$ip))
  train[i,'unique_url'] <- length(unique(bid.subset$url))
}

# Temporary to plot items
library(lattice)
bids$outcome <- 0
for (i in 1:length(bids$outcome)) {
  whic = which(train$bidder_id == bids$bidder_id[i])
  if (length(whic != 0)) {
    bids$outcome[i] = train$outcome[whic]
  } else {
    bids[i,'outcome'] <- NA
  }
}

# End temporary

test$num_bids <- 0
test$unique_auction <- 0
test$unique_merch <- 0
test$unique_device <- 0
test$unique_country <- 0
test$unique_ip <- 0
test$unique_url <- 0
for (i in 1:length(test$bidder_id)) {
  bid.subset <- bids[bidder_id==test$bidder_id[i]]
  test[i,'num_bids'] <- dim(bids[bidder_id==test$bidder_id[i]])[1]
  test[i,'unique_auction'] <- length(unique(bid.subset$auction))
  test[i,'unique_merch'] <- length(unique(bid.subset$merchandise))
  test[i,'unique_device'] <- length(unique(bid.subset$device))
  test[i,'unique_country'] <- length(unique(bid.subset$country))
  test[i,'unique_ip'] <- length(unique(bid.subset$ip))
  test[i,'unique_url'] <- length(unique(bid.subset$url))
}

y <- train[,outcome]
train[, outcome:=NULL]
test_id <- test[,bidder_id]

# Just for now
train[,address:=NULL]
train[,payment_account:=NULL]
test[,address:=NULL]
test[,payment_account:=NULL]

train[,bidder_id:=NULL]
test[,bidder_id:=NULL]

#Convert to matrix from datatable
trainMatrix <- train[,lapply(.SD,as.numeric)] %>% as.matrix
testMatrix <- test[,lapply(.SD,as.numeric)] %>% as.matrix

param <- list("objective" = "reg:linear",
              "eval_metric" = "auc")

cv.nround <- 100
cv.nfold <- 3
#bst.cv = xgb.cv(param=param, data = trainMatrix, label = y, nfold = cv.nfold, nrounds = cv.nround, max.depth=20,nthread=2,eta=0.1, min_child_weight=4, subsample=0.8, gamma = 0.8, colsample_bytree=0.5)
bst.cv = xgb.cv(param=param, data = trainMatrix, label = y, nfold = cv.nfold, nrounds=cv.nround, max.depth=100,nthread=4,eta=0.1, min_child_weight=4, subsample=0.8, gamma = 1, colsample_bytree=0.8)

nround = 100
bst = xgboost(param=param, data = trainMatrix, label = y, nrounds=nround, max.depth=100,nthread=4,eta=0.1, min_child_weight=4, subsample=0.8, gamma = 1, colsample_bytree=0.8)
#bst = xgboost(param=param, data = trainMatrix, label = y, nrounds=nround, max.depth=10,nthread=2,eta=0.05, min_child_weight=4, subsample=0.8, gamma = 1, colsample_bytree=0.5)
#bst = xgboost(param=param, data = trainMatrix, label = y, nrounds=nround, max.depth=20,nthread=2,eta=0.1, min_child_weight=4, subsample=0.8, gamma = 0.8, colsample_bytree=0.5)

pred = predict(bst,testMatrix)
pred = as.float(pred)
pred = data.frame(test_id,pred)
names(pred) = c('bidder_id', 'prediction')
write.csv(pred,file='submission_100.csv', quote=FALSE,row.names=FALSE)

