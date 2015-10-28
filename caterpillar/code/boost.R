require(xgboost)
require(caret)
require(qdapTools)
require(ggplot2)
require(data.table)
library(Matrix)

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

# Load data
test = read.csv("../input/test_set.csv", header=T)
train = read.csv("../input/train_set.csv", header=T)

train$id = -(1:nrow(train))
test$cost = 0

train = rbind(train, test)

### Merge datasets if only 1 variable in common
continueLoop = TRUE
while(continueLoop){
  continueLoop = FALSE
  for(f in dir("../input/")){
    d = read.csv(paste0("../input/", f))
    commonVariables = intersect(names(train), names(d))
    if(length(commonVariables) == 1){
      train = merge(train, d, by = commonVariables, all.x = TRUE)
      continueLoop = TRUE
      print(dim(train))
    }
  }
}

train$end_form_id <- NULL
train$name <- NULL
train$component_type_id <- NULL
train$component_id <- NULL
train$connection_type_id <- NULL
train$forming <- NULL
train$intended_nut_pitch <- NULL
train$intended_nut_thread <- NULL
train$orientation <- NULL
train$plating <- NULL
train$unique_feature <- NULL
train$weight <- NULL

train$quote_date  <- strptime(train$quote_date,format = "%Y-%m-%d", tz="GMT")
train$year <- year(as.IDate(train$quote_date))
train$month <- month(as.IDate(train$quote_date))
train$week <- week(as.IDate(train$quote_date))

# Remove unnecessary columns 
train$quote_date  <- NULL
train$tube_assembly_id  <- NULL

# converting NA in to '0' and '" "' for mode Matrix Generation
for(i in 1:ncol(train)){
  if(is.numeric(train[,i])){
    train[is.na(train[,i]),i] = 0
  }else{
    train[,i] = as.character(train[,i])
    train[is.na(train[,i]),i] = " "
    train[,i] = as.factor(train[,i])
  }
}

test = train[which(train$id > 0),]
train = train[which(train$id < 0),]

y = train$cost
y <- log(y+1)
test.ids <- test$id

# Remove unnecessary columns 
train$id <- NULL
test$id <- NULL
#train$cost <- NULL
test$cost <- NULL

tr.mf  <- model.frame(as.formula(paste("cost ~",paste(names(train),collapse = "+"))),train)
tr.m  <- model.matrix(attr(tr.mf,"terms"),data = train)
tr  <- Matrix(tr.m)
t(tr)


te.mf  <- model.frame(as.formula(paste("~",paste(names(test),collapse = "+"))),test)
te.m  <- model.matrix(attr(te.mf,"terms"),data = test)
te  <- Matrix(te.m)
t(te)

dtrain =  xgb.DMatrix(tr,label=y)

param <- list("objective" = "reg:linear")
cv.nround <- 500
cv.nfold <- 3
#Better
bst.cv = xgb.cv(param=param, data = dtrain, label = y, nfold = cv.nfold, nrounds=cv.nround, max.depth=5,nthread=4,eta=0.05, min_child_weight=0.5, subsample=0.7, gamma = 0.1, colsample_bytree=0.5)
#Old
bst.cv = xgb.cv(param=param, data = dtrain, label = y, nfold = cv.nfold, nrounds=cv.nround, max.depth=10,nthread=4, min_child_weight=6, subsample=0.85, gamma = 2, colsample_bytree=0.75, scale_pos_weight=1)

nround = 1500
bst = xgb.train(param=param, data = dtrain, label = y, nrounds=nround, max.depth=5,nthread=4,eta=0.05, min_child_weight=0.5, subsample=0.7, gamma = 0.1, colsample_bytree=0.5, verbose=1, watchlist=list(train=dtrain))
pred = exp(predict(bst,te))-1

importances <- xgb.importance(colnames(as.data.table(trainMatrix2)), model=bst)
xgb.plot.importance(importances)

predictions = data.frame(test.ids,pred)
predictions <- unique(predictions)
names(predictions) = c("id","cost")
write.csv(predictions,file=format(Sys.time(), "%m-%d_%H:%M_boost_submission.csv"), quote=FALSE,row.names=FALSE)
