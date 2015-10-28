# Kaggle-competititon "Avito Context Ad Clicks"
# See https://www.kaggle.com/c/avito-context-ad-clicks

# In order to run this script on Kaggle-scripts I had to limit the number of entries to read
# from the database as well as to decrease the sample-size. With the full dataset from the database as well
# as a sample of 10 millions entries I got a 0.05104 on the public leaderboard

require(xgboost)
library(data.table)
library(RSQLite)
library(caret)
require(qdapTools)

factorToNumeric <- function(train, test, response, variables, metrics){
  temp <- data.frame(c(rep(0,nrow(test))), row.names = NULL)

  for (variable in variables){
    for (metric in metrics) {
      x <- tapply(train[,response], train[,variable], metric)
      x <- data.frame(row.names(x),x, row.names = NULL)
      temp <- data.frame(temp,round(lookup(as.factor(test[,variable]), x),2))
      colnames(temp)[ncol(temp)] <- paste(metric,variable, sep = "_")
    }
  }
  return (temp[,-1])
}

# ----- Prepare database -------------------------------------------------------

db <- dbConnect(SQLite(), dbname="../input/database.sqlite")
dbListTables(db)

# ----- Utitlies ---------------------------------------------------------------

# Runs the query, fetches the given number of entries and returns a
# data.table
fetch  <- function(db, query, n = -1) {
  result <- dbSendQuery(db, query)
  data <- dbFetch(result, n)
  dbClearResult(result)
  return(as.data.table(data))
}

# Loss-function to evaluate result
# See https://www.kaggle.com/c/avito-context-ad-clicks/details/evaluation
logLoss <- function(y, yHat){
  
  threshold <- 10^(-15)
  yHat <- pmax(pmin(yHat, 1-threshold), threshold)
  
  loss <- -mean(y*log(yHat) + (1-y)*log(1-yHat))
  
  return(loss)
}

# ----- Simple Machine Learning ------------------------------------------------

# Select contextual Ads (OnjectType=3), results in 190.157.735 entries
# Warning: Takes a few minutes

train.search.stream <- fetch(db, "select SearchID,AdID,Position,HistCTR,IsClick from trainSearchStream", 1e7)
#phone.request.stream <- fetch(db, "select AdID,IPID,UserID from PhoneRequestsStream")
ads.info <- fetch(db, "select AdID,CategoryID,Title,Price from AdsInfo")
category.info <- fetch(db, "select * from Category")
#visits.stream <- fetch(db, "select * from VisitsStream") Too large
search.info <- fetch(db, "select * from SearchInfo")
user.info <- fetch(db, "select * from UserInfo")


setkey(train.search.stream,AdID)
setkey(ads.info,AdID)
train.db <- ads.info[train.search.stream]

setkey(train.db,CategoryID)
setkey(category.info,CategoryID)
train.db <- category.info[train.db]

setkey(train.db,SearchID)
setkey(search.info,SearchID)
train.db <- search.info[train.db]

setkey(train.db,UserID)
setkey(user.info,UserID)
train.db <- user.info[train.db]

train.factors <- c('UserAgentID','UserAgentOSID','UserDeviceID','UserAgentFamilyID',
                   'IPID','IsUserLoggedOn','SearchQuery','LocationID',
                   'CategoryID','SearchParams','AdID','Title','Position',
                   'ParentCategoryID','SubcategoryID','Level')
train.factor.stats <- factorToNumeric(as.data.frame(train.db), as.data.frame(train.db), 'IsClick', train.factors, 'mean')
for (train.factor in names(train.factor.stats)) {
  train.db[,eval(as.symbol(train.factor)):= train.factor.stats[,train.factor]]
}

test.search.stream <- fetch(db, "select TestId,SearchID,AdID,Position,HistCTR from testSearchStream")
setkey(test.search.stream,AdID)
setkey(ads.info,AdID)
test.db <- ads.info[test.search.stream]

setkey(test.db,CategoryID)
setkey(category.info,CategoryID)
test.db <- category.info[test.db]

setkey(test.db,SearchID)
setkey(search.info,SearchID)
test.db <- search.info[test.db]

setkey(test.db,UserID)
setkey(user.info,UserID)
test.db <- user.info[test.db]

test.factors <- train.factors
test.factor.stats <- factorToNumeric(as.data.frame(train.db), as.data.frame(test.db), 'IsClick', test.factors, 'mean')
for (test.factor in names(test.factor.stats)) {
  test.db[,eval(as.symbol(test.factor)):= test.factor.stats[,test.factor]]
}
train.db[,train.factors] <- NULL
test.db[,test.factors] <- NULL

y <- train.db$IsClick
train.db$IsClick <- NULL
test.ids <- test.db$TestId
test.db$TestId <- NULL

# Eliminate factors with length(unique(COLUMN_NAME))/length(COLUMN_NAME) > 0.1
# Also removing i.categoryID since it seems to be a somewhat modified version of CategoryID
useless.factors <- c('UserID','SearchID','SearchDate','i.CategoryID')
train.db[,useless.factors] <- NULL
test.db[,useless.factors] <- NULL

# Get rid of zero-variance variables
unique.lengths <- apply(train.db,2,function(x) length(unique(x)))
train.db <- train.db[,unique.lengths!=1, with=F]
test.db <- test.db[,unique.lengths!=1, with=F]

# Build a logistic regression ...
trainMatrix <- data.matrix(train.db)
testMatrix <- data.matrix(test.db)

dtrain <-  xgb.DMatrix(trainMatrix,label=as.numeric(y), missing=NA)
dtest <- xgb.DMatrix(testMatrix, missing=NA)

param <- list("objective" = "binary:logistic",
              "eval_metric" = "logloss")
cv.nround <- 500
cv.nfold <- 3
bst.cv = xgb.cv(param=param, data = dtrain, label = y, nfold = cv.nfold, nrounds=cv.nround, max.depth=5,nthread=4,eta=0.02, min_child_weight=1, subsample=0.7, gamma = 1, colsample_bytree=0.5)

nround = 500
bst = xgboost(param=param, data = dtrain, label = y, nrounds=nround, max.depth=5,nthread=4,eta=0.02, min_child_weight=1, subsample=0.7, gamma = 1, colsample_bytree=0.5)

# ----- Predict submission dataset ---------------------------------------------
prediction <- predict(bst, testMatrix)

submissionData <- data.frame(ID=test.search.stream$TestId, IsClick=prediction)
submissionFile <- paste0("glm", format(Sys.time(), "%Y-%m-%d-%H:%M:%S"), ".csv")
write.csv(submissionData, submissionFile, sep=",", dec=".", col.names=TRUE, row.names=FALSE)

# ----- Clean up ---------------------------------------------------------------

dbDisconnect(db)


