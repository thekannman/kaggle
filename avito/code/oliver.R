# Kaggle-competititon "Avito Context Ad Clicks"
# See https://www.kaggle.com/c/avito-context-ad-clicks

# In order to run this script on Kaggle-scripts I had to limit the number of entries to read
# from the database as well as to decrease the sample-size. With the full dataset from the database as well
# as a sample of 10 millions entries I got a 0.05104 on the public leaderboard

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
train.db[,train.factors] <- NULL

# Eliminate factors with length(unique(COLUMN_NAME))/length(COLUMN_NAME) > 0.1
# Also removing i.categoryID since it seems to be a somewhat modified version of CategoryID
useless.factors <- c('UserID','SearchID','SearchDate','i.CategoryID')
train.db[,useless.factors] <- NULL

# Get rid of zero-variance variables
unique.lengths <- apply(train.db,2,function(x) length(unique(x)))
train.db <- train.db[,unique.lengths!=1, with=F]

#temp <- fetch(db, "select HistCTR,UserAgentID,UserAgentFamilyID,UserAgentFamilyID,
#                          UserAgentOSID,UserDeviceID,IsClick 
#                     from trainSearchStream 
#                   INNER JOIN PhoneRequestsStream ON trainSearchStream.AdID=PhoneRequestsStream.AdID
#                   INNER Join UserInfo ON PhoneRequestsStream.UserID=UserInfo.UserID", 
#              1000)
#train.ads.info$HistCTR <- log(train.ads.info$HistCTR)
y <- train.db$IsClick
train.db$IsClick <- NULL
y <- as.factor(y)
levels(y) <- c("no","yes")

# Build a logistic regression ...
ctrl <- trainControl(method="cv", number=3, classProbs=T, savePred=T, summaryFunction = logloss)
mod <- train(x=as.matrix(train.db), y=y, method='glm', metric='LogLoss', maximize=F, family=binomial(), trControl=ctrl)

# Check that regression-coefficients have significant impact
summary(mod)

# ----- Predict submission dataset ---------------------------------------------
test.search.stream <- fetch(db, "select TestId,AdID,Position,HistCTR from testSearchStream")
setkey(train.search.stream,AdID)
test.ads.info <- ads.info[test.search.stream]

test.factors <- c('AdID','CategoryID','Title','Position')
test.factor.stats <- factorToNumeric(as.data.frame(train.ads.info), as.data.frame(test.ads.info), 'IsClick', test.factors, 'mean')
for (test.factor in names(test.factor.stats)) {
  test.ads.info[,eval(as.symbol(test.factor)):= test.factor.stats[,test.factor]]
}


prediction <- predict(mod, test.search.stream, type="prob")

submissionData <- data.frame(ID=test.search.stream$TestId, IsClick=prediction$yes)
submissionFile <- paste0("glm", format(Sys.time(), "%Y-%m-%d-%H:%M:%S"), ".csv")
write.csv(submissionData, submissionFile, sep=",", dec=".", col.names=TRUE, row.names=FALSE)

# ----- Clean up ---------------------------------------------------------------

dbDisconnect(db)


