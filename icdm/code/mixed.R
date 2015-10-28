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
require(plyr)
require(dplyr)

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


myFun <- function(x){
    tbl <- table(x$cookie_id)
    x$freq <- rep(names(tbl)[which.max(tbl)],nrow(x))
    x
}


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


# ----- Simple Machine Learning ------------------------------------------------

# Select contextual Ads (OnjectType=3), results in 190.157.735 entries
# Warning: Takes a few minutes

cookies <- fetch(db, "select * from cookies")
devices <- fetch(db, "select * from devices")

sample.submission <- read.csv('../input/sampleSubmission.csv')


devices <- devices[,c(2,5,8), with=F]
trainMatrix <- devices[,c(2,3),with=F]
devices$device_id <- as.factor(devices$device_id)
devices$country <- as.factor(trainMatrix$country)
devices$anonymous_c2 <- as.factor(trainMatrix$anonymous_c2)

ctrl <- trainControl(method="none", classProbs=T, savePred=T, summaryFunction = logloss)
mod <- train(device_id ~ country + anonymous_c2, data=devices[1:1000],method='knn',metric='LogLoss', maximize=F, trControl=ctrl)

test.subset <- devices$device_id %in% sample.submission$device_id
test.devices <- devices[test.subset,]

setkey(test.devices,anonymous_c2)
setkey(country_summary,anonymous_c2)

submission.cookie <- merge(test.devices,country_summary,all.x=TRUE)
submission.cookie <- data.frame(submission.cookie)
submission.cookie <- submission.cookie[,c("device_id","freq")]
names(submission.cookie) <- c("device_id","cookie_id")

submissionFile <- paste0("c2", format(Sys.time(), "%Y-%m-%d-%H:%M:%S"), ".csv")
write.csv(submission.cookie, submissionFile, row.names=FALSE)

# ----- Clean up ---------------------------------------------------------------

dbDisconnect(db)


