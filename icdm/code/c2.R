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

country_summary <- ddply(cookies, .(anonymous_c2), .fun=myFun)
country_summary <- country_summary[,c(8,12)]
country_summary <- unique(country_summary)
country_summary <- data.table(country_summary)

train.matrix <- devices[,c(2,3),with=F]
train.matrix <- as.matrix(train.matrix)
y <- devices[,1,with=F]
y <- as.vector(as.matrix(y))

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


