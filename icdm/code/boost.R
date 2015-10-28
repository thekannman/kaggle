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
id.ips <- fetch(db, "select * from id_ips")
id.properties <- fetch(db, "select * from id_properties")
ips <- fetch(db, "select * from ips")
property.categories <- fetch(db, "select * from property_categories")

id.ips.cookie <- id.ips[device_or_cookie_indicator==1]
id.ips.device <- id.ips[device_or_cookie_indicator==0]
id.ips.cookie$device_or_cookie_indicator <- NULL
id.ips.device$device_or_cookie_indicator <- NULL
names(id.ips.device)[1] <- names(devices)[2]
names(id.ips.cookie)[1] <- names(cookies)[2]
rm(id.ips)

id.properties.cookie <- id.properties[device_or_cookie_indicator==1]
id.properties.device <- id.properties[device_or_cookie_indicator==0]
id.properties.cookie$device_or_cookie_indicator <- NULL
id.properties.device$device_or_cookie_indicator <- NULL
names(id.properties.device)[1] <- names(devices)[2]
names(id.properties.cookie)[1] <- names(cookies)[2]
rm(id.properties)

setkey(cookies,cookie_id)
setkey(id.ips.cookie,cookie_id)
cookies.and.ips <- id.ips.cookie[cookies]

setkey(devices,device_id)
setkey(id.ips.device,device_id)
devices.and.ips <- id.ips.device[devices]

rm(cookies)
rm(devices)
rm(id.ips.cookie)
rm(id.ips.device)

names(ips)[1] <- 'ip'
setkey(cookies.and.ips,ip)
setkey(devices.and.ips,ip)
setkey(ips,ip)
cookies.and.ips.2 <- ips[cookies.and.ips]
devices.and.ips.2 <- ips[devices.and.ips]
rm(devices.and.ips)
rm(cookies.and.ips)

#train.cookies <- cookies.and.ips.2[drawbridge_handle!=-1]
#test.cookies <- cookies.and.ips.2[drawbridge_handle==-1]
#train.devices <- devices.and.ips.2[drawbridge_handle!=-1]
#test.devices <- devices.and.ips.2[drawbridge_handle==-1]
#rm(devices.and.ips.2)
#rm(cookies.and.ips.2)

#trainMatrix <- as.data.frame(train.devices)
trainMatrix <- as.data.frame(devices.and.ips.2)
trainMatrix$drawbridge_handle <- NULL

trainMatrix.subset <- trainMatrix[1:50000,]
trainMatrix = mutate(trainMatrix, 
       ip=as.factor(ip),is_cellular_ip=as.factor(is_cellular_ip),ip_total_freq=as.numeric(ip_total_freq),
       ip_anonymous_c0=as.factor(ip_anonymous_c0),ip_anonymous_c1=as.factor(ip_anonymous_c1),
       ip_anonymous_c2=as.factor(ip_anonymous_c2),device_id=as.factor(device_id),
       ip_freq_count=as.numeric(ip_freq_count),idxip_anonymous_c1=as.numeric(idxip_anonymous_c1),
       idxip_anonymous_c2=as.numeric(idxip_anonymous_c2),
       idxip_anonymous_c3=as.numeric(idxip_anonymous_c3),
       idxip_anonymous_c4=as.numeric(idxip_anonymous_c4),
       idxip_anonymous_c5=as.numeric(idxip_anonymous_c5),
       device_type=as.factor(device_type),device_os=as.factor(device_os),country=as.factor(country),
       anonymous_c0=as.factor(anonymous_c0),anonymous_c1=as.factor(anonymous_c1),
       anonymous_c2=as.factor(anonymous_c2),anonymous_5=as.numeric(anonymous_5),
       anonymous_6=as.numeric(anonymous_6),anonymous_7=as.numeric(anonymous_7))
       
mod <- lm(device_id ~ ip + is_cellular_ip + ip_total_freq + ip_anonymous_c0 + ip_anonymous_c1 +
                      ip_anonymous_c2 + ip_freq_count + idxip_anonymous_c1 + idxip_anonymous_c2 + 
                      idxip_anonymous_c3 + idxip_anonymous_c4 + idxip_anonymous_c5 + device_type + 
                      device_os + country + anonymous_c0 + anonymous_c1 + anonymous_c2 +
                      anonymous_5 + anonymous_6 + anonymous_7, trainMatrix)

sample.submission <- read.csv('../input/sampleSubmission.csv')

cookies.in.devices <- train.cookies$drawbridge_handle %in% train.devices$drawbridge_handle
train.cookies <- train.cookies[order(drawbridge_handle)]
train.devices <- train.devices[order(drawbridge_handle)]
train.cookies.2 <- train.cookies[cookies.in.devices,]

matching.cookies <- as.data.frame(matrix(0,dim(train.cookies.2)[1],dim(train.cookies.2)[2]))
matching.devices <- as.data.frame(matrix(0,dim(train.cookies.2)[1],dim(train.cookies.2)[2]))

i = j = k = 1
while(i <= dim(train.cookies.2)[1] & j <= dim(train.devices)[1]) {
  if (train.cookies.2$drawbridge_handle[i] < train.devices$drawbridge_handle[j]) {
    i = i + 1
  }
  else if (train.cookies.2$drawbridge_handle[i] > train.devices$drawbridge_handle[j]) {
    j = j + 1
  }
  else {
    matching.cookies[k] = train.cookies.2[k,]
    matching.devices[k] = train.devices[k,]
    k = k + 1
  }
}

matches <- as.data.table(matrix(0,dim(head(train.devices,n=1000))[1]*dim(head(train.cookies.2,n=1000))[1],2))
for (i in 1:dim(head(train.devices,n=1000))[1]) {
  for (j in 1:dim(head(train.cookies,n=1000))[1]) {
    matches[i*j,1] <- train.devices$country[i]==train.cookies.2$country[j]
    matches[i*j,2] <- train.devices$drawbridge_handle[i]==train.cookies.2$drawbridge_handle[j]
  }
} 

device.to.cookie <- sapply(head(devices.drawbridge),function(x) match(x,cookies.drawbridge))
device.to.cookie <- data.frame(1:length(device.to.cookie), device.to.cookie)


#setkey(id.properties.cookie,property_id)
#setkey(property.categories,property_id)
#property.categories.cookie <- property.categories[id.properties.cookie]
#property.count <- count(id.properties.cookie$cookie_id)


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


