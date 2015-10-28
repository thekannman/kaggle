require(xgboost)
require(lubridate)
require(methods)
require(data.table)
require(magrittr)
require(zoo)
require(plyr)

key <- fread('key.csv', header=T, stringsAsFactors=F)
setkey(key,'store_nbr','station_nbr')

weather <- fread('weather.csv', header=T, stringsAsFactors=F)
setkey(weather,'station_nbr','date')

# T (Trace) is < 0.01, treat as 0.005 
weather[preciptotal=="  T", preciptotal:="0.005"]
weather[snowfall=="  T", snowfall:="0.005"]

#M (missing data) treated as median
tmax_med <- median(as.numeric(weather[weather[,tmax!="M"]][,tmax]))
weather[tmax=="M", tmax:=as.character(tmax_med)]
tmin_med <- median(as.numeric(weather[weather[,tmin!="M"]][,tmin]))
weather[tmin=="M", tmin:=as.character(tmin_med)]
tavg_med <- median(as.numeric(weather[weather[,tavg!="M"]][,tavg]))
weather[tavg=="M", tavg:=as.character(tavg_med)]
dew_med <- median(as.numeric(weather[weather[,dewpoint!="M"]][,dewpoint]))
weather[dewpoint=="M", dewpoint:=as.character(dew_med)]
bulb_med <- median(as.numeric(weather[weather[,wetbulb!="M"]][,wetbulb]))
weather[wetbulb=="M", wetbulb:=as.character(bulb_med)]
heat_med <- median(as.numeric(weather[weather[,heat!="M"]][,heat]))
weather[heat=="M", heat:=as.character(heat_med)]
cool_med <- median(as.numeric(weather[weather[,cool!="M"]][,cool]))
weather[cool=="M", cool:=as.character(cool_med)]
pres_med <- median(as.numeric(weather[weather[,stnpressure!="M"]][,stnpressure]))
weather[stnpressure=="M", stnpressure:=as.character(pres_med)]
rspeed_med <- median(as.numeric(weather[weather[,resultspeed!="M"]][,resultspeed]))
weather[resultspeed=="M", resultspeed:=as.character(rspeed_med)]
rdir_med <- median(as.numeric(weather[weather[,resultdir!="M"]][,resultdir]))
weather[resultdir=="M", resultdir:=as.character(rdir_med)]
aspeed_med <- median(as.numeric(weather[weather[,avgspeed!="M"]][,avgspeed]))
weather[avgspeed=="M", avgspeed:=as.character(aspeed_med)]
precip_med <- median(as.numeric(weather[weather[,preciptotal!="M"]][,preciptotal]))
weather[preciptotal=="M", preciptotal:=as.character(precip_med)]
snow_med <- median(as.numeric(weather[weather[,snowfall!="M"]][,snowfall]))
weather[snowfall=="M", snowfall:=as.character(snow_med)]
dep_med <- median(as.numeric(weather[weather[,depart!="M"]][,depart]))
weather[depart=="M",depart:=as.character(dep_med)]
sea_med <- median(as.numeric(weather[weather[,sealevel!="M"]][,sealevel]))
weather[sealevel=="M", sealevel:=as.character(sea_med)]

#Sunrise/Sunset of '-' treated as median value
rise_med <- median(weather[,sunrise])
weather[sunrise=="-", sunrise:=as.character(rise_med)]
set_med <- median(weather[,sunset])
weather[sunset=="-", sunset:=as.character(set_med)]

codesum = unique(unlist(strsplit(weather$codesum, " ")))
codesum = codesum[codesum!=""]

for (code in codesum) {
  codes <- strsplit(weather$codesum, " ")
  code.col <- numeric(length(codes))
  for (i in 1:length(codes)) {
    if (code %in% codes[[i]]) {
      code.col[i] = 1
    }
  }
  weather[,eval(code):=code.col]
} 

train <- fread('train.csv', header=T, stringsAsFactors=F)
test <- fread('test.csv', header=T, stringsAsFactors=F)
setkey(train,'store_nbr')
setkey(test,'store_nbr')

train_key <- merge(key,train)
test_key <- merge(key,test)
setkey(train_key,'date','station_nbr')
setkey(test_key,'date','station_nbr')

train_weather <- merge(train_key,weather)
test_weather <- merge(test_key,weather)

#train_weather[, station_nbr := NULL]
#test_weather[, station_nbr := NULL]

train_id = paste(train_weather[,store_nbr],train_weather[,item_nbr],train_weather[,date], sep='_')
test_id = paste(test_weather[,store_nbr],test_weather[,item_nbr],test_weather[,date], sep='_')

train_weather$day<-day(as.POSIXlt(train_weather$date, format="%Y-%m-%d"))
train_weather$month<-month(as.POSIXlt(train_weather$date, format="%Y-%m-%d"))
train_weather$year<-year(as.POSIXlt(train_weather$date, format="%Y-%m-%d"))
train_weather$days_before_2015 <- as.POSIXlt('2015-01-01', format="%Y-%m-%d") - as.POSIXlt(train_weather$date, format="%Y-%m-%d")

#Other non-numberics to drop for now
train_weather[,codesum:=NULL]
test_weather[,codesum:=NULL]

test_weather$day<-day(as.POSIXlt(test_weather$date, format="%Y-%m-%d"))
test_weather$month<-month(as.POSIXlt(test_weather$date, format="%Y-%m-%d"))
test_weather$year<-year(as.POSIXlt(test_weather$date, format="%Y-%m-%d"))
test_weather$days_before_2015 <- as.POSIXlt('2015-01-01', format="%Y-%m-%d") - as.POSIXlt(test_weather$date, format="%Y-%m-%d")

train_weather <- train_weather[order(store_nbr,year,month)]
test_weather <- test_weather[order(store_nbr,year,month)]

train_id = paste(train_weather[,store_nbr],train_weather[,item_nbr],train_weather[,date], sep='_')
test_id = paste(test_weather[,store_nbr],test_weather[,item_nbr],test_weather[,date], sep='_')
test_weather[, date := NULL]
train_weather[, date := NULL]

#plot(rollmean(trainMatrix[,15], 3, mean(trainMatrix[,15])),y)

write.csv(train_weather,file='train_weather.csv', quote=FALSE,row.names=FALSE)
write.csv(test_weather,file='test_weather.csv', quote=FALSE,row.names=FALSE)


train_weather <- train_weather[, lapply(.SD, as.numeric)]
test_weather <- test_weather[, lapply(.SD, as.numeric)]

y <- train_weather[,units]
y=log(y+1)

train_weather[, units:=NULL]

avg <- ddply(train_weather, .(month, year, store_nbr), summarize, avg=ave(preciptotal))
avg = data.table(avg)
avg <- avg[order(store_nbr,year,month)]
train_weather[,precip_avg:=avg[,avg]]
avg <- ddply(train_weather, .(month, year, store_nbr), summarize, avg=ave(snowfall))
avg = data.table(avg)
avg <- avg[order(store_nbr,year,month)]
train_weather[,snow_avg:=avg[,avg]]
avg <- ddply(train_weather, .(month, year, store_nbr), summarize, avg=ave(stnpressure))
avg = data.table(avg)
avg <- avg[order(store_nbr,year,month)]
train_weather[,pres_avg:=avg[,avg]]

avg <- ddply(test_weather, .(month, year, store_nbr), summarize, avg=ave(preciptotal))
avg = data.table(avg)
avg <- avg[order(store_nbr,year,month)]
test_weather[,precip_avg:=avg[,avg]]
avg <- ddply(test_weather, .(month, year, store_nbr), summarize, avg=ave(snowfall))
avg = data.table(avg)
avg <- avg[order(store_nbr,year,month)]
test_weather[,snow_avg:=avg[,avg]]
avg <- ddply(test_weather, .(month, year, store_nbr), summarize, avg=ave(stnpressure))
avg = data.table(avg)
avg <- avg[order(store_nbr,year,month)]
test_weather[,pres_avg:=avg[,avg]]


train_weather$day_hours <- train_weather$sunset - train_weather$sunrise
train_weather$heat_min_cool <- train_weather$heat - train_weather$cool
train_weather$pres_diff <- train_weather$stnpressure - train_weather$sealevel

test_weather$day_hours <- test_weather$sunset - test_weather$sunrise
test_weather$heat_min_cool <- test_weather$heat - test_weather$cool
test_weather$pres_diff <- test_weather$stnpressure - test_weather$sealevel

#Convert to matrix from datatable
trainMatrix <- train_weather[,lapply(.SD,as.numeric)] %>% as.matrix
testMatrix <- test_weather[,lapply(.SD,as.numeric)] %>% as.matrix

param <- list("objective" = "reg:linear",
              "eval_metric" = "rmse")

cv.nround <- 100
cv.nfold <- 3
#bst.cv = xgb.cv(param=param, data = trainMatrix, label = y, nfold = cv.nfold, nrounds = cv.nround, max.depth=20,nthread=2,eta=0.1, min_child_weight=4, subsample=0.8, gamma = 0.8, colsample_bytree=0.5)
bst.cv = xgb.cv(param=param, data = trainMatrix, label = y, nfold = cv.nfold, nrounds=cv.nround, max.depth=100,nthread=4,eta=0.1, min_child_weight=4, subsample=0.8, gamma = 1, colsample_bytree=0.8)

pred = predict(bst.cv,testMatrix)
pred = exp(pred)-1
pred = as.integer(pred)
pred = data.frame(test_id,pred)
names(pred) = c('id', 'units')
write.csv(pred,file='submission_cv_100.csv', quote=FALSE,row.names=FALSE)


#nround = 100
#bst = xgboost(param=param, data = trainMatrix, label = y, nrounds=nround, max.depth=100,nthread=4,eta=0.1, min_child_weight=4, subsample=0.8, gamma = 1, colsample_bytree=0.8)
#bst = xgboost(param=param, data = trainMatrix, label = y, nrounds=nround, max.depth=10,nthread=2,eta=0.05, min_child_weight=4, subsample=0.8, gamma = 1, colsample_bytree=0.5)
#bst = xgboost(param=param, data = trainMatrix, label = y, nrounds=nround, max.depth=20,nthread=2,eta=0.1, min_child_weight=4, subsample=0.8, gamma = 0.8, colsample_bytree=0.5)

#pred = predict(bst,testMatrix)
#pred = exp(pred)-1
#pred = as.integer(pred)
#pred = data.frame(test_id,pred)
#names(pred) = c('id', 'units')
#write.csv(pred,file='submission_100.csv', quote=FALSE,row.names=FALSE)

