#library('e1071')
library('lubridate')
#library('randomForest')
#library('cvTools')
library('caret')
library(caretEnsemble)
train <- read.csv("train.csv",header=TRUE)
test <- read.csv("test.csv",header=TRUE)

train$day<-day(as.POSIXlt(train$Open.Date, format="%m/%d/%Y"))
train$month<-month(as.POSIXlt(train$Open.Date, format="%m/%d/%Y"))
train$year<-year(as.POSIXlt(train$Open.Date, format="%m/%d/%Y"))

train$days.open <- as.POSIXlt('01/01/2015', format="%m/%d/%Y") - as.POSIXlt(train$Open.Date, format="%m/%d/%Y")

test$day<-day(as.POSIXlt(test$Open.Date, format="%m/%d/%Y"))
test$month<-month(as.POSIXlt(test$Open.Date, format="%m/%d/%Y"))
test$year<-year(as.POSIXlt(test$Open.Date, format="%m/%d/%Y"))

test$days.open <- as.POSIXlt('01/01/2015', format="%m/%d/%Y") - as.POSIXlt(test$Open.Date, format="%m/%d/%Y")

train$City.Group <- as.numeric(factor(train$City.Group , levels=c("Other" ,
          "Big Cities")))
train$Type <- as.numeric(factor(train$Type), levels=c("DT","IL","FC"))
train$City <- as.numeric(factor(train$City), levels=c("Kırklareli", "Uşak",
							"Denizli","Konya",
							"Tokat","Amasya",
							"Kütahya","Samsun",
							"Şanlıurfa","Kastamonu",
							"Ankara","Tekirdağ",
							"Sakarya","Osmaniye",
							"Aydın","Antalya",
							"Diyarbakır","Kocaeli",
							"Karabük","Eskişehir",
							"Isparta","Bursa",
							"Muğla","Bolu",
							"Gaziantep","Kayseri",
							"Balıkesir","Adana",
							"Afyonkarahisar","Trabzon",
							"İzmir","Edirne",
							"Elazığ","İstanbul"))
test$City.Group <- as.numeric(factor(test$City.Group , levels=c("Other" ,
          "Big Cities")))
test$Type <- as.numeric(factor(test$Type), levels=c("DT","IL","FC"))
test$City <- as.numeric(factor(test$City), levels=c("Kırklareli", "Uşak",
                                                        "Denizli","Konya",
                                                        "Tokat","Amasya",
                                                        "Kütahya","Samsun",
                                                        "Şanlıurfa","Kastamonu",
                                                        "Ankara","Tekirdağ",
                                                        "Sakarya","Osmaniye",
                                                        "Aydın","Antalya",
                                                        "Diyarbakır","Kocaeli",
                                                        "Karabük","Eskişehir",
                                                        "Isparta","Bursa",
                                                        "Muğla","Bolu",
                                                        "Gaziantep","Kayseri",
                                                        "Balıkesir","Adana",
                                                        "Afyonkarahisar","Trabzon",
                                                        "İzmir","Edirne",
                                                        "Elazığ","İstanbul"))

train_cols<-train[,c(3:42,44:47)]
labels<-as.matrix(train[,43])
testdata<-test[,3:46]

train_cols <- data.frame(lapply(train_cols,as.numeric))
testdata<-data.frame(lapply(testdata,as.numeric))

train_cols$Pmax <- apply(train_cols[4:40],1,max)
train_cols$Pmin <- apply(train_cols[4:40],1,min)
train_cols$Pmed <- apply(train_cols[4:40],1,median)
train_cols$Psd <- apply(train_cols[4:40],1,sd)

train_cols$med_over_max <- train_cols$Pmed/train_cols$Pmax
train_cols$min_over_med <- train_cols$Pmin/train_cols$Pmed
train_cols$min_over_max <- train_cols$Pmin/train_cols$Pmax
train_cols$min_over_sd <- train_cols$Pmin/train_cols$Psd
train_cols$sd_over_max <- train_cols$Psd/train_cols$Pmax
train_cols$sd_over_med <- train_cols$Psd/train_cols$Pmed

for (i in 1:37)
{
    col.name <- paste("Porder",i,sep="")
    train_cols[,col.name] <- 0.0
}
P.names <- colnames(train_cols[,4:40])
for (i in 1:nrow(train_cols))
{
    train_cols[i,55:91] <- as.numeric(match(P.names, colnames(sort(train_cols[i,4:40], decreasing=T))))
}

testdata$Pmax <- apply(testdata[4:40],1,max)
testdata$Pmin <- apply(testdata[4:40],1,min)
testdata$Pmed <- apply(testdata[4:40],1,median)
testdata$Psd <- apply(testdata[4:40],1,sd)

testdata$med_over_max <- testdata$Pmed/testdata$Pmax
testdata$min_over_med <- testdata$Pmin/testdata$Pmed
testdata$min_over_max <- testdata$Pmin/testdata$Pmax
testdata$min_over_sd <- testdata$Pmin/testdata$Psd
testdata$sd_over_max <- testdata$Psd/testdata$Pmax
testdata$sd_over_med <- testdata$Psd/testdata$Pmed

for (i in 1:37)
{
    col.name <- paste("Porder",i,sep="")
    testdata[,col.name] <- 0.0
}
P.names <- colnames(testdata[,4:40])
for (i in 1:nrow(testdata))
{
    testdata[i,55:91] <- as.numeric(match(P.names, colnames(sort(testdata[i,4:40], decreasing=T))))
}


train_cols <- data.frame(lapply(train_cols,as.numeric))
testdata<-data.frame(lapply(testdata,as.numeric))

#fit.rf <- randomForest(revenue ~ ., data=train)
#print(fit.rf)
#importance(fit.rf)
#plot(fit.rf)
#plot( importance(fit.rf), lty=2, pch=16)
#lines(importance(fit.rf))
#imp = importance(fit.rf)

ctrl <- trainControl(method="cv", savePred=T, index=createResample(labels, 25))
#set.seed(1500)
#tuneGrid <- data.frame(ntree=50000,mtry=3)
#mod <- train(x=as.matrix(train_cols),y=as.vector(labels),method="rf",trControl=ctrl, ntree=5001)#,tuneGrid=tuneGrid)
#mod
#predictions<-as.data.frame(predict(mod,newdata=testdata))

#Used when Px are numeric
method_list = c('gbm','bdk','blackboost','bstLs','gaussprPoly','gcvEarth','kernelpls','kknn','knn','pls','rpart','rpart2','simpls','spls','svmPoly','svmRadial','svmRadialCost','treebag','widekernelpls','xyf','rf')
tuneList = list(rf1=caretModelSpec(method='rf', ntree=1000),
		bag1=caretModelSpec(method='treebag', nbagg=500),
		bag2=caretModelSpec(method='treebag', nbagg=1000),
		#spls1=caretModelSpec(method='spls', select='simpls'),
		svm1=caretModelSpec(method='svmRadialCost',tuneGrid=expand.grid(C=c(2.4))),
		rf1=caretModelSpec(method='rf', ntree=1000, preprocess='pca'),
                bag1=caretModelSpec(method='treebag', nbagg=500, preprocess='pca'),
                bag2=caretModelSpec(method='treebag', nbagg=1000, preprocess='pca'),
                #spls1=caretModelSpec(method='spls', select='simpls', preprocess='pca'),
                svm1=caretModelSpec(method='svmRadialCost', preprocess='ica'),
		rf1=caretModelSpec(method='rf', ntree=1000, preprocess='ica'),
                bag1=caretModelSpec(method='treebag', nbagg=500, preprocess='ica'),
                bag2=caretModelSpec(method='treebag', nbagg=1000, preprocess='ica'),
                #spls1=caretModelSpec(method='spls', select='simpls', preprocess='ica'),
                svm1=caretModelSpec(method='svmRadialCost', preprocess='ica'))
#method_list = c('gbm','bstLs','gcvEarth','kernelpls','kknn','knn','pls','rpart','rpart2','simpls','treebag','widekernelpls','rf')
mod <- caretList(x=train_cols,y=as.vector(labels),methodList=method_list,
                 trControl=ctrl, tuneList=tuneList)
greedy_ensemble <- caretEnsemble(mod)
summary(greedy_ensemble)
predictions<-as.data.frame(predict(greedy_ensemble,newdata=testdata))

submit<-as.data.frame(cbind(test[,1],predictions))
colnames(submit)<-c("Id","Prediction")

write.csv(submit,format(Sys.time(), "%m-%d_%H:%M_submission.csv"),row.names=FALSE,quote=FALSE)
