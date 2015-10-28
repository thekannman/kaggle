# Compute the average precision at k
apk <- function(k, actual, predicted)
{
    score <- 0.0
    cnt <- 0.0
    for (i in 1:min(k,length(predicted)))
    {
        if (predicted[i] %in% actual && !(predicted[i] %in% predicted[0:(i-1)]))
        {
            cnt <- cnt + 1
            score <- score + cnt/i 
        }
    }
    score <- score / min(length(actual), k)
    score
}

# Compute the mean average precision at k
mapk <- function (k, actual, predicted)
{
    scores <- rep(0, length(actual))
    for (i in 1:length(scores))
    {
        scores[i] <- apk(k, actual[[i]], predicted[[i]])
    }
    # This line added since users with 0 actual purchased coupons should contribute 0
    # to score by definition
    scores[is.nan(scores)] <- 0

    score <- mean(scores)
    score
}

only.validate=T

#read in all the input data
cpdtr <- read.csv("../input/coupon_detail_train.csv")
cpltr <- read.csv("../input/coupon_list_train.csv")
cplte <- read.csv("../input/coupon_list_test.csv")
ulist <- read.csv("../input/user_list.csv")

#making of the train set
train <- merge(cpdtr,cpltr)
train <- train[,c("COUPON_ID_hash","USER_ID_hash",
                  "GENRE_NAME","DISCOUNT_PRICE",
                  "USABLE_DATE_MON","USABLE_DATE_TUE","USABLE_DATE_WED","USABLE_DATE_THU",
                  "USABLE_DATE_FRI","USABLE_DATE_SAT","USABLE_DATE_SUN","USABLE_DATE_HOLIDAY",
                  "USABLE_DATE_BEFORE_HOLIDAY","ken_name","small_area_name")]
#combine the test set with the train
cplte$USER_ID_hash <- "dummyuser"
cpchar <- cplte[,c("COUPON_ID_hash","USER_ID_hash",
                   "GENRE_NAME","DISCOUNT_PRICE",
                   "USABLE_DATE_MON","USABLE_DATE_TUE","USABLE_DATE_WED","USABLE_DATE_THU",
                   "USABLE_DATE_FRI","USABLE_DATE_SAT","USABLE_DATE_SUN","USABLE_DATE_HOLIDAY",
                   "USABLE_DATE_BEFORE_HOLIDAY","ken_name","small_area_name")]

train <- rbind(train,cpchar)
#NA imputation
train[is.na(train)] <- 1
#feature engineering (binning the price into different buckets)
train$DISCOUNT_PRICE <- cut(train$DISCOUNT_PRICE,breaks=c(-0.01,0,1000,10000,50000,100000,Inf),labels=c("free","cheap","moderate","expensive","high","luxury"))
#convert the factors to columns of 0's and 1's
train <- cbind(train[,c(1,2)],model.matrix(~ -1 + .,train[,-c(1,2)]))

#separate the test from train
test <- train[train$USER_ID_hash=="dummyuser",]
test <- test[,-2]
train <- train[train$USER_ID_hash!="dummyuser",]
coupons <- unique(train$COUPON_ID_hash)
n.coupons <- length(coupons)
n.val.coupons <- 500
users <- unique(train$USER_ID_hash)
n.users <- length(users)

avg.score = 0
n.validate <- 10
if(only.validate) {
  for (v in 1:n.validate) {
    val.sample <- sample(n.coupons,n.val.coupons, replace=F)
    val.coupons <- coupons[val.sample]
    validate <- train[train$COUPON_ID_hash %in% val.coupons,]
    sub.train <- train[-(train$COUPON_ID_hash %in% val.coupons),]
    validate.list <- vector("list", n.users)
    names(validate.list) <- users
    for (i in 1:nrow(validate)) {
      validate.list[[as.character(validate[i,"USER_ID_hash"])]] <-
        c(validate.list[[as.character(validate[i,"USER_ID_hash"])]],
          as.character(validate[i,"COUPON_ID_hash"]))
    }
    validate <- validate[,-2]

    #data frame of user characteristics
    uchar <- aggregate(.~USER_ID_hash, data=sub.train[,-1],FUN=mean)
    #calculation of cosine similairties of users and coupons
    val.score <- as.matrix(uchar[,2:ncol(uchar)]) %*% t(as.matrix(validate[,2:ncol(validate)]))

    cutoff <- 10.0
    score.counts <- sapply(1:nrow(val.score), function(i) sum(val.score[i,order(val.score[i,], decreasing=T)]>=cutoff))
    #order the list of coupons according to similairties and take only first 10 coupons
    uchar$VAL_PURCHASED_COUPONS <- do.call(rbind, lapply(1:nrow(uchar),FUN=function(i){
      purchased_cp <- paste(validate$COUPON_ID_hash[order(val.score[i,], decreasing = TRUE)][1:score.counts[i]],collapse=" ")
      return(purchased_cp)
    }))

    uchar <- merge(ulist, uchar, all.x=TRUE)
    submission <- uchar[,c("USER_ID_hash","VAL_PURCHASED_COUPONS")]

    submit.list <- vector("list", n.users)
    names(submit.list) <- users
    for (i in 1:nrow(submission)) {
      submit.list[[as.character(submission[i,"USER_ID_hash"])]] <-
        strsplit(submission[i,"VAL_PURCHASED_COUPONS"], split=" ")
    }
    val.score = mapk(10,validate.list,submit.list)
    avg.score = avg.score + val.score
    print(paste("score",v,":",val.score))    
  }
  avg.score = avg.score/n.validate
  print(paste("Average score: ",avg.score))
} else {
  #data frame of user characteristics
  uchar <- aggregate(.~USER_ID_hash, data=train[,-1],FUN=mean)
  #calculation of cosine similairties of users and coupons
  score <- as.matrix(uchar[,2:ncol(uchar)]) %*% t(as.matrix(test[,2:ncol(test)]))

  cutoff <- 10.0
  score.counts <- sapply(1:nrow(score), function(i) sum(score[i,order(score[i,], decreasing=T)]>=cutoff))
  #order the list of coupons according to similairties and take only first 10 coupons
  uchar$PURCHASED_COUPONS <- do.call(rbind, lapply(1:nrow(uchar),FUN=function(i){
    purchased_cp <- paste(test$COUPON_ID_hash[order(score[i,], decreasing = TRUE)][1:score.counts[i]],collapse=" ")
    return(purchased_cp)
  }))

  #make submission
  uchar <- merge(ulist, uchar, all.x=TRUE)
  submission <- uchar[,c("USER_ID_hash","PURCHASED_COUPONS")]
  write.csv(submission, file="cosine_sim.csv", row.names=FALSE)
}
