k = 3


coupon_list_train <- read.csv("../input/coupon_list_train.csv")
coupon_list_test <- read.csv("../input/coupon_list_test.csv")

### Couple new features

coupon_list_train$year_from = as.numeric(format(as.POSIXlt(coupon_list_train$VALIDFROM, format = "%Y-%m-%d"), "%Y"))
coupon_list_train$month_from = as.numeric(format(as.POSIXlt(coupon_list_train$VALIDFROM, format = "%Y-%m-%d"), "%m"))
coupon_list_train$day_from = as.numeric(format(as.POSIXlt(coupon_list_train$VALIDFROM, format = "%Y-%m-%d"), "%d"))

coupon_list_train$year_end = as.numeric(format(as.POSIXlt(coupon_list_train$VALIDEND, format = "%Y-%m-%d"), "%Y"))
coupon_list_train$month_end = as.numeric(format(as.POSIXlt(coupon_list_train$VALIDEND, format = "%Y-%m-%d"), "%m"))
coupon_list_train$day_end = as.numeric(format(as.POSIXlt(coupon_list_train$VALIDEND, format = "%Y-%m-%d"), "%d"))

coupon_list_train = subset(coupon_list_train, select = -c(VALIDFROM, VALIDEND))

coupon_list_test$year_from = as.numeric(format(as.POSIXlt(coupon_list_test$VALIDFROM, format = "%Y-%m-%d"), "%Y"))
coupon_list_test$month_from = as.numeric(format(as.POSIXlt(coupon_list_test$VALIDFROM, format = "%Y-%m-%d"), "%m"))
coupon_list_test$day_from = as.numeric(format(as.POSIXlt(coupon_list_test$VALIDFROM, format = "%Y-%m-%d"), "%d"))

coupon_list_test$year_end = as.numeric(format(as.POSIXlt(coupon_list_test$VALIDEND, format = "%Y-%m-%d"), "%Y"))
coupon_list_test$month_end = as.numeric(format(as.POSIXlt(coupon_list_test$VALIDEND, format = "%Y-%m-%d"), "%m"))
coupon_list_test$day_end = as.numeric(format(as.POSIXlt(coupon_list_test$VALIDEND, format = "%Y-%m-%d"), "%d"))

coupon_list_test = subset(coupon_list_test, select = -c(VALIDFROM, VALIDEND))

### Scale numeric variables

for(i in 1:ncol(coupon_list_train)){
  if(is.numeric(coupon_list_train[,i])){
    scaled = as.numeric(scale(c(coupon_list_train[,i], coupon_list_test[,i])))
    coupon_list_train[,i] = scaled[1:nrow(coupon_list_train)]
    coupon_list_test[,i] = scaled[(nrow(coupon_list_train) + 1):length(scaled)]
  }
}

rm(scaled)

### Compute similarity between train coupons and test coupons

library(StatMatch)

join <- rbind(coupon_list_train, coupon_list_test)
coupon_list_train <- join[1:19413,]
coupon_list_test <- join[19414:19723,]

distances <- gower.dist(data.x = coupon_list_train, data.y = coupon_list_test)
distances <- as.matrix(distances)

### For each train coupon, get k = 10 most similar test coupons

num_row = dim(distances)[1]
num_col = dim(distances)[2]
recommended_coupons <- vector("list", num_row)

for(i in 1:num_row){
  row_distance = distances[i,]
  indices = numeric(k)
  for(j in 1:k){
    pos_min = which.min(row_distance)
    indices[j] = pos_min
    row_distance[pos_min] = 1
  }
  recommended_coupons[[i]] = as.character(coupon_list_test$COUPON_ID_hash[indices])
}

## Convert every element of the list to a single character vector, separated
## by spaces
recom = character(num_row)
i = 1
for(recommendation in recommended_coupons){
  recom[i] = paste(recommendation, collapse = " ")
  i = i + 1
}
rm(recommended_coupons, i, indices, j, k, num_col, num_row, pos_max, recommendation, row_distance, distances, join)

coupon_list_train$recom = recom

rm(coupon_list_test, recom)

## User-list 

user_list <- read.csv("../input/user_list.csv")
coupon_detail_train <- read.csv("../input/coupon_detail_train.csv")

join <- merge(coupon_detail_train, coupon_list_train)
join <- subset(join, select = c(USER_ID_hash, recom))

rm(coupon_detail_train, coupon_list_train)

## Preparing for submission

gc()
pos <- numeric(nrow(user_list))

for(i in 1:nrow(user_list)){
  print(i)
  position <- match(user_list$USER_ID_hash[i], join$USER_ID_hash, nomatch = 32284)[1]
  pos[i] <- position
}

save.image(file = "ponpare.RData")

join <- join[pos, ]

names(join) <- c("USER_ID_hash", "PURCHASED_COUPONS")

sub <- subset(user_list, select = c(USER_ID_hash))
sub$PURCHASED_COUPONS = join$PURCHASED_COUPONS

## Maybe ignore not known predictions

#to_ignore = which(as.character(user_list$USER_ID_hash) != as.character(join$USER_ID_hash))
#sub[to_ignore, 2] = ""


write.csv(sub, "submission.csv", row.names = F)

######
