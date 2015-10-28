# cpvtr <- read.csv("../input/coupon_visit_train.csv")
# cpdtr <- read.csv("../input/coupon_detail_train.csv")
# cpltr <- read.csv("../input/coupon_list_train.csv")
cplte <- read.csv("../input/coupon_list_test.csv")
ulist <- read.csv("../input/user_list.csv")

#match coupons by ken_name
submission <- merge(ulist, cplte, by.x="PREF_NAME", by.y="ken_name", all.x=TRUE)
submission <- aggregate(COUPON_ID_hash~USER_ID_hash, data=submission,FUN=function(x)paste(x, collapse = " "))
submission <- merge(ulist, submission, by="USER_ID_hash", all.x=TRUE)
submission <- submission[,c("USER_ID_hash","COUPON_ID_hash")]

colnames(submission) <- c("USER_ID_hash","PURCHASED_COUPONS")

write.csv(submission, file="ken_name_match.csv", row.names=FALSE)
