library(polycor)

coupon_area_test <- read.csv('../input/coupon_area_test.csv')
coupon_area_train <- read.csv('../input/coupon_area_train.csv')
coupon_detail_train <- read.csv('../input/coupon_detail_train.csv')
coupon_list_test <- read.csv('../input/coupon_list_test.csv')
coupon_list_train <- read.csv('../input/coupon_list_train.csv')
coupon_visit_train <- read.csv('../input/coupon_visit_train.csv')
prefecture_locations <- read.csv('../input/prefecture_locations.csv')
sample_submission <- read.csv('../input/sample_submission.csv')
user_list <- read.csv('../input/user_list.csv')

n_users <- nlevels(coupon_detail_train$USER_ID_hash)
n_coupons <- nlevels(coupon_detail_train$COUPON_ID_hash)
user_coupon_matrix <- matrix(nrow=n_users,ncol=n_coupons, 0)
rownames(user_coupon_matrix) <- levels(coupon_detail_train$USER_ID_hash)
colnames(user_coupon_matrix) <- levels(coupon_detail_train$COUPON_ID_hash)
user_coupon_matrix <- as.data.frame(user_coupon_matrix)

non_zeros <- matrix(nrow=nrow(coupon_detail_train),ncol=2,0)
non_zeros[,1] <- as.integer(coupon_detail_train$USER_ID_hash)
non_zeros[,2] <- as.integer(coupon_detail_train$COUPON_ID_hash)
for (i in 1:nrow(non_zeros)) {
  user_coupon_matrix[non_zeros[i,1],non_zeros[i,2]] = 1
}
