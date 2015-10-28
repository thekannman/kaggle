library(readr)
library(corrgram)
library(ellipse)
library(tidyr)


device <- read_csv("../input/dev_train_basic.csv")
cookie <- read_csv("../input/cookie_all_basic.csv")
ip <- read.table("../input/id_all_ip.csv",sep="{",col.names=c("first","second"))

## numerise device data
device$anonymous_c1      <- as.numeric(gsub("anonymous_c1_", "", device$anonymous_c1))
device$anonymous_c2      <- as.numeric(gsub("anonymous_c2_", "", device$anonymous_c2))
device$device_os         <- as.numeric(gsub("devos_"  , "", device$device_os))
device$drawbridge_handle <- as.numeric(gsub("handle_" , "", device$drawbridge_handle))
device$device_type       <- as.numeric(gsub("devtype_", "", device$device_type))
device$country           <- as.numeric(gsub("country_", "", device$country))
device$device_id     <- as.numeric(gsub("id_"     , "", device$device_id))

## numerise cookie data
cookie$anonymous_c1      <- as.numeric(gsub("anonymous_c1_", "", cookie$anonymous_c1))
cookie$anonymous_c2      <- as.numeric(gsub("anonymous_c2_", "", cookie$anonymous_c2))
cookie$computer_os_type        <- as.numeric(gsub("computer_os_type_"  , "", cookie$computer_os_type))
cookie$drawbridge_handle <- as.numeric(gsub("handle_" , "", cookie$drawbridge_handle))
cookie$computer_browser_version <- as.numeric(gsub("computer_browser_version_", "", cookie$computer_browser_version))
cookie$country                  <- as.numeric(gsub("country_", "", cookie$country))
cookie$cookie_id                <- as.numeric(gsub("id_"     , "", cookie$cookie_id))

## parse ip data
ip_parse <-separate(ip, first, into = c("device_id", "type","dummy"), sep = ",")

## separate device and cookie DF
ip_cookie <- ip_parse[which(ip_parse$type==1) ,c("device_id","second")]
names(ip_cookie) <- c("cookie_id", "ip_cookie_list")
ip_cookie$cookie_id                <- as.numeric(gsub("id_"     , "", ip_cookie$cookie_id))

## split IP tuple 
ip_device <- ip_parse[which(ip_parse$type==0),c("device_id","second")]
ip_device$second  <- gsub("\\),\\(", ";", ip_device$second)
ip_device$second  <- gsub("\\)|\\(", "" , ip_device$second)
ip_device$device_id     <- as.numeric(gsub("id_"     , "", ip_device$device_id))
names(ip_device) <- c("device_id", "ip_device_list")


m1 = merge(device, cookie, by="drawbridge_handle")
m2 = merge(m1, ip_cookie, by="cookie_id")
m3 = merge(m2, ip_device, by="device_id")

summary(m3)

# plot  correlation matrice
png("corrgram001.png", width = 2000, height = 2000, units = 'px')
corrgram(m3, order=TRUE, lower.panel=panel.shade,font.labels=1,
         upper.panel=panel.pie, text.panel=panel.txt,
         main="train device cookie ip correlation in PC2/PC1 Order") 


png("plotcorr001.png", width = 2000, height = 2000, units = 'px')
mamat = cor(m3[, 1:21])
plotcorr(mamat, col = colorRampPalette(c("firebrick3", "white", "navy"))(10))


