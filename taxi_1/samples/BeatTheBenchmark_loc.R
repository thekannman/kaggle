library(rjson)
library(data.table)

### Control the number of trips read for training (all=-1)
### Control the number of closest trips used to calculate trip duration
# N_read <- 100000
# N_trips <- 1000
N_read <- -1
N_trips <- 10000

### Get starting & ending longitude and latitude
get_coordinate <- function(row){
  lonlat <- fromJSON(row)
  snapshots <- length(lonlat)  
  start <- lonlat[[1]]
  end <- lonlat[[snapshots]]
  return(list(start[1], start[2], end[1], end[2], snapshots))
} 

### Get Haversine distance
get_dist <- function(lon1, lat1, lon2, lat2) {  
  lon_diff <- abs(lon1-lon2)*pi/360
  lat_diff <- abs(lat1-lat2)*pi/360
  a <- sin(lat_diff)^2 + cos(lat1) * cos(lat2) * sin(lon_diff)^2  
  d <- 2*6371*atan2(sqrt(a), sqrt(1-a))
  return(d)
}

### Read
train <- fread('../train.csv', select=c('POLYLINE'), stringsAsFactors=F, nrows=N_read)
test <- fread('../test.csv', select=c('TRIP_ID', 'POLYLINE'), stringsAsFactors=F)
train <- train[POLYLINE!='[]']
train[, r:=-seq(.N, 1, -1)]
test[, r:=1:.N]
setkey(train, r)
setkey(test, r)

### Get starting & ending position from POLYLINE
train[, c('lon', 'lat', 'lon_end', 'lat_end', 'snapshots'):=get_coordinate(POLYLINE), by=r]
test[, c('lon', 'lat', 'lon_end', 'lat_end', 'snapshots'):=get_coordinate(POLYLINE), by=r]
train[, POLYLINE:=NULL]
test[, POLYLINE:=NULL]

for (i in 1:nrow(test)) {
  ### Get the distance from each train trip to each test trip
  train[, c('lon2', 'lat2', 'snapshots2'):=test[i, list(lon, lat, snapshots)]]
  train[, d:=get_dist(lon, lat, lon2, lat2)]
  
  ### Get the closest trips to each test trip
  ### Bound below by 10 meters since we use 1/distance^2 as weight
  ### Trips must last as long as 80% of test duration
  ### Trimmed mean of x and y coordinate, assuming flat earth
  ds <- train[snapshots>=snapshots2*.8][order(d)][1:N_trips][!is.na(r), list(lon_end, lat_end)]  
  test[i, c('LONGITUDE', 'LATITUDE'):=ds[, list(mean(lon_end, trim=.1), mean(lat_end, trim=.1))]]    
}

write.csv(test[, list(TRIP_ID, LATITUDE, LONGITUDE)], 'same_start_all_90.csv', row.names=F)
