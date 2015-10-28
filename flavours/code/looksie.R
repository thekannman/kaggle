library(caret)
library(Metrics)
library(doMC)
library(ggbiplot)
library(tsne)
library(pryr)
library(lineprof)
library(fastICA)
registerDoMC(cores=1)

train <- read.csv('../input/training.csv', header=T)
test <- read.csv('../input/test.csv', header=T)

train$production <- NULL
train$mass <- NULL
train$min_ANNmuon <- NULL
train$SPDhits <- NULL

train$id <- NULL
test.ids <- test$id
test$ids <- NULL
test$SPDhits <- NULL

signal <- train$signal
train$signal <- NULL

pc <- prcomp(train, center=T,scale=T)

g <- ggbiplot(pc, obs.scale = 1, var.scale = 1, 
              groups = as.factor(signal), ellipse = TRUE, 
              circle = TRUE)
g <- g + scale_color_discrete(name = '')
g <- g + theme(legend.direction = 'horizontal', legend.position = 'top')
g

rand.1000 <- sample(1:length(signal),1000,replace=F)
tsn <- tsne(train[rand.1000,])

ic <- fastICA(train,2)
ggplot(as.data.frame(ic$S), aes(V1,V2,colour=as.factor(signal))) + geom_point()
pc.100 <- prcomp(train, center=T,scale=T)
