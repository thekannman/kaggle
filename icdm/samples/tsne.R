# This script shows a 2D t-SNE visualization of a sample of the devices data in this competition

library(ggplot2)
library(readr)
library(Rtsne)

train <- read_csv("../input/dev_train_basic.csv")
test  <- read_csv("../input/dev_test_basic.csv")
device <- rbind(train, test)

dev_sample <- device[sample(1:nrow(device), size = 5000),]
mat <- model.matrix(~device_os+country+anonymous_c0+anonymous_5+anonymous_6+anonymous_7, dev_sample)
tsne <- Rtsne(mat, check_duplicates = FALSE, pca = TRUE, perplexity=30, theta=0.5, dims=2)

embedding <- as.data.frame(tsne$Y)
embedding$device_type <- as.factor(dev_sample$device_type)

p <- ggplot(embedding, aes(x=V1, y=V2, color=device_type)) +
     geom_point(size=1.25) +
     guides(colour = guide_legend(override.aes = list(size=6))) +
     xlab("") + ylab("") +
     ggtitle("t-SNE 2D Embedding of Device Data") +
     theme_light(base_size=20) +
     theme(strip.background = element_blank(),
           strip.text.x     = element_blank(),
           axis.text.x      = element_blank(),
           axis.text.y      = element_blank(),
           axis.ticks       = element_blank(),
           axis.line        = element_blank(),
           panel.border     = element_blank())

ggsave("tsne.png", p, width=8, height=6, units="in")

