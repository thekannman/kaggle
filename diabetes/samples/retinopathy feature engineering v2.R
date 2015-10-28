#=============================================================================
# Script Name:  retinopathy starter v2.R
# Created on:   19/02/2015
# Author:       Alastair Muir
# Purpose:      read sample files and set up feature engineering
# History:      19/02/2015 - created
#               20/02/2015 - sped up using "jpeg" package for reading images
#               23/02/2015 - tested against entire test and train data images
#                            over 21 hours
#               23/02/2015 - using parLapply::doParallel for parallel processing
#                            factor of three faster
#=============================================================================
rm(list=ls())

#=============================================================================
# libraries
#=============================================================================
library("jpeg")
library("doParallel")

#=============================================================================
# specify your data directory and training and testing subdirectories below
# My structure is ./Kaggle/diabetic retinopathy/code
#                                              /data/train
#                                              /data/test
#=============================================================================
setwd("C:/Projects/Kaggle/diabetic retinopathy/code")

data_dir <- "../data"
train_data_dir <- paste(data_dir, "/train", sep = "")
test_data_dir <- paste(data_dir, "/test", sep = "")

#=============================================================================
# Function:     ImageStats
# Created on:   18/02/2015
# Author:       Alastair Muir
# Purpose:      Calculate some basic statistics about an image file
# Input:        image file name
# Return:       list of characteristics of the image, length, width,
#               aspect ratio, etc.
# History:      18/02/2015 - created
#               20/02/2015 - added ratio of black to total image
#=============================================================================
ImageStats <- function(fileName){
        # insert your own clever code here to detect edges, adjust constrast, resize
        # images, detect inverted images, and output resized image libraries
        #
        # This function is the workhorse for generating:
        #       1) feature library for classification or regression models
        #       2) resized, subsampled images for CNN models. The resizing
        #          should allow much more rapid modeling.
        imageInfo <- file.info(fileName)[1]
        imageFile <- readJPEG(fileName)
        imageLength <- nrow(imageFile)
        imageWidth <- ncol(imageFile)
        imageRatioBlack <- sum(imageFile[,,1] < 0.02)/(imageLength*imageWidth)
        imageDensity <- mean(imageFile)/(1 - imageRatioBlack)
        imageRatio <- imageLength/imageWidth
        return(c(file = fileName,
                 size = imageInfo,
                 length = imageLength,
                 width = imageWidth,
                 density = imageDensity,
                 ratio = imageRatio,
                 ratioBlack = imageRatioBlack))
}

#=============================================================================
# read in training data filenames
#=============================================================================
train.filenames <- list.files(train_data_dir, pattern = "*.jpeg", full.names = TRUE)
test.filenames <- list.files(test_data_dir, pattern = "*.jpeg", full.names = TRUE)

#=============================================================================
# Select a sample of images from both directories for testing and debugging
#=============================================================================
selection <- c(train.filenames, test.filenames)
# optional sample selection for testing
# set.seed(1)
# selection <- sample(selection, 100, replace = FALSE)

#=============================================================================
# Calculate statistics for images in both training and testing directories
## This took a while on my Intel 750@2.67 GHz, Quad processor with 12GBy memory
#    user   system  elapsed (21 hours, 21 minutes)
# 65079.45  7884.32 76904.35
#
# Intel 750@2.67 GHz, Windows 7, 64-bit, Quad processor
# 12GBy memory using three parallel CPUs
# elapsed (8 hours, 28 minutes)
# 30528.49 

#=============================================================================
# register cluster, export function and packages
# I have four processors so I use three for the cluster
cl <- makeCluster(3)
registerDoParallel(cl)
setDefaultCluster(cl)
clusterExport(cl, c("ImageStats"))
clusterEvalQ(cl, library("jpeg"))

system.time(fileStatsList <- parLapply(cl, selection, ImageStats))[3] 

stopCluster(cl)

fileStats <- data.frame(matrix(unlist(fileStatsList), ncol = 7, byrow = T))
colnames(fileStats) <- c("filename", "size", "height", "width",
                         "mean", "ratio", "ratio black")                


#=============================================================================
# Output tabulation of image sizes. Previous work on this problem used camara
# type as a feature for Random Forest classification. This table shows 28
# different combinations of resolution and aspect ration.
#=============================================================================
tableCount <- table(fileStats$width, fileStats$height)

#=============================================================================
# Write output files for later processing
#=============================================================================
write.table(fileStats, file = "SampleFileStats.csv", sep = ',', row.names = FALSE)
write.table(tableCount, file = "SampleTableCounts.csv", sep = ",", row.names = TRUE)