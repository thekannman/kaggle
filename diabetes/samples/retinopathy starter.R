#=============================================================================
# Script Name:  retinopathy starter.R
# Created on:   19/02/2015
# Author:       Alastair Muir
# Purpose:      read sample files and set up feature engineering
# History:      19/02/2015 - created
#               20/02/2015 - sped up using "jpeg" package for reading images
#               23/02/2015 - tested against entire test and train data images
#=============================================================================
rm(list=ls())

#=============================================================================
# libraries
#=============================================================================
library("jpeg")

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
# Calculate statistics for images in both training and testing directories
#=============================================================================
selection <- c(train.filenames, test.filenames)
# optional sample selection for testing
# set.seed(1)
# selection <- sample(selection, 1000, replace = FALSE)

system.time(fileStatsList <- lapply(selection, ImageStats))
# This took a while on my Intel 750@2.67 GHz, Quad processor with 12GBy memory
#    user   system  elapsed 
#65079.45  7884.32 76904.35
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
write.table(fileStats, file = "AllFileStats.csv", sep = ',', row.names = FALSE)
write.table(tableCount, file = "AllTableCounts.csv", sep = ",", row.names = TRUE)