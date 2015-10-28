# Code to semi-automatically remove eye-blinks using independent components analysis
#
# Dan Acheson (Maineiac)
# 2/10/2015



library(fastICA)
library(data.table)
library(prospectr)
library(stringr)

remove_eyeblinks_ica <- function(eeg_data, channel_names, threshold = 20) {
  #Function for automatic removal of eyeblinks from eeg data
  # Performs ICA on the data using the fastICA algorithm
  # Then calculates a gap derivative on the data with smoothing,
  # and uses a threshold to determine which ICs contain blinks
  # The removes the blinks from the IC matrix and reconstructs the data
  #
  # INPUT:
  #  eeg_data : data.frame organized as timeXchan
  #  channel_names : strig vector of channel names
  #  threshod : threshold used for detecting eyeblinks - count of events with big derivatives
  #
  # OUTPUT:
  # eeg_data with blink ICs removed
  
  library(fastICA)
  library(prospectr)
  
  
  data_chan_only <- eeg_data[,which(names(eeg_data) %in% channel_names)]
  print("Performing ICA")
  data_ica <- fastICA(data_chan_only, n.comp = ncol(data_chan_only), method = "C", verbose = F)
  blink_detect <- sapply(seq(1:ncol(data_ica$S)), 
                         function(x) sum(gapDer(data_ica$S[,x],s=11,w=3)>0.2))
  
  blink_components <- which(blink_detect > threshold)
  
  blink_to_remove <- c()
  #loop through each of the provided component and ask user if they want to remove it
  for(blink in blink_components) {
    par(mfrow = c(2,1))
    print_length = round(dim(data_ica$S)[1] / 10)
    #plot component data vs. EOG data
    plot(data_ica$S[1:print_length, blink], type = 'l', ylab = paste0("ICA component ", blink))
    plot(eeg_data$EOG[1:print_length], type = 'l', ylab = "EOG")
    remove <- readline("Remove this component? Y/N > ") 
    if("y" %in% tolower(remove)) {
      blink_to_remove <- c(blink_to_remove, blink)
    }
  }
  #remove components
  if(length(blink_to_remove) > 0) {
    print(paste0("Removing ", length(blink_to_remove), " components"))
    #ICA Sources retrieved, and eyeblinks set to 0
    ica_S <- data_ica$S
    ica_S[,blink_to_remove] == 0
    dat_reconstruct <- ica_S %*% data_ica$A
    
    #put the reconstructed data back in the data frame
    eeg_data[,which(names(eeg_data) %in% channel_names)] <- dat_reconstruct
  }
  return(eeg_data)
  
}


#Loop through train or test files to remove eyeblinks
# NOTE: saves the file as .rbin, not .csv

setwd("/home/zak/kaggle/bci/data")

ica_files <- list.files("./test/")
ica_files <- ica_files[str_detect(ica_files,".csv")]

file_count <- 1
total_files <- length(ica_files)
for(curr_file in ica_files) {
  dat <- as.data.frame(fread(paste0("./test/",curr_file)))
  channel_names <- names(dat)[2:57]
  file_prefix <- str_replace(curr_file,".csv","")
  dat_ica <- remove_eyeblinks_ica(dat, channel_names = channel_names, threshold =10)
  save(dat_ica, file = paste0("./test/", file_prefix, ".ica_remove.rbin"))
  print(paste0("Done with file: ", file_count, "/", total_files))
  file_count <- file_count + 1
}

