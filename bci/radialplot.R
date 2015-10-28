library('plotrix')

args <- commandArgs(trailingOnly = TRUE)
table <- data.frame(read.csv(args[1], header = TRUE, sep=","))
radial.plot(table$Radius, table$Phi)
