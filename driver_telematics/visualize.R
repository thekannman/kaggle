args <- commandArgs(trailingOnly = TRUE)
traj <- read.csv(args[1], header = TRUE, sep=",")
plot(traj$x,traj$y)
