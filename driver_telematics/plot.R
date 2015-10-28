args <- commandArgs(trailingOnly = TRUE)
file <- args[1]
if (length(args) > 1) {
    column <- as.integer(args[2])
}
if (length(args) > 2) {
    end <- as.integer(args[3])
}
table <- read.table(file)
plot(1:end,table[1:end,column])
