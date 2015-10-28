args <- commandArgs(trailingOnly = TRUE)
file <- args[1]
if (length(args) == 2) {
    column <- as.integer(args[2])
}
table <- read.table(file)
plot(density(table[,column]))
