#! /usr/bin/env Rscript

# print session info
sessionInfo()
cat("\n \n")
# print lib paths
.libPaths()
# load package from my local directory
library(cvtmleAUC, lib.loc = "/home/dbenkese/R/x86_64-pc-linux-gnu-library/3.4")

# load data.table
library(data.table)

# load glmnet
library(glmnet)

# try to use data.table
testDT <- as.data.table(data.frame(X = runif(100), Y = rnorm(100)))

# try to use glmnet
x <- model.matrix(~-1+X+ I(X^2) + I(X^3) , data = data.frame(X = testDT$X))
fit <- glmnet(x = x, y = testDT$Y)

