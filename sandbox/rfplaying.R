# playing around with random forests
# install.packages("~/Dropbox/R/cvtmleAUC", repos = NULL, type = "source")

library(cvtmleAUC)

makeData <- function(n, p){
	X <- matrix(rnorm(n*p), nrow = n, ncol = p)
	if(p < 10){
		Y <- rbinom(n, 1, plogis(X[,1]))		
	}else{
		Y <- rbinom(n, 1, plogis(0.25*X[,1] - 0.5*X[,10] + 0.125*X[,2]*X[,3]))		
	}
	return(list(X = data.frame(X), Y = Y))
}
n <- 100
K <- 10

dat <- makeData(n = n, p = 10)
# set seed
set.seed(1234)

# get tmle and regular estimates
fit <- cvauc_cvtmle(Y = dat$Y, X = dat$X, K = K, 
                    learner = "randomforest_wrapper")
