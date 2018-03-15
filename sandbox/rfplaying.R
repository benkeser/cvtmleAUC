# playing around with random forests
install.packages("~/Dropbox/R/cvtmleAUC", repos = NULL, type = "source")
q("no")

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
K <- 20
p <- 10

set.seed(123)
dat <- makeData(n = n, p = p)

# get tmle and regular estimates
fit <- cvauc_cvtmle(Y = dat$Y, X = dat$X, K = 10, 
                    learner = "superlearner_wrapper")

N <- 1e5
bigdat <- makeData(n = N, p = p)
big_valid_pred_list <- lapply(fit$models, function(x){
  predict(x, newdata = bigdat$X)[[1]]
})
big_label_list <- rep(list(bigdat$Y), K)
true_cvauc <- mean(cvAUC::AUC(predictions = big_valid_pred_list,
                        labels = big_label_list))



