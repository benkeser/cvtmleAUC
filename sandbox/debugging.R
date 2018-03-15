set.seed(124)
n <- 100
p <- 10
X <- data.frame(matrix(rnorm(n*p), nrow = n, ncol = p))
Y <- rbinom(n, 1, plogis(X[,1] + X[,10] + X[,2]*X[,3]))
K <- 5
learner <- "randomforest_wrapper"
debug(cvauc_cvtmle)
# debug(.makeLongDataNestedCV)
fit <- cvauc_cvtmle(Y = Y, X = X, K = 3, learner = "superlearner_wrapper",
					nested_cv = TRUE) 






grbg <- optim(fluc_mod_optim, method = "Brent", par = 0, 
              fld = full_long_data[full_long_data$Yi == 0,],
              lower = -20, upper = 20, control = list(reltol = 1e-14))



