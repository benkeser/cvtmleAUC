set.seed(124)
n <- 100
p <- 30
X <- matrix(rnorm(n*p), nrow = n, ncol = p)
Y <- rbinom(n, 1, plogis(10*X[,1] + 20*X[,10]))
K <- 5
learner <- "glm_wrapper"
debug(cvauc_cvtmle)
fit <- cvauc_cvtmle(Y = Y, X = X, K = 5, learner = "glmnet_wrapper") 




grbg <- optim(fluc_mod_optim, method = "Brent", par = 0, 
              fld = full_long_data[full_long_data$Yi == 0,],
              lower = -20, upper = 20, control = list(reltol = 1e-14))



