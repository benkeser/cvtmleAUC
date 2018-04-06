devtools::load_all("~/Dropbox/R/cvtmleAUC")
set.seed(11211125)
n <- 100
p <- 10
X <- data.frame(matrix(rnorm(n*p), nrow = n, ncol = p))
Y <- rbinom(n, 1, plogis(X[,1] + X[,10] + X[,2]*X[,3]))
K <- 50
# undebug(cvtn_cvtmle)
# debug(.makeLongDataNestedCV)
fit <- cvtn_cvtmle(Y = Y, X = X, K = 20, nested_K = 50, learner = "randomforest_wrapper",
				   nested_cv = TRUE, sens = 0.95) 



fit <- boot_corrected_cvtn(Y = Y, X = X, learner = "randomforest_wrapper")

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TO DO: 
# Should specify an inner K folds for the nested CV. 
# Seems like weird quantile values could be due to 
# using too few folds. K + K-1 is computationally useful
# but maybe not best when K is small. 
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# get the truth
big_n <- 1e5
bigX <- data.frame(matrix(rnorm(big_n*p), nrow = big_n, ncol = p))
bigY <- rbinom(big_n, 1, plogis(bigX[,1] + bigX[,10] + bigX[,2]*bigX[,3]))
bigpred <- lapply(fit$prediction_list[1:K], function(x){
	if("randomForest" %in% class(x$model)){
		predict(x$model, newdata = bigX, type = "prob")[,2]
	}else{
		predict(x$model, newdata = bigX, type = "response")
	}
})
bigquantile <- lapply(bigpred, function(x, Y){
	quantile(x[Y == 1], p = 0.05, type = 8)
}, Y = bigY)
big_testneg <- mapply(p = bigpred, q = bigquantile, function(p, q, Y){
	mean(p <= q)
}, MoreArgs = list(Y = bigY))
mean(big_testneg)

# fit on full data
# fit_full <- glm_wrapper(train = list(X = X, Y = Y), 
#                         test = list(X = X, Y = Y))
# bigpred_full <- predict(fit_full$model, newdata = bigX, type = "response")
fit_full <- randomforest_wrapper(train = list(X = X, Y = Y), 
                        test = list(X = X, Y = Y))
bigpred_full <- predict(fit_full$model, newdata = bigX, type = "prob")[,2]
bigquantile_full <- quantile(bigpred_full[bigY == 1], p = 0.05, type =8)
big_testneg_full <- mean(bigpred_full <= bigquantile_full)
big_testneg_full



grbg <- optim(fluc_mod_optim, method = "Brent", par = 0, 
              fld = full_long_data[full_long_data$Yi == 0,],
              lower = -20, upper = 20, control = list(reltol = 1e-14))



