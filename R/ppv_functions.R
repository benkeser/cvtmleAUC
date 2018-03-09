#' Compute CVTML estimates of cross-validated AUC
#' 
#' TO DO: Add
#' @param Y The outcome
#' @param X The predictors
#' @param K The number of folds
#' @param learner The learner wrapper
#' @param seed A random seed to set
#' @param parallel Compute the predictors in parallel?
#' @param maxIter Maximum number of iterations for cvtmle
#' @param icTol Iterate until maxIter is reach or mean of cross-validated
#' efficient influence function is less than \code{icTol}
#' @param ... other arguments, not currently used
#' @importFrom SuperLearner CVFolds
#' @importFrom cvAUC ci.cvAUC
#' @importFrom stats uniroot
#' @export
#' @return A list
#' @examples
#' n <- 200
#' p <- 10
#' X <- matrix(rnorm(n*p), nrow = n, ncol = p)
#' Y <- rbinom(n, 1, plogis(X[,1] + X[,10]))
#' fit <- cvauc_cvtmle(Y = Y, X = X, K = 5, learner = "glm_wrapper")
cvtn_cvtmle <- function(Y, X, K, sens = 0.95, learner = "glm_wrapper", 
                         seed = 1234,
                         parallel = FALSE, maxIter = 10, 
                         icTol = 1/length(Y), 
                         ...){
  n <- length(Y)
  set.seed(seed)
  folds <- SuperLearner::CVFolds(N = n, id = NULL, Y = Y, 
                                 cvControl = list(V = K, stratifyCV = TRUE, 
                                    shuffle = TRUE, validRows = NULL))
  prediction_list <- .getPredictions(learner = learner, Y = Y, X = X, 
                                 K = K, folds=folds, parallel = FALSE)
  # get quantile estimates
  quantile_list <- lapply(prediction_list, .getQuantile, p = sens)

  # get density estimate 
  density_list <- mapply(x = prediction_list, c0 = quantile_list, 
                         FUN = .getDensity, SIMPLIFY = FALSE)

  # make targeting data
  target_data <- .makeTargetingData(prediction_list, quantile_list, density_list, folds)
  target_data$weight <- with(target_data, 1 + Y/gn * f_ratio)
  target_data$logit_Fn <- SuperLearner::trimLogit(target_data$Fn, trim = 1e-5)
  fluc_mod <- glm(ind ~ offset(logit_Fn), data = target_data, family = "binomial",
                  weights = weight)
  target_data$Fnstar <- fluc_mod$fitted.values

  # compute parameter for each fold
  cv_estimates <- by(target_data, target_data$fold, function(x){
    x$Fn[x$Y==0][1] * (1-x$gn[1]) + x$Fn[x$Y==1][1] * x$gn[1]
  })

  

    # format output
    out <- list()
    out$est_cvtmle <- tmle_auc[iter]
    out$iter_cvtmle <- iter
    out$cvtmle_trace <- tmle_auc
    out$se_cvtmle <- sqrt(var(ic)/n)
    out$est_init <- init_auc
    out$est_empirical <- est_empirical
    out$se_empirical <- se_empirical
    out$est_onestep <- est_onestep
    out$se_onestep <- se_onestep
    out$est_esteq <- est_esteq
    out$se_esteq <- se_esteq

    out$models <- lapply(prediction_list, "[[", "model")
    class(out) <- "cvauc"
    return(out)
}

#' @param x An entry in prediction_list
#' @importFrom stats quantile 
.getQuantile <- function(x, p){
  stats::quantile(x$psi_nBn_trainx[x$train_y == 1], p = p, type = 1)
}

#' @param x An entry in prediction_list
#' @importFrom np npudensbw npudens
#' @importFrom stats predict
.getDensity <- function(x, c0, ... ){
  # density given y = 1
  fitbw <- np::npudensbw(x$psi_nBn_trainx[x$train_y == 1])
  fit <- np::npudens(fitbw)
  # estimate at c0
  f_10_c0 <- stats::predict(fit, edat = c0)

  # marginal density
  fitbw_marg <- np::npudensbw(x$psi_nBn_trainx)
  fit_marg <- np::npudens(fitbw_marg)
  # estimate at c0
  f_0_c0 <- stats::predict(fit_marg, edat = c0)

  # return both
  return(list(f_10_c0 = f_10_c0, f_0_c0 = f_0_c0))
}


.makeTargetingData <- function(prediction_list, quantile_list, density_list, folds){
  Y_vec <- Reduce(c, lapply(prediction_list, "[[", "test_y"))
  n <- length(Y_vec)
  fold_vec <- sort(rep(1:length(folds), unlist(lapply(folds, length))))
  gn_vec <- rep(mean(Y_vec), n)
  F_nBn_vec <- Reduce(c,mapply(FUN = function(x,c0){
    F_nBn_y1_at_c0 <- F_nBn_star(psi_x = c0, y = 1, Psi_nBn_0 = x$psi_nBn_trainx, Y_Bn = x$train_y)
    F_nBn_y0_at_c0 <- F_nBn_star(psi_x = c0, y = 0, Psi_nBn_0 = x$psi_nBn_trainx, Y_Bn = x$train_y)
    ifelse(x$test_y == 0, F_nBn_y0_at_c0, F_nBn_y1_at_c0)
  }, c0 = quantile_list, x = prediction_list))
  dens_ratio <- Reduce(c, mapply(FUN = function(x, dens){
    rep(dens[[2]]/dens[[1]], length(x$test_y))
  }, x = prediction_list, dens = density_list))
  ind <- Reduce(c, mapply(x = prediction_list, c0 = quantile_list, function(x, c0){
    as.numeric(x$psi_nBn_testx <= c0)
  }))
  out <- data.frame(fold = fold_vec, Y = Y_vec, gn = gn_vec, Fn = F_nBn_vec,
                    f_ratio = dens_ratio, ind = ind)
  return(out)
}