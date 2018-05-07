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
#' @importFrom cvma ci.cvAUC_withIC
#' @importFrom stats uniroot
#' @export
#' @return A list
#' @examples
#' n <- 200
#' p <- 10
#' X <- matrix(rnorm(n*p), nrow = n, ncol = p)
#' Y <- rbinom(n, 1, plogis(X[,1] + X[,10]))
#' fit <- cvauc_cvtmle(Y = Y, X = X, K = 5, learner = "glm_wrapper")
cvauc_cvtmle <- function(Y, X, K = 20, learner = "glm_wrapper", 
                         nested_cv = FALSE,
                         nested_K = K - 1,
                         parallel = FALSE, maxIter = 10, 
                         icTol = 1 / length(Y), 
                         prediction_list = NULL,
                         ...){
  n <- length(Y)
  folds <- SuperLearner::CVFolds(N = n, id = NULL, Y = Y, 
                                 cvControl = list(V = K, 
                                    stratifyCV = ifelse(K <= sum(Y) & K <= sum(!Y), TRUE, FALSE), 
                                    shuffle = TRUE, validRows = NULL))
  if(is.null(prediction_list)){
    prediction_list <- .getPredictions(learner = learner, Y = Y, X = X, 
                                   K = K, nested_K = nested_K, folds=folds, parallel = FALSE,
                                   nested_cv = nested_cv)
  }

  # initial distributions of psi in training samples
  # dist_psix_y0 <- lapply(prediction_list, .getPsiDistribution, y = 0)
  # dist_psix_y1 <- lapply(prediction_list, .getPsiDistribution, y = 1)
  
  # make long data for targeting step
  if(!nested_cv){
    long_data_list <- lapply(prediction_list, .makeLongData, gn = mean(Y))
  }else{
    long_data_list <- sapply(1:K, .makeLongDataNestedCV, gn = mean(Y), 
                             prediction_list = prediction_list, folds = folds,
                             simplify = FALSE)    
  }
  # full_long_data <- Reduce(rbind, long_data_list)
  # full_long_data$outcome <- with(full_long_data, as.numeric(psi <= u))
  # full_long_data$logit_Fn <- SuperLearner::trimLogit(full_long_data$Fn, .Machine$double.neg.eps)
  
  # targeting
  epsilon_0 <- rep(0, maxIter)
  epsilon_1 <- rep(0, maxIter)
  iter <- 0
  update_long_data_list <- long_data_list
  # combine list into data frame
  full_long_data <- Reduce(rbind, update_long_data_list)
  # compute mean of EIF
  D1 <- .Dy(full_long_data, y = 1)
  D0 <- .Dy(full_long_data, y = 0)
  ic <- ic_os <- D1 + D0 
  PnDstar <- mean(ic)

  # compute initial estimate of cvAUC
  # compute estimated cv-AUC 
  if(!nested_cv){
    dist_psix_y0_star <- lapply(prediction_list, .getPsiDistribution, 
                           y = 0, epsilon = epsilon_0)
    dist_psix_y1_star <- lapply(prediction_list, .getPsiDistribution, y = 1,
                           epsilon = epsilon_1)
  }else{
    dist_psix_y0_star <- sapply(1:K, .getPsiDistributionNestedCV, 
                           y = 0, epsilon = epsilon_0, folds = folds, 
                           prediction_list = prediction_list, simplify = FALSE)
    dist_psix_y1_star <- sapply(1:K, .getPsiDistributionNestedCV, 
                           y = 1, epsilon = epsilon_1, folds = folds, 
                           prediction_list = prediction_list, simplify = FALSE)
  }
  # get AUC
  init_auc <- mean(mapply(FUN = .getAUC, dist_y0 = dist_psix_y0_star, 
                     dist_y1 = dist_psix_y1_star))
  # hard code in for case that psi is constant and doesn't depend on Y
  if(identical(dist_psix_y0_star, dist_psix_y1_star)){
    est_onestep <- init_auc
    tmle_auc <- init_auc
    est_esteq <- init_auc
  }else{
    est_onestep <- init_auc + PnDstar
    se_onestep <- sqrt(var(ic_os)/n)

    # estimating equations
    if(!nested_cv){

    est_esteq <- tryCatch({stats::uniroot(.estimatingFn, interval = c(0, 1), 
                     prediction_list = prediction_list, gn = mean(Y))$root},
                  error = function(e){ return(NA) })
    se_esteq <- se_onestep
    }else{
      est_esteq <- tryCatch({stats::uniroot(.estimatingFnNestedCV, interval = c(0, 1), 
                   prediction_list = prediction_list, gn = mean(Y), folds = folds, K = K)$root},
                    error = function(e){ return(NA) })
      se_esteq <- se_onestep
    }

    tmle_auc <- rep(NA, maxIter)
    PnDstar <- Inf
    while(abs(PnDstar) > icTol & iter < maxIter){
      iter <- iter + 1
      # targeting with different epsilon
      # if(TRUE){
      ##############
      # target F0
      ##############
      # make weight for loss function
      full_long_data$targeting_weight_0 <- 
        as.numeric(full_long_data$Y == 0)/(2*full_long_data$gn) * full_long_data$dFn 
      # fit intercept only model with weights
      suppressWarnings(
        fluc_mod_0 <- glm(outcome ~ offset(logit_Fn), family = binomial(),
                          data = full_long_data[full_long_data$Yi == 0,], 
                          weights = full_long_data$targeting_weight_0[full_long_data$Yi == 0],
                          start = 0)
      )
      epsilon_0[iter] <- as.numeric(fluc_mod_0$coef[1])
      # if unstable glm fit refit using optim
      coef_tol <- 1e2
      if(abs(fluc_mod_0$coef) > coef_tol){
        # try a grid search
        eps_seq <- seq(-coef_tol, coef_tol, length = 1000)
        llik0 <- sapply(eps_seq, fluc_mod_optim_0, fld = full_long_data[full_long_data$Yi == 0,])
        idx_min <- which.min(llik0)
        epsilon_0[iter] <- eps_seq[idx_min]      
        # fluc_mod_0 <- optim(fluc_mod_optim_0, method = "Brent", par = 0, 
        #         fld = full_long_data[full_long_data$Yi == 0,],
        #         lower = -coef_tol, upper = coef_tol, 
        #         control = list(reltol = 1e-14))
        # epsilon_0[iter] <- as.numeric(fluc_mod_0$par)
      }
      # update values in long_data_list
      if(!nested_cv){
        update_long_data_list <- lapply(prediction_list, .makeLongData, gn = mean(Y),
                                 epsilon_0 = epsilon_0, epsilon_1 = epsilon_1,
                                 update = TRUE)
      }else{
        update_long_data_list <- sapply(1:K, .makeLongDataNestedCV, gn = mean(Y),
                                 epsilon_0 = epsilon_0, epsilon_1 = epsilon_1,
                                 prediction_list = prediction_list, folds = folds, 
                                 update = TRUE, simplify = FALSE)
      }
      # update full long data
      full_long_data <- Reduce(rbind, update_long_data_list)
      
      # sanity check
      D0_tmp <- c(.Dy(full_long_data, y = 0))

      # make weight for loss function
      full_long_data$targeting_weight_1 <- 
        as.numeric(full_long_data$Y == 1)/(2*full_long_data$gn) * full_long_data$dFn 
      # fit intercept only model with weights
      suppressWarnings(
        fluc_mod_1 <- glm(outcome ~ offset(logit_Fn), family = binomial(),
                          data = full_long_data[full_long_data$Yi == 1,], 
                          weights = full_long_data$targeting_weight_1[full_long_data$Yi == 1],
                          start = 0)
      )
      # update values in long_data_list
      epsilon_1[iter] <- as.numeric(fluc_mod_1$coef[1])
      if(abs(fluc_mod_1$coef) > coef_tol){
        eps_seq <- seq(-coef_tol, coef_tol, length = 1000)
        llik1 <- sapply(eps_seq, fluc_mod_optim_1, fld = full_long_data[full_long_data$Yi == 1,])
        idx_min <- which.min(llik1)
        epsilon_1[iter] <- eps_seq[idx_min] 
        # fluc_mod_1 <- optim(fluc_mod_optim_1, method = "Brent", par = 0, 
        #         fld = full_long_data[full_long_data$Yi == 1,],
        #         lower = -coef_tol, upper = coef_tol, 
        #         control = list(reltol = 1e-14))
        # epsilon_1[iter] <- as.numeric(fluc_mod_1$par)
      }
      if(!nested_cv){
        update_long_data_list <- lapply(prediction_list, .makeLongData, gn = mean(Y),
                                 epsilon_0 = epsilon_0, epsilon_1 = epsilon_1,
                                 update = TRUE)
      }else{
        update_long_data_list <- sapply(1:K, .makeLongDataNestedCV, gn = mean(Y),
                                 epsilon_0 = epsilon_0, epsilon_1 = epsilon_1,
                                 prediction_list = prediction_list, folds = folds, 
                                 update = TRUE, simplify = FALSE)
      }
      # update full long data
      full_long_data <- Reduce(rbind, update_long_data_list)
      
      # compute mean of EIF
      D1 <- .Dy(full_long_data, y = 1)
      D0 <- .Dy(full_long_data, y = 0)
      ic <- D1 + D0 
      PnDstar <- mean(ic)

      # compute estimated cv-AUC 
      if(!nested_cv){
        dist_psix_y0_star <- lapply(prediction_list, .getPsiDistribution, 
                               y = 0, epsilon = epsilon_0)
        dist_psix_y1_star <- lapply(prediction_list, .getPsiDistribution, y = 1,
                               epsilon = epsilon_1)
      }else{
        dist_psix_y0_star <- sapply(1:K, .getPsiDistributionNestedCV, 
                               y = 0, epsilon = epsilon_0, folds = folds, 
                               prediction_list = prediction_list, simplify = FALSE)
        dist_psix_y1_star <- sapply(1:K, .getPsiDistributionNestedCV, 
                               y = 1, epsilon = epsilon_1, folds = folds, 
                               prediction_list = prediction_list, simplify = FALSE)
      }

      # get AUC
      tmle_auc[iter] <- mean(mapply(FUN = .getAUC, dist_y0 = dist_psix_y0_star, 
                         dist_y1 = dist_psix_y1_star))

    }

      # fluc_mod_0 <- optim(par = 0, fn = .lossF0, long_data_list = long_data_list,
      #                   method = "Brent", lower = -0.05, upper = 0.05,
      #                   control = list(reltol = 1e-9))
      # ic_0 <- .D0(long_data_list, epsilon = fluc_mod_0$par)
      # fluc_mod_1 <- optim(par = 0, fn = .lossF1, long_data_list = long_data_list,
      #                   method = "Brent", lower = -0.05, upper = 0.05,
      #                   control = list(reltol = 1e-9))
      # ic_1 <- .D1(long_data_list, epsilon = fluc_mod_1$par)
    }
    se_cvtmle <- sqrt(var(ic)/n)

    # compute regular cvAUC
    valid_pred_list <- lapply(prediction_list[1:K], "[[", "psi_nBn_testx")
    valid_label_list <- lapply(prediction_list[1:K], "[[", "test_y")
    if(K < n - 1){
      regular_cvauc <- tryCatch({cvma::ci.cvAUC_withIC(predictions = valid_pred_list,
                                  labels = valid_label_list)}, error = function(e){
                                    return(list(cvAUC = NA, se = NA))})
    }else{
      # this is for computing the weird LOO CV estimator
      regular_cvauc <- tryCatch({cvma::ci.cvAUC_withIC(predictions = unlist(valid_pred_list),
                                  labels = unlist(valid_label_list))}, error = function(e){
                                    return(list(cvAUC = NA, se = NA))})
    }
    est_empirical <- regular_cvauc$cvAUC
    se_empirical <- regular_cvauc$se
    ic_emp <- regular_cvauc$ic
    if(init_auc == 0.5){
      se_onestep <- se_cvtmle <- se_esteq <- se_empirical
      iter <- 1
    }
    # # true CV AUC
    # N <- 1e5
    # p <- 100
    # bigX <- matrix(rnorm(N*p), nrow = N, ncol = p)
    # bigY <- rbinom(N, 1, plogis(bigX[,1] + bigX[,10] + bigX[,20]))
    # big_valid_pred_list <- lapply(prediction_list, function(x){
    #   predict(x$model, newx = bigX, type = "response")
    # })
    # big_label_list <- list(bigY,bigY,bigY,bigY,bigY,bigY,bigY,bigY,bigY,bigY,
    #                        bigY,bigY,bigY,bigY,bigY,bigY,bigY,bigY,bigY,bigY)
    # true_cvauc <- mean(cvAUC::AUC(predictions = big_valid_pred_list,
    #                         labels = big_label_list))

    # format output
    out <- list()
    out$est_cvtmle <- tmle_auc[iter]
    out$iter_cvtmle <- iter
    out$cvtmle_trace <- tmle_auc
    out$se_cvtmle <- se_cvtmle
    out$est_init <- init_auc
    out$est_empirical <- est_empirical
    out$se_empirical <- se_empirical
    out$est_onestep <- est_onestep
    out$se_onestep <- se_onestep
    out$est_esteq <- est_esteq
    out$se_esteq <- se_esteq
    out$folds <- folds
    out$ic_cvtmle <- ic
    out$ic_onestep <- ic_os
    out$ic_empirical <- ic_emp

    out$se_cvtmle_type <- "std"
    out$se_esteq_type <- "std"
    out$se_onestep_type <- "std"
    out$se_empirical_type <- "std"

    out$prediction_list <- prediction_list
    class(out) <- "cvauc"
    return(out)
}

#' Alternative fluctuation routine 
#' @param epsilon Fluctuation parameter 
#' @param fld full_long_data_list
fluc_mod_optim_0 <- function(epsilon, fld, tol = 1e-3){
  p_eps <- plogis(fld$logit_Fn + epsilon)
  p_eps[p_eps == 1] <- 1 - tol
  p_eps[p_eps == 0] <- tol
  loglik <- -sum(fld$targeting_weight_0 * (fld$outcome * log(p_eps) + (1-fld$outcome) * log(1 - p_eps)))
  return(loglik)
}
#' Alternative fluctuation routine 
#' @param epsilon Fluctuation parameter 
#' @param fld full_long_data_list
fluc_mod_optim_1 <- function(epsilon, fld, tol = 1e-3){
  p_eps <- plogis(fld$logit_Fn + epsilon)
  p_eps[p_eps == 1] <- 1 - tol
  p_eps[p_eps == 0] <- tol
  loglik <- -sum(fld$targeting_weight_1 * (fld$outcome * log(p_eps) + (1-fld$outcome) * log(1 - p_eps)))
  return(loglik)
}

#' An estimating function for cvAUC
#' @param auc The value of auc to find root for
#' @param prediction_list Entry in prediction_list
#' @param gn Marginal probability of outcome
.estimatingFn <- function(auc = 0.5, prediction_list, gn){
  # get first influence function piece for everyone
  ic_1 <- 
  Reduce("c",lapply(prediction_list, function(x){
    thisFn <- sapply(1:length(x$test_y), function(i){
      ifelse(x$test_y[i] == 1, 
             F_nBn_star(x$psi_nBn_testx[i], y = 0, Psi_nBn_0 = x$psi_nBn_trainx,
                        Y_Bn = x$train_y)/ gn , 
             (1 - F_nBn_star(x$psi_nBn_testx[i], y = 1, Psi_nBn_0 = x$psi_nBn_trainx,
                        Y_Bn = x$train_y))/(1 - gn))
    })
  }))
  all_y <- unlist(lapply(prediction_list, "[[", "test_y"))
  ic_2 <- rep(0, length(all_y))
  ic_2[all_y == 0] <- - auc / (1 - gn)
  ic_2[all_y == 1] <- - auc / gn
  return(mean(ic_1 + ic_2))
}

#' An estimating function for cvAUC with cv initial estimates
#' @param auc The value of auc to find root for
#' @param prediction_list Entry in prediction_list
#' @param gn Marginal probability of outcome
.estimatingFnNestedCV <- function(auc = 0.5, prediction_list, folds, gn, K){
  # get first influence function piece for everyone
  ic_1 <- 
  Reduce("c",sapply(1:K, function(x){
    valid_folds_idx <- which(unlist(lapply(prediction_list, function(z){ 
    x %in% z$valid_folds }), use.names = FALSE))

    # get only inner validation predictions
    inner_valid_prediction_and_y_list <- lapply(prediction_list[valid_folds_idx[-1]], 
                                        function(z){
      # pick out the fold that is not the outer validation fold
      inner_valid_idx <- which(!(z$valid_ids %in% folds[[x]]))
      # get predictions for this fold
      inner_pred <- z$psi_nBn_testx[inner_valid_idx]
      inner_y <- z$test_y[inner_valid_idx]
      return(list(inner_psi_nBn_testx = inner_pred, inner_test_y = inner_y))
    })
    thisFn <- sapply(1:length(prediction_list[[valid_folds_idx[1]]]$test_y), function(i){
      ifelse(prediction_list[[valid_folds_idx[1]]]$test_y[i] == 1, 
       F_nBn_star_nested_cv(prediction_list[[valid_folds_idx[1]]]$psi_nBn_testx[i], y = 0,
          inner_valid_prediction_and_y_list = inner_valid_prediction_and_y_list)/ gn , 
       (1 - F_nBn_star_nested_cv(prediction_list[[valid_folds_idx[1]]]$psi_nBn_testx[i], y = 1,
          inner_valid_prediction_and_y_list = inner_valid_prediction_and_y_list))/(1 - gn))
    })
  }))
  all_y <- unlist(lapply(prediction_list[1:K], "[[", "test_y"))
  ic_2 <- rep(0, length(all_y))
  ic_2[all_y == 0] <- - auc / (1 - gn)
  ic_2[all_y == 1] <- - auc / gn
  return(mean(ic_1 + ic_2))
}

# uniroot(.estimatingFn, interval = c(0, 1), prediction_list = prediction_list, gn = mean(Y))

#' Compute a portion of the efficient influence function
#' @param full_long_data A long form data set
#' @param y Which portion of the EIF to compute
#' @return Vector of EIF
.Dy <- function(full_long_data, y){
  by(full_long_data, full_long_data$id, function(x){
    sum((-1)^y * as.numeric(x$Y == y)/(x$gn) * (x$outcome - x$Fn) * x$dFn)
  })
}

#' Compute the (CV)TMLE cumulative dist at psi_x
#' @param psi_x Value to compute conditional (on Y=y) cdf of Psi
#' @param y Value of Y to condition on 
#' @param Psi_nBn_0 Values of Psi_nBn(X) from training sample
#' @param Y_Bn Values of Y from training sample
#' @param epsilon Vector of fluctuation parameter estimates
#' @return Numeric value of CDF
F_nBn_star <- function(psi_x, y, Psi_nBn_0, Y_Bn, epsilon = 0, 
                       # tol = .Machine$double.neg.eps
                       tol = 1e-3
                       ){
  plogis(SuperLearner::trimLogit(mean(Psi_nBn_0[Y_Bn %in% y] <= psi_x), tol) +
          sum(epsilon))
}

F_nBn_star_nested_cv <- function(psi_x, y, epsilon = 0, 
                                 inner_valid_prediction_and_y_list,
                       # tol = .Machine$double.neg.eps
                       tol = 1e-3
                       ){
  # get cdf estimated in each validation fold
  all_cv_est <- lapply(inner_valid_prediction_and_y_list, function(z){
    plogis(SuperLearner::trimLogit(mean(z$inner_psi_nBn_testx[z$inner_test_y %in% y] <= psi_x), tol) +
          sum(epsilon))
  })
  # average over folds
  return(mean(unlist(all_cv_est, use.names = FALSE), na.rm = TRUE))
}

#' Worker function to make long form data set needed for
#' CVTMLE targeting step 
#' 
#' @param x An entry in the "predictions list" that has certain
#' named values (see \code{?.getPredictions})
#' @param gn An estimate of the marginal dist. of Y
#' @param update Boolean of whether this is called for initial
#' construction of the long data set or as part of the targeting loop. 
#' If the former, empirical "density" estimates are used. If the latter
#' these are derived from the targeted cdf. 
#' @param epsilon_0 If \code{update = TRUE}, a vector of TMLE fluctuation
#' parameter estimates used to add the CDF and PDF of Psi(X) to the data set
#' @param epsilon_1 Ditto above
#' 
#' @return A long form data list of a particular set up. Columns are named id 
#' (multiple per obs. in validation sample), u (if Yi = 0, these are the values of psi(x) in the
#' training sample for obs with Y = 1, if Yi = 1, these are values of psi(x) in
#' the training sample for obs. with Y = 0), Yi (this id's value of Y), Fn (
#' estimated value of the cdf of psi(X) given Y = Yi in the training sample), 
#' dFn (estimated value of the density of psi(X) given Y = (1-Yi) in the 
#' training sample), psi (the value of this observations \hat{\Psi}(P_{n,B_n}^0)),
#' gn (estimate of marginal of Y e.g., computed in whole sample), outcome (indicator
#' that psix <= u), logit_Fn (the cdf estimate on the logit scale, needed for 
#' offset in targeting model).

.makeLongData <- function(x, gn, update = FALSE, epsilon_0 = 0, epsilon_1 = 0,
                          # tol = .Machine$double.neg.eps, 
                          tol = 1e-3
                          ){
  # first the dumb way, writing a loop over x$psi_nBn_testx
  uniq_train_psi_y0 <- sort(unique(x$psi_nBn_trainx[x$train_y == 0]))
  uniq_train_psi_y1 <- sort(unique(x$psi_nBn_trainx[x$train_y == 1]))
  # ord_train_psi_y0 <- order(x$psi_nBn_trainx[x$train_y == 0])
  # ord_train_psi_y1 <- order(x$psi_nBn_trainx[x$train_y == 1])
  
  n_valid <- length(x$psi_nBn_testx)
  n_train <- length(x$psi_nBn_trainx)
  n1_train <- sum(x$train_y)
  n1_uniq_train <- length(uniq_train_psi_y1)
  n0_train <- n_train - n1_train
  n0_uniq_train <- length(uniq_train_psi_y0)
  n1_valid <- sum(x$test_y)
  n0_valid <- n_valid - n1_valid
  valid_ids <- as.numeric(names(x$psi_nBn_testx))
  tot_length <- n1_valid * n0_uniq_train + n0_valid * n1_uniq_train

  idVec <- rep(NA, tot_length)
  uVec <- rep(NA, tot_length)
  YuVec <- rep(NA, tot_length)
  YiVec <- rep(NA, tot_length)
  F_1n_Bn_uVec <- rep(NA, tot_length)
  F_0n_Bn_uVec <- rep(NA, tot_length)
  dF_1n_Bn_uVec <- rep(NA, tot_length)
  dF_0n_Bn_uVec <- rep(NA, tot_length)
  FnVec <- rep(NA, tot_length)
  dFnVec <- rep(NA, tot_length)
  psiVec <- rep(NA, tot_length)

  # cumulative dist of psi_n | Y = 1 evaluated at all values of 
  # psi_n associated with Y = 0 observations
  F1nBn <- sapply(uniq_train_psi_y0, 
                  F_nBn_star, Psi_nBn_0 = x$psi_nBn_trainx, y = 1, 
                  Y_Bn = x$train_y, epsilon = epsilon_1)
  # cumulative dist of psi_n | Y = 0 evaluated at all values of 
  # psi_n associated with Y = 1 observations
  F0nBn <- sapply(uniq_train_psi_y1, 
                  F_nBn_star, Psi_nBn_0 = x$psi_nBn_trainx, y = 0, 
                  Y_Bn = x$train_y, epsilon = epsilon_0)

  # empirical dens of psi_n | Y = 1 and Y = 0 evaluated at all 
  if(!update){
    dF1nBn <- as.numeric(table(x$psi_nBn_trainx[x$train_y == 1])/n1_train)
    dF0nBn <- as.numeric(table(x$psi_nBn_trainx[x$train_y == 0])/n0_train)
  }else{
    # cumulative dist of psi_n | Y = 1 evaluated at all values of 
    # psi_n associated with Y = 0 observations
    F1nBn_1 <- sapply(uniq_train_psi_y1, 
                    F_nBn_star, Psi_nBn_0 = x$psi_nBn_trainx, y = 1, 
                    Y_Bn = x$train_y, epsilon = epsilon_1)
    dF1nBn <- diff(c(0, F1nBn_1))
    # cumulative dist of psi_n | Y = 0 evaluated at all values of 
    # psi_n associated with Y = 1 observations
    F0nBn_0 <- sapply(uniq_train_psi_y0, 
                    F_nBn_star, Psi_nBn_0 = x$psi_nBn_trainx, y = 0, 
                    Y_Bn = x$train_y, epsilon = epsilon_0)
    dF0nBn <- diff(c(0, F0nBn_0))
  }

  # loop over folks in validation fold
  cur_start <- 1
  for(i in seq_len(n_valid)){
    if(x$test_y[i] == 0){
      cur_end <- cur_start + n1_uniq_train - 1
      idVec[cur_start:cur_end] <- x$valid_ids[i]
      # ordered unique values of psi in training | y = 1
      uVec[cur_start:cur_end] <- uniq_train_psi_y1
      # value of this Y_i
      YiVec[cur_start:cur_end] <- 0
      # cdf of psi | y = 0 in training at each u
      FnVec[cur_start:cur_end] <- F0nBn
      # pdf of psi | y = 1 in training at each u
      dFnVec[cur_start:cur_end] <- dF1nBn
      # vector of this psi
      psiVec[cur_start:cur_end] <- x$psi_nBn_testx[i]
    }else{
      cur_end <- cur_start + n0_uniq_train - 1
      idVec[cur_start:cur_end] <- x$valid_ids[i]
      # ordered unique values of psi in training | y = 0
      uVec[cur_start:cur_end] <- uniq_train_psi_y0
      # value of this Y_i
      YiVec[cur_start:cur_end] <- 1
      # cdf of psi | y = 1 in training at each u
      FnVec[cur_start:cur_end] <- F1nBn
      # pdf of psi | y = 0 in training at each u
      dFnVec[cur_start:cur_end] <- dF0nBn
      # vector of this psi
      psiVec[cur_start:cur_end] <- x$psi_nBn_testx[i]
    }
    cur_start <- cur_end + 1
  }

  out <- data.frame(id = idVec, u = uVec,
                   Yi = YiVec, Fn = FnVec, dFn = dFnVec,
                   psi = psiVec)

  # add in gn
  out$gn <- NA
  out$gn[out$Yi == 1] <- gn
  out$gn[out$Yi == 0] <- 1 - gn
  # add in "outcome"
  out$outcome <- as.numeric(out$psi <= out$u)
  # add in logit(Fn)
  out$logit_Fn <- SuperLearner::trimLogit(out$Fn, tol)
  return(out)
}


#' Worker function to make long form data set needed for
#' CVTMLE targeting step when nested cv is used
#' 
#' @param x The outer validation fold
#' @param prediction_list The full prediction list
#' @param gn An estimate of the marginal dist. of Y
#' @param update Boolean of whether this is called for initial
#' construction of the long data set or as part of the targeting loop. 
#' If the former, cross-validated empirical "density" estimates are used. 
#' If the latter these are derived from the targeted cdf. 
#' @param epsilon_0 If \code{update = TRUE}, a vector of TMLE fluctuation
#' parameter estimates used to add the CDF and PDF of Psi(X) to the data set
#' @param epsilon_1 Ditto above
#' 
#' @return A long form data list of a particular set up. Columns are named id 
#' (multiple per obs. in validation sample), u (if Yi = 0, these are the unique 
#' values of psi(x) in the inner validation samples for psi fit on inner training
#' samples for obs with Y = 1, if Yi = 1, these are values of psi(x) in
#' the inner validation samples for psi fit on inner training samples for obs. 
#' with Y = 0), Yi (this id's value of Y), Fn (cross-validation estimated value 
#' of the cdf of psi(X) given Y = Yi in the training sample), 
#' dFn (cross-validated estimate of the density of psi(X) given Y = (1-Yi) in the 
#' training sample), psi (the value of this observations \hat{\Psi}(P_{n,B_n}^0)),
#' gn (estimate of marginal of Y e.g., computed in whole sample), outcome (indicator
#' that psix <= u), logit_Fn (the cdf estimate on the logit scale, needed for 
#' offset in targeting model).
.makeLongDataNestedCV <- function(x, prediction_list, folds, gn, update = FALSE, epsilon_0 = 0, epsilon_1 = 0,
                          # tol = .Machine$double.neg.eps, 
                          tol = 1e-3
                          ){
  # find all V-1 fold CV fits with this x in them. These will be the inner
  # CV fits that are needed. The first entry in this vector will correspond
  # to the outer V fold CV fit, which is what we want to make the outcome 
  # of the long data list with. 
  valid_folds_idx <- which(unlist(lapply(prediction_list, function(z){ 
    x %in% z$valid_folds }), use.names = FALSE))

  # get only inner validation predictions
  inner_valid_prediction_and_y_list <- lapply(prediction_list[valid_folds_idx[-1]], 
                                        function(z){
    # pick out the fold that is not the outer validation fold
    inner_valid_idx <- which(!(z$valid_ids %in% folds[[x]]))
    # get predictions for this fold
    inner_pred <- z$psi_nBn_testx[inner_valid_idx]
    inner_y <- z$test_y[inner_valid_idx]
    return(list(inner_psi_nBn_testx = inner_pred, inner_test_y = inner_y))
  })

  # now get all values of psi from inner validation with Y = 0 
  uniq_train_psi_y0 <- sort(unique(unlist(lapply(inner_valid_prediction_and_y_list, function(z){
    z$inner_psi_nBn_testx[z$inner_test_y == 0]    
  }), use.names = FALSE)))
  uniq_train_psi_y1 <- sort(unique(unlist(lapply(inner_valid_prediction_and_y_list, function(z){
    z$inner_psi_nBn_testx[z$inner_test_y == 1]    
  }), use.names = FALSE)))
  
  # number in outer validation sample 
  # NOTE: valid_folds_idx[1] is the fit on V-1 folds 
  n_valid <- length(prediction_list[[valid_folds_idx[1]]]$psi_nBn_testx)
  # number in outer training sample... is this what I want?
  n_train <- length(prediction_list[[valid_folds_idx[1]]]$psi_nBn_trainx)
  n1_train <- sum(prediction_list[[valid_folds_idx[1]]]$train_y)
  n1_uniq_train <- length(uniq_train_psi_y1)
  n0_train <- n_train - n1_train
  n0_uniq_train <- length(uniq_train_psi_y0)
  n1_valid <- sum(prediction_list[[valid_folds_idx[1]]]$test_y)
  n0_valid <- n_valid - n1_valid
  tot_length <- n1_valid * n0_uniq_train + n0_valid * n1_uniq_train

  idVec <- rep(NA, tot_length)
  uVec <- rep(NA, tot_length)
  YuVec <- rep(NA, tot_length)
  YiVec <- rep(NA, tot_length)
  F_1n_Bn_uVec <- rep(NA, tot_length)
  F_0n_Bn_uVec <- rep(NA, tot_length)
  dF_1n_Bn_uVec <- rep(NA, tot_length)
  dF_0n_Bn_uVec <- rep(NA, tot_length)
  FnVec <- rep(NA, tot_length)
  dFnVec <- rep(NA, tot_length)
  psiVec <- rep(NA, tot_length)

  # now need CV-averaged CDF
  # cumulative dist of psi_n | Y = 1 evaluated at all values of 
  # psi_n associated with Y = 0 observations
  F1nBn <- sapply(uniq_train_psi_y0, 
                  F_nBn_star_nested_cv, y = 1, 
                  epsilon = epsilon_1,
                  inner_valid_prediction_and_y_list = inner_valid_prediction_and_y_list)

  # cumulative dist of psi_n | Y = 0 evaluated at all values of 
  # psi_n associated with Y = 1 observations
  F0nBn <- sapply(uniq_train_psi_y1, 
                  F_nBn_star_nested_cv, y = 0, 
                  epsilon = epsilon_0,
                  inner_valid_prediction_and_y_list = inner_valid_prediction_and_y_list)

  # cv empirical density of psi_n | Y = 1 and Y = 0 evaluated at all 
  # cumulative dist of psi_n | Y = 1 evaluated at all values of 
  # psi_n associated with Y = 0 observations
  F1nBn_1 <- sapply(uniq_train_psi_y1, 
                  F_nBn_star_nested_cv, y = 1, 
                  epsilon = epsilon_1,
                  inner_valid_prediction_and_y_list = inner_valid_prediction_and_y_list)
  dF1nBn <- diff(c(0, F1nBn_1))
  # cumulative dist of psi_n | Y = 0 evaluated at all values of 
  # psi_n associated with Y = 1 observations
  F0nBn_0 <- sapply(uniq_train_psi_y0, 
                  F_nBn_star_nested_cv, y = 0, 
                  epsilon = epsilon_0,
                  inner_valid_prediction_and_y_list = inner_valid_prediction_and_y_list)
  dF0nBn <- diff(c(0, F0nBn_0))

  # loop over folks in validation fold
  cur_start <- 1
  for(i in seq_len(n_valid)){
    if(prediction_list[[valid_folds_idx[1]]]$test_y[i] == 0){
      cur_end <- cur_start + n1_uniq_train - 1
      idVec[cur_start:cur_end] <- prediction_list[[valid_folds_idx[1]]]$valid_ids[i]
      # ordered unique values of psi in training | y = 1
      uVec[cur_start:cur_end] <- uniq_train_psi_y1
      # value of this Y_i
      YiVec[cur_start:cur_end] <- 0
      # cdf of psi | y = 0 in training at each u
      FnVec[cur_start:cur_end] <- F0nBn
      # pdf of psi | y = 1 in training at each u
      dFnVec[cur_start:cur_end] <- dF1nBn
      # vector of this psi
      psiVec[cur_start:cur_end] <- prediction_list[[valid_folds_idx[1]]]$psi_nBn_testx[i]
    }else{
      cur_end <- cur_start + n0_uniq_train - 1
      idVec[cur_start:cur_end] <- prediction_list[[valid_folds_idx[1]]]$valid_ids[i]
      # ordered unique values of psi in training | y = 0
      uVec[cur_start:cur_end] <- uniq_train_psi_y0
      # value of this Y_i
      YiVec[cur_start:cur_end] <- 1
      # cdf of psi | y = 1 in training at each u
      FnVec[cur_start:cur_end] <- F1nBn
      # pdf of psi | y = 0 in training at each u
      dFnVec[cur_start:cur_end] <- dF0nBn
      # vector of this psi
      psiVec[cur_start:cur_end] <- prediction_list[[valid_folds_idx[1]]]$psi_nBn_testx[i]
    }
    cur_start <- cur_end + 1
  }
  out <- data.frame(id = idVec, u = uVec,
                   Yi = YiVec, Fn = FnVec, dFn = dFnVec,
                   psi = psiVec)

  # add in gn
  out$gn <- NA
  out$gn[out$Yi == 1] <- gn
  out$gn[out$Yi == 0] <- 1 - gn
  # add in "outcome"
  out$outcome <- as.numeric(out$psi <= out$u)
  # add in logit(Fn)
  out$logit_Fn <- SuperLearner::trimLogit(out$Fn, tol)
  return(out)
}

#' Compute the AUC given the cdf and pdf of psi 
#' 
#' See \code{?.getPsiDistribution} to understand expected input format
#' 
#' @param dist_y0 Distribution of psi given Y = 0
#' @param dist_y1 Distribution of psi given Y = 1
#' @return Numeric 
# TO DO: make more efficient
# TO DO: how are ties handled in findInterval?
.getAUC <- function(dist_y0, dist_y1){
  if(identical(dist_y0, dist_y1)){
    tot <- 0.5
  } else {
    tot <- 0
    for(i in seq_along(dist_y0$psix)){
      idx <- findInterval(x = dist_y0$psix[i], vec = dist_y1$psix)
      p1 <- ifelse(idx == 0, 1, (1 - dist_y1$Fn[idx]))
      p2 <- dist_y0$dFn[i]
      tot <- tot + p1 * p2
    }
  }
  return(tot)
}

#' Compute the conditional (given Y = y) estimated distribution of psi
#' 
#' @param x An entry in the output from .getPredictions
#' @param y What value of Y to compute dist. est. 
#' @param epsilon A vector of estimated coefficients form tmle fluctuation 
#' submodels. 
#' 
#' @return A data.frame with the distribution of psi given Y = y with names
#' psix (what value estimates are evaluated at), dFn (density estimates),
#' Fn (cdf estimates)
.getPsiDistribution <- function(x, y, epsilon = 0){
    this_n <- length(x$psi_nBn_trainx[x$train_y == y])
    uniq_train_psi_y <- sort(unique(x$psi_nBn_trainx[x$train_y == y]))
    FynBn_y <- sapply(uniq_train_psi_y, 
                F_nBn_star, Psi_nBn_0 = x$psi_nBn_trainx, y = y, 
                Y_Bn = x$train_y, epsilon = epsilon)
    dFynBn <- diff(c(0, FynBn_y))
    out <- data.frame(psix = uniq_train_psi_y, 
                      dFn = dFynBn,
                      Fn = FynBn_y)
    return(out)
}


#' Compute the conditional (given Y = y) CV-estimated distribution of psi
#' 
#' @param x The outer validation fold withheld
#' @param y What value of Y to compute dist. est. 
#' @param prediction_list List output from .getPredictions
#' @param epsilon A vector of estimated coefficients form tmle fluctuation 
#' submodels. 
#' 
#' @return A data.frame with the distribution of psi given Y = y with names
#' psix (what value estimates are evaluated at), dFn (density estimates),
#' Fn (cdf estimates)
.getPsiDistributionNestedCV <- function(x, y, prediction_list, folds, epsilon = 0){
  # find all V-1 fold CV fits with this x in them. These will be the inner
  # CV fits that are needed. The first entry in this vector will correspond
  # to the outer V fold CV fit, which is what we want to make the outcome 
  # of the long data list with. 
  valid_folds_idx <- which(unlist(lapply(prediction_list, function(z){ 
    x %in% z$valid_folds }), use.names = FALSE))

  # get only inner validation predictions
  inner_valid_prediction_and_y_list <- lapply(prediction_list[valid_folds_idx[-1]], 
                                        function(z){
    # pick out the fold that is not the outer validation fold
    inner_valid_idx <- which(!(z$valid_ids %in% folds[[x]]))
    # get predictions for this fold
    inner_pred <- z$psi_nBn_testx[inner_valid_idx]
    inner_y <- z$test_y[inner_valid_idx]
    return(list(inner_psi_nBn_testx = inner_pred, inner_test_y = inner_y))
  })

  # now get all values of psi from inner validation with Y = 0 
  uniq_train_psi_y <- sort(unique(unlist(lapply(inner_valid_prediction_and_y_list, function(z){
    z$inner_psi_nBn_testx[z$inner_test_y == y]    
  }), use.names = FALSE)))
  
  FynBn_y <- sapply(uniq_train_psi_y, 
                  F_nBn_star_nested_cv, y = y, 
                  epsilon = epsilon,
                  inner_valid_prediction_and_y_list = inner_valid_prediction_and_y_list)
    dFynBn <- diff(c(0, FynBn_y))
    out <- data.frame(psix = uniq_train_psi_y, 
                      dFn = dFynBn,
                      Fn = FynBn_y)
    return(out)
}

#' Worker function for fitting prediction functions (possibly in parallel)
#' 
#' @param learner The wrapper to use
#' @param Y The outcome
#' @param X The predictors
#' @param K The number of folds
#' @param parallel Whether to compute things in parallel using future
#' 
#' @return A list of the result of the wrapper executed in each fold
.getPredictions <- function(learner, Y, X, K = 10, folds, parallel, nested_cv = FALSE,
                            nested_K = K - 1){
  .doFit <- function(x, tmpX, Y, folds, learner, seed = 21){
    set.seed(seed)
    out <- do.call(learner, args=list(train = list(Y = Y[-unlist(folds[x])], X = tmpX[-unlist(folds[x]),,drop=FALSE]),
                                      test = list(Y = Y[unlist(folds[x])], X = tmpX[unlist(folds[x]),,drop=FALSE])))
    out$valid_ids <- unlist(folds[x], use.names = FALSE)
    out$valid_folds <- x
    return(out)
  }

  
  if(!nested_cv){
    valid_folds <- split(seq(K),factor(seq(K)))
  }else{
    if(nested_K == K - 1){
      combns <- combn(K, 2)
      valid_folds <- c(split(seq(K), factor(seq(K))),
                       split(combns, col(combns)))
    }
  }

  if(parallel){
    stop("Parallel processing code needs to be re-written.")
  }else{
    if(nested_K == K - 1){
      predFitList <- lapply(valid_folds ,FUN = .doFit, tmpX = X, Y = Y, folds = folds, learner = learner)
    }else if(nested_cv & nested_K != K - 1){
      inner_folds <- vector(mode = "list", length = K)
      for(k in seq_len(K)){
        train_idx <- unlist(folds[-k], use.names = FALSE)
        # these will just be numbers 1:length(train_idx)
        inner_folds[[k]] <- SuperLearner::CVFolds(N = length(train_idx), 
                                                        id = NULL, Y = Y[train_idx], 
                                   cvControl = list(V = nested_K, 
                                                    stratifyCV = ifelse(nested_K <= sum(Y[train_idx]) 
                                                                        & nested_K <= sum(!Y[train_idx]), 
                                                                        TRUE, FALSE), 
                                   shuffle = TRUE, validRows = NULL))
        # so replace them with actual ids
        inner_folds[[k]] <- lapply(inner_folds[[k]], function(x){
          train_idx[x]
        })
      }
      fold_combos <- expand.grid(outer_K = seq_len(K),
                                 inner_K = c(0,seq_len(nested_K)))
      # here x will be a data.frame with columns outer_K and inner_K
      .doFit2 <- function(x, tmpX, Y, folds, inner_folds, learner, seed = 21){
        if(x[2] == 0){
          set.seed(21)
          # in this case, just remove from folds
          out <- do.call(learner, args=list(train = list(Y = Y[-unlist(folds[x[1]])], 
                                                         X = tmpX[-unlist(folds[x[1]]),,drop=FALSE]),
                                      test = list(Y = Y[unlist(folds[x[1]])], 
                                                  X = tmpX[unlist(folds[x[1]]),,drop=FALSE])))
          out$valid_ids <- unlist(folds[x[1]], use.names = FALSE)
          out$valid_folds <- x[1]
        }else{
          # browser()
          # in this case, remove from folds and inner folds
          outer_valid_idx <- unlist(folds[x[1]], use.names = FALSE)
          inner_valid_idx <- unlist(inner_folds[[x[1]]][x[2]], use.names = FALSE)
          remove_idx <- c(outer_valid_idx, inner_valid_idx)
          out <- do.call(learner, args=list(train = list(Y = Y[-remove_idx], 
                                                         X = tmpX[-remove_idx,,drop=FALSE]),
                                      test = list(Y = Y[inner_valid_idx], 
                                                  X = tmpX[inner_valid_idx, , drop = FALSE])))
          out$valid_ids <- inner_valid_idx
          # leave this corresponding to outer validation fold?
          out$valid_folds <- x[1]
        }
        return(out)
      }
      predFitList <- apply(fold_combos, 1, FUN = .doFit2, 
                           tmpX = X, Y = Y, folds = folds, 
                           inner_folds = inner_folds, 
                           learner = learner)      
    }
  }
  
  # return results
  return(predFitList)
}


#' Function to do leave pair out AUC computation
#' @param Y The outcome
#' @param X The predictors
#' @param K The number of folds
#' @param learner The learner wrapper
#' @param seed A random seed to set
#' @param parallel Compute the predictors in parallel?
#' @export

leave_pair_out_auc <- function(Y, X, learner = "glm_wrapper", 
                         seed = 1234,
                         nested_cv = FALSE,
                         parallel = FALSE, ...){
  case_idx <- which(Y == 1)
  control_idx <- which(Y == 0)
  grid_idx <- expand.grid(case = case_idx, control = control_idx)
  folds <- split(grid_idx, seq_len(nrow(grid_idx)))

  prediction_list <- .getPredictions(learner = learner, Y = Y, X = X, 
                               K = length(folds), folds=folds, parallel = FALSE,
                               nested_cv = FALSE)

  zero_one_vec <- lapply(prediction_list, function(x){
    as.numeric(x$psi_nBn_testx[1] > x$psi_nBn_testx[2])
  })

  auc <- mean(unlist(zero_one_vec))

  return(list(auc = auc))
}

#' @export
boot_corrected_auc <- function(Y, X, B = 500, learner = "glm_wrapper", 
                         seed = 1234,
                         nested_cv = FALSE,
                         parallel = FALSE, ...){
  one_boot <- function(Y, X, n){
    idx <- sample(seq_len(n), replace = TRUE)
    train_Y <- Y[idx]
    train_X <- X[idx, , drop = FALSE]
    fit <- do.call(learner, args=list(train = list(Y = train_Y, X = train_X),
                                      test = list(Y = Y, X = X)))
    train_cvauc <- tryCatch({cvAUC::cvAUC(predictions = fit$psi_nBn_trainx,
                            labels = train_Y)$cvAUC}, error = function(e){
                              return(NA)})
    test_cvauc <- tryCatch({cvAUC::cvAUC(predictions = fit$psi_nBn_testx,
                            labels = Y)$cvAUC}, error = function(e){
                              return(NA)})
    optimism <- train_cvauc - test_cvauc
    return(optimism)
  }
  n <- length(Y)
  all_boot <- replicate(B, one_boot(Y = Y, X = X, n = n))
  mean_optimism <- mean(all_boot)

  full_fit <- do.call(learner, args=list(train = list(Y = Y, X = X),
                                      test = list(Y = Y, X = X)))
  naive_auc <- cvAUC::cvAUC(predictions = full_fit$psi_nBn_testx,
                            labels = Y)$cvAUC
  corrected_auc <- naive_auc - mean_optimism
  return(list(auc = corrected_auc))
}
