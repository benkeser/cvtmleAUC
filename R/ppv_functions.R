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
#' TO DO: Seems like this needs to return the quantiles in order to be useful
#' in a practical sense. 
#' @examples
#' n <- 200
#' p <- 10
#' X <- matrix(rnorm(n*p), nrow = n, ncol = p)
#' Y <- rbinom(n, 1, plogis(X[,1] + X[,10]))
#' fit <- cvtn_cvtmle(Y = Y, X = X, K = 5, learner = "glm_wrapper")
#' 
cvtn_cvtmle <- function(Y, X, K = 20, sens = 0.95, learner = "glm_wrapper", 
                        nested_cv = TRUE, nested_K = K - 1, 
                         parallel = FALSE, maxIter = 10, 
                         icTol = 1/length(Y), 
                         quantile_type = 8,
                         prediction_list = NULL, 
                         ...){
  n <- length(Y)
  folds <- SuperLearner::CVFolds(N = n, id = NULL, Y = Y, 
                                 cvControl = list(V = K, 
                                    stratifyCV = ifelse(K <= sum(Y) & K <= sum(!Y), TRUE, FALSE), 
                                    shuffle = TRUE, validRows = NULL))

  if(is.null(prediction_list)){
    prediction_list <- .getPredictions(learner = learner, Y = Y, X = X, nested_cv = nested_cv,
                                   K = K, folds = folds, nested_K = nested_K, 
                                   parallel = FALSE)
  }

  # get quantile estimates
  if(!nested_cv){
    quantile_list <- lapply(prediction_list[seq_len(K)], .getQuantile, p = 1 - sens,
                            quantile_type = quantile_type)
  }else{
    quantile_list <- sapply(1:K, .getNestedCVQuantile, quantile_type = quantile_type,
                             prediction_list = prediction_list, folds = folds,
                             p = 1 - sens, simplify = FALSE) 
  }

  # get density estimate 
  if(!nested_cv){
    density_list <- mapply(x = prediction_list[1:K], c0 = quantile_list, 
                           FUN = .getDensity, SIMPLIFY = FALSE)
  }else{
    density_list <- mapply(x = split(seq_len(K), seq_len(K)), c0 = quantile_list, 
                           FUN = .getDensity, SIMPLIFY = FALSE,
                           MoreArgs = list(prediction_list = prediction_list,
                                           folds = folds, nested_cv = nested_cv))
  }

  # make targeting data
  if(!nested_cv){
    target_and_pred_data <- .makeTargetingData(prediction_list = prediction_list, 
                                      quantile_list = quantile_list, 
                                      density_list = density_list, 
                                      folds = folds, gn = mean(Y))
    target_data <- target_and_pred_data$out
    pred_data <- target_and_pred_data$out_pred
  }else{
    target_and_pred_data <- sapply(seq_len(K), .makeTargetingData, 
                                        prediction_list = prediction_list, 
                                      quantile_list = quantile_list, 
                                      density_list = density_list, folds = folds,
                                      gn = mean(Y), nested_cv = TRUE, simplify = FALSE)
    target_data <- Reduce("rbind", lapply(target_and_pred_data, "[[", "out"))
    pred_data <- Reduce("rbind", lapply(target_and_pred_data, "[[", "out_pred"))
  }
  target_data$weight <- with(target_data, 1 + Y/gn * f_ratio)
  target_data$logit_Fn <- SuperLearner::trimLogit(target_data$Fn, trim = 1e-5)
  pred_data$logit_Fn <- SuperLearner::trimLogit(pred_data$Fn, trim = 1e-5)
  
  fluc_mod <- glm(ind ~ offset(logit_Fn), data = target_data, family = "binomial",
                  weights = weight, start = 0)
  target_data$Fnstar <- fluc_mod$fitted.values
  pred_data$Fnstar <- predict(fluc_mod, newdata = pred_data, type = "response")
  # compute initial non-targeted estimates
  init_estimates <- by(pred_data, pred_data$fold, function(x){
    x$Fn[x$Y==0][1] * (1-x$gn[1]) + x$Fn[x$Y==1][1] * x$gn[1]
  })

  # compute parameter for each fold
  cvtmle_estimates <- by(pred_data, pred_data$fold, function(x){
    x$Fnstar[x$Y==0][1] * (1-x$gn[1]) + x$Fnstar[x$Y==1][1] * x$gn[1]
  })

  target_data$F0nstar <- NaN
  target_data$F1nstar <- NaN
  for(k in seq_len(K)){
    target_data$F0nstar[target_data$fold == k] <- pred_data$Fnstar[pred_data$Y == 0 & pred_data$fold == k][1]
    target_data$F0n[target_data$fold == k] <- pred_data$Fn[pred_data$Y == 0 & pred_data$fold == k][1]
    target_data$F1nstar[target_data$fold == k] <- pred_data$Fnstar[pred_data$Y == 1 & pred_data$fold == k][1]
    target_data$F1n[target_data$fold == k] <- pred_data$Fn[pred_data$Y == 1 & pred_data$fold == k][1]
  }

  target_data$DY_cvtmle <- with(target_data, Fnstar - (gn*F1nstar + (1 - gn)*F0nstar))
  target_data$Dpsi_cvtmle <- with(target_data, weight * (ind - Fnstar))
  
  target_data$DY_os <- with(target_data, Fn - (gn*F1n + (1 - gn)*F0n))
  target_data$Dpsi_os <- with(target_data, weight * (ind - Fn))

  # cvtmle estimates
  est_cvtmle <- mean(cvtmle_estimates)
  se_cvtmle <- sqrt(var(target_data$DY_cvtmle + target_data$Dpsi_cvtmle) / n)
  # initial estimate
  est_init <- mean(unlist(init_estimates))
  est_onestep <- est_init + mean(target_data$DY_os + target_data$Dpsi_os)
  se_onestep <- sqrt(var(target_data$DY_os + target_data$Dpsi_os) / n)

  # lb_n <- with(target_data[DY_cvtmle < 0,], max((1/gn - 1) / DY_cvtmle))
  # ub_n <- with(target_data[DY_cvtmle < 0,], min(-1 / DY_cvtmle))
  # lb_p <- with(target_data[DY_cvtmle > 0,], max(-1 / DY_cvtmle))
  # ub_p <- with(target_data[DY_cvtmle > 0,], min((1/gn - 1) / DY_cvtmle))
  # lb <- max(c(lb_n, lb_p))
  # ub <- min(c(ub_n, ub_p))
  # gn_eps <- function(epsilon, data){
  #   (1 + data$DY_cvtmle * epsilon) * data$gn
  # }
  # log_lik <- function(epsilon, data){
  #   gn_fluc <- gn_eps(epsilon, data)
  #   sum(ifelse(data$Y == 1, -log(gn_fluc), -log(1 - gn_fluc)))
  # }
  # grid_eps <- seq(lb, ub, length = 1000)
  # llik <- sapply(grid_eps, log_lik, data = target_data)

  # get CV estimator
  cv_empirical_estimates <- .getCVEstimator(prediction_list[1:K], sens = sens, 
                                            gn = mean(Y), quantile_type = quantile_type)

  # sample split estimate
  est_empirical <- mean(unlist(lapply(cv_empirical_estimates, "[[", "est")))
  var_empirical <- mean(unlist(lapply(cv_empirical_estimates, function(x){
    var(x$ic)
  })))
  ic_empirical <- Reduce(c, lapply(cv_empirical_estimates, "[[", "ic"))
  se_empirical <- sqrt(var_empirical / n)
  
  
  ## !!!!!!!!!!!!!!!!!!!!!!! ##
  # TO DO: 
  # Make sure that the non-doubly nested CV code works as expected
  # Compute influence function; get CIs 
  # get estimator based on sample splitting
  ## !!!!!!!!!!!!!!!!!!!!!!! ##
  

    # format output
    out <- list()
    out$est_cvtmle <- est_cvtmle
    # out$iter_cvtmle <- iter
    # out$cvtmle_trace <- tmle_auc
    out$se_cvtmle <- se_cvtmle
    out$est_init <- est_init
    out$est_empirical <- est_empirical
    out$se_empirical <- se_empirical
    out$est_onestep <- est_onestep
    out$se_onestep <- se_onestep
    out$est_esteq <- est_onestep
    out$se_esteq <- se_onestep
    out$se_cvtmle_type <- out$se_esteq_type <- out$se_empirical_type <- out$se_onestep_type <- "std"
    out$ic_cvtmle <- target_data$DY_cvtmle + target_data$Dpsi_cvtmle
    out$ic_onestep <- target_data$DY_os + target_data$Dpsi_os
    out$ic_empirical <- ic_empirical
    out$prediction_list <- prediction_list
    class(out) <- "cvauc"
    return(out)
}

#' @param x An entry in prediction_list
.getOneFold <- function(x, sens, gn, quantile_type = 8, ...){
  # get quantile 
  c0 <- quantile(x$psi_nBn_testx[x$test_y == 1], p = 1 - sens, type = quantile_type)
  # get influence function
  F1nc0 <- mean(x$psi_nBn_testx[x$test_y == 1] <= c0)
  F0nc0 <- mean(x$psi_nBn_testx[x$test_y == 0] <= c0)
  FYnc0 <- ifelse(x$test_y == 1, F1nc0, F0nc0)
  Psi <- gn * F1nc0 + (1-gn) * F0nc0
  DY <- ifelse(x$test_y == 1, F1nc0, F0nc0) - Psi
  # get density estimate
  dens <- tryCatch({.getDensity(x = x, c0 = c0, 
                      bounded_kernel = FALSE,
                      x_name = "psi_nBn_testx", 
                      nested_cv = FALSE, prediction_list = NULL, 
                      folds = NULL)}, error = function(e){
    list(f_0_c0 = 1, f_10_c0 = 1)
  })
  weight <- (1 + x$test_y / gn * dens$f_0_c0/dens$f_10_c0)
  ind <- as.numeric(x$psi_nBn_testx <= c0)
  Dpsi <- weight * (ind - FYnc0)

  return(list(est = Psi, ic = DY + Dpsi))
}

.getCVEstimator <- function(prediction_list, sens, gn, quantile_type = 8, ...){
  allFolds <- lapply(prediction_list, .getOneFold, sens = sens, gn = gn,
                     quantile_type = quantile_type)
  return(allFolds)
}

#' @param x An entry in prediction_list
#' @importFrom stats quantile 
.getQuantile <- function(x, p, quantile_type = 8){
  stats::quantile(x$psi_nBn_trainx[x$train_y == 1], p = p, type = quantile_type)
}

.getNestedCVQuantile <- function(x, p, prediction_list, folds,
                                 quantile_type = 8){
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

  # now get all values of psi from inner validation with Y = 1
  train_psi_1 <- unlist(lapply(inner_valid_prediction_and_y_list, function(z){
    z$inner_psi_nBn_testx[z$inner_test_y == 1]    
  }), use.names = FALSE)

  # # the cdf 
  # FynBn_1 <- sapply(uniq_train_psi_1, 
  #               F_nBn_star_nested_cv, y = 1, 
  #               epsilon = 0,
  #               inner_valid_prediction_and_y_list = inner_valid_prediction_and_y_list)

  # # find the smallest index smaller than p
  # idx <- findInterval(x = p, vec = FynBn_1)

  # # value of psi at that id
  # c0 <- uniq_train_psi_1[idx]
  c0 <- quantile(train_psi_1, p = p, type = quantile_type)
  return(c0)
}

#' @param x An entry in prediction_list
#' @importFrom np npudensbw npudens
#' @importFrom stats predict
#' @importFrom bde bde
.getDensity <- function(x, c0, bounded_kernel = FALSE,
                        x_name = "psi_nBn_trainx", 
                        nested_cv = FALSE, prediction_list = NULL, 
                        folds = NULL, ... ){
  if(!nested_cv){
    if(!bounded_kernel){
      # density given y = 1
      fitbw <- np::npudensbw(x[[x_name]][x$train_y == 1])
      fit <- np::npudens(fitbw)
      # estimate at c0
      f_10_c0 <- stats::predict(fit, edat = c0)

      # marginal density
      fitbw_marg <- np::npudensbw(x[[x_name]])
      fit_marg <- np::npudens(fitbw_marg)
      # estimate at c0
      f_0_c0 <- stats::predict(fit_marg, edat = c0)
    }else{
      # density given Y = 1
      fit_1 <- bde::bde(dataPoints = x[[x_name]][x$train_y == 1],
                      dataPointsCache = c0,
                      estimator = "betakernel")
      f_10_c0 <- fit_1@densityCache
      fit_all <- bde::bde(dataPoints = x[[x_name]],
                      dataPointsCache = c0,
                      estimator = "betakernel")
      f_0_c0 <- fit_all@densityCache
    }
  }else{
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
    all_pred <- Reduce("c", lapply(inner_valid_prediction_and_y_list, "[[",
                                   "inner_psi_nBn_testx"))
    all_y <- Reduce("c", lapply(inner_valid_prediction_and_y_list, "[[",
                                   "inner_test_y"))
    if(bounded_kernel){
      # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      # TO DO: Implement CV bandwidth selection
      # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      # density given Y = 1
      fit_1 <- bde::bde(dataPoints = all_pred[all_y == 1],
                      dataPointsCache = c0,
                      estimator = "betakernel")
      f_10_c0 <- fit_1@densityCache
      fit_all <- bde::bde(dataPoints = all_pred,
                      dataPointsCache = c0,
                      estimator = "betakernel")
      f_0_c0 <- fit_all@densityCache
    }else{
      # density given y = 1
      fitbw <- np::npudensbw(all_pred[all_y == 1])
      fit <- np::npudens(fitbw)
      # estimate at c0
      f_10_c0 <- stats::predict(fit, edat = c0)

      # marginal density
      fitbw_marg <- np::npudensbw(all_pred)
      fit_marg <- np::npudens(fitbw_marg)
      # estimate at c0
      f_0_c0 <- stats::predict(fit_marg, edat = c0)
    }
  }
  # return both
  return(list(f_10_c0 = f_10_c0, f_0_c0 = f_0_c0))
}


.makeTargetingData <- function(x, prediction_list, quantile_list, density_list, folds,
                               nested_cv = FALSE, gn){
  K <- length(folds)
  if(!nested_cv){
    Y_vec <- Reduce(c, lapply(prediction_list, "[[", "test_y"))
    Y_vec_pred <- rep(c(0,1), K)
    n <- length(Y_vec)
    fold_vec <- sort(rep(seq_len(K), unlist(lapply(folds, length))))
    fold_vec_pred <- sort(rep(seq_len(K), 2))
    gn_vec <- gn
    F_nBn_vec <- Reduce(c, mapply(FUN = function(m, c0){
      F_nBn_y1_at_c0 <- F_nBn_star(psi_x = c0, y = 1, Psi_nBn_0 = m$psi_nBn_trainx, Y_Bn = m$train_y)
      F_nBn_y0_at_c0 <- F_nBn_star(psi_x = c0, y = 0, Psi_nBn_0 = m$psi_nBn_trainx, Y_Bn = m$train_y)
      ifelse(m$test_y == 0, F_nBn_y0_at_c0, F_nBn_y1_at_c0)
    }, c0 = quantile_list, m = prediction_list))
    F_nBn_vec_pred <- Reduce(c, mapply(FUN = function(m, c0){
      F_nBn_y1_at_c0 <- F_nBn_star(psi_x = c0, y = 1, Psi_nBn_0 = m$psi_nBn_trainx, Y_Bn = m$train_y)
      F_nBn_y0_at_c0 <- F_nBn_star(psi_x = c0, y = 0, Psi_nBn_0 = m$psi_nBn_trainx, Y_Bn = m$train_y)
      c(F_nBn_y0_at_c0, F_nBn_y1_at_c0)
    }, c0 = quantile_list, m = prediction_list))
    dens_ratio <- Reduce(c, mapply(FUN = function(m, dens){
      rep(dens[[2]]/dens[[1]], length(m$test_y))
    }, m = prediction_list, dens = density_list))
    ind <- Reduce(c, mapply(m = prediction_list, c0 = quantile_list, function(m, c0){
      as.numeric(m$psi_nBn_testx <= c0)
    }))
    ind_pred <- c(NA, NA)
  }else{
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
    
    Y_vec <- Reduce(c, lapply(prediction_list[x], "[[", "test_y"))
    Y_vec_pred <- c(0,1)    
    fold_vec <- rep(x, length(Y_vec))
    fold_vec_pred <- rep(x, length(Y_vec_pred))
    gn_vec <- gn
    n <- length(Y_vec)
    F_nBn_y1_at_c0 <- F_nBn_star_nested_cv(psi_x = quantile_list[[x]], y = 1, 
                                           inner_valid_prediction_and_y_list = inner_valid_prediction_and_y_list)
    F_nBn_y0_at_c0 <- F_nBn_star_nested_cv(psi_x = quantile_list[[x]], y = 0, 
                                           inner_valid_prediction_and_y_list = inner_valid_prediction_and_y_list)
    F_nBn_vec <- ifelse(prediction_list[[x]]$test_y == 0, F_nBn_y0_at_c0, F_nBn_y1_at_c0)
    F_nBn_vec_pred <- c(F_nBn_y0_at_c0, F_nBn_y1_at_c0)
    dens_ratio <- density_list[[x]][[2]]/density_list[[x]][[1]]
    if(dens_ratio == Inf){ dens_ratio <- 1e2 }
    ind <- as.numeric(prediction_list[[x]]$psi_nBn_testx <= quantile_list[[x]])
  }

  out <- data.frame(fold = fold_vec, Y = Y_vec, gn = gn_vec, Fn = F_nBn_vec,
                    f_ratio = dens_ratio, ind = ind)
  out_pred <- data.frame(fold = fold_vec_pred, Y = Y_vec_pred, gn = gn_vec, 
                         Fn = F_nBn_vec_pred)
  return(list(out = out, out_pred = out_pred))
}

#' @export
boot_corrected_cvtn <- function(Y, X, B = 200, learner = "glm_wrapper", 
                         seed = 1234,
                         nested_cv = FALSE,
                         parallel = FALSE, sens = 0.95, ...){
  one_boot <- function(Y, X, n){
    idx <- sample(seq_len(n), replace = TRUE)
    train_Y <- Y[idx]
    train_X <- X[idx, , drop = FALSE]
    fit <- do.call(learner, args=list(train = list(Y = train_Y, X = train_X),
                                      test = list(Y = Y, X = X)))
    train_c0 <- quantile(fit$psi_nBn_trainx[fit$train_y == 1], p = 1 - sens)
    test_c0 <- quantile(fit$psi_nBn_testx[fit$test_y == 1], p = 1 - sens)
    train_est <- mean(fit$psi_nBn_trainx <= train_c0)
    test_est <- mean(fit$psi_nBn_testx <= test_c0)

    optimism <- train_est - test_est
    return(optimism)
  }
  n <- length(Y)
  all_boot <- replicate(B, one_boot(Y = Y, X = X, n = n))
  mean_optimism <- mean(all_boot)

  full_fit <- do.call(learner, args=list(train = list(Y = Y, X = X),
                                      test = list(Y = Y, X = X)))
  full_c0 <- quantile(full_fit$psi_nBn_testx[full_fit$train_y == 1], p = 1 - sens)
  full_est <- mean(full_fit$psi_nBn_testx <= full_c0)

  corrected_est <- full_est - mean_optimism
  return(list(corrected_est = corrected_est))
}

#' @param result_list A list of cvtn_cvtmle objects
#' @export
.getMCAveragedResults <- function(result_list, logit = TRUE,
                                  estimators = c("cvtmle","onestep","empirical")){
  estimates <- colMeans(Reduce(rbind,lapply(result_list, function(x){
      unlist(x[grep("est_", names(x))])
  })))

  se <- mapply(which_estimator = estimators, 
               estimate = estimates[names(estimates) %in% paste0("est_",estimators)], .computeMCSE, 
               MoreArgs = list(result_list = result_list, logit = logit))
  out <- list()
  out$est_cvtmle <- estimates["est_cvtmle"]
  out$est_onestep <- estimates["est_onestep"]
  out$est_init <- estimates["est_init"]
  out$est_esteq <- estimates["est_esteq"]
  out$est_empirical <- estimates["est_empirical"]
  out$se_cvtmle <- se["cvtmle"]
  out$se_onestep <- se["onestep"]
  out$se_esteq <- se["onestep"]
  out$se_empirical <- se["empirical"]  
  out$se_cvtmle_type <- ifelse(logit, "logit", "std")
  out$se_onestep_type <- ifelse(estimates["est_onestep"] >= 0 & estimates["est_onestep"] <= 1 & logit, "logit", "std")
  out$se_esteq_type <- out$se_onestep_type
  out$se_empirical_type <- ifelse(logit, "logit", "std")

  class(out) <- "cvauc"
  return(out)
}

.computeMCSE <- function(which_estimator = "cvtmle", estimate, result_list, logit = TRUE){
  icMatrix <- Reduce(cbind, lapply(result_list, "[[", paste0("ic_", which_estimator)))
  covar <- cov(icMatrix)
  B <- length(result_list)
  a <- matrix(1/B, nrow = B, ncol = 1)
  if(estimate <= 0 | estimate >= 1){
    logit <- FALSE
  }
  if(!logit){
      se <- sqrt( t(a) %*% covar %*% a / dim(icMatrix)[2])
  }else{
    g <- 1 / (estimate - estimate^2)
    se <- sqrt( g^2 * t(a) %*% covar %*% a / dim(icMatrix)[2] )
  }
  return(se)
}

