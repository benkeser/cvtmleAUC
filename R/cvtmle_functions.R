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
#' @importFrom cvAUC AUC
#' @export
#' @return A list
#' @examples
#' n <- 200
#' p <- 10
#' X <- matrix(rnorm(n*p), nrow = n, ncol = p)
#' Y <- rbinom(n, 1, plogis(X[,1] + X[,10]))
#' fit <- cvauc_cvtmle(Y = Y, X = X, K = 5, learner = "glm_wrapper")
cvauc_cvtmle <- function(Y, X, K, learner = "glm_wrapper", 
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

  # initial distributions of psi in training samples
  dist_psix_y0 <- lapply(prediction_list, .getPsiDistribution, y = 0)
  dist_psix_y1 <- lapply(prediction_list, .getPsiDistribution, y = 1)
  
  # make long data for targeting step
  long_data_list <- lapply(prediction_list, .makeLongData, gn = mean(Y))
  # full_long_data <- Reduce(rbind, long_data_list)
  # full_long_data$outcome <- with(full_long_data, as.numeric(psi <= u))
  # full_long_data$logit_Fn <- SuperLearner::trimLogit(full_long_data$Fn, .Machine$double.neg.eps)
  
  # targeting
  PnDstar <- Inf
  epsilon_0 <- rep(0, maxIter)
  epsilon_1 <- rep(0, maxIter)
  iter <- 0
  update_long_data_list <- long_data_list
  # combine list into data frame
  full_long_data <- Reduce(rbind, update_long_data_list)
  

  # compute initial estimate of cvAUC
  # compute estimated cv-AUC 
  dist_psix_y0_star <- lapply(prediction_list, .getPsiDistribution, 
                         y = 0, epsilon = epsilon_0)
  dist_psix_y1_star <- lapply(prediction_list, .getPsiDistribution, y = 1,
                         epsilon = epsilon_1)

  # get AUC
  init_auc <- mean(mapply(FUN = .getAUC, dist_y0 = dist_psix_y0_star, 
                     dist_y1 = dist_psix_y1_star))

  tmle_auc <- rep(NA, maxIter)

  while(PnDstar > icTol & iter < maxIter){
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
    # update values in long_data_list
    epsilon_0[iter] <- as.numeric(fluc_mod_0$coef[1])
    update_long_data_list <- lapply(prediction_list, .makeLongData, gn = mean(Y),
                             epsilon_0 = epsilon_0, epsilon_1 = epsilon_1,
                             update = TRUE)
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
    update_long_data_list <- lapply(prediction_list, .makeLongData, gn = mean(Y),
                             epsilon_0 = epsilon_0, epsilon_1 = epsilon_1,
                             update = TRUE)
    # update full long data
    full_long_data <- Reduce(rbind, update_long_data_list)
    
    # compute mean of EIF
    D1 <- .Dy(full_long_data, y = 1)
    D0 <- .Dy(full_long_data, y = 0)
    ic <- D1 + D0 
    PnDstar <- mean(ic)

    # compute estimated cv-AUC 
    dist_psix_y0_star <- lapply(prediction_list, .getPsiDistribution, 
                           y = 0, epsilon = epsilon_0)
    dist_psix_y1_star <- lapply(prediction_list, .getPsiDistribution, y = 1,
                           epsilon = epsilon_1)

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


    # compute regular cvAUC
    valid_pred_list <- lapply(prediction_list, "[[", "psi_nBn_testx")
    valid_label_list <- lapply(prediction_list, "[[", "test_y")
    regular_cvauc <- mean(cvAUC::AUC(predictions = valid_pred_list,
                                labels = valid_label_list))

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
    out$est <- tmle_auc[iter]
    out$est_trace <- tmle_auc
    out$se <- sqrt(var(ic)/n)
    out$est_init <- init_auc
    out$est_empirical <- regular_cvauc
    out$models <- lapply(prediction_list, "[[", "model")
    return(out)
}

#' Compute a portion of the efficient influence function
#' @param full_long_data A long form data set
#' @param y Which portion of the EIF to compute
#' @return Vector of EIF
.Dy <- function(full_long_data, y){
  by(full_long_data, full_long_data$id, function(x){
    sum(as.numeric(x$Y == y)/(x$gn) * (x$outcome - x$Fn) * x$dFn)
  })
}

#' Compute the (CV)TMLE cumulative dist at psi_x
#' @param psi_x Value to compute conditional (on Y=y) cdf of Psi
#' @param y Value of Y to condition on 
#' @param Psi_nBn_0 Values of Psi_nBn(X) from training sample
#' @param Y_Bn Values of Y from training sample
#' @param epsilon Vector of fluctuation parameter estimates
#' @return Numeric value of CDF
F_nBn_star <- function(psi_x, y, Psi_nBn_0, Y_Bn, epsilon = 0){
  plogis(SuperLearner::trimLogit(mean(Psi_nBn_0[Y_Bn == y] <= psi_x), .Machine$double.neg.eps) +
          sum(epsilon))
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

.makeLongData <- function(x, gn, update = FALSE, epsilon_0 = 0, epsilon_1 = 0){
  # first the dumb way, writing a loop over x$psi_nBn_testx
  uniq_train_psi_y0 <- sort(unique(x$psi_nBn_trainx[x$train_y == 0]))
  uniq_train_psi_y1 <- sort(unique(x$psi_nBn_trainx[x$train_y == 1]))
  # ord_train_psi_y0 <- order(x$psi_nBn_trainx[x$train_y == 0])
  # ord_train_psi_y1 <- order(x$psi_nBn_trainx[x$train_y == 1])
  
  n_valid <- length(x$psi_nBn_testx)
  n_train <- length(x$psi_nBn_trainx)
  n1_train <- sum(x$train_y)
  n0_train <- n_train - n1_train
  n1_valid <- sum(x$test_y)
  n0_valid <- n_valid - n1_valid
  valid_ids <- as.numeric(names(x$psi_nBn_testx))
  tot_length <- n1_valid * n0_train + n0_valid * n1_train

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
      cur_end <- cur_start + n1_train - 1
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
      cur_end <- cur_start + n0_train - 1
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
  out$logit_Fn <- SuperLearner::trimLogit(out$Fn, .Machine$double.neg.eps)
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
  tot <- 0
  for(i in seq_along(dist_y0$psix)){
    idx <- findInterval(x = dist_y0$psix[i], vec = dist_y1$psix)
    p1 <- ifelse(idx == 0, 1, (1 - dist_y1$Fn[idx]))
    p2 <- dist_y0$dFn[i]
    tot <- tot + p1 * p2
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


#' Worker function for fitting prediction functions (possibly in parallel)
#' 
#' @param learner The wrapper to use
#' @param Y The outcome
#' @param X The predictors
#' @param K The number of folds
#' @param parallel Whether to compute things in parallel using future
#' 
#' @return A list of the result of the wrapper executed in each fold
.getPredictions <- function(learner, Y, X, K, folds, parallel){

  .doFit <- function(x, tmpX, Y, folds, learner){
    out <- do.call(learner, args=list(train = list(Y = Y[-folds[[x]]], X = tmpX[-folds[[x]],,drop=FALSE]),
                                      test = list(Y = Y[folds[[x]]], X = tmpX[folds[[x]],,drop=FALSE])))
    out$valid_ids <- folds[[x]]
    return(out)
  }

  if(parallel){
    stop("Parallel processing code needs to be re-written.")
    # cl <- makeCluster(detectCores())
    # registerDoParallel(cl)
    # predFitList <- foreach(v = 1:length(folds), .export=learner) %dopar% 
    #   .doFit(v, tmpX = X, Y = Y, folds = folds, learner = learner)
    # stopCluster(cl)
  }else{
    predFitList <- lapply(split(seq(K),factor(seq(K))),FUN = .doFit, tmpX = X, Y = Y, folds = folds, learner = learner)
  }
  
  # return results
  return(predFitList)
}

