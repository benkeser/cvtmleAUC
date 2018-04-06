#! /usr/bin/env Rscript

# get environment variables
MYSCRATCH <- Sys.getenv('MYSCRATCH')
RESULTDIR <- Sys.getenv('RESULTDIR')
STEPSIZE <- as.numeric(Sys.getenv('STEPSIZE'))
TASKID <- as.numeric(Sys.getenv('SLURM_ARRAY_TASK_ID'))

# set defaults if nothing comes from environment variables
MYSCRATCH[is.na(MYSCRATCH)] <- '.'
RESULTDIR[is.na(RESULTDIR)] <- '.'
STEPSIZE[is.na(STEPSIZE)] <- 1
TASKID[is.na(TASKID)] <- 0

# get command lines arguments
args <- commandArgs(trailingOnly = TRUE)
if(length(args) < 1){
  stop("Not enough arguments. Please use args 'listsize', 'prepare', 'run <itemsize>' or 'merge'")
}

ns <- c(50, 100, 250, 500)
bigB <- 500
K <- c(5,10,20,40)
wrappers <- c("glm_wrapper", "randomforest_wrapper")
# wrappers <- c("glmnet_wrapper")
p <- 10
# TO DO:
# Add a replicate argument for repeated cross-validation estimators
parm <- expand.grid(seed = 1:bigB,
                    n = ns, K = K, 
                    wrapper = wrappers,
                    stringsAsFactors = FALSE)

# load('~/cvtmleauc/out/allOut_new.RData')
# redo_idx <- which(is.na(out$est_dcvtmle))
# parm <- parm[redo_idx,]
# parm <- parm[1,,drop=FALSE]
# source in simulation Functions
source("~/cvtmleauc/makeData.R")
# load drinf
# library(glmnet)
# devtools::install_github("benkeser/cvtmleAUC", dependencies = TRUE)
library(cvtmleAUC, lib.loc = "/home/dbenkese/R/x86_64-pc-linux-gnu-library/3.4")
library(cvAUC)
library(SuperLearner)
library(data.table)
# library(glmnet)

# get the list size #########
if (args[1] == 'listsize') {
  cat(nrow(parm))
}

# execute prepare job ##################
if (args[1] == 'prepare') {
  parm_red <- parm[parm$K == parm$K[1] & parm$wrapper == parm$wrapper[1],]
  for(i in 1:nrow(parm_red)){
     set.seed(parm_red$seed[i])
     dat <- makeData(n = parm_red$n[i], p = p)
     save(dat, file=paste0("~/cvtmleauc/scratch/dataList",
                           "_n=",parm_red$n[i],
                           "_seed=",parm_red$seed[i],".RData"))
   }
   print(paste0('initial datasets saved to: ~/cvtmleauc/scratch/dataList ... .RData'))
}

# execute parallel job #################################################
if (args[1] == 'run') {
  if (length(args) < 2) {
    stop("Not enough arguments. 'run' needs a second argument 'id'")
  }
  id <- as.numeric(args[2])
  print(paste(Sys.time(), "arrid:" , id, "TASKID:",
              TASKID, "STEPSIZE:", STEPSIZE))
  for (i in (id+TASKID):(id+TASKID+STEPSIZE-1)) {
    print(paste(Sys.time(), "i:" , i))
    print(parm[i,])
    print(sessionInfo())
    # load data
    load(paste0("~/cvtmleauc/scratch/dataList_",
                "n=",parm$n[i],
                "_seed=",parm$seed[i], 
                ".RData"))
    
    # get estimates of dcvauc
    n_replicates <- 20
    fit_dcv <- vector(mode = "list", length = n_replicates)
    fit_cv <- vector(mode = "list", length = n_replicates)
    for(i in seq_len(n_replicates)){
      set.seed(i)
      fit_dcv[[i]] <- cvtn_cvtmle(Y = dat$Y, X = dat$X, K = parm$K[i], 
                          learner = parm$wrapper[i], nested_cv = TRUE,
                          nested_K = 39)
      set.seed(i)
    # get estimates of cvtn
      fit_cv[[i]] <- cvtn_cvtmle(Y = dat$Y, X = dat$X, K = parm$K[i], 
                          learner = parm$wrapper[i], nested_cv = FALSE,
                          prediction_list = fits_dcv$prediction_list[1:parm$K[i]])
    }

    # get the truth
    set.seed(parm$seed[i])
    big_n <- 1e5
    big_data <- makeData(n = big_n, p = 10)
    # bigpred <- lapply(fit$prediction_list[1:K], function(x){
    #   if("randomForest" %in% class(x$model)){
    #     predict(x$model, newdata = bigX, type = "prob")[,2]
    #   }else{
    #     predict(x$model, newdata = bigX, type = "response")
    #   }
    # })
    # bigquantile <- lapply(bigpred, function(x, Y){
    #   quantile(x[Y == 1], p = 0.05, type = 8)
    # }, Y = bigY)
    # big_testneg <- mapply(p = bigpred, q = bigquantile, function(p, q, Y){
    #   mean(p <= q)
    # }, MoreArgs = list(Y = bigY))
    # true_cvparameter <- mean(big_testneg)

    # fit on full data
    fit_full <- glm_wrapper(train = list(X = dat$X, Y = dat$Y), 
                            test = list(X = big_data$X, Y = big_data$Y))
    bigquantile_full <- quantile(fit_full$psi_nBn_testx[big_data$Y == 1], p = 0.05, type = 8)
    big_testneg_full <- mean(fit_full$psi_nBn_testx <= bigquantile_full)
    true_parameter <- big_testneg_full

    # bootstrap estimate 
    set.seed(parm$seed[i])
    fit_boot <- boot_corrected_cvtn(Y = dat$Y, X = dat$X, learner = parm$wrapper[i])

    # c together output
    out <- c( # cvtmle estimates of dcvauc
             fit_dcv[[1]]$est_cvtmle, fit_dcv[[1]]$se_cvtmle,
             # iterations of cvtmle for dcv
             fit_dcv[[1]]$iter, 
             # initial plug-in estimate of dcv
             fit_dcv[[1]]$est_init, 
             # one-step estimate of dcv
             fit_dcv[[1]]$est_onestep, fit_dcv[[1]]$se_onestep,
             # estimating eqn estimate of dcv
             fit_dcv[[1]]$est_esteq, fit_dcv[[1]]$se_esteq,
             # cvtmle estimate of cv
             fit_cv[[1]]$est_cvtmle, fit_cv[[1]]$se_cvtmle, 
             # iterations of cvtmle for cv
             fit_cv[[1]]$iter, fit_cv[[1]]$est_init, 
             # one-step estimate of cv
             fit_cv[[1]]$est_onestep, fit_cv[[1]]$se_onestep,
             # estimating eqn estimate of cv
             fit_cv[[1]]$est_esteq, fit_cv[[1]]$se_esteq,
             # full sample split estimate of cv
             fit_dcv[[1]]$est_empirical, fit_dcv[[1]]$se_empirical)

    # now add in MC averaged results for M = 5, 10, 20
    for(M in c(5, 10, 20)){
      avg_dcv <- .getMCAveragedResults(fit_dcv[1:M], logit = FALSE)
      avg_cv <- .getMCAveragedResults(fit_cv[1:M], logit = FALSE)
      out <- c(out, 
               avg_dcv$est_cvtmle, avg_dcv$se_cvtmle,
               # initial plug-in estimate of dcv
               avg_dcv$est_init, 
             # one-step estimate of dcv
             avg_dcv$est_onestep, avg_dcv$se_onestep,
             # estimating eqn estimate of dcv
             avg_dcv$est_esteq, avg_dcv$se_esteq,
             # cvtmle estimate of cv
             avg_cv$est_cvtmle, avg_cv$se_cvtmle, 
             # iterations of cvtmle for cv
             avg_cv$est_init, 
             # one-step estimate of cv
             avg_cv$est_onestep, avg_cv$se_onestep,
             # estimating eqn estimate of cv
             avg_cv$est_esteq, avg_cv$se_esteq,
             # full sample split estimate of cv
             avg_dcv$est_empirical, avg_dcv$se_empirical)
    }
    out <- c(out, fit_boot[[1]], true_parameter)

    # save output 
    save(out, file = paste0("~/cvtmleauc/out/out_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            ".RData.tmp"))
    file.rename(paste0("~/cvtmleauc/out/out_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],                            
                            ".RData.tmp"),
                paste0("~/cvtmleauc/out/out_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            ".RData"))
  }
}

# merge job ###########################
if (args[1] == 'merge') {   
  ns <- c(50, 75, 100, 250, 500, 750)
  bigB <- 500
  K <- c(5,10,20,40)
  wrappers <- c("glm_wrapper", "stepglm_wrapper", "randomforest_wrapper", "glmnet_wrapper")
  p <- 10
  parm <- expand.grid(seed = 1:bigB,
                      n = ns, K = K, 
                      wrapper = wrappers,
                      stringsAsFactors = FALSE)
  rslt <- matrix(NA, nrow = nrow(parm), ncol = 24)
  for(i in 1:nrow(parm)){
      tmp_1 <- tryCatch({
          load(paste0("~/cvtmleauc/out/out",
                      "_n=", parm$n[i],
                      "_seed=",parm$seed[i],
                      "_K=", parm$K[i],
                      "_wrapper=",parm$wrapper[i],
                      ".RData"))
          out
      }, error=function(e){
        rep(NA, 20)
      })
      rslt[i,] <- c(parm$seed[i], parm$n[i], parm$K[i], parm$wrapper[i], tmp_1)
  }
  # # format
  out <- data.frame(rslt, stringsAsFactors = FALSE)

  sim_names <- c("seed","n","K","wrapper",
                 "est_dcvtmle", "se_dcvtmle", "iter_dcvtmle",
                 "est_dinit", "est_donestep", "se_donestep",
                 "est_desteq","se_desteq","est_cvtmle","se_cvtmle",
                 "iter_cvtmle","est_init", "est_onestep", "se_onestep",
                 "est_esteq","se_esteq","est_emp","se_emp","true_cvauc",
                 "true_dcvauc")
  colnames(out) <- sim_names
  out[,c(1:3,5:ncol(out))] <- apply(out[,c(1:3,5:ncol(out))], 2, function(y){
    as.numeric(as.character(y))})

  save(out, file=paste0('~/cvtmleauc/out/allOut_new.RData'))
}


# local editing 
if(FALSE){
  # setwd("~/Dropbox/R/cvtmleauc/sandbox/simulation")
    load("~/cvtmleauc/out/allOut_new.RData")

    get_sim_rslt <- function(out, parm, wrapper, truth = "true_cvauc",
                             estimators = c("dcvtmle","donestep","desteq",
                                     "cvtmle","onestep","esteq",
                                     "emp"), ...){
      b <- v <- m <- co <- NULL
      for(i in seq_len(length(parm[,1]))){
        x <- out[out$n == parm$n[i] & out$K == parm$K[i] & out$wrapper == wrapper,]
        b <- rbind(b, colMeans(x[,paste0("est_",estimators)] - x[,truth], na.rm = TRUE))
        v <- rbind(v, apply(x[,paste0("est_",estimators)], 2, var, na.rm = TRUE))
        m <- rbind(m, colMeans((x[,paste0("est_",estimators)] - as.numeric(x[,truth]))^2, na.rm = TRUE))
        # coverage
        coverage <- rep(NA, length(estimators))
        ct <- 0
        for(est in estimators){
          ct <- ct + 1
          coverage[ct] <- mean(x[,paste0("est_",est)] - 1.96 * x[,paste0("se_",est)] < x[,truth] & 
                          x[,paste0("est_",est)] + 1.96 * x[,paste0("se_",est)] > x[,truth], na.rm = TRUE)
        }
        co <- rbind(co, coverage)
      }
      parm <- cbind(parm, b, v, m, co)
      colnames(parm) <- c("n", "K", paste0("bias_", estimators),
                          paste0("var_", estimators),
                          paste0("mse_", estimators),
                          paste0("cov_", estimators))
      return(parm)
    }
    parm <- expand.grid(n = c(100, 250, 500, 750),
                        K = c(5, 10, 20, 40))
    glm_rslt <- get_sim_rslt(out, parm, wrapper = "glm_wrapper")
    stepglm_rslt <- get_sim_rslt(out, parm, wrapper = "stepglm_wrapper")
    randomforest_rslt <- get_sim_rslt(out, parm, wrapper = "randomforest_wrapper")
    glmnet_rslt <- get_sim_rslt(out, parm, wrapper = "glmnet_wrapper")
    
    #--------------------------------
    # MSE plots
    #--------------------------------
    make_mse_compare_plot <- function(rslt, est1, est2, ns = c(100, 250, 500, 750),
                                      Ks = c(5, 10, 20, 40),...){
    # make matrix of relative MSE
    n_ct <- 0
    K_ct <- 0
    rel_mse <- matrix(NA, length(ns), length(Ks))
    for(n in ns){
      n_ct <- n_ct + 1
      for(K in Ks){
        K_ct <- K_ct + 1
        rel_mse[n_ct, K_ct] <- rslt[rslt$n == n & rslt$K == K, paste0("mse_",est1)] / 
                                  rslt[rslt$n == n & rslt$K == K, paste0("mse_",est2)]      
      }
      K_ct <- 0
    }
    row.names(rel_mse) <- ns
    colnames(rel_mse) <- Ks
    
    superheat::superheat(X = rel_mse, X.text = round(rel_mse, 2), scale = FALSE, 
              pretty.order.rows = FALSE, 
              pretty.order.cols = FALSE, heat.col.scheme = "red",
              row.title = "Sample size", column.title = "CV folds",
              title = paste0("MSE(",est1,")/MSE(",est2,")"),
    # plot_done()
              legend.breaks = seq(min(rel_mse), max(rel_mse), by = 0.1), ...)
    }

    # CV TMLE vs. empirical

    for(rslt in c("glm_rslt","randomforest_rslt","glmnet_rslt","stepglm_rslt")){
      pdf(paste0("~/cvtmleauc/",rslt,"_perf.pdf"))
      # comparing to emp
        make_mse_compare_plot(eval(parse(text = rslt)), est1 = "dcvtmle", est2 = "emp")
        make_mse_compare_plot(eval(parse(text = rslt)), est1 = "cvtmle", est2 = "emp")
        make_mse_compare_plot(eval(parse(text = rslt)), est1 = "desteq", est2 = "emp")
        make_mse_compare_plot(eval(parse(text = rslt)), est1 = "esteq", est2 = "emp")  
        make_mse_compare_plot(eval(parse(text = rslt)), est1 = "donestep", est2 = "emp")
        make_mse_compare_plot(eval(parse(text = rslt)), est1 = "onestep", est2 = "emp")
      # comparing to eachother
        make_mse_compare_plot(eval(parse(text = rslt)), est1 = "onestep", est2 = "cvtmle")
        make_mse_compare_plot(eval(parse(text = rslt)), est1 = "donestep", est2 = "dcvtmle")
        make_mse_compare_plot(eval(parse(text = rslt)), est1 = "onestep", est2 = "esteq")
        make_mse_compare_plot(eval(parse(text = rslt)), est1 = "donestep", est2 = "desteq")
        make_mse_compare_plot(eval(parse(text = rslt)), est1 = "esteq", est2 = "cvtmle")
        make_mse_compare_plot(eval(parse(text = rslt)), est1 = "desteq", est2 = "dcvtmle")
      # comparing cv to dcv 
        make_mse_compare_plot(eval(parse(text = rslt)), est1 = "onestep", est2 = "donestep")
        make_mse_compare_plot(eval(parse(text = rslt)), est1 = "cvtmle", est2 = "dcvtmle")
        make_mse_compare_plot(eval(parse(text = rslt)), est1 = "esteq", est2 = "desteq")
      dev.off()
    }
  #--------------------------------
  # CV TMLE vs. Empirical 
  #--------------------------------
  pdf("mse_results.pdf")
  superheat(X = rel_mse_cvtmle, X.text = round(rel_mse_cvtmle, 2), scale = FALSE, 
            pretty.order.rows = FALSE, 
            pretty.order.cols = FALSE, heat.col.scheme = "red",
            row.title = "Sample size", column.title = "CV folds",
            legend.breaks = c(0.7, 0.8, 0.9, 1),
            title = "MSE CVTMLE / MSE Empirical")
  #--------------------------------
  # One step vs. Empirical 
  #--------------------------------
  superheat(X = rel_mse_onestep, X.text = round(rel_mse_onestep, 2), scale = FALSE, 
            pretty.order.rows = FALSE, 
            pretty.order.cols = FALSE, heat.col.scheme = "red",
            row.title = "Sample size", column.title = "CV folds",
            legend.breaks = c(0.7, 0.8, 0.9, 1),
            title = "MSE CV One step / MSE Empirical")  
  #--------------------------------
  # CVTMLE vs. Onestep 
  #--------------------------------
  superheat(X = rel_mse_tmlevonestep, X.text = round(rel_mse_tmlevonestep, 2), 
            scale = FALSE, 
            pretty.order.rows = FALSE, 
            pretty.order.cols = FALSE, heat.col.scheme = "red",
            row.title = "Sample size", column.title = "CV folds",
            legend.breaks = c(0.7, 0.8, 0.9, 1),
            title = "MSE CV One step / MSE CVTMLE")
  dev.off()

  #--------------------------------
  # Coverage plots
  #--------------------------------
  # make matrix of relative MSE
  n_ct <- 0
  K_ct <- 0
  cov_cvtmle <- matrix(NA, 4, 4)
  cov_onestep <- matrix(NA, 4, 4)
  cov_tmlevonestep <- matrix(NA, 4, 4)
  for(n in c(100, 250, 500, 750)){
    n_ct <- n_ct + 1
    for(K in c(5, 10, 20, 30)){
      K_ct <- K_ct + 1
      cov_cvtmle[n_ct, K_ct] <- parm$cov_cvtmle[parm$n == n & parm$K == K]     
      cov_onestep[n_ct, K_ct] <- parm$cov_onestep[parm$n == n & parm$K == K] 
      cov_tmlevonestep[n_ct, K_ct] <- parm$cov_empirical[parm$n == n & parm$K == K] 
    }
    K_ct <- 0
  }
  row.names(cov_cvtmle) <- row.names(cov_onestep) <- row.names(cov_tmlevonestep) <- c(100, 250, 500, 750)
  colnames(cov_cvtmle) <- colnames(cov_onestep) <- colnames(cov_tmlevonestep) <- c(5, 10, 20, 30)
  
  #--------------------------------
  # CV TMLE vs. Empirical 
  #--------------------------------
  pdf("coverage_results.pdf")
  superheat(X = cov_cvtmle, X.text = round(cov_cvtmle, 2), scale = FALSE, 
            pretty.order.rows = FALSE, 
            pretty.order.cols = FALSE, heat.col.scheme = "red",
            row.title = "Sample size", column.title = "CV folds",
            legend.breaks = c(0.7, 0.8, 0.9, 1),
            title = "Coverage of nominal 95% CI CVTMLE")
  #--------------------------------
  # One step vs. Empirical 
  #--------------------------------
  superheat(X = cov_onestep, X.text = round(cov_onestep, 2), scale = FALSE, 
            pretty.order.rows = FALSE, 
            pretty.order.cols = FALSE, heat.col.scheme = "red",
            row.title = "Sample size", column.title = "CV folds",
            legend.breaks = c(0.7, 0.8, 0.9, 1),
            title = "Coverage of nominal 95% CI One step")  
  #--------------------------------
  # CVTMLE vs. Onestep 
  #--------------------------------
  superheat(X = cov_tmlevonestep, X.text = round(cov_tmlevonestep, 2), 
            scale = FALSE, 
            pretty.order.rows = FALSE, 
            pretty.order.cols = FALSE, heat.col.scheme = "red",
            row.title = "Sample size", column.title = "CV folds",
            legend.breaks = c(0.7, 0.8, 0.9, 1),
            title = "Coverage of nominal 95% CI Empirical")
  dev.off()
}