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

ns <- c(100, 200, 1000)
bigB <- 500
K <- c(5,20,50)
p <- 10
parm <- expand.grid(seed=1:bigB,
                    n=ns, K = K)

# parm <- parm[1,,drop=FALSE]
# source in simulation Functions
source("~/cvtmleauc/makeData.R")
# load drinf
# library(glmnet)
library(cvtmleAUC, lib.loc = "/home/dbenkese/R/x86_64-unknown-linux-gnu-library/3.2/")
library(SuperLearner)
library(glmnet)

# get the list size #########
if (args[1] == 'listsize') {
  cat(nrow(parm))
}

# execute prepare job ##################
if (args[1] == 'prepare') {
  for(i in 1:nrow(parm)){
     set.seed(parm$seed[i])
     dat <- makeData(n = parm$n[i], p = p)
     save(dat, file=paste0("~/cvtmleauc/scratch/dataList",
                           "_n=",parm$n[i],
                           "_K=",parm$K[i],
                           "_seed=",parm$seed[i],".RData"))
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
    
    # load data
    load(paste0("~/cvtmleauc/scratch/dataList_",
                "n=",parm$n[i],
                "_K=",parm$K[i],
                "_seed=",parm$seed[i], 
                ".RData"))
    
    # set seed
    set.seed(parm$seed[i])

    # get tmle and regular estimates
    fit <- cvauc_cvtmle(Y = dat$Y, X = dat$X, K = parm$K[i], 
                        learner = "randomforest_wrapper")
    # get true cvAUC
    N <- 1e5
    bigdat <- makeData(n = N, p = p)
    big_valid_pred_list <- lapply(fit$models, function(x){
      predict(x, newdata = bigdat$X, type = "response")
    })
    big_label_list <- rep(list(bigdat$Y), parm$K[i])
    true_cvauc <- mean(cvAUC::AUC(predictions = big_valid_pred_list,
                            labels = big_label_list))

    # now get auc of \hat{\Psi(P_n)}
    full_model <- glm_wrapper(train = dat, test = dat)
    big_valid_pred <- predict(full_model$model, newdata = bigdat$X, type = "response")
    true_auc_fullmodel <- cvAUC::AUC(predictions = big_valid_pred, labels = bigdat$Y)

    out <- c(fit$est, fit$se, fit$iter, fit$est_init, 
             fit$est_onestep, fit$se_onestep,
             fit$est_empirical, fit$se_empirical, true_cvauc, 
             true_auc_fullmodel)

    # save output 
    save(out, file = paste0("~/cvtmleauc/out/out_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            ".RData.tmp"))
    file.rename(paste0("~/cvtmleauc/out/out_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            ".RData.tmp"),
                paste0("~/cvtmleauc/out/out_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            ".RData"))    
    # save(fit$models, file = paste0("~/cvtmleauc/out/models_n=",
    #                         parm$n[i],"_seed=",parm$seed[i],
    #                         "_Q=",parm$Q[i],"_g=",parm$g[i],
    #                         "_cvFolds=",parm$cv[i],".RData.tmp"))
    # file.rename(paste0("~/cvtmleauc/out/models_n=",
    #                    parm$n[i],"_seed=",parm$seed[i],
    #                    "_Q=",parm$Q[i],"_g=",parm$g[i],"_cvFolds=",parm$cv[i],
    #                    ".RData.tmp"),
    #             paste0("~/cvtmleauc/out/models_n=",
    #                    parm$n[i],"_seed=",parm$seed[i],
    #                    "_Q=",parm$Q[i],"_g=",parm$g[i],"_cvFolds=",parm$cv[i],
    #                    ".RData"))
  }
}

# merge job ###########################
if (args[1] == 'merge') {   
  ns <- c(100, 250, 500, 750)
  bigB <- 500
  K <- c(5,10,20,30)
  p <- 10
  parm <- expand.grid(seed=1:bigB,
                      n=ns, K = K)
  rslt <- matrix(NA, nrow = nrow(parm), ncol = 13)
  for(i in 1:nrow(parm)){
      tmp_1 <- tryCatch({
          load(paste0("~/cvtmleauc/out/out",
                      "_n=", parm$n[i],
                      "_seed=",parm$seed[i],
                      "_K=", parm$K[i],
                      ".RData"))
          out
      }, error=function(e){
        rep(NA, 10)
      })
      rslt[i,] <- c(parm$seed[i], parm$n[i], parm$K[i], tmp_1)
  }
  # # format
  out <- data.frame(rslt)

  sim_names <- c("seed","n","K",
                 "cvtmle","se_cvtmle","iter_cvtmle",
                 "init",
                 "onestep","se_onestep",
                 "empirical","se_empirical",
                 "truth", "truth_full")
  colnames(out) <- sim_names

  save(out, file=paste0('~/cvtmleauc/out/allOut.RData'))
}
# local editing 
if(FALSE){
  setwd("~/Dropbox/R/cvtmleauc/sandbox/simulation")
  load("allOut.RData")
  # bias
  parm <- expand.grid(n = c(100, 250, 500, 750),
                      K = c(5, 10, 20, 30))
  b <- v <- m <- co <- NULL
  for(i in seq_len(length(parm[,1]))){
    x <- out[out$n == parm$n[i] & out$K == parm$K[i],]
    b <- rbind(b, colMeans(x[,c("cvtmle", "onestep", "empirical")] - x$truth))
    v <- rbind(v, apply(x[,c("cvtmle", "onestep", "empirical")], 2, var))
    m <- rbind(m, colMeans((x[,c("cvtmle", "onestep", "empirical")] - x$truth)^2))
    # coverage
    cov_tmle <- mean(x$cvtmle - 1.96 * x$se_cvtmle < x$truth & 
                      x$cvtmle + 1.96 * x$se_cvtmle > x$truth)
    cov_onestep <- mean(x$onestep - 1.96 * x$se_onestep < x$truth & 
                      x$onestep + 1.96 * x$se_onestep > x$truth)
    cov_empirical <- mean(x$empirical - 1.96 * x$se_empirical < x$truth & 
                      x$empirical + 1.96 * x$se_empirical > x$truth)
    co <- rbind(co, c(cov_tmle, cov_onestep, cov_empirical))
  }
  parm <- cbind(parm, b, v, m, co)
  colnames(parm) <- c("n", "K", paste0("bias_", c("cvtmle","onestep","empirical")),
                      paste0("var_", c("cvtmle","onestep","empirical")),
                      paste0("mse_", c("cvtmle","onestep","empirical")),
                      paste0("cov_", c("cvtmle","onestep","empirical")))

  #--------------------------------
  # MSE plots
  #--------------------------------
  # make matrix of relative MSE
  n_ct <- 0
  K_ct <- 0
  rel_mse_cvtmle <- matrix(NA, 4, 4)
  rel_mse_onestep <- matrix(NA, 4, 4)
  rel_mse_tmlevonestep <- matrix(NA, 4, 4)
  for(n in c(100, 250, 500, 750)){
    n_ct <- n_ct + 1
    for(K in c(5, 10, 20, 30)){
      K_ct <- K_ct + 1
      rel_mse_cvtmle[n_ct, K_ct] <- parm$mse_cvtmle[parm$n == n & parm$K == K] / 
                                parm$mse_empirical[parm$n == n & parm$K == K]      
      rel_mse_onestep[n_ct, K_ct] <- parm$mse_onestep[parm$n == n & parm$K == K] / 
                                parm$mse_empirical[parm$n == n & parm$K == K]
      rel_mse_tmlevonestep[n_ct, K_ct] <- parm$mse_cvtmle[parm$n == n & parm$K == K] / 
                                parm$mse_onestep[parm$n == n & parm$K == K]
    }
    K_ct <- 0
  }
  row.names(rel_mse_cvtmle) <- row.names(rel_mse_onestep) <- row.names(rel_mse_tmlevonestep) <- c(100, 250, 500, 750)
  colnames(rel_mse_cvtmle) <- colnames(rel_mse_onestep) <- colnames(rel_mse_tmlevonestep) <- c(5, 10, 20, 30)
  
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