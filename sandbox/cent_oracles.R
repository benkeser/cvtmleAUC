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

ns <- c(100, 250, 500, 750)
bigB <- 500
K <- c(5,10,20,40)

parm <- expand.grid(seed = 1:bigB,
                    n = ns, K = K, 
                    stringsAsFactors = FALSE)

# parm <- parm[1,,drop=FALSE]
# source in simulation Functions
source("~/cvtmleauc/makeData.R")
# source("~/cvtmleauc/wrapper_functions.R")
# load drinf
# devtools::install_github("benkeser/cvtmleAUC")
library(cvtmleAUC, lib.loc = "/home/dbenkese/R/x86_64-pc-linux-gnu-library/3.4")
library(glmnet)
library(randomForest)
library(cvAUC)

# load('~/cvtmleauc/out/allOut_oracles.RData')
# idx <- which(is.na(out[,5]))
# ceil_idx <- ceiling(idx/4)
# parm <- parm[ceil_idx,]
# library(SuperLearner, lib.loc = '/home/dbenkese/R/x86_64-pc-linux-gnu-library/3.4')

# get the list size #########
if (args[1] == 'listsize') {
  cat(nrow(parm))
}

# execute prepare job ##################
if (args[1] == 'prepare') {
  # for(i in 1:nrow(parm)){
  #    set.seed(parm$seed[i])
  #    dat <- makeData(n = parm$n[i], p = p)
  #    save(dat, file=paste0("~/cvtmleauc/scratch/dataList",
  #                          "_n=",parm$n[i],
  #                          "_K=",parm$K[i],
  #                          "_seed=",parm$seed[i],".RData"))
  #  }
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
  # for(i in ceil_idx){
    cat("i \n")
    print(paste(Sys.time(), "i:" , i))
    print(parm[i,])
    
    # load data
    data_suffix <- paste0("n=",parm$n[i],
                "_seed=",parm$seed[i], 
                ".RData")

    out_suffix <- paste0("n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            ".RData")

    load(paste0("~/cvtmleauc/scratch/dataList_", data_suffix))

    N <- 1e4
    bigdat <- makeData(n = N, p = 10)
    my_predict <- function(x, newdata){
      if("glm" %in% class(x$model)){
        predict(x$model, newdata = newdata, type = "response")
      }else if("randomForest" %in% class(x$model)){
        predict(x$model, newdata = newdata, type = "vote")[, 2]
      }else if("cv.glmnet" %in% class(x$model)){
        newx <- model.matrix(~.-1,data = newdata)
        predict(x$model, newx = newx, type = "response", s = "lambda.min")
      }else if("glmnet" %in% class(x$model)){
        newx <- model.matrix(~.-1,data = newdata)
        predict(x$model, newx = newx, type = "response", s = x$model$my_lambda)
      }else if("xgboost" %in% class(x$model)){
        predict(x$model, newdata = newdata)
      }else if("polyclass" %in% class(x$model)){
        polspline::ppolyclass(cov = newdata, fit = x$model)[, 2]
      }else if("svm" %in% class(x$model)){
        attr(predict(x$model, newdata = newdata, probability = TRUE), "prob")[, "1"]
      }
    }

    # load results for each wrapper
    wrappers <- c("glm", "stepglm", "randomforest", "glmnet")
    rslt <- NULL
    for(w in wrappers){
      # load wrapper results
      eval(parse(text = paste0("out_",w," <- tryCatch({get(load(paste0('~/cvtmleauc/out/out_',",
                               paste0("'n=", parm$n[i],"',"),
                                paste0("'_seed=",parm$seed[i],"',"),
                                paste0("'_K=",parm$K[i],"',"),
                                paste0("'_wrapper=",paste0(w,"_wrapper"),"',"),
                                paste0("'.RData","'"),
                               ")))}, error = function(e){ print(e) })")))
      # fit to full data
      tmp <- do.call(paste0(w,"_wrapper"), args = list(test = dat, train = dat))
      # compute AUC of Psi(P_n)
      big_pred <- my_predict(x = tmp, newdata = bigdat$X)
      true_cvauc <- mean(cvAUC::AUC(predictions = big_pred,
                        labels = bigdat$Y))
      if(!("error" %in% class(eval(parse(text = paste0("out_",w)))))){
        rslt <- rbind(rslt, c(parm[i,], out, true_cvauc))
      }else{
        rslt <- rbind(rslt, c(parm[i,], rep(NA, 20), true_cvauc))
      }
    }
    rslt <- cbind(wrappers, data.frame(rslt))
    colnames(rslt) <- c("learner","seed","n","K","est_dcvtmle", "se_dcvtmle", "iter_dcvtmle",
                        "est_dinit", "est_donestep", "se_donestep",
                        "est_desteq","se_desteq","est_cvtmle","se_cvtmle",
                        "iter_cvtmle","est_init", "est_onestep", "se_onestep",
                        "est_esteq","se_esteq","est_emp","se_emp","true_cvauc",
                        "true_dcvauc","true_auc")

    est <- c("dcvtmle","donestep","desteq","cvtmle","onestep","esteq","emp")
    cvauc_cv_select <- rep(NA, length(est))
    bestcvauc_cv_select <- rep(NA, length(est))
    auc_cv_select <- rep(NA, length(est))
    bestauc_cv_select <- rep(NA, length(est))
    ct <- 0
    bestcvauc <- which.max(rslt$true_cvauc)
    bestauc <- which.max(rslt$true_auc)

    for(e in est){
      ct <- ct + 1
      cv_select_idx <- which.max(rslt[,paste0("est_",e)])
      if(length(cv_select_idx) > 0){
        cvauc_cv_select[ct] <- rslt$true_cvauc[cv_select_idx]      
        auc_cv_select[ct] <- rslt$true_auc[cv_select_idx]
        bestcvauc_cv_select[ct] <- as.numeric(cv_select_idx == bestcvauc)
        bestauc_cv_select[ct] <- as.numeric(cv_select_idx == bestauc)
      }else{
        cvauc_cv_select[ct] <- NA
        auc_cv_select[ct] <- NA
        bestcvauc_cv_select[ct] <- NA
        bestauc_cv_select[ct] <- NA
      }
    }
    names(cvauc_cv_select) <- est
    names(bestcvauc_cv_select) <- est
    names(auc_cv_select) <- est
    names(bestauc_cv_select) <- est

    out <- list(rslt = rslt, 
                cvauc_cv_select = cvauc_cv_select,
                auc_cv_select = auc_cv_select,
                bestcvauc_cv_select = bestcvauc_cv_select,
                bestauc_cv_select = bestauc_cv_select)

    # save output 
    save(out, file = paste0("~/cvtmleauc/out/oracleout_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            ".RData.tmp"))
    file.rename(paste0("~/cvtmleauc/out/oracleout_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],                            
                            ".RData.tmp"),
                paste0("~/cvtmleauc/out/oracleout_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            ".RData"))
  }
}

# merge job ###########################
if (args[1] == 'merge') {   
  ns <- c(100, 250, 500, 750)
  bigB <- 500
  K <- c(5,10,20,40)

  parm <- expand.grid(seed = 1:bigB,
                      n = ns, K = K, 
                      stringsAsFactors = FALSE)
  prev_out <- get(load('~/cvtmleauc/out/allOut_new.RData'))
  full_rslt <- matrix(NA, nrow = 32000, ncol = 25)
  ct <- 4
  for(i in seq_len(nrow(parm))){
    # load result file 
      tmp <- tryCatch({get(load(paste0("~/cvtmleauc/out/oracleout_",
                              "n=", parm$n[i],
                              "_seed=",parm$seed[i],
                              "_K=",parm$K[i],
                              "_wrapper=",parm$wrapper[i],
                              ".RData")))},
                      error = function(e){
                        grbg <- list(rslt = data.frame(matrix(NA, nrow = 4, ncol = 25)))
                        colnames(grbg$rslt) <- colnames(full_rslt)
                        return(grbg)
                      })
      # if(!is.na(tmp$rslt[1,1])){
      #   tmp$rslt$wrapper <- c("glm_wrapper","stepglm_wrapper","randomforest_wrapper",
      #                         "glmnet_wrapper")
      # }
      full_rslt[(ct-3):ct,] <- data.matrix(tmp$rslt)
      ct <- ct + 4
  }
  # ns <- c(100, 250, 500, 750)
  # bigB <- 500
  # K <- c(5,10,20,30)
  # p <- 10
  # parm <- expand.grid(seed=1:bigB,
  #                     n=ns, K = K)
  # rslt <- matrix(NA, nrow = nrow(parm), ncol = 13)
  # for(i in 1:nrow(parm)){
  #     tmp_1 <- tryCatch({
  #         load(paste0("~/cvtmleauc/out/out",
  #                     "_n=", parm$n[i],
  #                     "_seed=",parm$seed[i],
  #                     "_K=", parm$K[i],
  #                     ".RData"))
  #         out
  #     }, error=function(e){
  #       rep(NA, 10)
  #     })
  #     rslt[i,] <- c(parm$seed[i], parm$n[i], parm$K[i], tmp_1)
  # }
  # # # format
  # out <- data.frame(rslt)

  # sim_names <- c("seed","n","K",
  #                "cvtmle","se_cvtmle","iter_cvtmle",
  #                "init",
  #                "onestep","se_onestep",
  #                "empirical","se_empirical",
  #                "truth", "truth_full")
  # colnames(out) <- sim_names
  sim_names <- c("wrapper","seed","n","K",
                 "est_dcvtmle", "se_dcvtmle", "iter_dcvtmle",
                 "est_dinit", "est_donestep", "se_donestep",
                 "est_desteq","se_desteq","est_cvtmle","se_cvtmle",
                 "iter_cvtmle","est_init", "est_onestep", "se_onestep",
                 "est_esteq","se_esteq","est_emp","se_emp","true_cvauc",
                 "true_dcvauc","true_auc")
  colnames(full_rslt) <- sim_names
  out <- data.frame(full_rslt)
  out$wrapper <- c("glm", "stepglm", "randomforest", "glmnet")
  save(full_rslt, file=paste0('~/cvtmleauc/out/allOut_oracles.RData'))
}
# local editing 
if(FALSE){
  load("~/cvtmleauc/out/allOut_oracles.RData")

  get_sim_rslt <- function(out, parm, wrapper, truth = "true_auc",
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
  glm_rslt <- get_sim_rslt(out, parm, wrapper = "glm")
  stepglm_rslt <- get_sim_rslt(out, parm, wrapper = "stepglm")
  randomforest_rslt <- get_sim_rslt(out, parm, wrapper = "randomforest")
  glmnet_rslt <- get_sim_rslt(out, parm, wrapper = "glmnet")
  
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
    pdf(paste0("~/cvtmleauc/",rslt,"_perfvstrueauc.pdf"))
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
}