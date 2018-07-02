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

data_sets <- c("adult", "bank", "cardio", "default",
               "drugs", "magic", "wine")
bigB <- 100
ns <- c(50, 100, 250, 500)
K <- c(5, 10, 20, 40)
wrappers <- c("randomforest_wrapper", "glmnet_wrapper")

parm <- expand.grid(data_set = data_sets,
                    seed = 1:bigB,
                    K = K, 
                    n = ns,
                    wrapper = wrappers, 
                    stringsAsFactors = FALSE)
full_parm <- parm
load("~/cvtmleauc/scratch/redo_parm_tn_realdata.RData")
parm <- redo_parm_tn

library(cvtmleAUC, lib.loc = "/home/dbenkese/R/x86_64-pc-linux-gnu-library/3.4")
library(glmnet)

# get the list size #########
if (args[1] == 'listsize') {
  cat(nrow(parm))
}

# execute prepare job ##################
if (args[1] == 'prepare') {
  parm_red <- full_parm[full_parm$K == full_parm$K[1] & full_parm$wrapper == full_parm$wrapper[1],]
  for(i in 1:nrow(parm_red)){
     set.seed(parm_red$seed[i])
     eval(parse(text = paste0("data(",parm_red$data_set[i],")")))
     dat <- eval(parse(text = parm_red$data_set[i]))
     sum_Y <- 0
     iter <- 0
     while(sum_Y < 5){
      iter <- iter + 1
      train_idx <- sample(seq_len(length(dat[,1])), parm_red$n[i])
      sum_Y <- sum(dat[train_idx,"outcome"])
     }
     # save what observations are in the training sample
     save(train_idx, file=paste0("~/cvtmleauc/scratch/real_data_idx",
                           "_n=",parm_red$n[i],"_data=",parm_red$data_set[i],
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
    options(np.messages = FALSE)

    # load data
    data_suffix <- paste0("n=",parm$n[i],"_data=",parm$data_set[i],
                "_seed=",parm$seed[i], 
                ".RData")

    out_suffix <- paste0("n=", parm$n[i],"_data=",parm$data_set[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            ".RData")

    # load data set from package 
    eval(parse(text = paste0("data(",parm$data_set[i],")")))
    # rename to dat for simplicity
    dat <- eval(parse(text = parm$data_set[i]))
    # load training observations
    load(paste0("~/cvtmleauc/scratch/real_data_idx_", data_suffix))
    # training data
    train_dat <- dat[train_idx, ]
    # test data
    test_dat <- dat[-train_idx, ]
    # column named outcome
    outcome_idx <- which(colnames(dat) == "outcome")

    # get estimates of dcvauc
    # options(np.messages = FALSE)
    n_replicates <- 1
    fitauc_dcv <- vector(mode = "list", length = n_replicates)
    fitauc_cv <- vector(mode = "list", length = n_replicates)    
    fittn_dcv <- vector(mode = "list", length = n_replicates)
    fittn_cv <- vector(mode = "list", length = n_replicates)
    for(j in seq_len(n_replicates)){
      set.seed(j)
      fitauc_dcv[[j]] <- cvauc_cvtmle(Y = train_dat[,outcome_idx], 
                                      X = train_dat[,-outcome_idx], 
                                      K = parm$K[i], 
                          learner = parm$wrapper[i], nested_cv = TRUE,
                          nested_K = 39)
      set.seed(j)
    # get estimates of cvtn
      fitauc_cv[[j]] <- cvauc_cvtmle(Y = train_dat[,outcome_idx], 
                                      X = train_dat[,-outcome_idx], 
                                      K = parm$K[i], 
                          learner = parm$wrapper[i], nested_cv = FALSE,
                          prediction_list = fitauc_dcv$prediction_list[1:parm$K[i]])

      set.seed(j)
    # get estimates of cvtn
      fittn_dcv[[j]] <- cvtn_cvtmle(Y = train_dat[,outcome_idx], 
                                      X = train_dat[,-outcome_idx], 
                                      K = parm$K[i], 
                          learner = parm$wrapper[i], nested_cv = TRUE,
                          prediction_list = fitauc_dcv$prediction_list)      
      set.seed(j)
    # get estimates of cvtn
      fittn_cv[[j]] <- cvtn_cvtmle(Y = train_dat[,outcome_idx], 
                                      X = train_dat[,-outcome_idx], 
                                       K = parm$K[i], 
                          learner = parm$wrapper[i], nested_cv = FALSE,
                          prediction_list = fitauc_cv$prediction_list)
    }

   fit_full <- do.call(parm$wrapper[i], args = list(train = list(X = train_dat[,-outcome_idx], Y = train_dat[,outcome_idx]), 
                            test = list(X = test_dat[,-outcome_idx], Y = test_dat[,outcome_idx])))

   true_auc <- cvAUC::AUC(predictions = fit_full$psi_nBn_testx, labels = test_dat[,outcome_idx])

   bigquantile_full <- quantile(fit_full$psi_nBn_testx[test_dat[,outcome_idx] == 1], p = 0.05, type = 8)
   big_testneg_full <- mean(fit_full$psi_nBn_testx <= bigquantile_full)
   true_tn <- big_testneg_full

    # bootstrap estimate 
    # only needed for K = 5 runs
    # and will be put back in later
    if(parm$K[i] == 5){
      set.seed(parm$seed[i])
      # fit_boot <- boot_corrected_auc(Y = train_dat[,outcome_idx], X = train_dat[,-outcome_idx], learner = parm$wrapper[i])
      # fit_boot <- boot_corrected_tn(Y = train_dat[,outcome_idx], X = train_dat[,-outcome_idx], learner = parm$wrapper[i])
      fit_lpo <- leave_pair_out_auc(Y = train_dat[,outcome_idx], X = train_dat[,-outcome_idx], learner = parm$wrapper[i])
    }else{
      fit_boot <- list(NA)
      fit_lpo <- list(NA)
    }

    # c together output
    for(param in c("auc","tn")){
      eval(parse(text = paste0(
        "out_",param," <- ", 
        'c( # cvtmle estimates of dcvauc
             fit',param,'_dcv[[1]]$est_cvtmle, fit',param,'_dcv[[1]]$se_cvtmle,
             # iterations of cvtmle for dcv
             # fit',param,'_dcv[[1]]$iter, 
             # initial plug-in estimate of dcv
             fit',param,'_dcv[[1]]$est_init, 
             # one-step estimate of dcv
             fit',param,'_dcv[[1]]$est_onestep, fit',param,'_dcv[[1]]$se_onestep,
             # estimating eqn estimate of dcv
             fit',param,'_dcv[[1]]$est_esteq, fit',param,'_dcv[[1]]$se_esteq,
             # cvtmle estimate of cv
             fit',param,'_cv[[1]]$est_cvtmle, fit',param,'_cv[[1]]$se_cvtmle, 
             # iterations of cvtmle for cv
             # fit',param,'_cv[[1]]$iter, 
             fit',param,'_cv[[1]]$est_init, 
             # one-step estimate of cv
             fit',param,'_cv[[1]]$est_onestep, fit',param,'_cv[[1]]$se_onestep,
             # estimating eqn estimate of cv
             fit',param,'_cv[[1]]$est_esteq, fit',param,'_cv[[1]]$se_esteq,
             # full sample split estimate of cv
             fit',param,'_dcv[[1]]$est_empirical, fit',param,'_dcv[[1]]$se_empirical)'
      )))
    }
    
  #   # now add in MC averaged results for M = 5, 10, 20
  #   for(M in c(5, 10, 20)){
  #     for(param in c("auc","tn")){
  #       this_dcvfit <- if(param == "auc"){
  #         fitauc_dcv
  #       }else{
  #         fittn_dcv
  #       }        
  #       this_cvfit <- if(param == "auc"){
  #         fittn_dcv
  #       }else{
  #         fittn_dcv
  #       }
  #     avg_dcv <- .getMCAveragedResults(this_dcvfit[1:M], logit = FALSE)
  #     avg_cv <- .getMCAveragedResults(this_cvfit[1:M], logit = FALSE)
  #     eval(parse(text = paste0('out_',param,' <- c(out_',param,',',
  #                              'avg_dcv$est_cvtmle, avg_dcv$se_cvtmle,
  #              # initial plug-in estimate of dcv
  #              avg_dcv$est_init, 
  #            # one-step estimate of dcv
  #            avg_dcv$est_onestep, avg_dcv$se_onestep,
  #            # estimating eqn estimate of dcv
  #            avg_dcv$est_esteq, avg_dcv$se_esteq,
  #            # cvtmle estimate of cv
  #            avg_cv$est_cvtmle, avg_cv$se_cvtmle, 
  #            # iterations of cvtmle for cv
  #            avg_cv$est_init, 
  #            # one-step estimate of cv
  #            avg_cv$est_onestep, avg_cv$se_onestep,
  #            # estimating eqn estimate of cv
  #            avg_cv$est_esteq, avg_cv$se_esteq,
  #            # full sample split estimate of cv
  #            avg_dcv$est_empirical, avg_dcv$se_empirical)')))
  #   }
  # }
    out_auc <- c(out_auc, fit_lpo[[1]], true_auc)
    out_tn <- c(out_tn, true_tn)

    # save output 
    save(out_auc, file = paste0("~/cvtmleauc/out/realdataaucout_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            "_data=",parm$data_set[i],
                            ".RData.tmp"))
    file.rename(paste0("~/cvtmleauc/out/realdataaucout_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            "_data=",parm$data_set[i],                            
                            ".RData.tmp"),
                paste0("~/cvtmleauc/out/realdataaucout_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            "_data=",parm$data_set[i],
                            ".RData"))    
    save(out_tn, file = paste0("~/cvtmleauc/out/realdatatnout_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            "_data=",parm$data_set[i],
                            ".RData.tmp"))
    file.rename(paste0("~/cvtmleauc/out/realdatatnout_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            "_data=",parm$data_set[i],                            
                            ".RData.tmp"),
                paste0("~/cvtmleauc/out/realdatatnout_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            "_data=",parm$data_set[i],
                            ".RData"))
    grbg <- c(NULL)
    save(grbg, file = paste0("~/cvtmleauc/out/",i,"-run_realdata.dat")) 
  }
}

# merge job ###########################
if (args[1] == 'merge') {   

  data_sets <- c("adult", "bank", "cardio", "default",
                 "drugs", "magic", "wine")
  bigB <- 100
  ns <- c(50, 100, 250, 500)
  K <- c(5, 10, 20, 40)
  wrappers <- c("randomforest_wrapper", "glmnet_wrapper")

  parm <- expand.grid(data_set = data_sets,
                    seed = 1:bigB,
                    K = K, 
                    n = ns,
                    wrapper = wrappers, 
                    stringsAsFactors = FALSE)
  rslt_tn <- matrix(NA, nrow = nrow(parm), ncol = 17 +5)
  rslt_auc <- matrix(NA, nrow = nrow(parm), ncol = 18 +5)
  redo_parm_tn <- redo_parm_auc <- NULL

  for(i in 1:nrow(parm)){
    if(file.exists(paste0("~/cvtmleauc/out/realdatatnout_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            "_data=",parm$data_set[i],
                            ".RData"))){
      tmp_1 <- get(load(paste0("~/cvtmleauc/out/realdatatnout_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            "_data=",parm$data_set[i],
                            ".RData")))
    }else{
      redo_parm_tn <- rbind(redo_parm_tn, parm[i,])
      tmp_1 <- rep(NA, 17)
    }
    rslt_tn[i,] <- c(parm$seed[i], parm$n[i], parm$K[i], parm$wrapper[i], 
                     parm$data_set[i], tmp_1)
    # now for cvauc
    if(file.exists(paste0("~/cvtmleauc/out/realdataaucout_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            "_data=",parm$data_set[i],
                            ".RData"))){
      tmp_1 <- get(load(paste0("~/cvtmleauc/out/realdataaucout_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            "_data=",parm$data_set[i],
                            ".RData")))
    }else{
      redo_parm_auc <- rbind(redo_parm_auc, parm[i,])
      tmp_1 <- rep(NA, 18)
    }
    rslt_auc[i,] <- c(parm$seed[i], parm$n[i], parm$K[i], parm$wrapper[i], 
                     parm$data_set[i], tmp_1)

    if(is.na(tmp_1[length(tmp_1) - 1])){
      if(file.exists(paste0("~/cvtmleauc/out/outtn_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=5",
                            "_wrapper=",parm$wrapper[i],
                            "_data=",parm$data_set[i],
                            ".RData"))){
      lpo_rslt_out <- get(load(paste0("~/cvtmleauc/out/outtn_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=5",
                            "_wrapper=",parm$wrapper[i],
                            ".RData")))
      lpo_idx <- length(lpo_rslt_out) - 1
      rslt_auc[i,lpo_idx] <- lpo_rslt_out[lpo_idx]
      }
    }
  }

  # # format
  out_tn <- data.frame(rslt_tn, stringsAsFactors = FALSE)
  out_auc <- data.frame(rslt_auc, stringsAsFactors = FALSE)

  repeat_names <- c("est_dcvtmle", "se_dcvtmle", 
                 "est_dinit", "est_donestep", "se_donestep",
                 "est_desteq","se_desteq","est_cvtmle","se_cvtmle",
                 "est_init", "est_onestep", "se_onestep",
                 "est_esteq","se_esteq","est_emp","se_emp")
  repeat_names2 <- c("est_dcvtmle", "se_dcvtmle","est_dinit",
                  "est_donestep", "se_donestep",
                 "est_desteq","se_desteq","est_cvtmle","se_cvtmle",
                 "est_init", "est_onestep", "se_onestep",
                 "est_esteq","se_esteq","est_emp","se_emp")

  sim_names_tn <- c("seed","n","K","wrapper","data",
                 paste0(repeat_names, 1),
                 "truth")  
  sim_names_auc <- c("seed","n","K","wrapper","data",
                 paste0(repeat_names, 1),"lpo",
                 "truth")  

  colnames(out_tn) <- sim_names_tn
  colnames(out_auc) <- sim_names_auc

  out_tn[,c(1:3,6:ncol(out_tn))] <- apply(out_tn[,c(1:3,6:ncol(out_tn))], 2, function(y){
    as.numeric(as.character(y))})
  out_auc[,c(1:3,6:ncol(out_auc))] <- apply(out_auc[,c(1:3,6:ncol(out_auc))], 2, function(y){
    as.numeric(as.character(y))})

  save(out_tn, file=paste0('~/cvtmleauc/out/allOut_cvtn_realdata.RData'))
  save(out_auc, file=paste0('~/cvtmleauc/out/allOut_cvauc_realdata.RData'))
  save(redo_parm_tn, file = "~/cvtmleauc/scratch/redo_parm_tn_realdata.RData")
  save(redo_parm_auc, file = "~/cvtmleauc/scratch/redo_parm_auc_realdata.RData")
}

if(FALSE){
  setwd("~/Dropbox/R/cvtmleauc/sandbox/simulation/")
  load("allOut_cvtn_realdata.RData")
  load("allOut_cvauc_realdata.RData")
  colnames(out_auc)[ncol(out_auc)-1] <- "est_lpo"
    get_sim_rslt <- function(out, parm, wrapper, truth = "truth",
                             estimators = c("dcvtmle1","donestep1",
                                     "cvtmle1","onestep1","emp1"
                                     ), ...){
      b <- bp <- v <- m <- cv <- co <- mad <- NULL
      for(i in seq_len(length(parm[,1]))){
        x <- out[out$n == parm$n[i] & out$K == parm$K[i] & out$wrapper == wrapper,]
        b <- rbind(b, colMeans(x[,paste0("est_",estimators)] - x[,truth], na.rm = TRUE))
        bp <- rbind(bp, colMeans((x[,paste0("est_",estimators)] - x[,truth])/x[,truth], na.rm = TRUE))
        v <- rbind(v, apply(x[,paste0("est_",estimators)], 2, var, na.rm = TRUE))
        cv <- rbind(cv, apply(x[,paste0("est_",estimators)], 2, function(y){
          sd(y, na.rm = TRUE) / mean(y, na.rm = TRUE)
        }))
        m <- rbind(m, colMeans((x[,paste0("est_",estimators)] - as.numeric(x[,truth]))^2, na.rm = TRUE))
        mad <- rbind(mad, apply(x[,paste0("est_",estimators)], 2, function(y){ 
          median(y - as.numeric(x[,truth]), na.rm = TRUE)
        }))
        # coverage
        # coverage <- rep(NA, length(estimators))
        # ct <- 0
        # for(est in estimators){
        #   ct <- ct + 1
        #   coverage[ct] <- mean(x[,paste0("est_",est)] - 1.96 * x[,paste0("se_",est)] < x[,truth] & 
        #                   x[,paste0("est_",est)] + 1.96 * x[,paste0("se_",est)] > x[,truth], na.rm = TRUE)
        # }
        # co <- rbind(co, coverage)
      }
      parm <- cbind(parm, b, v, m, mad, cv, bp) #, 
                    # co)
      colnames(parm) <- c("n", "K", paste0("bias_", estimators),
                          paste0("var_", estimators),
                          paste0("mse_", estimators),
                          paste0("mad_", estimators),
                          paste0("cv_", estimators),
                          paste0("bp_", estimators)
                          ) #,
                          # paste0("cov_", estimators))
      return(parm)
    }
    parm <- expand.grid(n = c(50, 100, 250, 500),
                        K = c(5, 10, 20, 40))
    glm_rslt <- get_sim_rslt(out_tn, parm, wrapper = "glmnet_wrapper")
    glm_rslt_auc <- get_sim_rslt(out_auc, parm, wrapper = "glmnet_wrapper",
                                 estimators = c("dcvtmle1","donestep1",
                                                             "desteq1","cvtmle1","onestep1",
                                                             "esteq1",
                                                             "emp1","lpo"))
    randomforest_rslt <- get_sim_rslt(out_tn, parm, wrapper = "randomforest_wrapper")
    randomforest_rslt_auc <- get_sim_rslt(out_auc, parm, wrapper = "randomforest_wrapper",
                                          estimators = c("dcvtmle1","donestep1",
                                                             "desteq1","cvtmle1","onestep1",
                                                             "esteq1",
                                                             "emp1","lpo"))
    

    # box plots 
    make_one_box_plot_row <- function(rslt, metric = "bias", 
                                      est, add_legend = FALSE, 
                                      est_labels = c("CVEMP", "CVTMLE","CVOS","BOOT"),
                                      rm_last = TRUE,
                                      leg_x = "topright", leg_y = NULL, 
                                      grid = FALSE, scale = 1, 
                                      add_text = FALSE, 
                                      nx_grid = NULL, ny_grid = BULL, 
                                      pred_algo = "Logistic regression",
                                            yaxis_label, add_ratio = TRUE, 
                                            xaxis_label = "Number CV Folds",
                                            transpose = FALSE, 
                                            log = "y", print_n = FALSE,
                                            relative_est = NULL,
                                            relative_K = NULL, 
                                            absolute_val = TRUE, col, ...){
      for(n in c(50, 100, 250, 500)){
        par(las = 1)
        tmp <- rslt[rslt$n == n, ]
        tmp5 <- tmp[tmp$K == 5,]
        tmp10 <- tmp[tmp$K == 10,]
        tmp20 <- tmp[tmp$K == 20,]
        tmp40 <- tmp[tmp$K == 40,]
        # est <- c("mse_dcvtmle1","mse_donestep1","mse_emp1","mse_bootstrap")
        grbg <- as.matrix(rbind(tmp5[,est],tmp10[,est],tmp20[,est],tmp40[,est]))
        if(absolute_val){
          grbg <- abs(grbg)
        }
        if(rm_last){
          grbg[2:4,4] <- NA
        }
        row.names(grbg) <- c(5,10,20,40)
        if(n == 50){
          leg.text <- est_labels
          sp <- c(0,0,0,0,0,0,0,0,1,0,0,0)
        }else{
          leg.text <- FALSE
          sp <- c(0,0,0,0,1,0,0,0,1,0,0,0)
        }
        xl <- c(0, 12 + sum(sp))
        if(transpose){
          grbg <- t(grbg)
        }
        if(!is.null(relative_est)){
          grbg <- grbg / grbg[paste(relative_K),relative_est]
        }
        grbg <- grbg * scale
        txt <- c(5,10,20,40)
        if(n == 50) txt <- c(5,10,20,"",5,10,20,40,5,10,20,40)
      
        tmp <- barplot(grbg, legend.text = FALSE, 
          beside=TRUE, log = log, yaxt = "n", names.arg = rep("",3), col = col, 
          space = sp, xlim = xl, ... )

        if(grid){
          grid(nx = nx_grid, ny = ny_grid, lty = 1, col = "gray75", equilogs = FALSE)
          par(new = TRUE)
          tmp <- barplot(grbg, legend.text = FALSE, 
          beside=TRUE, log = log, yaxt = "n", names.arg = rep("",3), col = col, 
          space = sp, xlim = xl, ... )
        }
        mtext(side = 1, outer = FALSE, line = 0.02, text = txt, cex =0.5, 
        at = c(tmp[,1],tmp[,2],tmp[,3]))
      
        mtext(side = 1, outer = FALSE, line = 0.02, text = "K = ", cex =0.5, 
              at = par()$usr[1])
        if(add_text){
          # grbg2 <- format(grbg, digits = 2, zero.print = TRUE, scientific = FALSE)
          grbg2 <- formatC(grbg, digits = 2, format = "f")
          grbg2[grbg > 10] <- ">10"
          grbg2[grepl("N",grbg2)] <- ""          
          text(x = tmp, y = 2.2*10^par()$usr[3], srt = 90, grbg2)
               # adj = c(0,0.001),
        }
        if(n == 50 & add_legend){
          legend(x = leg_x, y= leg_y, xpd = TRUE, fill = unique(col), legend = leg.text, ncol = 3)
        }
      
        if(n == 50){
          axis(side = 2)
          par(las = 0)
          mtext(outer = FALSE, side = 2, line = 3, yaxis_label, cex = 0.75)
          # mtext(outer = FALSE, side = 2, line = 4.5, pred_algo, cex = 1)
        }else{
          par(las = 0)
          axis(side = 2, labels = FALSE)
        }
        if(print_n){
          mtext(outer = FALSE, side = 3, line = 1.5, paste0("n = ", n))
        }
        if(add_ratio){
        }
        if(!is.null(relative_est)){
          abline(h = 1, lty = 3)
        }
      }
    }

    library(RColorBrewer)
    my_col <- brewer.pal(9,"Greys")[c(1,2,3,4,5,6)]


    pdf("~/Dropbox/Emory/cross-validated-prediction-metrics/realdata_cvtn.pdf",
        height = 6*2/3, width = 11)
    layout(matrix(1:8, nrow = 2, ncol = 4,  byrow = TRUE))
    par(mar = c(1.6, 0.6, 1.6, 0.6), mgp = c(2.1, 0.5, 0),
        oma = c(2.1, 7.1, 2.1, 2.1))
    make_one_box_plot_row(rslt = glm_rslt, rm_last = FALSE, print_n = TRUE,
                              est = c("mse_emp1","mse_dcvtmle1",
                                            "mse_donestep1"),
                                relative_est = "mse_emp1",
                                relative_K = 5,    add_text = TRUE,
                                grid = TRUE, ny_grid = NULL, nx_grid = 0,                           
                                    est_label = c("CVEMP", "CVTMLE","CVOS"),
                                    ylim = c(0.01,5), yaxis_label = "Relative MSE",
                                    col = my_col[sort(rep(c(1:3,5),4))])
    mtext(side = 2, line = 5, "LASSO", outer = TRUE, at = 0.75)
    make_one_box_plot_row(rslt = randomforest_rslt,  
                          est = c("mse_emp1","mse_dcvtmle1",
                                        "mse_donestep1"),
                            relative_est = "mse_emp1",
                            relative_K = 5,add_text = TRUE,rm_last = FALSE, 
                            grid = TRUE, ny_grid = NULL, nx_grid = 0,                           
                                est_label = c("CVEMP", "CVTMLE","CVOS"),
                                ylim = c(0.01,5), yaxis_label = "Relative MSE",
                                col = my_col[sort(rep(c(1:3,5),4))])
    mtext(side = 2, line = 5, "Random forest", outer = TRUE, at = 0.25)
    dev.off()

    # pdf("~/Dropbox/Emory/cross-validated-prediction-metrics/rf_realdata_cvtn.pdf",
    #     height = 6, width = 11)
    # layout(matrix(1:12, nrow = 3, ncol = 4,  byrow = TRUE))
    # par(mar = c(1.6, 0.6, 1.6, 0.6), mgp = c(2.1, 0.5, 0),
    #     oma = c(2.1, 5.1, 2.1, 2.1))
    # make_one_box_plot_row(rslt = randomforest_rslt, print_n = TRUE, 
    #                       est = c("bp_emp1","bp_dcvtmle1",
    #                                     "bp_donestep1"),
    #                       leg_x = 0, leg_y = 500,
    #                       grid = TRUE, ny_grid = NULL, nx_grid = 0,                           
    #                       add_legend = TRUE, rm_last = FALSE, 
    #                             est_label = c("CVEMP", "CVTMLE","CVOS"),
    #                             ylim = c(0.000001,100), yaxis_label = "Absolute Bias",
    #                             col = my_col[sort(rep(c(1:3,5),4))])
    # make_one_box_plot_row(rslt = randomforest_rslt, 
    #                             grid = TRUE, ny_grid = NULL, nx_grid = 0,                           
    #                           est = c("cv_emp1","cv_dcvtmle1",
    #                                         "cv_donestep1"), rm_last = FALSE, 
    #                                 est_label = c("CVEMP", "CVTMLE","CVOS"),
    #                                 ylim = c(0.000001,10), yaxis_label = "Coefficient of variation",
    #                                 col = my_col[sort(rep(c(1:3,5),4))])

    # dev.off()


    pdf("~/Dropbox/Emory/cross-validated-prediction-metrics/rf_realdata_cvtn_bydata.pdf",
                height = 6, width = 11)
    for(d in unique(out_tn$data)){
      randomforest_rslt_d <- get_sim_rslt(out_tn[out_tn$data == d,], parm, wrapper = "randomforest_wrapper")

      layout(matrix(1:12, nrow = 3, ncol = 4,  byrow = TRUE))
      par(mar = c(1.6, 0.6, 1.6, 0.6), mgp = c(2.1, 0.5, 0),
          oma = c(2.1, 5.1, 4.1, 2.1))
      make_one_box_plot_row(rslt = randomforest_rslt_d, print_n = TRUE, 
                            est = c("bp_emp1","bp_dcvtmle1",
                                          "bp_donestep1"),
                            leg_x = 0, leg_y = 500,
                            grid = TRUE, ny_grid = NULL, nx_grid = 0,                           
                            add_legend = TRUE, rm_last = FALSE, 
                                  est_label = c("CVEMP", "CVTMLE","CVOS"),
                                  ylim = c(0.000001,100), yaxis_label = "Absolute Bias",
                                  col = my_col[sort(rep(c(1:3,5),4))])
      mtext(side = 3, outer = TRUE, line = 2.5, d)
      make_one_box_plot_row(rslt = randomforest_rslt_d, 
                                  grid = TRUE, ny_grid = NULL, nx_grid = 0,                           
                                est = c("cv_emp1","cv_dcvtmle1",
                                              "cv_donestep1"), rm_last = FALSE, 
                                      est_label = c("CVEMP", "CVTMLE","CVOS"),
                                      ylim = c(0.000001,10), yaxis_label = "Coefficient of variation",
                                      col = my_col[sort(rep(c(1:3,5),4))])
      make_one_box_plot_row(rslt = randomforest_rslt_d, 
                                est = c("mse_emp1","mse_dcvtmle1",
                                              "mse_donestep1"),
                                  relative_est = "mse_emp1",
                                  relative_K = 5,add_text = TRUE,rm_last = FALSE, 
                                  grid = TRUE, ny_grid = NULL, nx_grid = 0,                           
                                      est_label = c("CVEMP", "CVTMLE","CVOS"),
                                      ylim = c(0.01,5), yaxis_label = "Relative MSE",
                                      col = my_col[sort(rep(c(1:3,5),4))])
    }
    dev.off()

  pdf("~/Dropbox/Emory/cross-validated-prediction-metrics/glm_realdata_cvtn_bydata.pdf",
                height = 6, width = 11)
    for(d in unique(out_tn$data)){
      glm_rslt_d <- get_sim_rslt(out_tn[out_tn$data == d,], parm, wrapper = "glmnet_wrapper")

      layout(matrix(1:12, nrow = 3, ncol = 4,  byrow = TRUE))
      par(mar = c(1.6, 0.6, 1.6, 0.6), mgp = c(2.1, 0.5, 0),
          oma = c(2.1, 5.1, 4.1, 2.1))
      make_one_box_plot_row(rslt = glm_rslt_d, print_n = TRUE, 
                            est = c("bp_emp1","bp_dcvtmle1",
                                          "bp_donestep1"),
                            leg_x = 0, leg_y = 500,
                            grid = TRUE, ny_grid = NULL, nx_grid = 0,                           
                            add_legend = TRUE, rm_last = FALSE, 
                                  est_label = c("CVEMP", "CVTMLE","CVOS"),
                                  ylim = c(0.000001,100), yaxis_label = "Absolute Bias",
                                  col = my_col[sort(rep(c(1:3,5),4))])
      mtext(side = 3, outer = TRUE, line = 2.5, d)
      make_one_box_plot_row(rslt = glm_rslt_d, 
                                  grid = TRUE, ny_grid = NULL, nx_grid = 0,                           
                                est = c("cv_emp1","cv_dcvtmle1",
                                              "cv_donestep1"), rm_last = FALSE, 
                                      est_label = c("CVEMP", "CVTMLE","CVOS"),
                                      ylim = c(0.000001,10), yaxis_label = "Coefficient of variation",
                                      col = my_col[sort(rep(c(1:3,5),4))])
      make_one_box_plot_row(rslt = glm_rslt_d, 
                                est = c("mse_emp1","mse_dcvtmle1",
                                              "mse_donestep1"),
                                  relative_est = "mse_emp1",
                                  relative_K = 5,add_text = TRUE,rm_last = FALSE, 
                                  grid = TRUE, ny_grid = NULL, nx_grid = 0,                           
                                      est_label = c("CVEMP", "CVTMLE","CVOS"),
                                      ylim = c(0.01,5), yaxis_label = "Relative MSE",
                                      col = my_col[sort(rep(c(1:3,5),4))])
    }
    dev.off()


    make_one_box_plot_row_auc <- function(rslt, metric = "bias", 
                                  est, add_legend = FALSE, 
                                  est_labels = c("CVEMP", "CVTMLE","CVOS"),
                                  rm_last = TRUE,
                                  grid = FALSE, 
                                  scale = 1, mult =1.15, 
                                  leg_y = NULL, 
                                  leg_x = "topleft",
                                  add_text = FALSE, 
                                  nx_grid = NULL, ny_grid = BULL, 
                                  pred_algo = "Logistic regression",
                                        yaxis_label, add_ratio = TRUE, 
                                        xaxis_label = "Number CV Folds",
                                        transpose = FALSE, 
                                        log = "y", print_n = FALSE,
                                        relative_est = NULL,
                                        relative_K = NULL, 
                                        absolute_val = TRUE, col, ...){
      for(n in c(50, 100, 250, 500)){
          par(las = 0)
        tmp <- rslt[rslt$n == n, ]
        tmp5 <- tmp[tmp$K == 5,]
        tmp10 <- tmp[tmp$K == 10,]
        tmp20 <- tmp[tmp$K == 20,]
        tmp40 <- tmp[tmp$K == 40,]
        # est <- c("mse_dcvtmle1","mse_donestep1","mse_emp1","mse_bootstrap")
        grbg <- as.matrix(rbind(tmp5[,est],tmp10[,est],tmp20[,est],tmp40[,est]))
        if(absolute_val){
          grbg <- abs(grbg)
        }
        if(rm_last){
          grbg[2:4,5] <- NA
          if(n==50){
            grbg[3:4,1] <- NA
          }else if(n == 100){
            grbg[4,1] <- NA
          }
          # grbg[2:4,6] <- NA
        }
        row.names(grbg) <- c(5,10,20,40)

        if(!is.null(relative_est)){
          grbg <- grbg / grbg[paste(relative_K),relative_est]
        }
        if(n == 50){
          leg.text <- est_labels
          sp <- c(0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0)
        }else{
          leg.text <- FALSE
          sp <- c(0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0)
        }
        txt <- c(5,10,20,40)
        if(n == 50) txt <- c(5,10,"","",5,10,20,40,5,10,20,40,5,10,20,40)
        if(n == 100) txt <- c(5,10,20,"",5,10,20,40,5,10,20,40,5,10,20,40)
        grbg <- grbg * scale
        xl <- c(0, 17 + sum(sp))
        tmp <- barplot(grbg, legend.text = FALSE,
                       beside=TRUE, log = "y", yaxt = "n", 
                       names.arg = rep("",dim(grbg)[2]), 
                       col = col, space = sp, xlim = xl, ... )
        mtext(side = 1, outer = FALSE, line = 0.02, text = txt, cex =0.5, 
              at = c(tmp[,1],tmp[,2],tmp[,3],tmp[,4]))
        mtext(side = 1, outer = FALSE, line = 0.02, text = "K = ", cex =0.5, 
              at = par()$usr[1])
        if(grid){
          grid(nx = nx_grid, ny = ny_grid, lty = 1, col = "gray75", equilogs = FALSE)
          par(new = TRUE)
          tmp <- barplot(grbg, legend.text = FALSE, space = sp, xlim = xl, 
          beside=TRUE, log = log, yaxt = "n", names.arg = rep("",dim(grbg)[2]), col = col, 
          ... )
        }
        if(add_text){
          # grbg2 <- format(grbg, digits = 2, zero.print = TRUE, scientific = FALSE)
          grbg2 <- formatC(grbg, digits = 2, format = "f")
          grbg2[grbg > 10] <- ">10"
          grbg2[grepl("N",grbg2)] <- ""          
          text(x = tmp, y = mult*10^par()$usr[3], srt = 90, grbg2)
               # adj = c(0,0.001),
        }
        if(add_legend){
          if(n == 50){
            legend(x = leg_x, y = leg_y, xpd = TRUE, fill = unique(col), legend = leg.text, ncol = 2)
          }
        }
        if(n == 50){
          par(las = 2)
          axis(side = 2)
          par(las = 0)
          mtext(outer = FALSE, side = 2, line = 3, yaxis_label, cex = 0.75)
          # mtext(outer = FALSE, side = 2, line = 4, "Logistic Regression", cex = 1)
        }else{
          par(las = 2)
          axis(side = 2, labels = FALSE)
        }
        if(print_n){
          par(las = 0)
          mtext(outer = FALSE, side = 3, line = 1.5, paste0("n = ", n))
        }        
        if(!is.null(relative_est)){
          abline(h = 1, lty = 3)
        }
      }
    }

    pdf("~/Dropbox/Emory/cross-validated-prediction-metrics/realdata_cvauc.pdf",
        height = 6*2/3, width = 12)
    layout(matrix(1:8, nrow = 2, ncol = 4,  byrow = TRUE))
    par(mar = c(1.6, 0.6, 1.6, 0.6), mgp = c(2.1, 0.5, 0),
        oma = c(2.1, 7.1, 2.1, 2.1))

    make_one_box_plot_row_auc(rslt = glm_rslt_auc, print_n = TRUE,
                              est = c("mse_emp1","mse_dcvtmle1",
                                            "mse_donestep1","mse_desteq1", "mse_lpo"),
                                relative_est = "mse_emp1", 
                                relative_K = 5,add_text = TRUE,mult = 1.4,
                                grid = TRUE, ny_grid = NULL, nx_grid = 0,                           
                                    est_label = c("CVEMP", "CVTMLE","CVOS","CVEE","LPO"),
                                    ylim = c(0.2,5), yaxis_label = "Relative MSE",
                                    col = my_col[sort(rep(c(1:4,6),4))])
    par(las = 0)
    mtext(side = 2, line = 5, "LASSO", outer = TRUE, at = 0.75)

    make_one_box_plot_row_auc(rslt = randomforest_rslt_auc, 
                              est = c("mse_emp1","mse_dcvtmle1",
                                        "mse_donestep1","mse_desteq1","mse_lpo"),
                                relative_est = "mse_emp1",rm_emp = TRUE, 
                                relative_K = 5,add_text = TRUE, mult = 1.45,
                                grid = TRUE, ny_grid = NULL, nx_grid = 0,                      
                                    est_label = c("CVEMP", "CVTMLE","CVOS","CVEE","BOOT","LPO"),
                                    ylim = c(0.2,5), yaxis_label = "Relative MSE",
                                    col = my_col[sort(rep(1:6,4))])
    par(las = 0)
    mtext(side = 2, line = 5, "Random forest", outer = TRUE, at = 0.25)
    dev.off()


    pdf("~/Dropbox/Emory/cross-validated-prediction-metrics/rf_realdata_cvauc.pdf",
        height = 6, width = 12)
    layout(matrix(1:12, nrow = 3, ncol = 4,  byrow = TRUE))
    par(mar = c(1.6, 0.6, 1.6, 0.6), mgp = c(2.1, 0.5, 0),
        oma = c(2.1, 5.1, 2.1, 2.1))
    make_one_box_plot_row_auc(rslt = randomforest_rslt_auc, print_n = TRUE, 
                              est = c("bp_emp1","bp_dcvtmle1",
                                        "bp_donestep1","bp_desteq1","bp_lpo"),
                          grid = TRUE, ny_grid = NULL, nx_grid = 0,                      
                          add_legend = TRUE, scale = 100,  leg_x = 0, leg_y = 150, 
                                est_label = c("CVEMP", "CVTMLE","CVOS","CVEE","BOOT","LPO"),
                                ylim = c(1e-2,50), yaxis_label = "Absolute Bias (%)",
                                col = my_col[sort(rep(1:6,4))])
    make_one_box_plot_row_auc(rslt = randomforest_rslt_auc, 
                                grid = TRUE, ny_grid = NULL, nx_grid = 0,                      
                              est = c("cv_emp1","cv_dcvtmle1",
                                        "cv_donestep1","cv_desteq1","cv_lpo"),
                                    est_label = c("CVEMP", "CVTMLE","CVOS","CVEE","BOOT","LPO"),
                                    ylim = c(0.05,0.2), yaxis_label = "Coefficient of variation",
                                    col = my_col[sort(rep(1:6,4))])
    make_one_box_plot_row_auc(rslt = randomforest_rslt_auc, 
                              est = c("mse_emp1","mse_dcvtmle1",
                                        "mse_donestep1","mse_desteq1","mse_lpo"),
                                relative_est = "mse_emp1",
                                relative_K = 5,add_text = TRUE,
                                grid = TRUE, ny_grid = NULL, nx_grid = 0,                      
                                    est_label = c("CVEMP", "CVTMLE","CVOS","CVEE","BOOT","LPO"),
                                    ylim = c(0.5,1.5), yaxis_label = "Relative MSE",
                                    col = my_col[sort(rep(1:6,4))])
    dev.off()


    pdf("~/Dropbox/Emory/cross-validated-prediction-metrics/rf_realdata_cvauc_bydata.pdf",
                height = 6, width = 11)
    for(d in unique(out_auc$data)){
      randomforest_rslt_d <- get_sim_rslt(out_auc[out_auc$data == d,], parm, 
                                          wrapper = "randomforest_wrapper",
                                          estimator = c("dcvtmle1","donestep1",
                                                             "desteq1","cvtmle1","onestep1",
                                                             "esteq1",
                                                             "emp1","lpo"))

      layout(matrix(1:12, nrow = 3, ncol = 4,  byrow = TRUE))
      par(mar = c(1.6, 0.6, 1.6, 0.6), mgp = c(2.1, 0.5, 0),
          oma = c(2.1, 5.1, 4.1, 2.1))
      make_one_box_plot_row_auc(rslt = randomforest_rslt_d, print_n = TRUE, 
                            est = c("bp_emp1","bp_dcvtmle1",
                                          "bp_donestep1","bp_desteq1","bp_lpo"),
                            leg_x = 0, leg_y = 500, mult = 1.8,
                            grid = TRUE, ny_grid = NULL, nx_grid = 0,                           
                            add_legend = TRUE, rm_last = FALSE, 
                                  est_label = c("CVEMP", "CVTMLE","CVOS","LPO"),
                                  ylim = c(0.000001,100), yaxis_label = "Absolute Bias",
                                  col = my_col[sort(rep(c(1:4,6),4))])
      mtext(side = 3, outer = TRUE, line = 2.5, d)
      make_one_box_plot_row_auc(rslt = randomforest_rslt_d, 
                                  grid = TRUE, ny_grid = NULL, nx_grid = 0,mult = 1.8,
                                est = c("cv_emp1","cv_dcvtmle1",
                                              "cv_donestep1","cv_desteq1","cv_lpo"), rm_last = FALSE, 
                                      est_label = c("CVEMP", "CVTMLE","CVOS","LPO"),
                                      ylim = c(0.000001,10), yaxis_label = "Coefficient of variation",
                                      col = my_col[sort(rep(c(1:4,6),4))])
      make_one_box_plot_row_auc(rslt = randomforest_rslt_d, 
                                est = c("mse_emp1","mse_dcvtmle1",
                                              "mse_donestep1","mse_desteq1","mse_lpo"),
                                  relative_est = "mse_emp1",mult = 1.8,
                                  relative_K = 5,add_text = TRUE,rm_last = FALSE, 
                                  grid = TRUE, ny_grid = NULL, nx_grid = 0,                           
                                      est_label = c("CVEMP", "CVTMLE","CVOS","LPO"),
                                      ylim = c(0.01,5), yaxis_label = "Relative MSE",
                                      col = my_col[sort(rep(c(1:4,6),4))])
    }
    dev.off()

  pdf("~/Dropbox/Emory/cross-validated-prediction-metrics/glm_realdata_cvauc_bydata.pdf",
                height = 6, width = 11)
    for(d in unique(out_auc$data)){
      glm_rslt_d <- get_sim_rslt(out_auc[out_auc$data == d,], 
                                 parm, 
                                 estimator =c("dcvtmle1","donestep1",
                                                             "desteq1","cvtmle1","onestep1",
                                                             "esteq1",
                                                             "emp1","lpo"),
                                 wrapper = "glmnet_wrapper")

      layout(matrix(1:12, nrow = 3, ncol = 4,  byrow = TRUE))
      par(mar = c(1.6, 0.6, 1.6, 0.6), mgp = c(2.1, 0.5, 0),
          oma = c(2.1, 5.1, 4.1, 2.1))
      make_one_box_plot_row_auc(rslt = glm_rslt_d, print_n = TRUE, 
                            est = c("bp_emp1","bp_dcvtmle1",
                                          "bp_donestep1","bp_desteq1","bp_lpo"),
                            leg_x = 0, leg_y = 500,mult = 1.8,
                            grid = TRUE, ny_grid = NULL, nx_grid = 0,                           
                            add_legend = TRUE, rm_last = FALSE, 
                                  est_label = c("CVEMP", "CVTMLE","CVOS","LPO"),
                                  ylim = c(0.000001,100), yaxis_label = "Absolute Bias",
                                  col = my_col[sort(rep(c(1:4,6),4))])
      mtext(side = 3, outer = TRUE, line = 2.5, d)
      make_one_box_plot_row_auc(rslt = glm_rslt_d, 
                                  grid = TRUE, ny_grid = NULL, nx_grid = 0,   mult = 1.8,                        
                                est = c("cv_emp1","cv_dcvtmle1",
                                              "cv_donestep1","cv_desteq1","cv_lpo"), rm_last = FALSE, 
                                      est_label = c("CVEMP", "CVTMLE","CVOS","LPO"),
                                      ylim = c(0.000001,10), yaxis_label = "Coefficient of variation",
                                      col = my_col[sort(rep(c(1:4,6),4))])
      make_one_box_plot_row_auc(rslt = glm_rslt_d, 
                                est = c("mse_emp1","mse_dcvtmle1",
                                              "mse_donestep1","mse_desteq1","mse_lpo"),
                                  relative_est = "mse_emp1",mult = 1.8,
                                  relative_K = 5,add_text = TRUE,rm_last = FALSE, 
                                  grid = TRUE, ny_grid = NULL, nx_grid = 0,                           
                                      est_label = c("CVEMP", "CVTMLE","CVOS","LPO"),
                                      ylim = c(0.01,5), yaxis_label = "Relative MSE",
                                      col = my_col[sort(rep(c(1:4,6),4))])
    }
    dev.off()



    # some descriptive plots
    # box plot of true AUC for each data/wrapper/n combo
    # reorder data
    out_auc_small <- out_auc[out_auc$K == 5,]
    out_auc_small$my_int <- interaction(out_auc_small$n,out_auc_small$wrapper,out_auc_small$data)
    # ord <- NULL
    # for(i in 1:14){
    #   ord <- c(ord, seq(i,length(unique(my_int)),by = 14))
    # }
    # levels(out_auc_small$my_int) <- levels(out_auc_small$my_int)[ord]

    lab <- sapply(levels(out_auc_small[order(out_auc_small$my_int),"my_int"]), function(x){
      tmp <- paste0(unlist(strsplit(as.character(x),"_wrapper")), collapse = "")
      return(gsub("\\.","/",tmp))
    })
    my_grays <- brewer.pal(7, "Greys")
    my_red <- rgb(1,0,0,0.8)
    my_blue <- rgb(0,0,1,0.8)

    pdf("~/Dropbox/Emory/cross-validated-prediction-metrics/real_data_true_auc.pdf",
        width = 10, height = 5.5)
    par(mar = c(10.2, 4.1, 0.2, 0.2))
    boxplot(truth ~ my_int,
            data = out_auc_small[order(out_auc_small$my_int),], 
            las = 2, mar = c(20.1, 4.1, 0.1, 0.1),
            ylab = "AUC", col = my_grays[sort(rep(1:7,8))],
            border = rep(c(rep(my_red,4), rep(my_blue,4)), 7),
            names = lab)
    dev.off()

    out_tn_small <- out_tn[out_tn$K == 5,]
    out_tn_small$my_int <- interaction(out_tn_small$n,out_tn_small$wrapper,out_tn_small$data)
    # ord <- NULL
    # for(i in 1:14){
    #   ord <- c(ord, seq(i,length(unique(my_int)),by = 14))
    # }
    # levels(out_tn_small$my_int) <- levels(out_tn_small$my_int)[ord]

    lab <- sapply(levels(out_tn_small[order(out_tn_small$my_int),"my_int"]), function(x){
      tmp <- paste0(unlist(strsplit(as.character(x),"_wrapper")), collapse = "")
      return(gsub("\\.","/",tmp))
    })
    my_grays <- brewer.pal(7, "Greys")
    my_red <- rgb(1,0,0,0.8)
    my_blue <- rgb(0,0,1,0.8)

    pdf("~/Dropbox/Emory/cross-validated-prediction-metrics/real_data_true_tn.pdf",
        width = 10, height = 5.5)
    par(mar = c(10.2, 4.1, 0.2, 0.2))
    boxplot(truth ~ my_int,
            data = out_tn_small[order(out_tn_small$my_int),], 
            las = 2, mar = c(20.1, 4.1, 0.1, 0.1),
            ylab = "Test negative probability", col = my_grays[sort(rep(1:7,8))],
            border = rep(c(rep(my_red,4), rep(my_blue,4)), 7),
            names = lab)
    dev.off()





}