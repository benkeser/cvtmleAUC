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
K <- c(5, 10, 20, 40)
wrappers <- c("glm_wrapper", "randomforest_wrapper")
# wrappers <- c("glmnet_wrapper")
p <- 10
# TO DO:
# Add a replicate argument for repeated cross-validation estimators
parm <- expand.grid(seed = 1:bigB,
                    n = ns, K = K, 
                    wrapper = wrappers,
                    stringsAsFactors = FALSE)
# load("~/cvtmleauc/scratch/redo_parm_newest.RData")
# parm <- redo_parm
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

 # library(np, lib.loc = "/home/dbenkese/R/x86_64-pc-linux-gnu-library/3.4")
# library(cvAUC)
# library(SuperLearner)
# library(data.table)
# library(glmnet)

# get the list size #########
if (args[1] == 'listsize') {
  cat(nrow(parm))
}

# execute prepare job ##################
if (args[1] == 'prepare') {
  # parm_red <- parm[parm$K == parm$K[1] & parm$wrapper == parm$wrapper[1],]
  # for(i in 1:nrow(parm_red)){
  #    set.seed(parm_red$seed[i])
  #    dat <- makeData(n = parm_red$n[i], p = p)
  #    save(dat, file=paste0("~/cvtmleauc/scratch/dataList",
  #                          "_n=",parm_red$n[i],
  #                          "_seed=",parm_red$seed[i],".RData"))
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
    print(paste(Sys.time(), "i:" , i))
    print(parm[i,])
    print(sessionInfo())
    # load data
    load(paste0("~/cvtmleauc/scratch/dataList_",
                "n=",parm$n[i],
                "_seed=",parm$seed[i], 
                ".RData"))
    
    # get estimates of dcvauc
    options(np.messages = FALSE)
    n_replicates <- 20
    fit_dcv <- vector(mode = "list", length = n_replicates)
    fit_cv <- vector(mode = "list", length = n_replicates)
    for(j in seq_len(n_replicates)){
      set.seed(j)
      fit_dcv[[j]] <- cvtn_cvtmle(Y = dat$Y, X = dat$X, K = parm$K[i], 
                          learner = parm$wrapper[i], nested_cv = TRUE,
                          nested_K = 39)
      set.seed(j)
    # get estimates of cvtn
      fit_cv[[j]] <- cvtn_cvtmle(Y = dat$Y, X = dat$X, K = parm$K[i], 
                          learner = parm$wrapper[i], nested_cv = FALSE,
                          prediction_list = fit_dcv$prediction_list[1:parm$K[i]])
    }

    # get the truth
    set.seed(parm$seed[i])
    big_n <- 1e5
    big_data <- makeData(n = big_n, p = 10)
    
    # fit on full data
    fit_full <- do.call(parm$wrapper[i], args = list(train = list(X = dat$X, Y = dat$Y), 
                            test = list(X = big_data$X, Y = big_data$Y)))
    bigquantile_full <- quantile(fit_full$psi_nBn_testx[big_data$Y == 1], p = 0.05, type = 8)
    big_testneg_full <- mean(fit_full$psi_nBn_testx <= bigquantile_full)
    true_parameter <- big_testneg_full

    # bootstrap estimate 
    # only needed for K = 5 runs
    # and will be put back in later
    if(parm$K[i] == 5){
      set.seed(parm$seed[i])
      fit_boot <- cvtmleAUC:::boot_corrected_cvtn(Y = dat$Y, X = dat$X, learner = parm$wrapper[i])
    }else{
      fit_boot <- list(NA)
    }
    # c together output
    out <- c( # cvtmle estimates of dcvauc
             fit_dcv[[1]]$est_cvtmle, fit_dcv[[1]]$se_cvtmle,
             # iterations of cvtmle for dcv
             # fit_dcv[[1]]$iter, 
             # initial plug-in estimate of dcv
             fit_dcv[[1]]$est_init, 
             # one-step estimate of dcv
             fit_dcv[[1]]$est_onestep, fit_dcv[[1]]$se_onestep,
             # estimating eqn estimate of dcv
             fit_dcv[[1]]$est_esteq, fit_dcv[[1]]$se_esteq,
             # cvtmle estimate of cv
             fit_cv[[1]]$est_cvtmle, fit_cv[[1]]$se_cvtmle, 
             # iterations of cvtmle for cv
             # fit_cv[[1]]$iter, 
             fit_cv[[1]]$est_init, 
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
    save(out, file = paste0("~/cvtmleauc/out/outtn_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            ".RData.tmp"))
    file.rename(paste0("~/cvtmleauc/out/outtn_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],                            
                            ".RData.tmp"),
                paste0("~/cvtmleauc/out/outtn_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            ".RData"))

    grbg <- c(NULL)
    save(grbg, file = paste0("~/cvtmleauc/out/",i,"-run.dat"))  
  }
}

# merge job ###########################
if (args[1] == 'merge') {   
  ns <- c(50, 100, 250, 500)
  bigB <- 500
  K <- c(5,10,20,40)
  wrappers <- c("glm_wrapper", "randomforest_wrapper")
  # wrappers <- c("glmnet_wrapper")
  p <- 10
  redo_parm <- NULL
  # TO DO:
  # Add a replicate argument for repeated cross-validation estimators
  parm <- expand.grid(seed = 1:bigB,
                      n = ns, K = K, 
                      wrapper = wrappers,
                      stringsAsFactors = FALSE)
  rslt <- matrix(NA, nrow = nrow(parm), ncol = 66 + 4)
  for(i in 1:nrow(parm)){
    if(file.exists(paste0("~/cvtmleauc/out/outtn_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            ".RData"))){
      tmp_1 <- get(load(paste0("~/cvtmleauc/out/outtn_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=",parm$K[i],
                            "_wrapper=",parm$wrapper[i],
                            ".RData")))
    }else{
      redo_parm <- rbind(redo_parm, parm[i,])
      tmp_1 <- rep(NA, 66)
    }
    rslt[i,] <- c(parm$seed[i], parm$n[i], parm$K[i], parm$wrapper[i], tmp_1)
    if(is.na(tmp_1[length(tmp_1) - 1])){
      if(file.exists(paste0("~/cvtmleauc/out/outtn_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=5",
                            "_wrapper=",parm$wrapper[i],
                            ".RData"))){
      boot_rslt_out <- get(load(paste0("~/cvtmleauc/out/outtn_",
                            "n=", parm$n[i],
                            "_seed=",parm$seed[i],
                            "_K=5",
                            "_wrapper=",parm$wrapper[i],
                            ".RData")))
      boot_idx <- length(boot_rslt_out) - 1
      rslt[i,boot_idx] <- boot_rslt_out[boot_idx]
      }
    }
  }
  # # format
  out <- data.frame(rslt, stringsAsFactors = FALSE)

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

  sim_names <- c("seed","n","K","wrapper",
                 paste0(repeat_names, 1),
                 paste0(repeat_names2, 5),
                 paste0(repeat_names2, 10),
                 paste0(repeat_names2, 20),
                 "est_bootstrap",
                 "truth")  
  colnames(out) <- sim_names
  out[,c(1:3,5:ncol(out))] <- apply(out[,c(1:3,5:ncol(out))], 2, function(y){
    as.numeric(as.character(y))})

  save(out, file=paste0('~/cvtmleauc/out/allOut_cvtn.RData'))
  save(redo_parm, file = "~/cvtmleauc/scratch/redo_parm_newest.RData")
}


# local editing 
if(FALSE){
  setwd("~/Dropbox/R/cvtmleauc/sandbox/simulation/")
  sim <- "cvtn"
  load(paste0("allOut_",sim,".RData"))
    # load("~/cvtmleauc/out/allOut_new.RData")

    get_sim_rslt <- function(out, parm, wrapper, truth = "truth",
                             estimators = c("dcvtmle1","donestep1",
                                     "cvtmle1","onestep1","emp1",
                                     "dcvtmle5","donestep5",
                                     "cvtmle5","onestep5","emp5",
                                     "dcvtmle10","donestep10",
                                     "cvtmle10","onestep10","emp10",
                                     "dcvtmle20","donestep20",
                                     "cvtmle20","onestep20","emp20",
                                     "bootstrap"
                                     ), ...){
      b <- v <- m <- co <- NULL
      for(i in seq_len(length(parm[,1]))){
        x <- out[out$n == parm$n[i] & out$K == parm$K[i] & out$wrapper == wrapper,]
        b <- rbind(b, colMeans(x[,paste0("est_",estimators)] - x[,truth], na.rm = TRUE))
        v <- rbind(v, apply(x[,paste0("est_",estimators)], 2, var, na.rm = TRUE))
        m <- rbind(m, colMeans((x[,paste0("est_",estimators)] - as.numeric(x[,truth]))^2, na.rm = TRUE))
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
      parm <- cbind(parm, b, v, m) #, 
                    # co)
      colnames(parm) <- c("n", "K", paste0("bias_", estimators),
                          paste0("var_", estimators),
                          paste0("mse_", estimators)) #,
                          # paste0("cov_", estimators))
      return(parm)
    }
    parm <- expand.grid(n = c(50, 100, 250, 500),
                        K = c(5, 10, 20, 40))
    glm_rslt <- get_sim_rslt(out, parm, wrapper = "glm_wrapper")
    randomforest_rslt <- get_sim_rslt(out, parm, wrapper = "randomforest_wrapper")
    
    #---------------------------------
    # bar plots
    #---------------------------------

    make_side_by_side_bar_plots <- function(glm_rslt, randomforest_rslt,
                                            est, est_labels, rm_last = TRUE,
                                            yaxis_label, add_ratio = TRUE, 
                                            xaxis_label = "Number CV Folds",
                                            transpose = FALSE, 
                                            absolute_val = TRUE, col, ...){
      layout(matrix(1:8, nrow = 2, ncol = 4,  byrow = TRUE))
      par(mar = c(1.6, 0.6, 1.6, 0.6), mgp = c(2.1, 0.5, 0),
          oma = c(2.1, 5.1, 2.1, 2.1))
      for(n in c(50, 100, 250, 500)){
        tmp <- glm_rslt[glm_rslt$n == n, ]
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
        }else{
          leg.text <- FALSE
        }
        if(transpose){
          grbg <- t(grbg)
        }
        tmp <- barplot(grbg, legend.text = FALSE,
          beside=TRUE, log = "y", yaxt = "n", names.arg = rep("",4), col = col, ... )
          mtext(side = 1, outer = FALSE, line = 0.02, text = c(5,10,20,40), cex =0.5, 
        at = c(tmp[,1],tmp[,2],tmp[,3]))
        if(n == 50){
          legend(x = "topright", fill = unique(col), legend = leg.text)
        }
        if(n == 50){
          axis(side = 2)
          mtext(outer = FALSE, side = 2, line = 2, yaxis_label, cex = 0.75)
          mtext(outer = FALSE, side = 2, line = 4, "Logistic Regression", cex = 1)
        }else{
          axis(side = 2, labels = FALSE)
        }
        mtext(outer = FALSE, side = 3, line = 0.5, paste0("n = ", n))
        if(add_ratio){
        }
      }
      for(n in c(50, 100, 250, 500)){
        tmp <- randomforest_rslt[randomforest_rslt$n == n, ]
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
        if(transpose){
          grbg <- t(grbg)
        }
        tmp <- barplot(grbg,  beside=TRUE, log = "y", yaxt = "n", names.arg = rep("",4), 
                       col = col, ...)
        mtext(side = 1, outer = FALSE, line = 0.02, text = c(5,10,20,40), cex =0.5, 
              at = c(tmp[,1],tmp[,2],tmp[,3]))
        # mtext(side = 1, outer = FALSE, line = 0.02, text = c(5,10,20,40), cex =0.5, 
        #       at = c(mean(tmp[2:3,1]),mean(tmp[2:3,2]),mean(tmp[2:3,3])))
        if(n == 50){
          axis(side = 2)
          mtext(outer = FALSE, side = 2, line = 2, yaxis_label, cex = 0.75)
          mtext(outer = FALSE, side = 2, line = 4, "Random Forest", cex = 1)
        }else{
          axis(side = 2, labels = FALSE)
        }
      }
    }

    library(RColorBrewer)
    my_col <- brewer.pal(5,"Greys")


    # bias
    make_side_by_side_bar_plots(glm_rslt, randomforest_rslt, 
                                est = c("bias_dcvtmle1","bias_donestep1",
                                        "bias_emp1","bias_bootstrap"),
                                est_label = c("CVTMLE", "CVOS","Empirical","Bootstrap"),
                                ylim = c(0.0001,100), yaxis_label = "Absolute Bias",
                                col = my_col[sort(rep(1:4,4))])
    # variance
    make_side_by_side_bar_plots(glm_rslt, randomforest_rslt, 
                                est = c("var_dcvtmle1","var_donestep1",
                                        "var_emp1","var_bootstrap"),
                                est_label = c("CVTMLE", "CVOS","Empirical","Bootstrap"),
                                ylim = c(0.0001,50), yaxis_label = "Variance",
                                col = my_col[sort(rep(1:4,4))])
    # mse
    # debug(make_side_by_side_bar_plots)
    make_side_by_side_bar_plots(glm_rslt, randomforest_rslt, 
                                est = c("mse_donestep1","mse_dcvtmle1",
                                        "mse_emp1","mse_bootstrap"),
                                est_label = c("CVOS", "CVTMLE","Empirical","Bootstrap"),
                                ylim = c(0.0001,100), yaxis_label = "Mean squared-error",
                                col = my_col[sort(rep(1:4,4))])


    # bias by CV Repeats for CVTMLE
    make_side_by_side_bar_plots(glm_rslt, randomforest_rslt, 
                                est = c("bias_dcvtmle1","bias_dcvtmle5",
                                        "bias_dcvtmle10","bias_dcvtmle20"),
                                rm_last = FALSE, transpose = TRUE, 
                                est_label = paste0("CVTMLE ", c(1,5,10,20)," repeats"),
                                ylim = c(0.00001,100), yaxis_label = "Absolute bias",,
                                col = my_col[sort(rep(1:4,4))])

    # variance by CV Repeats for CVTMLE
    make_side_by_side_bar_plots(glm_rslt, randomforest_rslt, 
                                est = c("var_dcvtmle1","var_dcvtmle5",
                                        "var_dcvtmle10","var_dcvtmle20"),
                                rm_last = FALSE, transpose = TRUE, 
                                est_label = paste0("CVTMLE ", c(1,5,10,20)," repeats"),
                                ylim = c(0.0001,0.01), yaxis_label = "Variance",
                                col = my_col[sort(rep(1:4,4))])

    # mse by CV Repeats for CVTMLE
    make_side_by_side_bar_plots(glm_rslt, randomforest_rslt, 
                                est = c("mse_dcvtmle1","mse_dcvtmle5",
                                        "mse_dcvtmle10","mse_dcvtmle20"),
                                rm_last = FALSE, transpose = TRUE, 
                                est_label = paste0("CVTMLE ", c(1,5,10,20)," repeats"),
                                ylim = c(0.0001,0.1), yaxis_label = "Variance",
                                col = my_col[sort(rep(1:4,4))])





    # compare mse of cvtmle with 40 folds to mse of empirical with 5 folds
    make_mse_compare_one_repeat <- function(rslt, B, legend = FALSE){
      cvt_n_glm <- rslt[,paste0("mse_dcvtmle",B)][rslt$K == 40][2:4]
      cvo_n_glm <- rslt[,paste0("mse_donestep",B)][rslt$K == 40][2:4]
      emp_n_glm <- rslt[,paste0("mse_emp",B)][rslt$K == 5][2:4]

      plot(y = cvt_n_glm/emp_n_glm, x = 1:3, xaxt = "n", yaxt = "n", bty = "n",
           xlim = c(1,3), ylim = c(0,2), type = "b",
           xlab = "Sample size", ylab = "MSE / Empirical with K = 5")
      axis(side = 1, at = 1:3, labels = c(100, 250, 500))
      axis(side = 2)
      abline(h = 1, lty = 3)
      points(y = cvo_n_glm/emp_n_glm, x = 1:3, pch = 2, type = "b", lty = 2)
      if(legend){
        legend(x = "topleft", c("CVTMLE, K = 40", "CVOS, K = 40"), pch = 1:2,
               bty = "n")
      }
    }

    layout(matrix(1:8, nrow = 2, ncol = 4,  byrow = TRUE))
    for(b in c(1, 5, 10, 20)){
      make_mse_compare_one_repeat(rslt = glm_rslt, b, legend = ifelse(b == 1, TRUE, FALSE))
      mtext(side = 3, text = paste0("MC Repeats = ", b))
    }
    for(b in c(1, 5, 10, 20)){
      make_mse_compare_one_repeat(rslt = randomforest_rslt, b, legend = ifelse(b == 1, TRUE, FALSE))
    }




    cvt_n_glm <- glm_rslt$mse_dcvtmle1[glm_rslt$K == 40]
    cvt_n_glm <- glm_rslt$mse_dcvtmle1[glm_rslt$K == 40]

    ratio_cvtmle_to_mse <- 


    #---------------------------------
    # Bias plots
    #---------------------------------
    # top row = glm bias ~ K for each n 
    # bottom row = random forest bias ~ K for each n

    layout(matrix(1:8, nrow = 2, ncol = 4,  byrow = TRUE))
    par(mar = c(1.6, 0.6, 0.6, 0.6), mgp = c(2.1, 0.5, 0),
        oma = c(2.1, 5.1, 2.1, 2.1))
    for(n in c(50, 100, 250, 500)){
      tmp <- glm_rslt[glm_rslt$n == n, ]
      tmp5 <- tmp[tmp$K == 5,]
      tmp10 <- tmp[tmp$K == 10,]
      tmp20 <- tmp[tmp$K == 20,]
      tmp40 <- tmp[tmp$K == 40,]
      est <- c("bias_dcvtmle1","bias_donestep1","bias_emp1")
      grbg <- t(abs(as.matrix(rbind(tmp5[,est],tmp10[,est],tmp20[,est],tmp40[,est]))))
      colnames(grbg) <- c(5,10,20,40)
      barplot(grbg,
        beside=TRUE, log = "y", yaxt = "n", ylim = c(0.0001, 200))
      if(n == 50){
        axis(side = 2)
        mtext(outer = FALSE, side = 2, line = 2, "Absolute bias", cex = 0.75)
        mtext(outer = FALSE, side = 2, line = 4, "Logistic Regression", cex = 1)
      }else{
        axis(side = 2, labels = FALSE)
      }
      mtext(outer = FALSE, side = 3, line = 0.5, paste0("n = ", n))
    }
    for(n in c(50, 100, 250, 500)){
      tmp <- randomforest_rslt[randomforest_rslt$n == n, ]
      tmp5 <- tmp[tmp$K == 5,]
      tmp10 <- tmp[tmp$K == 10,]
      tmp20 <- tmp[tmp$K == 20,]
      tmp40 <- tmp[tmp$K == 40,]
      est <- c("bias_dcvtmle1","bias_donestep1","bias_emp1")
      grbg <- t(abs(as.matrix(rbind(tmp5[,est],tmp10[,est],tmp20[,est],tmp40[,est]))))
      colnames(grbg) <- c(5,10,20,40)
      barplot(grbg,
        beside=TRUE, log = "y", yaxt = "n", ylim = c(0.0001, 200))
      mtext(side = 1, outer = FALSE, line = 2, "Number of CV Folds", cex =0.75)
      if(n == 50){
        axis(side = 2)
        mtext(outer = FALSE, side = 2, line = 2, "Absolute bias", cex = 0.75)
        mtext(outer = FALSE, side = 2, line = 4, "Random Forest", cex = 1)
      }else{
        axis(side = 2, labels = FALSE)
      }
    }


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