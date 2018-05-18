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

# parm <- parm[1,,drop=FALSE]
# source in simulation Functions
# source("~/cvtmleauc/makeData.R")
# load drinf
# library(glmnet)
# devtools::install_github("benkeser/cvtmleAUC")
library(cvtmleAUC, lib.loc = "/home/dbenkese/R/x86_64-pc-linux-gnu-library/3.4")
# library(SuperLearner, lib.loc = '/home/dbenkese/R/x86_64-pc-linux-gnu-library/3.4')
library(glmnet)
# library(xgboost)
# library(polspline)

# get the list size #########
if (args[1] == 'listsize') {
  cat(nrow(parm))
}

# execute prepare job ##################
if (args[1] == 'prepare') {
  parm_red <- parm[parm$K == parm$K[1] & parm$wrapper == parm$wrapper[1],]
  for(i in 1:nrow(parm_red)){
     set.seed(parm_red$seed[i])
     eval(parse(text = paste0("data(",parm_red$data_set[i],")")))
     dat <- eval(parse(text = parm_red$data_set[i]))
     train_idx <- sample(seq_len(length(dat[,1])), parm_red$n[i])
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


}