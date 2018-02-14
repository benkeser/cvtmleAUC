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

# ns <- c(100,200,500,1000)
# bigB <- 500
# K <- c(5,10,20,30)
ns <- c(200)
bigB <- 10
K <- 20
p <- 50
parm <- expand.grid(seed=1:bigB,
                    n=ns, K = K)
i <- 1

# parm <- parm[1,,drop=FALSE]
# source in simulation Functions
source("~/cvtmleauc/makeData.R")
# load drinf
library(glmnet)
library(cvtmleAUC, lib.loc = "/home/dbenkese/R/x86_64-unknown-linux-gnu-library/3.2/")
library(SuperLearner)

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
    # load(paste0("~/cvtmleauc/scratch/dataList_n=",parm$n[i],
    #             "_seed=",parm$seed[i], ".RData"))
    do.one <- function(){
    # set seed
    # set.seed(parm$seed[i])

    dat <- makeData(n = parm$n[i], p = p)

    # get tmle and regular estimates
    fit <- cvauc_cvtmle(Y = dat$Y, X = dat$X, K = parm$K[i], 
                        learner = "glmnet_wrapper")
    # get true cvAUC
    N <- 5e5
    bigdat <- makeData(n = N, p = p)
    big_valid_pred_list <- lapply(fit$models, function(x){
      predict(x, newx = bigdat$X, type = "response")
    })
    big_label_list <- rep(list(bigdat$Y), parm$K[i])
    true_cvauc <- mean(cvAUC::AUC(predictions = big_valid_pred_list,
                            labels = big_label_list))

    # now get auc of \hat{\Psi(P_n)}
    full_model <- glmnet_wrapper(train = dat, test = dat)
    big_valid_pred <- predict(full_model$model, newx = bigdat$X, type = "response")
    true_auc_fullmodel <- cvAUC::AUC(predictions = big_valid_pred, labels = bigdat$Y)

    out <- c(fit$est, fit$se, fit$iter, fit$est_init, 
             fit$est_empirical, true_cvauc, true_auc_fullmodel)
    return(out)
    }
    system.time(do.one())
    rslt <- replicate(10, do.one)

    # save output 
    save(out, file = paste0("~/cvtmleauc/out/out_n=",
                            parm$n[i],"_seed=",parm$seed[i],
                            "_Q=",parm$Q[i],"_g=",parm$g[i],
                            "_cvFolds=",parm$cv[i],".RData.tmp"))
    file.rename(paste0("~/cvtmleauc/out/out_n=",
                       parm$n[i],"_seed=",parm$seed[i],
                       "_Q=",parm$Q[i],"_g=",parm$g[i],"_cvFolds=",parm$cv[i],
                       ".RData.tmp"),
                paste0("~/cvtmleauc/out/out_n=",
                       parm$n[i],"_seed=",parm$seed[i],
                       "_Q=",parm$Q[i],"_g=",parm$g[i],"_cvFolds=",parm$cv[i],
                       ".RData"))    
    save(fit$models, file = paste0("~/cvtmleauc/out/models_n=",
                            parm$n[i],"_seed=",parm$seed[i],
                            "_Q=",parm$Q[i],"_g=",parm$g[i],
                            "_cvFolds=",parm$cv[i],".RData.tmp"))
    file.rename(paste0("~/cvtmleauc/out/models_n=",
                       parm$n[i],"_seed=",parm$seed[i],
                       "_Q=",parm$Q[i],"_g=",parm$g[i],"_cvFolds=",parm$cv[i],
                       ".RData.tmp"),
                paste0("~/cvtmleauc/out/models_n=",
                       parm$n[i],"_seed=",parm$seed[i],
                       "_Q=",parm$Q[i],"_g=",parm$g[i],"_cvFolds=",parm$cv[i],
                       ".RData"))
  }
}

# merge job ###########################
if (args[1] == 'merge') {   
  
}