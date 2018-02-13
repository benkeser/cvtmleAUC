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

ns <- c(100,200,500,1000)
bigB <- 500
K <- c(5,10,20,30)

parm <- expand.grid(seed=1:bigB,
                    n=ns, K = K)

# parm <- parm[1,,drop=FALSE]
# source in simulation Functions
source("~/cvtmleauc/makeData.R")
# load drinf
library(SuperLearner)
library(cvAUC)
library(glmnet)
library(cvtmleAUC)

# get the list size #########
if (args[1] == 'listsize') {
  cat(nrow(parm))
}

# execute prepare job ##################
if (args[1] == 'prepare') {
  for(i in 1:nrow(parm)){
     set.seed(parm$seed[i])
     dat <- makeData(n = parm$n[i], p = 20)
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
    load(paste0("~/cvtmleauc/scratch/dataList_n=",parm$n[i],
                "_seed=",parm$seed[i], ".RData"))
    
    # set seed
    set.seed(parm$seed[i])

    fit <- cvauc_cvtmle(Y = dat$Y, X = dat$X, K = parm$K[i], 
                        learner = "glmnet_wrapper")
    

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
  }
}

# merge job ###########################
if (args[1] == 'merge') {   
  ns <- c(500,1000,5000,7000)
  bigB <- 500
  g <- c("SL.hal9001","SL.glm")
  Q <- c("SL.hal9001","SL.glm")
  cv <- c(1)
  # g <- c("SL.glm.interaction")
  # Q <- c("SL.glm.interaction")
  parm <- expand.grid(seed=1:bigB,
                      n=ns, g = g, Q = Q, cv = cv, 
                      stringsAsFactors = FALSE)
  # n = 500 => up to 49 seconds => get about 20 done per run
  # n = 1000 => up to 147 seconds = 2.5 minutes => get about 10 done per run
  # n = 5000 => just do one per run
  parm$g[(parm$g == "SL.glm" & parm$Q == "SL.glm")] <- "SL.glm.interaction"
  parm$Q[(parm$g == "SL.glm.interaction" & parm$Q == "SL.glm")] <- "SL.glm.interaction"

    rslt <- matrix(NA, nrow = nrow(parm), ncol = 43)
    for(i in 1:nrow(parm)){
        tmp_1 <- tryCatch({
            load(paste0("~/drinf/out/out_n=",
                        parm$n[i],"_seed=",parm$seed[i],
                       "_Q=",parm$Q[i],"_g=",parm$g[i],
                       "_cvFolds=1.RData"))
            out
        }, error=function(e){
          c(parm$seed[i], parm$n[i], NA, parm$Q[i], parm$g[i], rep(NA, 43 - 5))
        })
        # tmp_5 <- tryCatch({
        #     load(paste0("~/drinf/out/out_n=",
        #                 parm$n[i],"_seed=",parm$seed[i],
        #                "_Q=",parm$Q[i],"_g=",parm$g[i],
        #                "_cvFolds=5.RData"))
        #     out[-(1:5)]
        # }, error=function(e){
        #   rep(NA, 17 + 25*2 - 5)
        # })
        # tmp <- c(tmp_1, tmp_5)
        rslt[i,] <- tmp_1
    }
    # # format
    out <- data.frame(rslt)
    sim_names <- c("seed","n","truth","Q","g",
                   paste0("max_n_", c("est","cil","ciu","cov")),
                   paste0("max_sqrt_n_", c("est","cil","ciu","cov")),
                   paste0("norm_n_", c("est","cil","ciu","cov")),
                   paste0("norm_sqrt_n_", c("est","cil","ciu","cov")),
                   paste0("drtmle_maxIter",1:5),
                   paste0("se_drtmle_maxIter",1:5),
                   "ltmle","ltmle_cil","ltmle_ciu",
                   # "ltmleboot_cil", 
                   # "ltmleboot_ciu", 
                   "ltmle_cov", 
                   # "ltmleboot_cov",
                   paste0(c("total_","sqrt_n_max_","n_max_","n_norm_","sqrt_n_norm_"),"iter"),
                   "origIC","missQIC","missgIC")
    colnames(out) <- sim_names

    out[,(1:ncol(out))[c(-4,-5)]] <- apply(out[,(1:ncol(out))[c(-4,-5)]], 2, as.numeric)
    save(out, file=paste0('~/drinf/out/noboot_allOut_nocv_newest.RData'))

    # # post processing
    # getBias <- function(out, n, Q, g){
    #   rslt <- out[out$n %in% n & out$Q %in% Q & out$g %in% g, ]
    #   bias <- by(rslt, rslt$n, function(x){
    #     bias_drtmle <- mean(x$drtmle - x$truth, na.rm = TRUE)
    #     bias_drtmle_1 <- mean(x$drtmle_maxIter1 - x$truth, na.rm = TRUE)
    #     bias_ltmle <- mean(x$ltmle - x$truth, na.rm = TRUE)
    #     c(nrow(x), bias_drtmle, bias_drtmle_1, bias_ltmle)
    #   })
    #   bias
    # }
    # getBias(out, n = c(500,1000,5000), Q = "SL.hal9001", g = "SL.glm")
    # getBias(out, n = c(500,1000,5000), g = "SL.hal9001", Q = "SL.glm")
    # getBias(out, n = c(500,1000,5000), Q = "SL.hal9001", g = "SL.hal9001")
    # getBias(out, n = c(500,1000,5000), Q = "SL.glm.interaction", g = "SL.glm.interaction")
    getRootNBias <- function(out, n, Q, g, est = c("max_sqrt_n_est",
                                                   "norm_sqrt_n_est",
                                                   paste0("drtmle_maxIter",1:5),
                                                   "ltmle")){
      rslt <- out[out$n %in% n & out$Q %in% Q & out$g %in% g, ]
      rootn_bias <- by(rslt, rslt$n, function(x){
        o <- matrix(c(nrow(x), rep(NA, length(est))), nrow = 1)
        ct <- 1
        for(e in est){
          # browser()
          ct <- ct + 1
          o[ct] <- sqrt(x$n[1])*mean(x[,e] - x$truth, na.rm = TRUE)
        }
        colnames(o) <- c("nsim", est)
        o
      })
      ou <- Reduce(rbind, rootn_bias)
      ou <- cbind(unique(rslt$n), ou)
      ou
    }
    getRootNBias(out, n = c(500,1000,5000,7000), Q = "SL.hal9001", g = "SL.glm")
                 # est = paste0("cv_drtmle_maxIter", 1:25))
    getRootNBias(out, n = c(500,1000,5000,7000), g = "SL.hal9001", Q = "SL.glm")
                 # est = paste0("cv_drtmle_maxIter", 1:25))
    getRootNBias(out, n = c(500,1000,5000,7000), Q = "SL.hal9001", g = "SL.hal9001")
                 # est = paste0("cv_drtmle_maxIter", 1:25))
    getRootNBias(out, n = c(500,1000,5000,7000), Q = "SL.glm.interaction", g = "SL.glm.interaction")
                 # est = paste0("cv_drtmle_maxIter", 1:25))

    getCov <- function(out, n, Q, g,est = c("max_sqrt_n",
                                             "norm_sqrt_n",
                                             paste0("drtmle_maxIter",1:5),
                                             "ltmle")){
      rslt <- out[out$n %in% n & out$Q %in% Q & out$g %in% g, ]
      cov <- by(rslt, rslt$n, function(x){
        o <- matrix(c(nrow(x), rep(NA, length(est) + 1)), nrow = 1)
        ct <- 1
        for(e in est){
          # browser()
          ct <- ct + 1
          cov_avail <- any(grepl(paste0(e,"_cov"), colnames(rslt)))
          if(cov_avail){
            o[,ct] <- mean(x[,paste0(e,"_cov")], na.rm = TRUE)
          }else{
            this_est <- x[,paste0(e)]
            this_se <- x[,paste0("se_",e)]
            cil <- this_est - 1.96*this_se; ciu <- this_est + 1.96*this_se
            o[,ct] <- mean(cil < x$truth[1] & ciu > x$truth[1])
          }
        }
        # add in ltmle with mc standard deviation interval
        sd_ltmle <- sd(x$ltmle, na.rm = TRUE)
        cil <- x$ltmle - 1.96 * sd_ltmle
        ciu <- x$ltmle + 1.96 * sd_ltmle
        o[,ct + 1] <- mean(cil < x$truth[1] & ciu > x$truth[1])

        colnames(o) <- c("nsim", est, "ltmle_mc")
        o
      })
      ou <- Reduce(rbind, cov)
      ou <- cbind(unique(rslt$n), ou)
      ou
    }
    getCov(out, n = c(500,1000,5000,7000), Q = "SL.hal9001", g = "SL.glm")
    getCov(out, n = c(500,1000,5000,7000), g = "SL.hal9001", Q = "SL.glm")
    getCov(out, n = c(500,1000,5000,7000), Q = "SL.hal9001", g = "SL.hal9001")
    getCov(out, n = c(500,1000,5000,7000), Q = "SL.glm.interaction", g = "SL.glm.interaction")

    # getIC <- function(out, n, Q, g){
    #   rslt <- out[out$n %in% n & out$Q %in% Q & out$g %in% g, ]
    #   ic <-  by(rslt, rslt$n, function(x){
    #     colMeans(x[ , grepl("IC", colnames(x))])
    #   })
    #   ic
    # }
    # getIC(out, n = c(500,1000,5000), Q = "SL.hal9001", g = "SL.glm")
    # getIC(out, n = c(500,1000,5000), g = "SL.hal9001", Q = "SL.glm")
    # getIC(out, n = c(500,1000,5000), Q = "SL.hal9001", g = "SL.hal9001")
    # getIC(out, n = c(500,1000,5000), Q = "SL.glm.interaction", g = "SL.glm.interaction")


}