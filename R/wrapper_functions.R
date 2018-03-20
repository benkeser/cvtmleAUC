#' Wrapper for fitting a main terms random forest
#' 
#' @param train ...
#' @param test ...
#' @param SL.library super learner library
#' @return A list
#' @export
#' @importFrom SuperLearner SuperLearner 
#' @importFrom stats predict
#' @examples
#' # TO DO: Add
superlearner_wrapper <- function(train, test,
                                 SL.library = c("SL.mean"), 
                                 ...){
    sl_fit <- SuperLearner::SuperLearner(Y = train$Y, 
                                         X = train$X, SL.library = SL.library,
                                         newX = rbind(test$X,train$X),
                                         family = binomial())
    all_pred <- sl_fit$SL.pred
    ntest <- length(test$Y)
    ntrain <- length(train$Y)
    psi_nBn_testx <- all_pred[1:ntest]
    psi_nBn_trainx <- all_pred[(ntest+1):(ntest+ntrain)]
    return(list(psi_nBn_trainx = psi_nBn_trainx, psi_nBn_testx = psi_nBn_testx,
                model = sl_fit, train_y = train$Y, test_y = test$Y))
}

#' Wrapper for fitting a main terms random forest
#' 
#' @param train ...
#' @param test ...
#' @param mtry ...
#' @param ntree ...
#' @param nodesize ...
#' @param maxnodes ...
#' @param importance ...
#' @param ... ...
#' @return A list
#' @export
#' @importFrom randomForest randomForest 
#' @importFrom stats predict
#' @examples
#' # TO DO: Add
randomforest_wrapper <- function(train, test,
                                 mtry = floor(sqrt(ncol(train$X))), 
    ntree = 1000, nodesize = 1, maxnodes = NULL, importance = FALSE,...){
    rf_fit <- randomForest::randomForest(y = as.factor(train$Y), 
            x = train$X, ntree = ntree, xtest = rbind(test$X, train$X), 
            keep.forest = TRUE, mtry = mtry, nodesize = nodesize, 
            maxnodes = maxnodes, importance = importance, ...)
    all_psi <- rf_fit$test$votes[,2]
    ntest <- length(test$Y)
    ntrain <- length(train$Y)
    psi_nBn_testx <- all_psi[1:ntest]
    psi_nBn_trainx <- all_psi[(ntest+1):(ntest+ntrain)]
    return(list(psi_nBn_trainx = psi_nBn_trainx, psi_nBn_testx = psi_nBn_testx,
                model = rf_fit, train_y = train$Y, test_y = test$Y))
}

#' Wrapper for fitting a main terms GLM
#' 
#' @param train ...
#' @param test ...
#' @return A list
#' @export
#' @importFrom stats glm predict
#' @examples
#' # TO DO: Add
glm_wrapper <- function(train, test){
    glm_fit <- stats::glm(train$Y ~ ., data = train$X, family = binomial())
    Psi_nBn_0 <- function(x){
      stats::predict(glm_fit, newdata = x, type = "response")
    }
    psi_nBn_trainx <- Psi_nBn_0(train$X)
    psi_nBn_testx <- Psi_nBn_0(test$X)
    return(list(psi_nBn_trainx = psi_nBn_trainx, psi_nBn_testx = psi_nBn_testx,
                model = glm_fit, train_y = train$Y, test_y = test$Y))
}

#' Wrapper for fitting a main terms GLM
#' 
#' @param train ...
#' @param test ...
#' @return A list
#' @export
#' @importFrom stats glm predict
#' @examples
#' # TO DO: Add
stepglm_wrapper <- function(train, test){
    glm_full <- stats::glm(train$Y ~ ., data = train$X, family = binomial())
    glm_fit <- step(glm(train$Y ~ 1, data = train$X, family = binomial()), scope = formula(glm_full), 
        direction = "forward", trace = 0, k = 2)
    Psi_nBn_0 <- function(x){
      stats::predict(glm_fit, newdata = x, type = "response")
    }
    psi_nBn_trainx <- Psi_nBn_0(train$X)
    psi_nBn_testx <- Psi_nBn_0(test$X)
    return(list(psi_nBn_trainx = psi_nBn_trainx, psi_nBn_testx = psi_nBn_testx,
                model = glm_fit, train_y = train$Y, test_y = test$Y))
}

#' Wrapper for fitting lasso 
#' @param train ...
#' @param test ...
#' @return A list
#' @export
#' @importFrom glmnet cv.glmnet
#' @examples
#' # TO DO: Add
glmnet_wrapper <- function(train, test, lambda.select = "ncoef", ncoef = 5){
    x <- model.matrix(~ -1 + ., data = train$X)
    if(lambda.select == "cv"){
        glmnet_fit <- glmnet::cv.glmnet(x = x, y = train$Y,
            lambda = NULL, type.measure = "deviance", nfolds = 5, 
            family = "binomial", alpha = 1, nlambda = 100)
        Psi_nBn_0 <- function(x){
          newx <- model.matrix(~ -1 + ., data = x)
          stats::predict(glmnet_fit, newx = newx, type = "response", s = "lambda.min")
        }
    }else if (lambda.select == "ncoef"){
        glmnet_fit <- glmnet::glmnet(x = x, y = train$Y,
            lambda = NULL, family = "binomial", alpha = 1, nlambda = 100)
        n_nonzero_coef <- apply(glmnet_fit$beta, 2, function(x){ sum(abs(x) > 0) })
        lambda_idx <- which(n_nonzero_coef == ncoef)[1]
        lambda_select <- glmnet_fit$lambda[lambda_idx]
        glmnet_fit$my_lambda <- lambda_select
        Psi_nBn_0 <- function(x){
          newx <- model.matrix(~ -1 + ., data = x)
          stats::predict(glmnet_fit, newx = data.matrix(x), type = "response", s = lambda_select)
        }
    }
    psi_nBn_trainx <- Psi_nBn_0(train$X)
    psi_nBn_testx <- Psi_nBn_0(test$X)
    return(list(psi_nBn_trainx = psi_nBn_trainx, psi_nBn_testx = psi_nBn_testx,
                model = glmnet_fit, train_y = train$Y, test_y = test$Y))
}


# #' Wrapper for fitting dbarts
# #' 
# #' @param train ...
# #' @param test ...
# #' @return A list
# #' @export
# #' @importFrom dbarts bart
# #' @importFrom stats pnorm
# #' @examples
# #' # TO DO: Add
# bart_wrapper <- function(train, test, sigest = NA, sigdf = 3, 
#     sigquant = 0.9, k = 2, power = 2, base = 0.95, binaryOffset = 0, 
#     ntree = 200, ndpost = 1000, nskip = 100, printevery = 100, 
#     keepevery = 1, keeptrainfits = TRUE, usequants = FALSE, numcut = 100, 
#     printcutoffs = 0, nthread = 1, keepcall = TRUE, verbose = FALSE){
    
#     bart_fit <- dbarts::bart(x.train = train$X, y.train = train$Y, 
#         x.test = rbind(train$X, test$X), sigest = sigest, sigdf = sigdf, 
#         sigquant = sigquant, k = k, power = power, base = base, 
#         binaryOffset = binaryOffset, ntree = ntree, 
#         ndpost = ndpost, nskip = nskip, printevery = printevery, 
#         keepevery = keepevery, keeptrainfits = keeptrainfits, usequants = usequants, 
#         numcut = numcut, printcutoffs = printcutoffs, nthread = nthread, 
#         keepcall = keepcall, verbose = verbose)
#     ntest <- length(test$Y)
#     ntrain <- length(train$Y)
#     all_psi <- colMeans(stats::pnorm(bart_fit$yhat.test))
#     psi_nBn_testx <- all_psi[1:ntest]
#     psi_nBn_trainx <- all_psi[(ntest+1):(ntest+ntrain)]

#     return(list(psi_nBn_trainx = psi_nBn_trainx, psi_nBn_testx = psi_nBn_testx,
#                 model = bart_fit, train_y = train$Y, test_y = test$Y))
# }

#' Wrapper for fitting xgboost
#' 
#' @param train ...
#' @param test ...
#' @return A list
#' @export
#' @importFrom xgboost xgboost xgb.DMatrix
#' @examples
#' # TO DO: Add
#' 
xgboost_wrapper <- function(test, train, ntrees = 500, 
    max_depth = 4, shrinkage = 0.1, minobspernode = 10, params = list(), 
    nthread = 1, verbose = 0, save_period = NULL){
    x <- model.matrix(~. - 1, data = train$X)
    xgmat <- xgboost::xgb.DMatrix(data = x, label = train$Y)
    xgboost_fit <- xgboost::xgboost(data = xgmat, objective = "binary:logistic", 
            nrounds = ntrees, max_depth = max_depth, min_child_weight = minobspernode, 
            eta = shrinkage, verbose = verbose, nthread = nthread, 
            params = params, save_period = save_period)
    newx <- model.matrix(~. - 1, data = test$X)

    psi_nBn_testx <- predict(xgboost_fit, newdata = newx)
    psi_nBn_trainx <- predict(xgboost_fit, newdata = x)

    return(list(psi_nBn_trainx = psi_nBn_trainx, psi_nBn_testx = psi_nBn_testx,
                model = xgboost_fit, train_y = train$Y, test_y = test$Y))
}


#' Wrapper for fitting polymars
#' 
#' @param train ...
#' @param test ...
#' @return A list
#' @export
#' @importFrom polspline polyclass
#' @examples
#' # TO DO: Add
#' 
polymars_wrapper <- function(test, train){
    mars_fit <- polspline::polyclass(train$Y, train$X, cv = 5)
    psi_nBn_trainx <- polspline::ppolyclass(cov = train$X, fit = mars_fit)[,2]
    psi_nBn_testx <- polspline::ppolyclass(cov = test$X, fit = mars_fit)[,2]
    return(list(psi_nBn_trainx = psi_nBn_trainx, psi_nBn_testx = psi_nBn_testx,
                model = mars_fit, train_y = train$Y, test_y = test$Y))
}



#' Wrapper for fitting svm
#' 
#' @param train ...
#' @param test ...
#' @return A list
#' @export
#' @importFrom e1071 svm
#' @examples
#' # TO DO: Add
#' 
svm_wrapper <- function(test, train, type.class = "nu-classification", 
    kernel = "radial", nu = 0.5, degree = 3, cost = 1, coef0 = 0, ...){
    svm_fit <- e1071::svm(y = as.factor(train$Y), x = train$X, nu = nu, 
            type = type.class, fitted = FALSE, probability = TRUE, 
            kernel = kernel, degree = degree, cost = cost, coef0 = coef0)
    psi_nBn_trainx <- attr(predict(svm_fit, newdata = train$X, probability = TRUE), 
        "prob")[, "1"]    
    psi_nBn_testx <- attr(predict(svm_fit, newdata = test$X, probability = TRUE), 
        "prob")[, "1"]
    return(list(psi_nBn_trainx = psi_nBn_trainx, psi_nBn_testx = psi_nBn_testx,
                model = svm_fit, train_y = train$Y, test_y = test$Y))
}