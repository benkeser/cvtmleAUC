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
glmnet_wrapper <- function(train, test){
    glmnet_fit <- glmnet::cv.glmnet(x = data.matrix(train$X), y = train$Y,
        lambda = NULL, type.measure = "deviance", nfolds = 5, 
        family = "binomial", alpha = 1, nlambda = 100)
    Psi_nBn_0 <- function(x){
      stats::predict(glmnet_fit, newx = x, type = "response", s = "lambda.min")
    }
    psi_nBn_trainx <- Psi_nBn_0(train$X)
    psi_nBn_testx <- Psi_nBn_0(test$X)
    return(list(psi_nBn_trainx = psi_nBn_trainx, psi_nBn_testx = psi_nBn_testx,
                model = glmnet_fit, train_y = train$Y, test_y = test$Y))
}
