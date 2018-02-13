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

#' @param train ...
#' @param test ...
#' @return A list
#' @export
#' @importFrom glmnet cv.glmnet
#' @examples
#' # TO DO: Add
glmnet_wrapper <- function(train, test){
    glmnet_fit <- glmnet::cv.glmnet(x = train$X, y = train$Y,
        lambda = NULL, type.measure = "deviance", nfolds = 5, 
        family = "binomial", alpha = 0, nlambda = 100)
    Psi_nBn_0 <- function(x){
      stats::predict(glmnet_fit, newx = x, type = "response")
    }
    psi_nBn_trainx <- Psi_nBn_0(train$X)
    psi_nBn_testx <- Psi_nBn_0(test$X)
    return(list(psi_nBn_trainx = psi_nBn_trainx, psi_nBn_testx = psi_nBn_testx,
                model = glmnet_fit, train_y = train$Y, test_y = test$Y))
}
