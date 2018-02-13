makeData <- function(n, p){
	X <- matrix(rnorm(n*p), nrow = n, ncol = p)
	if(p < 10){
		Y <- rbinom(n, 1, plogis(X[,1]))		
	}else{
		Y <- rbinom(n, 1, plogis(X[,1] - X[,10]))		
	}
	return(list(X = X, Y = Y))
}