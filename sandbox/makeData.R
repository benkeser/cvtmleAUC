makeData <- function(n, p){
	X <- matrix(rnorm(n*p), nrow = n, ncol = p)
	if(p < 10){
		Y <- rbinom(n, 1, plogis(X[,1]))		
	}else{
		Y <- rbinom(n, 1, plogis(0.25*X[,1] - 0.5*X[,10] + 0.5*X[,2]*X[,3]))		
	}
	return(list(X = X, Y = Y))
}