#' Print results of cvauc_cvtmle
#' @param x An object of class "cvauc"
#' @export
print.cvauc <- function(x, ...){
	out <- data.frame(rbind(
	 c(x$est_cvtmle, x$se_cvtmle, x$est_cvtmle - 1.96*x$se_cvtmle, x$est_cvtmle + 1.96*x$se_cvtmle),
	 c(x$est_onestep, x$se_onestep, x$est_onestep - 1.96*x$se_onestep, x$est_onestep + 1.96*x$se_onestep),
	 c(x$est_esteq, x$se_esteq, x$est_esteq - 1.96*x$se_esteq, x$est_esteq + 1.96*x$se_esteq),
	 c(x$est_empirical, x$se_empirical, x$est_empirical - 1.96*x$se_empirical, x$est_empirical + 1.96*x$se_empirical)
  	))
  	colnames(out) <- c("est","se","cil","ciu")
  	row.names(out) <- c("cvtmle","onestep","esteq","empirical")
  	print(out)
}