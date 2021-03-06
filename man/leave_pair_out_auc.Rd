% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cvtmle_functions.R
\name{leave_pair_out_auc}
\alias{leave_pair_out_auc}
\title{Function to do leave pair out AUC computation}
\usage{
leave_pair_out_auc(Y, X, learner = "glm_wrapper", seed = 1234,
  max_pairs = NULL, nested_cv = FALSE, parallel = FALSE, ...)
}
\arguments{
\item{Y}{The outcome}

\item{X}{The predictors}

\item{learner}{The learner wrapper}

\item{seed}{A random seed to set}

\item{parallel}{Compute the predictors in parallel?}

\item{K}{The number of folds}
}
\description{
Function to do leave pair out AUC computation
}
