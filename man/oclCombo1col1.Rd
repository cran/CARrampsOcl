\name{oclCombo1col1}
\alias{oclCombo1col1}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Function to call OpenCL to calculate posterior means and
standard deviations of random effects in models with 1 structure
matrix.

}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
Function to call OpenCL to calculate posterior means and
standard deviations of random effects in models with 1 structure
matrix.

}
\usage{
oclCombo1col1(a, D, tausqy, tausqphi, By)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{a}{Eigenvector matrix of 1st Q matrix.
%%     ~~Describe \code{a} here~~
}
  \item{D}{atrix of eigenvalues from the structure matrix.
%%     ~~Describe \code{D} here~~
}
  \item{tausqy}{Vector of samples of measurement error precision.
%%     ~~Describe \code{tausqy} here~~
}
  \item{tausqphi}{Matrix sampled values of spatial precisions.
%%     ~~Describe \code{tausqphi} here~~
} 
  \item{By}{ Vector resulting from premultiplication of data vector y b
transpose of kronecker sum of eigenvector matrices.

%genvector matrix of 1st Q matrix.%     ~~Describe \code{By} here~~
}
}
%%\details{
%%  ~~ If necessary, more details than the description above ~~
%%}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
\item{phimean }{Vector of means of posterior densities of random effects}
 \item{phisd }{Vector of standard deviations of marginal posterior}
}
%%\references{
%% ~put references to the literature/web site here ~
%%}
\author{ Michael Seedorff
%%  ~~who you are~~
}
%%\note{
%%  ~~further notes~~
%%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%%}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (a, D, tausqy, tausqphi, By) 
{
    if (!is.numeric(a)) 
        stop("a must be a numeric matrix")
    na1 <- nrow(a)
    nc1 <- length(tausqy)
    if (is.null(ncol(tausqphi))) 
        F1 <- 2
    else F1 <- ncol(tausqphi) + 1
    mresults <- rep(0, 2 * na1)
    out <- .C("oclCombo1col1", a = as.double(as.vector(t(a))), 
        D = as.double(as.vector(t(D))), tausqy = as.double(tausqy), 
        tausqphi = as.double(as.vector(t(tausqphi))), By = as.double(By), 
        results = as.double(mresults), na1 = as.integer(na1), 
        nc1 = as.integer(nc1), F1 = as.integer(F1))
    return(list(phimean = out$results[1:na1], phisd = sqrt(out$results[(na1 + 
        1):(2 * na1)]/(nc1 - 1))))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
