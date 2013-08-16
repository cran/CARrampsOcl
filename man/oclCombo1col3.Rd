\name{oclCombo1col3}
\alias{oclCombo1col3}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Function to call OpenCL  to calculate posterior means and
standard deviations of random effects in models with 2 structure
matrices.

%%  ~~function to do ... ~~
}
\description{
Function to call OpenCL  to calculate posterior means and
standard deviations of random effects in models with 2 structure
matrices.
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
oclCombo1col3(a, b, b2, D, tausqy, tausqphi, By)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{a}{ Eigenvector matrix of 1st Q matrix.}
  \item{b}{ Eigenvector matrix of 2nd Q matrix.}
  \item{b2}{ Eigenvector matrix of 3rd Q matrix.}
  \item{D}{ Matrix of eigenvalues from the 2 structure matrices.}
 \item{tausqy}{
Vector of samples of measurement error precision.}
  \item{tausqphi}{
Matrix sampled values of spatial precisions.}
  \item{By}{
Vector resulting from premultiplication of data vector y b
transpose of kronecker sum of eigenvector matrices.
}

}
%%\details{
%%  ~~ If necessary, more details than the description above ~~
%%}
\value{
 \item{phimean }{Vector of means of posterior densities of random effects}
 \item{phisd }{Vector of standard deviations of marginal posterior}
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
%%\references{
%% ~put references to the literature/web site here ~
%%}
\author{Michael Seedorff
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
function (a, b, b2, D, tausqy, tausqphi, By) 
{
    if (!is.numeric(a) | !is.numeric(b) | !is.numeric(b2)) 
        stop("a, b, and b2 must be numeric matrices")
    na1 <- nrow(a)
    nb1 <- nrow(b)
    nb21 <- nrow(b2)
    nc1 <- length(tausqy)
    F1 <- ncol(tausqphi) + 1
    nab <- na1 * nb1 * nb21
    mresults <- rep(0, 2 * nab)
    out <- .C("oclCombo1col3", a = as.double(as.vector(t(a))), 
        b = as.double(as.vector(t(b))), b2 = as.double(as.vector(t(b2))), 
        D = as.double(as.vector(t(D))), tausqy = as.double(tausqy), 
        tausqphi = as.double(as.vector(t(tausqphi))), By = as.double(By), 
        results = as.double(mresults), na1 = as.integer(na1), 
        nb1 = as.integer(nb1), nb21 = as.integer(nb21), nc1 = as.integer(nc1), 
        F1 = as.integer(F1))
    return(list(phimean = out$results[1:nab], phisd = sqrt(out$results[(nab + 
        1):(2 * nab)]/(nc1 - 1))))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
