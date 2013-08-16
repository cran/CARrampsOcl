\name{plot3Q}
\alias{plot3Q}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{ Plot 3-dimensional data modeled with 3 separate structure matrices
%%  ~~function to do ... ~~
}
\description{Plot 3-dimensional data modeled with 3 separate structure matrices.


%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
plot3Q(objname, numcols = 64, col = rev(terrain.colors(numcols)), 
plotdims = c(1, 2), rev.inds = c(FALSE, FALSE), blocks = NULL, 
animate = FALSE, intv = 3, 
title = c("Raw data", "Estimated underlying truth"), sub = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{objname}{ name of output object produced by \code{CARrampsOcl.fit}
%%     ~~Describe \code{objname} here~~
}
  \item{numcols}{ number of shades from the color palette to be used
%%     ~~Describe \code{numcols} here~~
}
  \item{col}{ color palette to be used in plotting; the default plots high
values in green and low values in pink. 
%%     ~~Describe \code{col} here~~
}
  \item{plotdims}{ This function produces a sequence of two-dimensional plots.
The \code{plotdims} argument is a 2-vector identifying which two dimensions
appear in the rows and columns of each two-dimensional plot.  The numbers refer
to the order of the structure matrices specified in the \code{Q} argument to
\code{CARrampsOcl.fit}.  The default is the first two dimensions.
%%     ~~Describe \code{plotdims} here~~
}
  \item{rev.inds}{ Should the plotting indices on each two-dimensional plot
be reversed?  Setting \code{rev.inds = c(TRUE,FALSE)} flips the plots from left to right;  \code{rev.inds = c(FALSE,TRUE)} turns the plots upside down.
%%     ~~Describe \code{rev.inds} here~~
}
  \item{blocks}{Subset which two-dimensional plots are displayed.  Default
is that the corresponding two-dimensional plot is displayed for each value of
the third dimension.  To subset, set \code{blocks} to a set of integers.  For
example, \code{blocks = 1:5} will display the first 5 plots only.
%%     ~~Describe \code{blocks} here~~
}
  \item{animate}{If \code{animate = FALSE}, the user is prompted to press
a key to move to the next plot.  Otherwise, the next plot will automatically
appear after \code{intv} seconds.
%%     ~~Describe \code{animate} here~~
}
  \item{intv}{ How many seconds to wait before displaying next plot (ignored
if \code{animate=FALSE}).
%%     ~~Describe \code{intv} here~~
}
  \item{title}{ Vector of character values; the titles of the two plots.
%%     ~~Describe \code{title} here~~
}
  \item{sub}{ Vector of subtitles for plots.
%%     ~~Describe \code{sub} here~~
}
}
\details{ For each value of a third dimension, this function plots two
two-dimensional plots side-by-side.  The left plot is of the raw data
input into the CARrampsOcl.fit function, and the right plot is of the 
estimated means of the of the posterior distributions of the 
corresponding random effects.
%%  ~~ If necessary, more details than the description above ~~
}
\value{ For each value in \code{blocks}, this function plots two
two-dimensional plots side-by-side.  The left plot is of the raw data
input into the CARrampsOcl.fit function, and the right plot is of the 
estimated means of the of the posterior distributions of the 
corresponding random effects.

%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
%%\references{
%% ~put references to the literature/web site here ~
%%}
\author{ Alex Sawyer and Kate Cowles
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
\dontrun{
data(iowaSW97_06small)
na<- 17
nb <- 12
nc <- 10
Q <- list( list(type="RW1",content=na), list(type="RW1",content=nb), 
list(type="RW1",content=10))
alpha <- beta <- rep(0.01,4)
X<- matrix( rep(1, na * nb * nc), ncol=1)

# Add noise to data to see whether fitting method can extract true image
y <- iowaSW97_06small/100 + rnorm(na*nb*nc,sd=6)

outputSW <- CARrampsOcl.fit(alpha, beta, Q, y, nsamp=50,                  
     seed=2, fixed = FALSE, coefs = TRUE,
     randeffs = TRUE, designMat = X, mult = 2000, filename = "params.txt")
require(coda)
summary( as.mcmc(1/sqrt(outputSW$params)))

plot3Q( objname=outputSW, numcols=64, col=rev(terrain.colors(64)),
plotdims=c(1,2), rev.inds=c(FALSE,TRUE), blocks=NULL, animate=TRUE,
intv=3, title=c("Raw data","Estimated underlying truth"), sub=1997:2006 )
}
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ graphics }
%%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
