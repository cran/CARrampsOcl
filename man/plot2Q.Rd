\name{plot2Q}
\alias{plot2Q}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Function to produce image plot of 2-dimensional data modeled with 2
separate structure matrices.
%%  ~~function to do ... ~~
}
\description{Function to produce image plot of 2-dimensional data modeled with 2
separate structure matrices. 
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
plot2Q(objname, numcols = 64, col = rev(terrain.colors(numcols)), 
rev.inds = c(FALSE, FALSE))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{objname}{name of output object produced by \code{CARrampsOcl.fit}
%%     ~~Describe \code{objname} here~~
}
  \item{numcols}{number of shades from the color palette to be used
%%     ~~Describe \code{numcols} here~~
}
  \item{col}{ color palette to be used in plotting; the default plots high
values in green and low values in pink.
%%     ~~Describe \code{col} here~~
}
  \item{rev.inds}{ Should the plotting indices on the two-dimensional plot
be reversed?  Setting \code{rev.inds = c(TRUE,FALSE)} flips the plot from left to right;  \code{rev.inds = c(FALSE,TRUE)} turns the plot upside down.
%%     ~~Describe \code{rev.inds} here~~
}
}
\details{  This function plots two
two-dimensional plots side-by-side.  The left plot is of the raw data
input into the CARrampsOcl.fit function, and the right plot is of the 
estimated means of the posterior distributions of the 
corresponding random effects.

%%  ~~ If necessary, more details than the description above ~~
}
\value{  This function plots two
two-dimensional plots side-by-side.  The left plot is of the raw data
input into the CARrampsOcl.fit function, and the right plot is of the 
estimated means of the posterior distributions of the 
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
\author{Kate Cowles
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
# load data
  data(iowaSW06)

# construct structure matrix
  Q1<- makeRW2Q(33)       # for rows
  Q2<- makeRW2Q(24)       # for columns


# dimensions of Q1, Q2,  in that order
    na<- nrow(Q1)
    nb<- nrow(Q2)

Q <- list( list(type="Gen",content=Q1), list(type="Gen",content=Q2) )

# construct the design matrix with with as many columns as there are
# in null space of kronecker prod of Q's

X2 <- cbind( rep(1,nb), 1:nb)
X1 <- cbind( rep(1,na), 1:na)
X <-  kronecker( X2, X1)

# parameters of gamma prior densities on tausqy, tausqphi[1], tausqphi[2]
alpha2 = beta2 <- c(.1, .1, .1)

# number of samples
nsamp = 100

#random seed
myseed = 314

output <- CARrampsOcl.fit(alpha=alpha2,
            beta=beta2, Q=Q, y=iowaSW06,  nsamp=nsamp,
            seed=myseed,
            fixed = FALSE, randeffs=TRUE, coefs=TRUE,designMat=X,
            mult= 50)

# plot the raw data and the posterior means of the site-specific random effects

plot2Q( output, numcols=32, col = rev(terrain.colors(32)), rev.inds = c(FALSE, TRUE))

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ graphics }
%%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
