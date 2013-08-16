\name{plotCAR1}
\alias{plotCAR1}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{ Function to plot 2-dimensional data modeled using a single structure
matrix.
%%  ~~function to do ... ~~
}
\description{ Function to plot 2-dimensional data modeled using a single structure
matrix.

%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
plotCAR1(objname, numcols = 64, col = rev(terrain.colors(numcols)), cardims, 
   rev.inds = c(FALSE, FALSE), 
   title = c("Raw data", "Estimated underlying truth"),
   sub = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{objname}{ name of output object produced by \code{CARrampsOcl.fit}
%%     ~~Describe \code{objname} here~~
}
  \item{numcols}{  number of shades from the color palette to be used
%%     ~~Describe \code{numcols} here~~
}
  \item{col}{ color palette to be used in plotting; the default plots high
values in green and low values in pink.
%%     ~~Describe \code{col} here~~
}
  \item{cardims}{ Two-vector specifying the number of rows and columns in the
CAR1 structure matrix; same as dimension of CAR1 Q matrix provided to
\code{CARrampsOcl.fit}.
%%     ~~Describe \code{cardims} here~~
}
  \item{rev.inds}{ Should the plotting indices on each two-dimensional plot
be reversed?  Setting \code{rev.inds = c(TRUE,FALSE)} flips the plots from left to right;  \code{rev.inds = c(FALSE,TRUE)} turns the plots upside down.
%%     ~~Describe \code{rev.inds} here~~
} 
  \item{title}{ Vector of character values; the titles of the two plots.
%%     ~~Describe \code{title} here~~
}
  \item{sub}{subtitle for plots; vector of length equal to the number
of pages of plots to be displayed
%%     ~~Describe \code{sub} here~~
}
}
\details{This function plots two
two-dimensional plots side-by-side.  The left plot is of the raw data
input into the CARrampsOcl.fit function, and the right plot is of the
estimated means of the of the posterior distributions of the
corresponding random effects.

%%  ~~ If necessary, more details than the description above ~~
} 
\value{ This function plots two
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
\author{ Kate Cowles
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
\dontrun{
data(iowaSW06)

Q <- list( list(type="CAR1",content=c(33,24)) )

# construct the design matrix with with as many columns as there are
# in null space of kronecker prod of Q's

X <-  matrix( rep(1,33*24), ncol=1)

# parameters of gamma prior densities on tausqy, tausqphi[1], tausqphi[2]
alpha2 = beta2 <- c(.1, .1)
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

plotCAR1( output, numcols=32, col = rev(terrain.colors(32)),
    cardims = c(33 ,24 ), rev.inds = c(FALSE, TRUE))
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ graphics }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
