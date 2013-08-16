\name{plotem}
\alias{plotem}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{ Utility function used by plotting functions.
%%  ~~function to do ... ~~
}
\description{ Utility function used by plotting function.
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
plotem(range1, range2, objname, range3, nrow, numcols, col, rev.inds, breaks, title, sub)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{range1}{indices for first plotting dimension 
%%     ~~Describe \code{range1} here~~
}
  \item{range2}{ indices for second plotting dimension
%%     ~~Describe \code{range2} here~~
}
  \item{objname}{ name of output object from CARrampsOcl.fit to be plotted
%%     ~~Describe \code{objname} here~~
}
  \item{range3}{ which coordinate to plot at
%%     ~~Describe \code{range3} here~~
}
  \item{nrow}{  number of rows in matrix format for "image"
%%     ~~Describe \code{nrow} here~~
}
  \item{numcols}{Number of shades from color palette.
%%     ~~Describe \code{numcols} here~~
}
  \item{col}{ Color palette.
%%     ~~Describe \code{col} here~~
}
  \item{rev.inds}{  Reverse indices on either axis?
%%     ~~Describe \code{rev.inds} here~~
}
  \item{breaks}{ Cut points for quantitative variable determining color shades.
%%     ~~Describe \code{breaks} here~~
}
  \item{title}{ Titles for each of the two plots
%%     ~~Describe \code{title} here~~
}
  \item{sub}{ Subtitle
%%     ~~Describe \code{sub} here~~
}
}
%%\details{
%%  ~~ If necessary, more details than the description above ~~
%%}
\value{  Produces an image plot
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
%%\references{
%% ~put references to the literature/web site here ~
%%a}
\author{ Kate Cowles
%%  ~~who you are~~
}
\note{
Uses \code{image.plot} from the \code{fields} package.
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%%}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
\dontrun{
function (range1, range2, objname, range3, nrow, numcols, col, 
    rev.inds, breaks, title, sub) 
{
    inds1 <- range1
    inds2 <- range2
    if (rev.inds[1]) 
        inds1 <- rev(inds1)
    if (rev.inds[2]) 
        inds2 <- rev(inds2)
    image.plot(range1, range2, matrix(objname$y[range3], nrow = nrow)[inds1, 
        inds2], col = col, breaks = breaks, xlab = "", ylab = "", 
        main = title[1], sub = sub)
    image.plot(range1, range2, matrix(objname$phi$phimean[range3], 
        nrow = nrow)[inds1, inds2], col = col, xlab = "", ylab = "", 
        breaks = breaks, main = title[2], sub = sub)
  }
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
