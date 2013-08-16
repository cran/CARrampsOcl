\name{oclKronVectMult1col3Q}
\alias{oclKronVectMult1col3Q}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to call OpenCL program to premultiply a matrix by a
kronecker product of 3 matrices.
}
\description{
Function to call OpenCL program to premultiply a matrix by a 
kronecker product of 3 matrices.
}
\usage{
oclKronVectMult1col3Q(kernel, a, b, c, y, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{kernel}{ OpenCL kernel to be run }

  \item{a}{ first matrix in kronecker product
%%     ~~Describe \code{a} here~~
}
  \item{b}{second matrix in kronecker product
%%     ~~Describe \code{b} here~~
}
  \item{c}{third matrix in kronecker product
%%     ~~Describe \code{b2} here~~
}
  \item{y}{ matrix to premultiply by kronecker product
%%     ~~Describe \code{c} here~~
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
matrix; the product of the kronecker product of the first three arguments
and the matrix y
}
%%\references{
%% ~put references to the literature/web site here ~
%%}
\author{ Kate Cowles and Michael Seedorff
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
function(a,b,b2,c)
{
   if(!is.numeric(a) | !is.numeric(b) | !is.numeric(c) )
          stop("a and b must be numeric matrices")

   na1 <- nrow(a)
   nb1 <- nrow(b)
   nb21 <- nrow(b2)
   nc1 <- ncol(c)
   a<- as.vector(t(a))
   b<- as.vector(t(b))
   b2<- as.vector(t(b2))
   c<- as.vector(c)     # change; not transposed before sent
   retvect <- rep(0, na1 * nb1 * nb21* nc1)

   out <- .C("doKronVectMult1col3QD", a=as.double(a), b = as.double(b),
              b2 = as.double(b2),
              c = as.double(c),
               retvect=as.double(retvect), na1 = as.integer(na1),
               nb1 = as.integer(nb1), nb21=as.integer(nb21), 
               nc1 = as.integer(nc1), PACKAGE="CARramps" )
   return(matrix(out$retvect, nrow=na1*nb1*nb21,byrow=F) ) # change byrow
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
