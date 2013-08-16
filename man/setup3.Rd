\name{setup3}
\alias{setup3}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{ Function to invoke OpenCL package and create kernels
%%  ~~function to do ... ~~
}
\description{ This function sets up the OpenCL environment and creates kernels.
It depends on the OpenCL package.
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
setup3(dev)
}
\arguments{
\item{dev}{device list from OpenCL package function oclDevices}
}

%- maybe also 'usage' for other objects documented here.
%%\details{
%%  ~~ If necessary, more details than the description above ~~
%%}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
List of OpenCL kernels
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
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.
\dontrun{
require(OpenCL)
plat <- oclPlatforms()
dev <- oclDevices( plat[[1]] )
if(!any(grepl("cl_khr_fp64", oclInfo(dev[[1]])$exts)))
    stop("GPU with double precision and Open CL capabilities required.")

k.sampling <- setup3(dev=dev)
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal }
