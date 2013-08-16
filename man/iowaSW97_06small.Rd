\name{iowaSW97_06small}
\alias{iowaSW97_06small}
\docType{data}
\title{Southwest Iowa 10-year normalized difference vegetation index NDVI values}
%%   ~~ data name/kind ... ~~

\description{
Normalized difference vegetation index (NDVI) values derived from satellite
image data from southwest Iowa and eastern Nebraska in July of each
year from 1997 through 2006.  These are 2040 values,
representing NDVI at each pixel on a rectangle with 17 rows and 12 columns
at each of 10 times.  
}
\usage{data(iowaSW97_06small)}
\format{
A vector of 2040 integer values.  The data are in row-major order within
year.
}
%%\details{
%%  ~~ If necessary, more details than the __description__ above ~~
%%}
\source{
http://glcf.umiacs.umd.edu/data/gimms/
}
\references{
Pinzon, J., Brown, M.E. and Tucker, C.J., 2005. Satellite time series correction of orbital drift artifacts using empirical mode decomposition. In: N. Huang (Editor), Hilbert-Huang Transform: Introduction and Applications, pp. 167-186.

Tucker, C.J., J. E. Pinzon, M. E. Brown, D. Slayback, E. W. Pak, R. Mahoney, E. Vermote and N. El Saleous (2005), An Extended AVHRR 8-km NDVI Data Set Compatible with MODIS and SPOT Vegetation NDVI Data. International Journal of Remote Sensing, Vol 26:20, pp 4485-5598.
}
\seealso{
\code{\link{plot3Q}}
}


\examples{
data(iowaSW97_06small)
}
\keyword{datasets}
