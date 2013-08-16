\name{iowaSW06}
\alias{iowaSW06}
\docType{data}
\title{Iowa normalized difference vegetation index (NDVI) data}
\description{
Normalized difference vegetation index (NDVI) values derived from satellite
image data from southwest Iowa and eastern Nebraska in July 2006.  
These are 792 values,
representing NDVI at each pixel on a rectangle with 33 rows and 24 columns.
The cities of Des Moines, Omaha/Council Bluffs, and Lincoln are included in
the rectangular area.
}
\usage{data(iowaSW06)}
\format{
  A vector of 792 integer values.  The data are in row-major order.
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
\code{\link{plot2Q}}
}
\examples{
data(iowaSW06)
}
\keyword{datasets}
