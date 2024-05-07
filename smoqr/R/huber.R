hramp.prime <- function(x){
  1
}
hramp <- function(x){
  x
}

Gaussian.kernel <- function(x, tau, h){
  obj <- h * dnorm((x/h)) + x * (tau - pnorm((-x/h)))
  obj[is.nan(obj)] <- 0
  obj
}
Gaussian.prime <- function(x, tau, h){
  obj <- tau - pnorm((-x/h))
  obj[is.nan(obj)] <- 0
  obj
}

Logistic.kernel <- function(x, tau, h){
  obj <- tau * x + h * log(1 + exp(-x/h))
  obj[is.nan(obj)] <- 0
  return(obj)
}
Logistic.prime <- function(x, tau, h){
  obj <- tau - 1/(1 + exp(x/h))
  obj[is.nan(obj)] <- 0
  return(obj)
}

Uniform.kernel <- function(x, tau, h){
  obj <- (h/2)*((((x/h)^2 + 1)/2) * (abs(x) <= h) + abs(x/h) * (abs(x) > h)) + x*(tau - 0.5)
  obj[is.nan(obj)] <- 0
  return(obj)
}
Uniform.prime <- function(x, tau, h){
  x[x< -h] <- tau - 1
  x[abs(x) <= h] <- x/(2*h) + tau -1
  x[x > h] <- tau
  x[is.nan(x)] <- 0
  x
}

Epanechnikov.kernel <- function(x, tau, h){
  obj <- (tau - 0.5)*x + (h/2)*((0.75*(x/h) - ((x/h)**4)/8 + 0.375)*(abs(x) <= h) + abs(x)*(abs(x) > h))
  obj[is.nan(obj)] <- 0
  return(obj)
}
Epanechnikov.prime <- function(x, tau, h){
  x[x< -h] <- tau - 1
  x[abs(x) <= h] <-  ((3*x[abs(x) <= h])/h - (x[abs(x) <= h]/h)**3)/4 + tau - 0.5
  x[x > h] <- tau
  x[is.nan(x)] <- 0
  x
}

Triangular.kernel <- function(x, tau, h){
  obj <- (tau - 0.5)*x + (h/2)*(((x/h)^2 - (abs(x/h)^3)/h + 1/3)*(abs(x <=h)) + abs(x/h) * (abs(x) > h))
  obj[is.nan(obj)] <- 0
  return(obj)
}
Triangular.prime <- function(x, tau, h){
  x[x <= -h] <- tau - 1
  x[x <= h & x > 0] <- x/h - ((x/h)^2)/2 + tau - 0.5
  x[x <= 0 & x > -h] <- x/h + ((x/h)^2)/2 + tau - 0.5
  x[x > h] <- tau
  x[is.nan(x)] <- 0
  x
}
