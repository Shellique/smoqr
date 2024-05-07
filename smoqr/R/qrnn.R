qrnn.cost <- function(weights, x, y, h, n.hidden, w, tau, monotone, additive,
           Th, Th.prime, tilted.approx, tilted.approx.prime, penalty, unpenalized){
  ## Weights Treatments
  # 降维惩罚项
  penalty2 <- ifelse(identical(Th, linear), penalty, 0)
  # 隐藏层和输出层的权重矩阵
  w1w2 <- qrnn.reshape(x, y, weights, n.hidden)
  W1 <- w1w2$W1; rW1 <- nrow(W1); cW1 <- ncol(W1)
  W2 <- w1w2$W2; rW2 <- nrow(W2); cW2 <- ncol(W2)
  # 是否monotone和additive
  if (!is.null(monotone)){
    W1[monotone,] <- exp(W1[monotone,])
    W2[1:(rW2-1), ] <- exp(W2[1:(rW2-1),])
  }
  if(!is.logical(additive)){
    W1 <- W1*additive
  }
  
  ## Forward pass
  x <- cbind(x, 1) # 原始数据N*(I+1)
  h1 <- x %*% W1 # 隐藏层的输入N*n.hidden
  y1 <- Th(h1)
  
  aug.y1 <- cbind(y1, 1) # 隐藏层的输出N*(n.hidden+1)
  h2 <- aug.y1 %*% W2 # 输出层的输入N*1
  y2 <- hramp(h2) # 输出层的输出，即yhat，N*1
  E <- y-y2 # 损失函数的输入（y-yhat）N*1
  
  ## Backward pass
  # output layer gradient
  delta2 <- hramp.prime(h2)*tilted.approx.prime(E, tau, h) # N*1
  gradient.W2 <- -(t(aug.y1) %*% sweep(delta2, 1, w, '*')) # (n.hidden+1)*1，输出层权重的梯度
  
  if (!is.null(monotone)){
    gradient.W2[1:(rW2-1),] <- gradient.W2[1:(rW2-1),]*W2[1:(rW2-1),]
  }
  gradient.W2.penalty <- 2*penalty2*rbind(W2[1:(rW2-1),,drop=FALSE], 0)/
    (length(W2)-cW2)
  
  # hidden layer gradient
  E1 <- delta2 %*% t(W2[1:(rW2-1),,drop=FALSE])
  delta1 <- Th.prime(h1)*E1
  
  gradient.W1 = -(t(x) %*% sweep(delta1, 1, w, '*')) # 隐藏层权重梯度
  
  if (!is.null(monotone)){
    gradient.W1[monotone,] <- gradient.W1[monotone,]*W1[monotone,]
  }
  W1p <- W1; W1p[c(unpenalized, rW1),] <- 0
  gradient.W1.penalty <- 2*penalty*W1p/sum(W1p != 0)
  
  ## Error & gradient
  cost <- sum(w*tilted.approx(E, tau, h)) +
    penalty*sum(W1p^2)/sum(W1p != 0) +
    penalty2*sum(W2[1:(rW2-1),,drop=FALSE]^2)/(length(W2)-cW2)
  gradient <- c(gradient.W1 + gradient.W1.penalty,
                gradient.W2 + gradient.W2.penalty)
  attr(cost, "gradient") <- gradient
  cost
}
qrnn.initialize <- function(x, y, n.hidden, init.range=c(-0.5, 0.5, -0.5, 0.5)){
  if(!is.list(init.range)){
    if(length(init.range)==4){
      r11 <- init.range[1]
      r12 <- init.range[2]
      r21 <- init.range[3]
      r22 <- init.range[4]
    } else{
      r11 <- r21 <- init.range[1]
      r12 <- r22 <- init.range[2]
    }
    W1 <- matrix(runif((ncol(x)+1)*n.hidden, r11, r12), ncol(x)+1, n.hidden)
    W2 <- matrix(runif((n.hidden+1)*ncol(y), r21, r22), n.hidden+1, ncol(y))
    weights <- c(W1, W2)
  } else{
    weights <- unlist(init.range)
  }
  weights
}
qrnn.nlm <- function(x, y, h, n.hidden, w, tau, iter.max, n.trials, bag, init.range,
           monotone, additive, Th, Th.prime, tilted.approx, tilted.approx.prime, 
           penalty, unpenalized, trace, ...){
  # Data Treatments
  cases <- seq(nrow(x))
  if (bag) cases <- sample(nrow(x), replace=TRUE)
  x <- x[cases,,drop=FALSE]
  y <- y[cases,,drop=FALSE]
  w <- w[cases]
  if(length(tau) > 1) tau <- tau[cases]
  cost.best <- Inf
  
  # Optimization
  for(i in seq(n.trials)){
    weights <- qrnn.initialize(x, y, n.hidden, init.range)
    fit <- suppressWarnings(nlm(qrnn.cost, weights, iterlim=iter.max,
                                x=x, y=y, h=h, n.hidden=n.hidden, w=w, tau=tau,
                                monotone=monotone, additive=additive, Th=Th,
                                Th.prime=Th.prime, tilted.approx=tilted.approx, tilted.approx.prime=tilted.approx.prime,
                                penalty=penalty, unpenalized=unpenalized,
                                check.analyticals=FALSE, ...))
    weights <- fit$estimate
    cost <- fit$minimum
    if(trace) cat(i, cost, "\n")
    if(cost < cost.best){
      cost.best <- cost
      weights.best <- fit$estimate
    }
  }
  if(trace) cat("*", cost.best, "\n")
  weights.best <- qrnn.reshape(x, y, weights.best, n.hidden)
  if(!is.logical(additive)){
    weights.best$W1 <- weights.best$W1*additive
  }
  weights.best
}
qrnn.reshape <- function(x, y, weights, n.hidden){
  N11 <- ncol(x)+1
  N12 <- n.hidden
  N1 <- N11*N12
  W1 <- weights[1:N1]
  W1 <- matrix(W1, N11, N12)
  N21 <- n.hidden+1
  N22 <- ncol(y)
  N2 <- N1 + N21*N22
  W2 <- weights[(N1+1):N2]
  W2 <- matrix(W2, N21, N22)
  list(W1=W1, W2=W2)
}
qrnn.predict <- function(x, parms){
  if (!is.matrix(x)) stop("\"x\" must be a matrix")
  weights <- parms$weights
  monotone <- parms$monotone
  additive <- parms$additive
  Th <- parms$Th
  x.center <- parms$x.center
  x.scale <- parms$x.scale
  y.center <- parms$y.center
  y.scale <- parms$y.scale
  x <- sweep(x, 2, x.center, "-")
  x <- sweep(x, 2, x.scale, "/")
  y.bag <- matrix(0, ncol=length(weights), nrow=nrow(x))
  for (i in seq_along(weights)){
    y.bag[,i] <- qrnn.eval(x, weights[[i]]$W1, weights[[i]]$W2,
                           monotone, additive, Th)
    y.bag[,i] <- y.bag[,i]*y.scale + y.center
  }
  y.bag
}
qrnn.eval <- function(x, W1, W2, monotone, additive, Th){
  if (!is.null(monotone)){
    W1[monotone, ] <- exp(W1[monotone, ])
    W2[1:(nrow(W2) - 1), ] <- exp(W2[1:(nrow(W2) - 1),])
  }
  if(!is.logical(additive)){
    W1 <- W1*additive
  }
  x <- cbind(x, 1)
  h1 <- x %*% W1
  y1 <- Th(h1)
  
  aug.y1 <- cbind(y1, 1)
  y2 <- aug.y1 %*% W2
  y2 <- hramp(y2)
  y2
}
qrnn.fit <- function(x, y, h, n.hidden, w=NULL, tau=0.5, n.ensemble=1, iter.max=5000,
         n.trials=5, bag=FALSE, init.range=c(-0.5, 0.5, -0.5, 0.5),
         monotone=NULL, additive=FALSE, Th=sigmoid, Th.prime=sigmoid.prime, penalty=0,
         tilted.approx=Epanechnikov.kernel, tilted.approx.prime=Epanechnikov.prime,
         unpenalized=NULL, n.errors.max=10, trace=TRUE, scale.y=TRUE, ...){
  # 声明
  if (!is.matrix(x)) stop("\"x\" must be a matrix")
  if (!is.matrix(y)) stop("\"y\" must be a matrix")
  if (h <= 0) stop("bandwidth must be greater then zero")
  if (any(is.na(c(x, y)))) stop("missing values in \"x\" or \"y\"")
  if (any(apply(x, 2, sd) < .Machine$double.eps^0.5)) stop("zero variance column(s) in \"x\"")
  if (ncol(y) != 1) stop("\"y\" must be univariate")
  if (any((tau > 1) | (tau < 0))) stop("invalid \"tau\"")
  if (!identical(Th, linear) && missing(n.hidden)) stop("must specify \"n.hidden\"")
  if (identical(Th, linear)) n.hidden <- 1
  is.whole <- function(x, tol = .Machine$double.eps^0.5) 
      abs(x - round(x)) < tol
  if (additive && !is.whole(n.hidden/ncol(x))) stop("\"n.hidden\" must be an integer multiple of \"ncol(x)\" when \"additive=TRUE\"")
  if(is.null(w)) w <- rep(1/nrow(y), nrow(y))
  if (any(w < 0)) stop("invalid \"w\"")
  
  # 数据中心化
  x <- scale(x)
  x.center <- attr(x, "scaled:center")
  x.scale <- attr(x, "scaled:scale")
  if(scale.y){
      y <- scale(y)
  } else{
      attr(y, "scaled:center") <- 0
      attr(y, "scaled:scale") <- 1
  }
  y.center <- attr(y, "scaled:center")
  y.scale <- attr(y, "scaled:scale")
  
  # 某些数据处理
  if(additive)
      additive <- gam.mask(x, n.hidden)
  weights <- vector("list", n.ensemble)
  if(trace){
      if(length(unique(tau)) <= 100){
          cat("tau =", unique(tau), "\n", sep=" ")
      } else{
          cat("tau = [", range(tau), "]; n.tau =", length(tau), "\n", sep=" ") 
      }
  }
  
  # 主体
  for (i in seq(n.ensemble)){
      if(trace) cat(i, "/", n.ensemble, "\n", sep="")
      w.tmp <- NA
      class(w.tmp) <- "try-error"
      n.errors <- 0
      while (inherits(w.tmp, "try-error")){
          w.tmp <- try(qrnn.nlm(x, y, h, n.hidden, w, tau, iter.max,
                                n.trials, bag, init.range, monotone, 
                                additive, Th, Th.prime, tilted.approx, tilted.approx.prime,
                                penalty, unpenalized, trace, ...),
                      silent = TRUE)
          n.errors <- n.errors + 1
          if (n.errors > n.errors.max) stop("nlm optimization failed")
      }
      weights[[i]] <- w.tmp
  }
  if(trace) cat("\n")
  parms <- list(weights=weights, tau=tau, Th=Th, x.center=x.center,
                x.scale=x.scale, y.center=y.center, y.scale=y.scale,
                monotone=monotone, additive=additive, h=h)
  parms
}
