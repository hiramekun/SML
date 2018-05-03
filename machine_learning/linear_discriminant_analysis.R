library(mvtnorm)
sigma = matrix(c(1, 1, 1, 9), ncol = 2)
rand = rmvnorm(n = 1000, mean = c(2, 0), sigma)
rand = rmvnorm(n = 1000, mean = c(-2, 0), sigma)
x1 = rand[, 1]
x2 = rand[, 2]
plot(x1, x2)

x1 = seq(-3, 3, length = 50)
x2 = x1
f = function(x1, x2, mean) {
  dmvnorm(matrix(c(x1, x2), ncol = 2), mean = mean, sigma = sigma)
}
mu1 = c(2, 0)
mu2 = c(-2, 0)

alpha = 0.1
n = 600
n1 = sum(runif(600) < alpha)
n2 = n - n1
x1 = c(rnorm(n1, mean = 2), 1 * rnorm(n1))
x2 = c(rnorm(n2, mean = -2), 3 * rnorm(n2))

a = solve(sigma) %*% (mu1 - mu2)
b = -1 / 2 * (t(mu1) %*% solve(sigma) %*% mu1 - t(mu2) %*% solve(sigma) %*% mu2) + log(n1 /
                                                                                         n2)

linear = function(x) {
  if (a[2] == 0) {
    rep(0, length(x))
  } else {
    -(a[1] / a[2]) * x + rep(b / a[2], length(x))
  }
}
plot(
  rnorm(n1, mean = 2),
  3 * rnorm(n1),
  xlim = c(-6, 6),
  ylim = c(-10, 10),
  col = "blue",
  pch = 20
)
par(new = T)
plot(
  rnorm(n2, mean = -2),
  3 * rnorm(n2),
  xlim = c(-6, 6),
  ylim = c(-10, 10),
  col = "red",
  pch = 20
)

par(new = T)
if (a[2] == 0) {
  abline(v = 0)
} else{
  plot(
    -6:6,
    linear(-6:6),
    xlim = c(-6, 6),
    ylim = c(-10, 10),
    type = "l"
  )
}