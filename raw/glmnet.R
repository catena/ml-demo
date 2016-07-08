library(glmnet)

# multinomial
set.seed(100)
X <- matrix(rnorm(100 * 20), 100, 20)
Y <- rnorm(100)

ans1 <- cv.glmnet(X, Y, alpha = 0) # ridge
plot(ans1$glmnet.fit, "lambda", label = FALSE)

ans2 <- cv.glmnet(X, Y, alpha = 1) # lasso
plot(ans2$glmnet.fit, "lambda", label = FALSE)

ans3 <- cv.glmnet(X, Y, alpha = 0.5) # elastic net
plot(ans3$glmnet.fit, "lambda", label = FALSE)
