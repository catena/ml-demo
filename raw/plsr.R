library(pls)

data(yarn)

# partial least squares regression
yarn.pls <- plsr(density ~ NIR, 6, data = yarn, validation = "CV")
summary(yarn.pls)

# Using caret
library(caret)
SLC14_1 <- function(n = 100) {
  dat <- matrix(rnorm(n*20, sd = 3), ncol = 20)
  foo <- function(x) x[1] + sin(x[2]) + log(abs(x[3])) + x[4]^2 + x[5]*x[6] + 
    ifelse(x[7]*x[8]*x[9] < 0, 1, 0) +
    ifelse(x[10] > 0, 1, 0) + x[11]*ifelse(x[11] > 0, 1, 0) + 
    sqrt(abs(x[12])) + cos(x[13]) + 2*x[14] + abs(x[15]) + 
    ifelse(x[16] < -1, 1, 0) + x[17]*ifelse(x[17] < -1, 1, 0) -
    2 * x[18] - x[19]*x[20]
  dat <- as.data.frame(dat)
  colnames(dat) <- paste0("Var", 1:ncol(dat))
  dat$y <- apply(dat[, 1:20], 1, foo) + rnorm(n, sd = 3)
  dat
}

set.seed(1)
training <- SLC14_1(100)
nTrain <- 1:50
trainX <- training[nTrain, -ncol(training)]
trainY <- training$y[nTrain]
testX <- training[-nTrain, -ncol(training)]
testY <- training$y[-nTrain]

ctrl <- trainControl(method = "cv", number = 5, returnResamp = "all",
                     verboseIter = TRUE)
set.seed(849)
model <- train(trainX, trainY, 
               method = "kernelpls", 
               trControl = ctrl,
               preProc = c("center", "scale"),
               tuneGrid = data.frame(ncomp = c(2, 4, 8, 16)))

preds <- predict(model, testX)
cor(preds, testY)^2

