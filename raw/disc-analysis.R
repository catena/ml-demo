##### Discriminant Analysis

## Linear Discriminant Analysis
library(MASS)
Iris <- data.frame(rbind(iris3[, , 1], iris3[, , 2], iris3[, , 3]),
                   Sp = rep(c("s", "c", "v"), rep(50, 3)))
train <- sample(1:150, 75)
table(Iris$Sp[train])
z <- lda(Sp ~ ., Iris, prior = c(1, 1, 1) / 3, subset = train)
predict(z, Iris[-train,])$class
(z1 <- update(z, . ~ . - Petal.W.))


## Quadratic Discriminant Analysis
library(MASS)
tr <- sample(1:50, 25)
train <- rbind(iris3[tr, , 1], iris3[tr, , 2], iris3[tr, , 3])
test <- rbind(iris3[-tr, , 1], iris3[-tr, , 2], iris3[-tr, , 3])
cl <- factor(c(rep("s", 25), rep("c", 25), rep("v", 25)))
z <- qda(train, cl)
predict(z, test)$class

