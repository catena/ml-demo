library(mlbench)
data(BostonHousing)
data(HouseVotes84)

mod1 <- cubist(x = BostonHousing[, -14], y = BostonHousing$medv)
summary(mod1)

mod2 <- M5Rules(mpg ~ ., data = mtcars)
print(mod2)

mod3 <- OneR(Class~., HouseVotes84)
print(mod3)

mod4 <- JRip(Class~., HouseVotes84)
print(mod4)
