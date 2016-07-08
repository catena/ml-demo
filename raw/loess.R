library(ggplot2)

cars.lo <- loess(dist ~ speed, cars)
speed.test <- data.frame(speed = seq(5, 30, 1))
cars.pred <- predict(cars.lo, speed.test, se = T)
with(cars, plot(speed, dist))
lines(speed.test$speed, cars.pred$fit, col = "red", lwd = 2)
