library(pls)

# principal component regression
yarn.pcr <- pcr(density ~ NIR, 6, data = yarn, validation = "CV")
summary(yarn.pcr)
