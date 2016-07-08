library(party)

set.seed(1010)

airq <- subset(airquality, !is.na(Ozone))
airct <- ctree(Ozone ~ ., data = airq, 
               controls = ctree_control(minsplit = 0.1 * nrow(airq)))
print(airct)