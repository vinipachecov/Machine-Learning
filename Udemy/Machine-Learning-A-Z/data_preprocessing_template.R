# data preprocessing

dataset = read.csv('Data.csv')

#dataset = dataset[, 2:3]

#Splitting the dataset into the training set and Test Set
# install.packages('caTools') 
library(caTools)
set.seed(123)
##Aqui ele separa a lógica onde 80% vai ser o meu traning set  e 20% o meu test_set
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])



