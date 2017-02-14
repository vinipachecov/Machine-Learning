dataset = read.csv('50_Startups.csv')

#dataset = dataset[, 2:3]

#encoding categorical data Countries and  Purchased column
dataset$State = factor(dataset$State, 
                       levels = c('New York','California', 'Florida'),
                       labels = c(1,2,3))#Splitting the dataset into the training set and Test Set
# install.packages('caTools') 
library(caTools)
set.seed(123)
##Aqui ele separa a lógica onde 80% vai ser o meu traning set  e 20% o meu test_set
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)