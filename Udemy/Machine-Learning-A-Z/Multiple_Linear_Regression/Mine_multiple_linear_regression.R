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

## Fitting Multiple Linear Regression to the Training Set
#regressor = lm(formula = Profit ~ R.D.Spend +  Administration + Marketing.Spend + State)
regressor = lm(formula = Profit ~  ., 
               data = training_set)

#Predict The Test set Results
y_pred = predict(regressor, newdata = test_set)


##building the optimal using backward Elimitation

regressor = lm(formula = Profit ~  R.D.Spend + Administration + Marketing.Spend + State, 
               data = dataset)

summary(regressor)

regressor = lm(formula = Profit ~  R.D.Spend + Administration + Marketing.Spend, 
               data = dataset)

summary(regressor)

regressor = lm(formula = Profit ~  R.D.Spend +  Marketing.Spend, 
               data = dataset)

summary(regressor)

regressor = lm(formula = Profit ~  R.D.Spend , 
               data = dataset)

summary(regressor)

y_pred = predict(regressor, newdata = test_set)

