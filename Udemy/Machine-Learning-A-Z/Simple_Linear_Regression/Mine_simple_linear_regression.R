dataset = read.csv('Salary_Data.csv')

#dataset = dataset[, 2:3]

#Splitting the dataset into the training set and Test Set
# install.packages('caTools') 
library(caTools)
set.seed(123)
##Aqui ele separa a lógica onde 80% vai ser o meu traning set  e 20% o meu test_set
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fitting SImple  Linear Regression to the Training set
regressor = lm(formula = Salary ~YearsExperience, data = training_set)

#prediting the test set results
y_pred = predict(regressor, newdata = test_set)

#visualising the Traning set results
#install.packages('ggplot2')
library(ggplot2)
ggplot()  +web

  geom_point(aes(x = training_set$YearsExperience, y =  training_set$Salary),
                      colour = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)')+
  xlab('YearsExperience')+
  ylab('Salary')

ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y =  test_set$Salary),
             colour = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)')+
  xlab('YearsExperience')+
  ylab('Salary')

