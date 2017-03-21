rm(list = ls())

library(mlbench)
library(nnet)
library(caret)

data("BostonHousing")

## Model using linear regression
lm_fit <- lm(medv ~ ., data = BostonHousing)
lm_pred <- predict(lm_fit)

## Mean squared error
print(mean((lm_pred - BostonHousing$medv)^2))
plot(BostonHousing$medv, lm_pred, main = 'LM Pred vs. Actual', xlab = 'Actual')

## Model using neural network: one hidden layer with 3 neurons
## Scale the input to get 0-1 range
nnet_fit <- nnet(medv / 50 ~ ., data = BostonHousing, size = 3)

## Multiply 50 to restore the original scale
nnet_pred <- predict(nnet_fit) * 50

## Mean squared error
print(mean((nnet_pred - BostonHousing$medv)^2))
plot(BostonHousing$medv, nnet_pred, main = 'LM Pred vs. Actual', xlab = 'Actual')

## Optimize the NN hyperparams
my_grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(4, 5, 6))
nnet_fit <- train(medv / 50 ~ ., data = BostonHousing, method = 'nnet', maxit = 1000,
                  tuneGrid = my_grid, trace = FALSE)
print(nnet_fit)
