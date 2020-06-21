
# CLASSIFICATION USING DECISION TREE ENSEMBLES
# Jochen Hartmann, Hamburg University

# DATA SOURCE: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing (accessed August 30, 2019)
# Moro, S., Cortez, P., & Rita, P.. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014.
# Dua, D. & Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science, available at: http://archive.ics.uci.edu/ml.

# load packages
library(rio)
library(caret)
library(rpart.plot)
library(dplyr)
library(rattle)
library(parallel)
library(doParallel)
library(partykit)

# import data
df <- import("../data/uci/bank/bank-full.csv") # set path to location of bank-full.csv
df$day <- as.character(df$day)

# explore data
anyNA(df)
str(df)
head(df,3)
tbl <- table(df$y); tbl
round(prop.table(tbl)*100,2)

# balance data
rows <- tbl[c("yes")]
set.seed(0); df_bal <- rbind(df %>% filter(y == "yes") %>% sample_n(rows), 
                             df %>% filter(y == "no") %>% sample_n(rows))

# shuffle rows
set.seed(0); shuffle_index <- sample(nrow(df_bal))
df_bal <- df_bal[shuffle_index,]
head(df_bal,3)

# create summary statistics for numeric variables
num_var <- c("age", "balance", "duration", "campaign", "pdays", "previous")
summary(df_bal[,num_var]) # ignore -1 for pdays
round(sapply(df_bal[,num_var], sd), 2)

# partition data
partition <- .8
set.seed(0); train_index <- createDataPartition(y = df_bal$y, p = partition, list = FALSE)
train <- df_bal[train_index,]
test <- df_bal[-train_index,]
sqrt_cols <- round(sqrt(ncol(train)))
table(train$y)

# specify training procedure
folds = 10
repeats = 1
params = 2

# populate seeds vector
set.seed(0); seeds <- vector(mode = "list", length = (folds*repeats)+1)
for(i in 1:(length(seeds)-1)) seeds[[i]] <- sample.int(n=1000, params)
seeds[[length(seeds)]] <- sample.int(1000, 1)
tail(seeds,3)

# define training control
ctrl <- trainControl(method = "repeatedcv", 
                     number = folds,
                     repeats = repeats, 
                     seeds = seeds,
                     selectionFunction = "best", 
                     savePredictions = "final",
                     classProbs = TRUE,
                     verboseIter = TRUE)

# set hyperparameter grids
grid_svm <- expand.grid(C = c(0.01,0.1))
grid_dt <- expand.grid(cp = c(0.01, 0.1))
grid_rf <- expand.grid(mtry = c(sqrt_cols,sqrt_cols*2), splitrule = "gini", min.node.size = 1)
grid_ab <- expand.grid(nIter = c(100,200), method="Adaboost.M1")
grid_xgb <- expand.grid(nrounds = c(100,200), max_depth = 6, eta = 0.3, gamma = 0, 
                        colsample_bytree = 1, min_child_weight = 1, subsample = 1)

# store models and hyperparameters
models <- c("svmLinear", "rpart", "ranger", "adaboost", "xgbTree")
nb_models <- length(models)
param_grids <- as.data.frame(matrix(nrow = nb_models, ncol = 2))
colnames(param_grids) <- c("model", "param_grid")
param_grids$model = models
param_grids$param_grid = list(grid_svm, grid_dt, grid_rf, grid_ab, grid_xgb)
param_grids

# initialize parallel processing
nb_cores <- detectCores()
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# train models
m <- list()
set.seed(0); system.time(
for(i in 1:nb_models){
  
  model <- models[[i]]
  param_grid <- param_grids$param_grid[[i]]
  print(paste0(i,": ", model))
  
  if(model == "ranger"){
  
    m[[i]] <- train(y ~ ., data = train,
                    method = model,
                    tuneGrid = param_grid,
                    metric = "Accuracy",
                    importance = "impurity", # specify variable importance measure
                    trControl = ctrl)
  }
  
  else{
    
    m[[i]] <- train(y ~ ., data = train, 
                    method = model,
                    tuneGrid = param_grid,
                    metric = "Accuracy",
                    trControl = ctrl)
  }
  
})

# make predictions
p <- list()
cm <- list()
for(i in 1:nb_models){
  
  model <- m[[i]]
  p[[i]] <- predict(model, test[,-which(colnames(test) == "y")])
  cm[[i]] <- confusionMatrix(p[[i]], factor(test$y))

}

# store tuned hyperparameters and test accuracy
final_params <- unlist(lapply(lapply(m, '[[', "bestTune"), function(x) x[1]))
test_acc <- unlist(lapply(lapply(cm, '[[', "overall"), function(x) x[1]))
true_name <- c("Support Vector Machine", "Decision Tree", "Random Forest", "AdaBoost", "XGBoost")
summary <- data.frame(model_name = true_name, model = models, param_name = names(final_params), 
                      param_value = final_params, test_acc = test_acc)
rownames(summary) <- c()

# print summary
summary

# plot decision tree (Fig. 1)
rpart.plot(m[[2]]$finalModel, 
           box.palette="Blues",
           tweak = 1.5,
           fallen.leaves = TRUE,
           round = 0,
           type=1)

# translate decision tree to rules
rpart.rules(m[[2]]$finalModel, extra = 4, cover = TRUE)

# plot variable importance for rf (Fig. 2)
var_imp <- varImp(m[[3]])$importance
var_imp <- data.frame(variable = rownames(var_imp), importance = var_imp$Overall)
var_imp <- var_imp %>% arrange(importance)
top_features <- 10
ggplot(tail(var_imp,top_features), aes(x=variable, y=importance, group=1)) + 
  geom_bar(stat="identity", width=0.7, fill="steelblue") +
  labs(x = "\nVariable", 
       y = "Relative Importance\n") +
  scale_x_discrete(limits=tail(var_imp,top_features)$variable) +
  coord_flip() +
  theme_minimal()

# define scale function to adjust decimal points in ggplot
scaleFUN2 <- function(x) sprintf("%.2f", x)
scaleFUN4 <- function(x) sprintf("%.4f", x)

# plot accuracies (Fig. 4)
ggplot(summary, aes(x=model_name, y=test_acc, group=1)) + 
  geom_bar(stat="identity", width=0.7, fill="steelblue") +
  coord_cartesian(ylim=c(0.50,0.90)) +
  scale_y_continuous(labels=scaleFUN2) + 
  geom_text(aes(label=scaleFUN4(test_acc)), vjust=-0.5, size=3.5) +
  labs(x = "\nMethod", 
       y = "Accuracy\n") +
  scale_x_discrete(limits=summary$model_name[c(2,1,3,4,5)]) +
  geom_hline(yintercept=.5, linetype="dashed", color = "red") + 
  theme_minimal()

# run rf with varying number of trees
trees <- c(1,5,10,50,100,500)
trees
rf <- list()
for(i in 1:length(trees)){
  num_trees <- trees[i]
  print(paste0(i,": ", num_trees))
  rf[[i]] <- train(y ~ ., data = train,
                   method = "ranger",
                   tuneGrid = grid_rf,
                   metric = "Accuracy",
                   importance = "impurity",
                   num.trees = num_trees,
                   trControl = ctrl)
}

# make predictions
p2 <- list()
cm2 <- list()
for(i in 1:length(trees)){
  
  model <- rf[[i]]
  p2[[i]] <- predict(model, test[,-which(colnames(test) == "y")])
  cm2[[i]] <- confusionMatrix(p2[[i]], factor(test$y))
  
}

test_acc2 <- unlist(lapply(lapply(cm2, '[[', "overall"), function(x) x[1]))
df_rf <- data.frame(trees = factor(trees), acc = test_acc2)

# plot relationship between accuracy and number of trees in the rf (Fig. 3)
ggplot(df_rf, aes(trees, acc, group=1)) + 
  geom_line(color="grey") +
  geom_point(color="steelblue", size=3) +
  ylim(.70, .90) +
  geom_text(aes(label=scaleFUN4(acc)), vjust=-1, size=3.5) +
  labs(x = "\nNumber of Trees in the Random Forest", 
       y = "Accuracy\n") +
  theme_minimal()

# end parallel processing
stopCluster(cl)
