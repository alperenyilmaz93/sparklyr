## configuration with spark
install.packages("config")

config <- config::get()
config$sparklyr.cores.local
config$`sparklyr.shell.driver-memory`
config$spark.executor.memory

library(sparklyr)
sc <- spark_connect(master = "local")
spark_web(sc)
spark_log(sc)
spark_disconnect(sc)


## installing packages and libraries

if(!require("dplyr")) install.packages("dplyr")
if(!require("purrr")) install.packages("purrr")
install.packages("tidyverse")
install.packages('ggplot2')


library(dplyr)
library(purrr)
library(ggplot2)
library(rpart)
library(tidyverse)
library(tidyr)
library(jsonlite)
library(readr)

install.packages('pander')

library(pander)

## Data Preparation

data=read.table("bank-additional-full.csv",header=TRUE,sep=";")
data$pdays <- NULL


data$y<-ifelse(data$y=='yes', 1,0)


data2<-data[data$job!="unknown",]
data3<-data2[data2$marital!="unknown",]
data4<-data3[data3$education!="unknown",]
data5<-data4[data4$default!="unknown",]
data6<-data5[data5$housing!="unknown",]
data7<-data6[data6$loan!="unknown",]

data_tbl<-data7


data2_tbl <- data_tbl %>% 
  mutate(campaign = as.numeric(campaign)) %>%
  mutate(previous = as.numeric(previous)) %>%
  mutate(y=as.numeric(y)) %>% 
  as_data_frame()


## Exploratory Data Analysis

data2_tbl %>%
  select(age, job, marital,education) %>%
  summary() %>% pander(caption = "Feature Summary: Cilent Demographics")


data2_tbl %>%
  select(marital,education,y) %>%
  gather(key, value, -y) %>%
  group_by(key, value, y) %>%
  count() %>%
  group_by(key) %>%
  mutate(n = n / sum(n)) %>%
  ungroup() %>%
  ggplot(aes(x = value, y = n, fill = y)) +
  geom_bar(stat = 'identity') +
  facet_wrap(~key, scales = 'free', nrow = 2) +
  guides(fill = guide_legend(reverse = TRUE)) +
  theme(legend.position = "top") + 
  labs(title = "Cilent Demographics", x = NULL, y = NULL) +
  scale_y_continuous(labels =  scales::percent)


data2_tbl %>%
  select(job,y) %>%
  gather(key, value, -y) %>%
  group_by(key, value, y) %>%
  count() %>%
  group_by(key) %>%
  mutate(n = n / sum(n)) %>%
  ungroup() %>%
  ggplot(aes(x = value, y = n, fill = y)) +
  geom_bar(stat = 'identity') +
  facet_wrap(~key, scales = 'free', nrow = 2) +
  guides(fill = guide_legend(reverse = TRUE)) +
  theme(legend.position = "top") + 
  labs(title = "Cilent Demographics", x = NULL, y = NULL) +
  scale_y_continuous(labels =  scales::percent)

data2_tbl %>%
  select(poutcome,y) %>%
  gather(key, value, -y) %>%
  group_by(key, value, y) %>%
  count() %>%
  group_by(key) %>%
  mutate(n = n / sum(n)) %>%
  ungroup() %>%
  ggplot(aes(x = value, y = n, fill = y)) +
  geom_bar(stat = 'identity') +
  facet_wrap(~key, scales = 'free', nrow = 2) +
  guides(fill = guide_legend(reverse = TRUE)) +
  theme(legend.position = "top") + 
  labs(title = "Cilent Demographics", x = NULL, y = NULL) +
  scale_y_continuous(labels =  scales::percent)



## Copying the Dataset into Spark 

data2_tbl <- sdf_copy_to(sc, data2_tbl, name = "data2_tbl", overwrite = TRUE)



## Min-Max Normalization

features2 <- c("emp_var_rate")

data3_tbl <- data2_tbl %>%
  ft_vector_assembler(input_col = features2,
                      output_col = "features2_temp") %>%
  ft_min_max_scaler(input_col = "features2_temp",
                    output_col = "emp_var_rate_scaled") %>% 
  sdf_register('data3')

features3 <- c("cons_price_idx")

data4_tbl <- data3_tbl %>%
  ft_vector_assembler(input_col = features3,
                      output_col = "features3_temp") %>%
  ft_min_max_scaler(input_col = "features3_temp",
                    output_col = "cons_price_idx_scaled") %>% 
  sdf_register('data4')

features4 <- c("cons_conf_idx")

data5_tbl <- data4_tbl %>%
  ft_vector_assembler(input_col = features4,
                      output_col = "features4_temp") %>%
  ft_min_max_scaler(input_col = "features4_temp",
                    output_col = "cons_conf_idx_scaled") %>% 
  sdf_register('data5')


features5 <- c("nr_employed")

data6_tbl <- data5_tbl %>%
  ft_vector_assembler(input_col = features5,
                      output_col = "features5_temp") %>%
  ft_min_max_scaler(input_col = "features5_temp",
                    output_col = "nr_employed_scaled") %>% 
  sdf_register('data6')




data_final <- select(data6_tbl, -c('cons_conf_idx', 'nr_employed','emp_var_rate','cons_price_idx','euribor3m',
                                   'features2_temp','features3_temp','features4_temp','features5_temp')) %>% 
  sdf_register('datafinal')




##################
## Modelling 

ml_formula <- formula(y ~ .)
set.seed(42)
vect=rep(0,6)
for(i in 1:1) {
  
  partition <- data_final %>% 
    sdf_partition(train = 0.80, test = 0.20, seed = 42)
  
  train_tbl <- partition$train
  test_tbl <- partition$test
  
  set.seed(123)
  ml_log <- ml_logistic_regression(train_tbl, ml_formula) 
  
  ml_dt <- ml_decision_tree_classifier(train_tbl, ml_formula,max_depth = 6, max_bins = 19,
                            type = 'classification',impurity = "entropy")
  
  ml_rf <- train_tbl %>%
    ml_random_forest_classifier(ml_formula, type =c("classification"),
                                num_tress= 20, max_depth= 20, impurity ='entropy', seed=2017)
 
  ml_gbt <- train_tbl %>% 
  ml_gradient_boosted_trees(ml_formula ,type = c("classification"), max_depth = 5, max_iter = 30L, seed = 2017)
  
  ml_nb <- ml_naive_bayes(train_tbl, ml_formula)
  
  ml_svm <- ml_linear_svc(train_tbl,ml_formula)
  
  
  
  pred_log <- sdf_predict(test_tbl,ml_log)
  p_log <- ml_binary_classification_evaluator(pred_log)
  
  pred_dt <- sdf_predict(test_tbl,ml_dt)
  p_dt <- ml_binary_classification_evaluator(pred_dt)
  
  pred_rf <- sdf_predict(test_tbl,ml_rf)
  p_rf <- ml_binary_classification_evaluator(pred_rf)
  
  pred_gbt <- sdf_predict(test_tbl,ml_gbt)
  p_gbt <- ml_binary_classification_evaluator(pred_gbt)
  
  pred_nb <- sdf_predict(test_tbl, ml_nb)
  p_nb <- ml_binary_classification_evaluator(pred_nb)
  
  pred_svm <- sdf_predict(test_tbl,ml_svm)
  p_svm <- ml_binary_classification_evaluator(pred_svm)
  
  
  
  p_vect=c(p_log,p_dt, p_rf, p_gbt, p_nb,p_svm)
  
  vect=vect+(1/1)*p_vect
}
vect



#% 75-%25
0.9341522 0.8663507 0.9292572 0.9397532 0.3590689 0.9264606

# %80-%20
0.9321689 0.8643599 0.9300976 0.9390376 0.3592501 0.9303841

# %85-%15
0.9314750 0.8439787 0.9277912 0.9377009 0.3615899 0.9230051

# %90-%10
0.9236839 0.8515727 0.9228807 0.9322218 0.3753521 0.9175540


## finding the best ratio for splitting the dataset 
## as well as the best algorithm for the dataset

Vect_75=c(0.9341522,0.8663507,0.9292572,0.9397532,0.3590689,0.9264606)
Vect_80=c(0.9321689,0.8643599,0.9300976,0.9390376,0.3592501,0.9303841)
Vect_85=c(0.9314750,0.8439787,0.9277912,0.9377009,0.3615899,0.9230051)
Vect_90=c(0.9236839,0.8515727,0.9228807,0.9322218,0.3753521,0.9175540)

Mat=cbind(Vect_75,Vect_80,Vect_85,Vect_90)


xdata <- c(75,80,85,90)
log <- Mat[1,]
dt <- Mat[2,] 
rf <- Mat[3,]
gbt <- Mat[4,] 
nb <- Mat[5,] 
svm <- Mat[6,]



# First curve is plotted
plot(xdata, log , type="o", col="gold", pch="*", lty=1,  ylim=c(0.5,1), 
     main = 'comparison of models by the ratio of splitting the dataset' , 
     xlab = 'the ratio of splitting the dataset' , ylab = 'accuracy')


# Decision Tree
points(xdata, dt,col="red", pch="*")
lines(xdata, dt,col="red",lty=1)

# Random Forest
points(xdata, rf, col="green4",pch="*")
lines(xdata, rf, col="green4", lty=1)

# Gradient Boosted Trees
points(xdata, gbt, col="blue",pch="*")
lines(xdata, gbt, col="blue", lty=1)

# Naive Bayes
points(xdata, nb, col="turquoise1",pch="*")
lines(xdata, nb, col="turquoise1", lty=1)

# Support Vector Machines
points(xdata, svm, col="darkmagenta",pch="*")
lines(xdata, svm, col="darkmagenta", lty=1)

legend(
  "bottomleft", 
  lty=c(1,1,1,1), 
  col=c("gold", "red","green4", "blue", "turquoise1",'darkmagenta'), 
  legend = c("log", "dt", "rf", "gbt","nb",'svm')
)


################################
# Validation data

ml_models <- list(
  "Logistic" = ml_log,
  "Decision Tree" = ml_dt,
  "Random Forest" = ml_rf,
  "Gradient Boosted Trees" = ml_gbt,
  "Naive Bayes" = ml_nb,
  'Support Vector Machines' = ml_svm
)

# Create a function for scoring
score_test_data <- function(model, data = test_tbl){
  pred <- sdf_predict(data, model)
  select(pred, y , prediction)
}

# Score all the models
ml_score <- map(ml_models, score_test_data)


#################################
# Function for calculating accuracy

calc_accuracy <- function(data, cutpoint = 0.5){
  data %>% 
    mutate(prediction = if_else(prediction > cutpoint, 1.0, 0.0)) %>%
    mutate(y = as.numeric(y)) %>%
    ml_classification_eval("prediction", "y", "f1")
}

calc_accuracy2 <- function(data, cutpoint = 0.5){
  data %>% 
    mutate(prediction = if_else(prediction > cutpoint, 1.0, 0.0)) %>%
    mutate(y = as.numeric(y)) %>%
    ml_classification_eval("prediction", "y", "weightedPrecision")
}

calc_accuracy3 <- function(data, cutpoint = 0.5){
  data %>% 
    mutate(prediction = if_else(prediction > cutpoint, 1.0, 0.0)) %>%
    mutate(y = as.numeric(y)) %>%
    ml_classification_eval("prediction", "y", "weightedRecall")
}


# Calculate  precision, f1, recall 
perf_metrics <- data_frame(
  model = names(ml_score),
  f1 = 100 * map_dbl(ml_score, calc_accuracy),
  precision = 100 * map_dbl(ml_score, calc_accuracy2),
  recall = 100 * map_dbl(ml_score, calc_accuracy3)
)
perf_metrics %>%
  pander(caption = "Comparison of 6 Spark Models")



gather(perf_metrics, metric, value,f1, precision,recall) %>%
  ggplot(aes(reorder(model, value), value, fill = metric)) + 
  geom_bar(stat = "identity", position = "dodge") + 
  coord_flip() +
  xlab("") +
  ylab("Percent") +
  ggtitle("Performance Metrics")


# Calculate feature importance
feature_importance <- data_frame()


for(i in c("Decision Tree", "Random Forest", "Gradient Boosted Trees")){
  feature_importance <- ml_tree_feature_importance(ml_models[[i]]) %>%
    mutate(Model = i) %>%
    rbind(feature_importance, .)
}

# Plot results
feature_importance %>%
  ggplot(aes(reorder(feature, importance), importance, fill = Model)) + 
  facet_wrap(~Model) +
  geom_bar(stat = "identity") + 
  coord_flip() +
  labs(title = "Feature importance",
       x = NULL) +
  theme(legend.position = "none")

