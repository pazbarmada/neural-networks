#BINARY CLASSIFFICATION USING TENSORFLOW
rm(list=ls())

library(ggplot2)
library(cowplot)
library(caret)
library(keras)
library(nn2poly)
library(patchwork)
library(tensorflow)

#Data Preparation
x_train <- as.matrix(read.csv("/Users/clo/Downloads/X_train.csv"))
y_train <- as.matrix(read.csv("/Users/clo/Downloads/y_train.csv"))
y_train <- as.numeric(y_train) 

x_test <- as.matrix(read.csv("/Users/clo/Downloads/x_test.csv"))
y_test <- as.matrix(read.csv("/Users/clo/Downloads/y_test.csv"))
y_test <- as.numeric(y_test) 

#1: Neural Network 

#Build Neural Network

n_inputs <- ncol(x_train)
n_hidden <- 50  #number of neurons in the hidden layer

#hyperparameters obtained from the previous trained model in Python 
set.seed(45)
nn_model <- keras_model_sequential() %>%
  layer_dense(units = 50, activation = "tanh", input_shape = n_inputs) %>%
  layer_dense(units = 1, activation = "linear") # 

nn_model <- nn2poly::add_constraints(object = nn_model, type = "l1_norm")

nn_model
 

#model compile
nn_model %>% compile(
  loss = loss_binary_crossentropy(from_logits = TRUE),
  optimizer = optimizer_adam(),
  metrics = "accuracy")

history <- fit(nn_model,
               x_train,
               y_train,
               verbose = 1,
               epochs = 40,
               validation_split = 0.3
)

plot(history)


#NN Predictions

probability_model <- keras_model(
  inputs = nn_model$input,
  outputs = nn_model$output %>% layer_activation("sigmoid"))

prediction_NN_class <- predict(probability_model, x_test)
prediction_NN_class


prediction_NN_class <- ifelse(prediction_NN_class > 0.5, 1, 0)


prediction_NN <- predict(nn_model, x_test)
prediction_NN

cm_nn<-caret::confusionMatrix(as.factor(prediction_NN_class), as.factor(y_test))
cm_nn

#2:Polynomial Representation

#Using nn2poly
poly_model <- nn2poly(nn_model, polmax_order = 3)
poly_model

# Obtaining polynomial predictions 

#obtain the predicted values fort the test data
prediction_poly <- predict(poly_model, newdata = x_test)

#define sigmoid funtion to convert the result into probabilities
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

#convert the results to probabilities
prediction_poly_prob <- sigmoid(prediction_poly)

#class prediction with the polynomial outputs
prediction_poly_class <- ifelse(prediction_poly_prob > 0.5, 1, 0)

#Evaluate performance. Visualize results in a confusion matrix
prediction_poly_class <- factor(prediction_poly_class, levels = c(0, 1))
y_test_factor <- factor(y_test, levels = c(0, 1))

#Visualizing results

#Confussion matrix between NN class prediction and polynomial class prediction, to see how well the model performs
cm_2<-caret::confusionMatrix(as.factor(prediction_NN_class), as.factor(prediction_poly_class))
cm_2

#Diagonal line: linear outputs obtained directly from the polynomial and NN predictions
nn2poly:::plot_diagonal(
  x_axis = prediction_NN,
  y_axis = prediction_poly,
  xlab = "NN Prediction",
  ylab = "Polynomial prediction"
)

# top 10 most significant variables 
plot(poly_model, n = 10)

#performance of  taylor expansion
train <- data.frame(x_train, target = y_train)

plots <- nn2poly:::plot_taylor_and_activation_potentials(object = nn_model,
                                                data = train,
                                                max_order = 3,
                                                constraints = TRUE)

# Show plot for layer 1 (hidden layer) 
print(plots[[1]])

# Show plot for layer 2 (output layer)
print(plots[[2]]) 

#Interpretation of coefficients 

# obtain terms as text
terms <- sapply(poly_model$labels, function(x) paste(x, collapse = ","))  # +1 porque R indexa desde 1

#obtain the coefficients
coefs <- poly_model$values[,1]

#create a data frame
coefs_df <- data.frame(
  Term = terms,
  Coefficient = coefs,
  Abs_Coef = abs(coefs)
)

#uplaod real names
column_names <- read.csv("/Users/clo/Downloads/X_columns.csv", header = FALSE)[,1]

#translate each term to real name 
traducir_termino <- function(termino) {
  indices <- as.numeric(unlist(strsplit(termino, ",")))
  nombres <- column_names[indices]
  paste(nombres, collapse = " * ")
}

coefs_df$Variable <- sapply(coefs_df$Term, traducir_termino)


top_vars <- coefs_df[order(-coefs_df$Abs_Coef), ]

# Top 10 significant terms
head(top_vars[, c("Variable", "Coefficient")], 10)

library(ggplot2)

ggplot(head(top_vars, 10), aes(x = reorder(Variable, Abs_Coef), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "#69b3a2") +
  coord_flip() +
  labs(title = "Top 10 most Influential Polynomial Terms",
       x = "Variables or interactions",
       y = "Coefficients") +
  theme_minimal()

