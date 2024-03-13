rm(list=ls()) 
library(rstudioapi)
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library(MLmetrics)
library(randomForest)
library(caret)
library(ISLR)
library(MASS)
library(glmnet)
library(ModelMatrixModel)
library(stats)
library(pdp)
library(gbm)
library(xgboost)
library(doParallel)
library(parallel)
library(lme4)
library(e1071)
library(ROCR)



func.dir = "/Users/vasant/Desktop/NPS Data Science Certificate/OS4106/cell_project/code/"       
val.file = "validation_functions.R"
val.file.name = paste(func.dir,val.file,sep="")
source(val.file.name)
class.file = "classification_metrics.R"
class.file.name = paste(func.dir,class.file,sep="")
source(class.file.name)

##### Load data file
data.dir = "/Users/vasant/Desktop/NPS Data Science Certificate/OS4106/cell_project/data/"       
data.file = "cell_plan_cancellations.csv"
cell.data <- read.csv(paste(data.dir,data.file,sep=""), 
                        stringsAsFactors = TRUE)
dim(cell.data)
names(cell.data)
summary(cell.data)
sum(is.na(cell.data))
################# Data Splitting
## First set reference on cancel variable
## Split into train and validation indices
cell.data$Cancel <- relevel(cell.data$Cancel, ref = "Yes")

#cell.data$Cancel01 =  (cell.data$Cancel=="Yes")*1

set.seed(248)

# creating training data as fraction of the dataset 
p = .99
indices <- train_val_test_split(cell.data$Cancel,p,0,1-p) # 50% train, 50% validation
train_index = indices$train
test_index = indices$test
#train_index = c(1:dim(cell.data)[1])
#test_index = train_index

train_data = cell.data[train_index,]
test_data = cell.data[test_index,]

("Dimension of Original Dataset")
(dim(cell.data))
("Dimension of Train Dataset")
(dim(train_data))
("Dimension of Test Dataset")
(dim(test_data))
train_data = cell.data
##############################
## Define trainControl() object. This object is to be used for all
## models except random forest.
set.seed(2423)
num_folds = 5
num_repeats = 3
train.control <- caret::trainControl(
  method = "repeatedcv", 
  number = num_folds,
  repeats = num_repeats,
  savePredictions = TRUE,
  index = createMultiFolds(train_data$Cancel, k=num_folds, times=num_repeats) ,
  classProbs = TRUE, 
  summaryFunction=caret_classification_metrics,
  returnResamp = "all",
  allowParallel = TRUE
)
cv.results = data.frame(model = c(
  "Linear, all terms,",
  "ElasticNet interactions",
  "K-Nearest Neighbors",
  "Random Forest",
  "Gradient Boosting Method",
  "SVM",
  "Subset interaction model"
  ),
  cvLogLoss = matrix(0,nrow=length(5),ncol=1),
  cvBrier = matrix(0,nrow=length(5),ncol=1),
  cvROCAUC = matrix(0,nrow=length(5),ncol=1),
  cvPRAUC = matrix(0,nrow=length(5),ncol=1),
  cvGMean = matrix(0,nrow=length(5),ncol=1)
)

###############################
## Linear Model
###############################
cell.lin <- caret::train(Cancel ~ .,
                         data=train_data,
                         method="glm",
                         family="binomial",
                         trControl=train.control,
                         preProcess = c("range"),
                         metric="log.loss",
                         maximize=FALSE
                         )


cv.results$cvLogLoss[[1]] <- min(cell.lin$results$log.loss)
cv.results$cvBrier[[1]] <- min(cell.lin$results$Brier)
cv.results$cvROCAUC[[1]] <- min(cell.lin$results$roc.auc)
cv.results$cvPRAUC[[1]] <- min(cell.lin$results$pr.auc)
cv.results$cvGMean[[1]] <- min(cell.lin$results$Gmean)

lgrid <- 10^seq(6,-5,length=25)
agrid <- seq(0,1,length=4)

cell.ela.int <- caret::train(Cancel ~ .^2,
                    data = train_data,
                    method = "glmnet",
                    family = "binomial",
                    trControl = train.control,
                    preProcess = c("range"),
                    tuneGrid = expand.grid(alpha =agrid, lambda=lgrid),
                    metric = "log.loss",
                    maximize = FALSE
                  )



cv.results$cvLogLoss[[2]] <- min(cell.ela.int$results$log.loss)
cv.results$cvBrier[[2]] <- min(cell.ela.int$results$Brier)
cv.results$cvROCAUC[[2]] <- min(cell.ela.int$results$roc.auc)
cv.results$cvPRAUC[[2]] <- min(cell.ela.int$results$pr.auc)
cv.results$cvGMean[[2]] <- min(cell.ela.int$results$Gmean)

### Convert dataset into matrix for suitable for final ElasticNet
xint <- model.matrix(Cancel ~ .^2, data = cell.data)
yint <- cell.data$Cancel
x <- model.matrix(Cancel ~., data = cell.data)

# split matrices into train and validation
x.train <- x[train_index,] 
y.train <- yint[train_index]

# validation
x.test <- x[test_index,]
y.test <- yint[test_index]

###########
## K-Nearest Neighbors Classification
###########
knn.control <- caret::trainControl(
  method = "repeatedcv", 
  number = num_folds,
  repeats = num_repeats,
  savePredictions = TRUE,
  index = createMultiFolds(train_data$Cancel, k=num_folds, times=num_repeats) ,
  classProbs = TRUE, 
  summaryFunction=caret_classification_metrics,
  returnResamp = "all",
  allowParallel = TRUE
)

train.x <- subset(train_data,select=-Cancel)
test.x <- subset(test_data, select=-Cancel)

preProc <- caret::preProcess(x=train.x, method=c("range"))
#use predict to normalize the data, use same process to normalize bth
#train  and test
train.norm <- predict(preProc, train.x)
test.norm <- predict(preProc, test.x)

kmax = 100
f <- function(kmax) {
  k <- seq(from=5, to=kmax, by=8)
  k.tune <- data.frame(K = matrix(0,nrow=length(k)),
                      bestk = matrix(0,nrow=length(k)))
  for (i in 1:length(k)) {
  kmax = k[i]
  print("Tuning up to: ")
  print(kmax)
  print("Iteration ")
  print(i)
  knn.cv <- caret::train(Cancel ~ ., data = train_data, method = "knn",
                         trControl = knn.control, preProcess = c("range"),
                         metric="logLoss",
                         maximize=FALSE,
                         tuneGrid = expand.grid(k = 1:kmax))
  
  k.tune$K[i] <- k[i]
  k.tune$bestk[i] <- knn.cv$bestTune
  }
  return(k.tune)
  return(knn.cv)
}  
cl <- makeCluster(8)
system.time(k.tune<-mclapply(kmax,f,mc.cores=8))
stopCluster(cl)
k.tune <- as.data.frame(k.tune[1])

plot(k.tune$K,k.tune$bestk,xlab="Maximum K-value",  ylab="Best k",
     cex=1.5,pch=10,lwd=1, type="o")

knn.cv <- caret::train(Cancel ~ ., data = train_data, method = "knn",
                       trControl = knn.control, preProcess = c("range"),
                       metric="log.loss",
                       maximize=FALSE,
                       tuneGrid = expand.grid(k = 1:kmax))

plot(knn.cv$results$k,knn.cv$results$logLoss,xlab="k",  ylab="Validation Log-loss",
     cex=1.5,pch=10,lwd=1, type="o")

knn.fin <- caret::knn3(x = train.norm, y = train_data$Cancel, k=knn.cv$bestTune)

cv.results$cvLogLoss[[3]] <- min(knn.cv$results$log.loss)
cv.results$cvBrier[[3]] <- min(knn.cv$results$Brier)
cv.results$cvROCAUC[[3]] <- min(knn.cv$results$roc.auc)
cv.results$cvPRAUC[[3]] <- min(knn.cv$results$pr.auc)
cv.results$cvGMean[[3]] <- min(knn.cv$results$Gmean)

#################### 
### Apply Random Forest, using its own trainControl object.
####################
set.seed(248)
mtry.vec = 1:25
num_folds = 5
num_repeats = 3
train.control.rf <- caret::trainControl(
  method = "repeatedcv", 
  number = num_folds,
  repeats = num_repeats,
  savePredictions = TRUE,
  classProbs = TRUE,
  verboseIter=TRUE,
  summaryFunction= mnLogLoss,
  allowParallel = TRUE,
  index = createMultiFolds(train_data$Cancel, k=num_folds, times=num_repeats) ,
  returnResamp = "all"
)

ntree <- c(25,50,75,100,125,200,500)
rf <- function(ntree) {
  rflist <- list()
  print(rflist)
  for (i in 1:length(ntree)) {
    print("ntree is ")
    print(ntree)
    rf.cv.ntree <- caret::train(Cancel ~ ., data = train_data,
                                method = "rf",
                                trControl = train.control,
                                ntree=ntree[i],
                                metric = "log.loss",
                                maximize = FALSE,
                                tuneGrid = expand.grid(.mtry=mtry.vec))
    rflist <- list(rflist, rf.cv.ntree)
  }
  return(rflist)
}
cl <- makeCluster(4)
system.time(rflist<-mclapply(ntree,rf,mc.cores=8))
stopCluster(cl)

## Plot mtry vs. logloss for the various number of trees
rfntree.legend <- data.frame(ntree=ntree,
                             color=rainbow(length(ntree))
)
yrange <- data.frame(ntree=ntree,
                     min = matrix(0,nrow=length(ntree)),
                     max = matrix(0,nrow=length(ntree))
)
for (i in 1:length(ntree)){
  
  yrange$ntree[i] <- ntree[i]
  yrange$min[i] <- min(rflist[[i]][[2]]$results$log.loss)
  yrange$max[i] <- max(rflist[[i]][[2]]$results$log.loss)
  if (i==1) {
    plot(rflist[[i]][[2]]$results$mtry,rflist[[1]][[2]]$results$log.loss,type="b",pch=16,xlab="mtry",ylab="logLoss",ylim = c(min(yrange$min),max(yrange$max)),col=rfntree.legend$color[1])
  }
  else{
  print(rflist[[i]][[2]]$finalModel$ntree)
  if (i==length(ntree)) {
  lines(rflist[[i]][[2]]$results$mtry,rflist[[i]][[2]]$results$log.loss,type="b",pch=16,col = rfntree.legend$color[i])
  }
  else{
    lines(rflist[[i]][[2]]$results$mtry,rflist[[i]][[2]]$results$log.loss,type="b",pch=16,col = rfntree.legend$color[i])
  }
  }
}
legend("bottomright",legend=rfntree.legend$ntree,fill=rfntree.legend$color)

yrange_mtry <- data.frame(ntree=ntree,
                     mtry = matrix(0,nrow=length(ntree))
)
for (i in 1:length(ntree)) {
  yrange_mtry$ntree[i] <- ntree[i]
  yrange_mtry$mtry[i] <- rflist[[i]][[2]]$bestTune[['mtry']]
}
## Plot ntree vs. minimum log-loss for each ntree
for (i in 1:length(ntree)){
  print("ntree = ")
  print(ntree)

  if (i==1) {
    plot(ntree[i],min(rflist[[i]][[2]]$bestTune[['mtry']]),type="b",pch=16,xlab="ntree",ylab="Best mtry",xlim=c(ntree[1],ntree[length(ntree)]),ylim=c(min(yrange_mtry$mtry)-1,max(yrange_mtry$mtry)+1),col=rfntree.legend$color[1])
  }
  else{
    if (i==length(ntree)) {
      lines(ntree[i],min(rflist[[i]][[2]]$bestTune[['mtry']]),type="b",pch=16,col = rfntree.legend$color[i])
    }
    else{
      lines(ntree[i],min(rflist[[i]][[2]]$bestTune[['mtry']]),type="b",pch=16,col = rfntree.legend$color[i])
    }
  }
}
legend("bottomright",legend=rfntree.legend$ntree,fill=rfntree.legend$color)

ntree_fin = 500
rf.cv <- caret::train(Cancel ~ ., data = train_data,
                          method = "rf",
                          trControl = train.control,
                          ntree=ntree_fin,
                          metric = "log.loss",
                          maximize = FALSE,
                          tuneGrid = expand.grid(.mtry=mtry.vec))

rf.cv.2 <- caret::train(Cancel ~ ., data = train_data,
                      method = "rf",
                      trControl = train.control.rf,
                      ntree=ntree_fin,
                      metric = "logLoss",
                      maximize = FALSE,
                      tuneGrid = expand.grid(.mtry=mtry.vec))

rf.cv.fin <- randomForest::randomForest(Cancel ~ ., data=train_data,ntree=500,mtry=rf.cv$bestTune[['mtry']],importance=TRUE)
cv.results$cvLogLoss[[4]] <- min(rf.cv$results$log.loss)
cv.results$cvBrier[[4]] <- min(rf.cv$results$Brier)
cv.results$cvROCAUC[[4]] <- min(rf.cv$results$roc.auc)
cv.results$cvPRAUC[[4]] <- min(rf.cv$results$pr.auc)
cv.results$cvGMean[[4]] <- min(knn.cv$results$Gmean)



########### Gradient Boosting
hyper_grid <- expand.grid(
  n.trees = c(100,200,500,1000),
  shrinkage = c(0.01,0.05,0.25, 0.5),
  interaction.depth=c(1, 3,5,10),
  n.minobsinnode = c(2,4,8,16)
)

cl <- makePSOCKcluster(8)
registerDoParallel(cl)

gbm.cv <- caret::train(Cancel ~ ., data = cell.data,
                     method="gbm",
                     distribution="bernoulli", 
                     verbose = TRUE,
                     trControl = train.control,
                     tuneGrid = hyper_grid,
                     metric="log.loss",
                     maximize=FALSE)

stopCluster(cl)
cv.results$cvLogLoss[[5]] <- min(gbm.cv$results$log.loss)
cv.results$cvBrier[[5]] <- min(gbm.cv$results$Brier)
cv.results$cvROCAUC[[5]] <- min(gbm.cv$results$roc.auc)
cv.results$cvPRAUC[[5]] <- min(gbm.cv$results$pr.auc)
cv.results$cvGMean[[5]] <- min(gbm.cv$results$Gmean)

C.vec <- seq(0.01,1,0.005)

cl <- makePSOCKcluster(8)
registerDoParallel(cl)
svm.cv <- caret::train(Cancel ~ ., data = train_data,
                       method="svmRadialCost",
                       distribution="bernoulli", 
                       verbose = TRUE,
                       trControl = train.control,
                       tuneGrid = expand.grid(.C=C.vec),
                       metric="log.loss",
                       maximize=FALSE)

stopCluster(cl)
cv.results$cvLogLoss[[6]] <- min(svm.cv$results$log.loss)
cv.results$cvBrier[[6]] <- min(svm.cv$results$Brier)
cv.results$cvROCAUC[[6]] <- min(svm.cv$results$roc.auc)
cv.results$cvPRAUC[[6]] <- min(svm.cv$results$pr.auc)
cv.results$cvGMean[[6]] <- min(svm.cv$results$Gmean)

#### BART



############## Get secondary metrics on models
pred_data = cell.data
pred_data01 <- (pred_data$Cancel=="Yes")*1
# Generate final models on train data
final.lin <- glm(Cancel ~ ., data=pred_data,family="binomial")
final.lin.pred.prob <- predict(final.lin,pred_data,type="response")
lin.logloss <- MLmetrics::LogLoss(y_pred=final.lin.pred.prob,y_true=pred_data01)

final.ela.int <- glmnet::glmnet(x.train,y.train, 
                                family = "binomial",
                                alpha = cell.ela.int$bestTune[['alpha']], 
                                lambda = cell.ela.int$bestTune[['lambda']],
                                
)


final.ela.int.pred.prob <- predict(final.ela.int,x.train,type="response")
ela.logloss <- MLmetrics::LogLoss(y_pred=final.ela.int.pred.prob,y_true=pred_data01)

final.rf <- randomForest::randomForest(Cancel ~ ., data=pred_data,ntree=500,mtry=rf.cv$bestTune[['mtry']],importance=TRUE)
final.rf.prob <- predict(final.rf,pred_data,type="prob")
final.rf.prob <- final.rf.prob[,1]
final.rf.resp <- predict(final.rf,pred_data,type="response")
rf.logloss <- MLmetrics::LogLoss(y_pred=final.rf.prob,y_true=pred_data01)
caret::confusionMatrix(data=final.rf.resp,reference=factor(pred_data$Cancel),
                       mode="everything", positive = primary.class)



imp <- randomForest::importance(final.rf)
randomForest::varImpPlot(final.rf)
pred_data_gbm <- pred_data
pred_data_gbm$Cancel  = (pred_data_gbm$Cancel=="Yes")*1
final.gbm <- gbm::gbm(Cancel ~ ., data=pred_data_gbm,
                      distribution = "bernoulli",
                      n.trees=gbm.cv$bestTune[['n.trees']],
                      interaction.dept=gbm.cv$bestTune[['interaction.depth']],
                      shrinkage=gbm.cv$bestTune[['shrinkage']],
                      n.minobsinnode = gbm.cv$bestTune[['n.minobsinnode']])
final.gbm.prob <- predict(final.gbm, pred_data_gbm,type="response")
gbm.logloss <- MLmetrics::LogLoss(y_pred=final.gbm.prob,y_true=pred_data01)

pred_data_svm <- as.data.frame()
final.svm <- svm(Cancel~.,data=pred_data,type="C",kernel="radial",
                 cost = svm.cv$bestTune[['C']])

## ---------------------------------------------------------------------------

final.svm.prob <- predict(final.svm,pred_data,type="prob")

svm.confmat <- caret::confusionMatrix(data=factor(final.svm.prob), reference = factor(pred_data$Cancel),
                       mode="everything") 


##--------------------------------------------------------------- 
# Compute threshold value on final model.
#look at several different candidate threshold to determine best one
thresh_vec = seq(from=0,to=1,by=0.01)

#validation df to store gmean and f1 for each threshold
val_df = data.frame(thresh = thresh_vec, gmean = rep(0,length(thresh_vec)), f1 = rep(0,length(thresh_vec))  )

#can be helpful to keep track of the two classes for bookeeping purposes
primary.class = levels(pred_data$Cancel)[1]
ref.class = levels(pred_data$Cancel)[2]
#get predicted probabilites
pred.prob.val <- final.gbm.prob


## ---- warning=FALSE---------------------------------------------------------
#loop over compute these
for(i in 1:dim(val_df)[1]) {
  
  thresh = val_df[i,1] 
  
  #need to convert the predicted probability to factor label prediction based
  #on threshold
  pred.class <- factor( x = ifelse(pred.prob.val > thresh, primary.class,ref.class))
  
  #only relevel if ref level in list
  #can throw error if threshold close to 0 or 1 when predicted call all one 
  #class or another
  if(ref.class %in% levels(pred.class) ) {
    #relevel factor for consistency. should not matter
    pred.class <- relevel(pred.class, ref = ref.class)
    
  }
  
  #get confusion matrix
  cm <- caret::confusionMatrix(data=pred.class, reference = factor(pred_data$Cancel),
                               mode="everything", positive = primary.class) 
  
  #get gmean from  confusion matrix by get sens and spex
  val_df[i,2] = sqrt(cm$byClass[['Sensitivity']]*cm$byClass[['Specificity']])
  #extract F1 from confusion matrix
  val_df[i,3] = cm$byClass[['F1']]
}


## ---------------------------------------------------------------------------

#print out best gmean and f1 and corresponding threshold
val_df[which.max(val_df$gmean),]
val_df[which.max(val_df$f1),]

#get best thrshold. we use gmean 
(best_thresh <- val_df$thresh[which.max(val_df$gmean)])

pred.class.final  <- factor( x = ifelse(pred.prob.val > best_thresh, "Yes","No"))    
caret::confusionMatrix(data=pred.class.final,reference=factor(pred_data$Cancel),
                       mode="everything", positive = primary.class)


################## Final model
############################
data.dir = "/Users/vasant/Desktop/NPS Data Science Certificate/OS4106/cell_project/data/"       
data.file = "cell_plan_cancellations_test.csv"
test.data <- read.csv(paste(data.dir,data.file,sep=""), 
                      stringsAsFactors = TRUE)

gbm.test.pred <- predict(final.gbm,test.data,type="response")
gbm.test.class.final  <- factor( x = ifelse(pred.prob.val > best_thresh, "Yes","No"))

gbm.pred.file <- "predictions.csv"
filepath <- paste0(data.dir,gbm.pred.file)
write.csv(file=filepath,row.names=FALSE)

