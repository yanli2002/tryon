llkjadkjh

# Setup
library(survey)
library(tableone)
library(ranger)
library(partykit)
library(gbm)

# Set directory
setwd("C:/Users/wangl29/Box/Research/Lingxiao Projects/Machine learning methods/Nonprobability weighting")
# Read aarp data
aarp_syn = read.table("aarp_orig.txt", head=T)# change to "aarp_orig.txt" for the original aarp data
# Read nhis data
nhis_m = read.table("nhis.txt", head=T)
# Load R functions for pseudo weights calculation
source("weighting_functions1119.R")
# Check variable names in the two data sets. Please see "variable dictionary.xlsx" for data dictionary
names(aarp_syn)
names(nhis_m)
# Number of records in AARP and NHIS
n_c=dim(aarp_syn)[1]
n_s=dim(nhis_m)[1]

# Combine NHIS and AARP data 
psa_dat = rbind(nhis_m, aarp_syn)
psa_dat$wt = c(nhis_m$elig_wt, rep(1, n_c))
psa_dat$trt = c(rep(0, n_s), rep(1, n_c))
# Name of data source indicator in the combined sample 
rsp_name="trt" # 1 for AARP, 0 for NHIS

# Data prep
psa_dat$trt_f <- as.factor(psa_dat$trt)
psa_dat$sex_f <- as.factor(psa_dat$sex)
psa_dat$race_f <- as.factor(psa_dat$race)
psa_dat$martl_f <- as.factor(psa_dat$martl)
psa_dat$smk1_f <- as.factor(psa_dat$smk1)
cols <- ncol(model.matrix(trt_f ~ age+sex_f+race_f+martl_f+educ+bmi+smk1_f+phys+health, data = psa_dat))
covars <- c("age", "sex", "race", "martl", "educ", "bmi", "smk1", "phys", "health")
catcovars <- c("sex", "race", "martl")

# Covariate balance before adjustment
ds <- svydesign(ids = ~1, weight = ~ wt, data = psa_dat)
tab_pre_adjust <- svyCreateTableOne(vars = covars, factorVars = catcovars, strata = "trt", data = ds, test = FALSE)
tab_pre_adjust_smd <- rbind(attr(tab_pre_adjust$ContTable, "smd"), attr(tab_pre_adjust$CatTable, "smd"))
summary(tab_pre_adjust_smd)

###########################################################################
#### Calculate propensity scores and pseudo weights based on logistic regression

#Fitted propensity score model
Formula_fit = as.formula("trt ~ age+sex+as.factor(race)+as.factor(martl)+educ+bmi+as.factor(smk1)+phys+health+
                  sex:as.factor(martl)+educ:health+as.factor(race):educ+age:phys+age:as.factor(martl)+
                 phys:health+as.factor(race):health+age:as.factor(race)+sex:as.factor(smk1)+
                 as.factor(race):as.factor(smk1)+age:health+educ:phys+age:as.factor(smk1)+age:bmi+
                 educ:as.factor(smk1)+as.factor(martl):health+sex:educ+age:educ+as.factor(smk1):phys+
                 as.factor(race):as.factor(martl)+sex:health+sex:as.factor(race)+as.factor(race):phys+
                 bmi:as.factor(smk1)+as.factor(martl):phys+as.factor(smk1):health+bmi:phys+as.factor(martl):educ+
                 sex:bmi+educ:bmi+sex:phys")
# unweighted propensity score model
svyds = svydesign(ids =~1, weight = rep(1, n_c+n_s), data = psa_dat)
lgtreg = svyglm(Formula_fit, family = binomial, design = svyds)
p_score = predict.glm(lgtreg, type = "response")
# Propensity scores for the cohort
p_score.c = as.data.frame(p_score[psa_dat[,rsp_name]==1])
# Propensity scores for the survey sample
p_score.s = as.data.frame(p_score[psa_dat[,rsp_name]==0])
#### Fit logistic regression model to cohort and weighted survey sample
ds = svydesign(ids=~1, weight = ~ wt, data = psa_dat)
lgtreg.w = svyglm(Formula_fit, family = binomial, design = ds)
# Predict propensity scores
p_score.w = predict.glm(lgtreg.w, type = "response")
p_score.w.c = as.data.frame(p_score.w[psa_dat[,rsp_name]==1])

# calculate IPSW weights
aarp_syn$ipsw = ipsw.wt(p_score.c = p_score.w.c[,1], svy.wt = nhis_m$elig_wt)
# calculate PSAS weights
aarp_syn$psas = psas.wt(p_score.c = p_score.c[,1], p_score.s = p_score.s[,1], svy.wt = nhis_m$elig_wt, nclass = 5)$pswt
# calculate KW weights
aarp_syn$kw.1 = kw.wt(p_score.c = p_score.c[,1], p_score.s = p_score.s[,1], svy.wt = nhis_m$elig_wt, Large=T)$pswt

###########################################################################
#### Calculate propensity scores and KW weights based on ML methods

#### Model-based recursive partitioning (MOB)

# Set try-out values and prepare loop
tune_maxdepth <- c(3, 5, 7)
psa_dat$wt_kw <- psa_dat$wt
p_scores <- data.frame(matrix(ncol = length(tune_maxdepth), nrow = nrow(psa_dat)))
p_score_c <- data.frame(matrix(ncol = length(tune_maxdepth), nrow = n_c))
p_score_s <- data.frame(matrix(ncol = length(tune_maxdepth), nrow = n_s))
smds <- rep(NA, length(tune_maxdepth))

# Loop over try-out values
for (i in seq_along(tune_maxdepth)) {
  maxdepth <- tune_maxdepth[i]
  mob <- glmtree(trt_f ~ age+sex_f+race_f+martl_f+educ+bmi+smk1_f+phys+health | age+sex_f+race_f+martl_f+educ, 
               data = psa_dat,
               family = binomial,
               alpha = 0.01,
               minsplit = 1000,
               maxdepth = maxdepth,
               prune = "AIC")
  p_scores[, i] <- predict(mob, psa_dat, type = "response")
  p_score_c[, i] <- p_scores[psa_dat$trt == 1, i]
  p_score_s[, i] <- p_scores[psa_dat$trt == 0, i]
  # Calculate KW weights
  aarp_syn$kw <- kw.wt(p_score.c = p_score_c[,i], p_score.s = p_score_s[,i], svy.wt = nhis_m$elig_wt, Large=T)$pswt
  # Calculate covariate balance
  psa_dat$wt_kw[psa_dat$trt == 1] <- aarp_syn$kw
  ds_kw <- svydesign(ids = ~1, weight = ~ wt_kw, data = psa_dat)
  tab_post_adjust <- svyCreateTableOne(vars = covars, factorVars = catcovars, strata = "trt", data = ds_kw, test = FALSE)
  tab_post_adjust_smd <- rbind(attr(tab_post_adjust$ContTable, "smd"), attr(tab_post_adjust$CatTable, "smd"))
  smds[i] <- mean(tab_post_adjust_smd)
  names(aarp_syn)[dim(aarp_syn)[2]] <- paste0("kw.", "mob.", i, collapse = "")
}

# Select KW weights with best average covariate balance
best <- which.min(smds)
names(aarp_syn)[names(aarp_syn) == paste0("kw.", "mob.", best, collapse = "")] <- "kw.2"
aarp_syn[, grep("kw.mob", names(aarp_syn))] <- NULL

#### Random Forest (RF)

# Set try-out values and prepare loop
tune_mtry <- c(floor(sqrt(cols)), floor(log(cols)))
psa_dat$wt_kw <- psa_dat$wt
p_scores <- data.frame(matrix(ncol = length(tune_mtry), nrow = nrow(psa_dat)))
p_score_c <- data.frame(matrix(ncol = length(tune_mtry), nrow = n_c))
p_score_s <- data.frame(matrix(ncol = length(tune_mtry), nrow = n_s))
smds <- rep(NA, length(tune_mtry))

# Loop over try-out values, calculate KW weights and covariate balance
for (i in seq_along(tune_mtry)) {
  mtry <- tune_mtry[i]
  rf <- ranger(trt_f ~ age+sex_f+race_f+martl_f+educ+bmi+smk1_f+phys+health,
             data = psa_dat,
             splitrule = "gini",
             num.trees = 500,
             mtry = mtry,
             min.node.size = 15,
             write.forest = T,
             probability = T)
  p_scores[, i] <- predict(rf, psa_dat, type = "response")$predictions[, 2]
  p_score_c[, i] <- p_scores[psa_dat$trt == 1, i]
  p_score_s[, i] <- p_scores[psa_dat$trt == 0, i]
  # Calculate KW weights
  aarp_syn$kw <- kw.wt(p_score.c = p_score_c[,i], p_score.s = p_score_s[,i], svy.wt = nhis_m$elig_wt, Large=T)$pswt
  # Calculate covariate balance
  psa_dat$wt_kw[psa_dat$trt == 1] <- aarp_syn$kw
  ds_kw <- svydesign(ids = ~1, weight = ~ wt_kw, data = psa_dat)
  tab_post_adjust <- svyCreateTableOne(vars = covars, factorVars = catcovars, strata = "trt", data = ds_kw, test = FALSE)
  tab_post_adjust_smd <- rbind(attr(tab_post_adjust$ContTable, "smd"), attr(tab_post_adjust$CatTable, "smd"))
  smds[i] <- mean(tab_post_adjust_smd)
  names(aarp_syn)[dim(aarp_syn)[2]] <- paste0("kw.", "rf.", i, collapse = "")
}

# Select KW weights with best average covariate balance
best <- which.min(smds)
names(aarp_syn)[names(aarp_syn) == paste0("kw.", "rf.", best, collapse = "")] <- "kw.3"
aarp_syn[, grep("kw.rf", names(aarp_syn))] <- NULL

#### Extremely Randomized Trees (XTREE)

# Set try-out values and prepare loop
tune_mtry <- c(floor(sqrt(cols)), floor(log(cols)))
psa_dat$wt_kw <- psa_dat$wt
p_scores <- data.frame(matrix(ncol = length(tune_mtry), nrow = nrow(psa_dat)))
p_score_c <- data.frame(matrix(ncol = length(tune_mtry), nrow = n_c))
p_score_s <- data.frame(matrix(ncol = length(tune_mtry), nrow = n_s))
smds <- rep(NA, length(tune_mtry))

# Loop over try-out values
for (i in seq_along(tune_mtry)) {
  mtry <- tune_mtry[i]
  xtree <- ranger(trt_f ~ age+sex_f+race_f+martl_f+educ+bmi+smk1_f+phys+health,
               data = psa_dat,
               splitrule = "extratrees",
               num.random.splits = 1,
               num.trees = 500,
               mtry = mtry,
               min.node.size = 15,
               write.forest = T,
               probability = T)
  p_scores[, i] <- predict(xtree, psa_dat, type = "response")$predictions[, 2]
  p_score_c[, i] <- p_scores[psa_dat$trt == 1, i]
  p_score_s[, i] <- p_scores[psa_dat$trt == 0, i]
  # Calculate KW weights
  aarp_syn$kw <- kw.wt(p_score.c = p_score_c[,i], p_score.s = p_score_s[,i], svy.wt = nhis_m$elig_wt, Large=T)$pswt
  # Calculate covariate balance
  psa_dat$wt_kw[psa_dat$trt == 1] <- aarp_syn$kw
  ds_kw <- svydesign(ids = ~1, weight = ~ wt_kw, data = psa_dat)
  tab_post_adjust <- svyCreateTableOne(vars = covars, factorVars = catcovars, strata = "trt", data = ds_kw, test = FALSE)
  tab_post_adjust_smd <- rbind(attr(tab_post_adjust$ContTable, "smd"), attr(tab_post_adjust$CatTable, "smd"))
  smds[i] <- mean(tab_post_adjust_smd)
  names(aarp_syn)[dim(aarp_syn)[2]] <- paste0("kw.", "xtree.", i, collapse = "")
}

# Select KW weights with best average covariate balance
best <- which.min(smds)
names(aarp_syn)[names(aarp_syn) == paste0("kw.", "xtree.", best, collapse = "")] <- "kw.4"
aarp_syn[, grep("kw.xtree", names(aarp_syn))] <- NULL

#### Gradient Boosting (GBM)

# Set try-out values and prepare loop
tune_ntree <- c(250, 500, 1000)
psa_dat$wt_kw <- psa_dat$wt
p_scores <- data.frame(matrix(ncol = length(tune_ntree), nrow = nrow(psa_dat)))
p_score_c <- data.frame(matrix(ncol = length(tune_ntree), nrow = n_c))
p_score_s <- data.frame(matrix(ncol = length(tune_ntree), nrow = n_s))
smds <- rep(NA, length(tune_ntree))

# Loop over try-out values
for (i in seq_along(tune_ntree)) {
  ntree <- tune_ntree[i]
  boost <- gbm(trt ~ age+sex_f+race_f+martl_f+educ+bmi+smk1_f+phys+health,
             data = psa_dat,
             distribution = "bernoulli",
             n.trees = ntree,
             interaction.depth = 3,
             shrinkage = 0.05,
             bag.fraction = 0.5,
             verbose = TRUE)
  p_scores[, i] <- predict(boost, psa_dat, n.trees = ntree, type = "response")
  p_score_c[, i] <- p_scores[psa_dat$trt == 1, i]
  p_score_s[, i] <- p_scores[psa_dat$trt == 0, i]
  # Calculate KW weights
  aarp_syn$kw <- kw.wt(p_score.c = p_score_c[,i], p_score.s = p_score_s[,i], svy.wt = nhis_m$elig_wt, Large=T)$pswt
  # Calculate covariate balance
  psa_dat$wt_kw[psa_dat$trt == 1] <- aarp_syn$kw
  ds_kw <- svydesign(ids = ~1, weight = ~ wt_kw, data = psa_dat)
  tab_post_adjust <- svyCreateTableOne(vars = covars, factorVars = catcovars, strata = "trt", data = ds_kw, test = FALSE)
  tab_post_adjust_smd <- rbind(attr(tab_post_adjust$ContTable, "smd"), attr(tab_post_adjust$CatTable, "smd"))
  smds[i] <- mean(tab_post_adjust_smd)
  names(aarp_syn)[dim(aarp_syn)[2]] <- paste0("kw.", "gbm.", i, collapse = "")
}

# Select KW weights with best average covariate balance
best <- which.min(smds)
names(aarp_syn)[names(aarp_syn) == paste0("kw.", "gbm.", best, collapse = "")] <- "kw.5"
aarp_syn[, grep("kw.gbm", names(aarp_syn))] <- NULL

###########################################################################
#### Calcute weighted estimates 

# NHIS estimate of 9-year all-cause mortality
est_nhis = sum(nhis_m$mtlty*nhis_m$elig_wt)/sum(nhis_m$elig_wt)
age_c.m_nhis = outer(nhis_m$age_c4, c(1:4), function(a, b)as.integer(a==b))
grp_wt.m_nihs = nhis_m$elig_wt*age_c.m_nhis
est_nhis = c(est_nhis, apply(nhis_m$mtlty*grp_wt.m_nihs, 2, sum)/apply(grp_wt.m_nihs, 2, sum))*100
round(est_nhis, 2)

# Record the number of methods used for propensity score calculation
n_pw = length(grep("ipsw", names(aarp_syn), value = T))  # from weighted model, for IPSW
n_p1 = length(grep("psas", names(aarp_syn), value = T)) # from unweighted models, for PSAS
n_p2 = length(grep("kw", names(aarp_syn), value = T)) # from unweighted models, for KW

# Setup a matrix storing the weighted estimates
est = matrix(0, (n_pw+n_p1+n_p2), 5)
# Convert categorial age group variable to a matrix of dummy variables
age_c.m = outer(aarp_syn$age_c4, c(1:4), function(a, b)as.integer(a==b))

#### Inverse of propensity score method
# estimate overall estimate of mortality rate
est[n_pw,1] = sum(aarp_syn$mtlty*aarp_syn$ipsw)/sum(aarp_syn$ipsw)*100
# estimate mortality rate by age group 
grp_wt.m = aarp_syn$ipsw*age_c.m
est[n_pw,c(2:5)] = apply(aarp_syn$mtlty*grp_wt.m, 2, sum)/apply(grp_wt.m, 2, sum)*100

#### Sub-classification
# estimate overall estimate of mortality rate
est[n_pw+n_p1,1] = sum(aarp_syn$mtlty*aarp_syn$psas)/sum(aarp_syn$psas)*100
# estimate mortality rate by age group 
grp_wt.m = aarp_syn$psas*age_c.m
est[n_pw+n_p1,c(2:5)] = apply(aarp_syn$mtlty*grp_wt.m, 2, sum)/apply(grp_wt.m, 2, sum)*100

#### Kernel-weighting 
for(i in 1:n_p2){
  # estimate overall estimate of mortality rate
  est[(i+n_pw+n_p1),1] = sum(aarp_syn$mtlty*eval(parse(text = paste0("aarp_syn$kw.", i))))/sum(eval(parse(text = paste0("aarp_syn$kw.", i))))*100
  # estimate mortality rate by age group 
  grp_wt.m = eval(parse(text = paste0("aarp_syn$kw.", i)))*age_c.m
  est[(i+n_pw+n_p1),c(2:5)] = apply(aarp_syn$mtlty*grp_wt.m, 2, sum)/apply(grp_wt.m, 2, sum)*100
}

###########################################################################
#### Compare balance and weighted estimates
  
# Covariate balance after adjustment (KW)
tab_post_adjust_smd <- data.frame(matrix(ncol = n_p2, nrow = length(covars)))
for(i in 1:n_p2){
  psa_dat$wt_kw <- psa_dat$wt
  psa_dat$wt_kw[psa_dat$trt == 1] <- eval(parse(text = paste0("aarp_syn$kw.", i)))
  ds_kw <- svydesign(ids = ~1, weight = ~ wt_kw, data = psa_dat)
  tab_post_adjust <- svyCreateTableOne(vars = covars, factorVars = catcovars, strata = "trt", data = ds_kw, test = FALSE)
  tab_post_adjust_smd[,i] <- rbind(attr(tab_post_adjust$ContTable, "smd"), attr(tab_post_adjust$CatTable, "smd"))
}
summary(tab_post_adjust_smd)

# Naive cohort estimate 
est.cht=mean(aarp_syn$mtlty)*100
est.cht=c(est.cht, apply(aarp_syn$mtlty*age_c.m, 2, mean)/apply(age_c.m, 2, mean)*100)
est=rbind(est.cht, est)
# Relative difference from weighted NHIS estimate
rel.diff = t((t(est)-est_nhis)/est_nhis*100)
colnames(rel.diff)=c("Overall", "50-54", "55-59", "60-64", "64+")
# Please change the row names (weighting method)
rownames(rel.diff)= c("NIH-AARP", "IPSW", "PSAS", "KW-Logit", "KW-MOB", "KW-RF", "KW-XTREE", "KW-GBM")
round(rel.diff, 3)
# bias reduction%
bias.r = t((rel.diff[1,]-t(rel.diff))/rel.diff[1,])*100
colnames(bias.r)=c("Overall", "50-54", "55-59", "60-64", "64+")
# Please change the row names (weighting method)
rownames(bias.r)= c("NIH-AARP", "IPSW", "PSAS", "KW-Logit", "KW-MOB", "KW-RF", "KW-XTREE", "KW-GBM")
round(bias.r, 3)
