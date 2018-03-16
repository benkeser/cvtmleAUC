#-------------------------------------
# curating data sets for cvauc paper
#-------------------------------------

data_dir <- "~/Dropbox/R/cvtmleAUC/data/raw/"
setwd(data_dir)
save_dir <- "~/Dropbox/R/cvtmleAUC/data/"
# function to add a train and test variable
add_train <- function(dat, n = c(100, 250, 500)){
	nrows <- length(dat[,1])
	for(i in n){
		dat[,paste0("train_",i)] <- 0
		dat[sample(1:nrows, size = i), paste0("train_",i)] <- 1
	}
	return(dat)
}

#--------------------
# 1. Adult data set
#--------------------
adult <- read.table("adult.data", sep = ",")

colnames(adult) <- c("age", "workclass","fnlwgt","education","education_num",
                     "marital_status","occupation","relationship","race",
                     "sex","capital_gain","capital_loss","hours_per_week",
                     "native_country","income_50")

# add column named outcome
adult$outcome <- as.numeric(as.numeric(adult$income_50) == 2)

# remove income 50 column
adult <- adult[,-which(colnames(adult) %in% c("education","income_50"))] 

# add train and test variables for n = 100, 250, 500
adult <- add_train(adult)

# save to data folder
save(adult, file = paste0(save_dir,"adult.RData"))

#--------------------
# 2. Autism Adult data set
#--------------------
library(foreign)

asd <- read.arff("Autism-Adult-Data.arff")

# make numeric outcome column
asd$outcome <- as.numeric(as.numeric(asd[,'Class/ASD']) == 2)

# remove original outcome column
asd <- asd[,-which(colnames(asd) %in% c('Class/ASD'))]

# add train variables
asd <- add_train(asd)

# replace NA's in ethnicitiy
asd$ethnicity[is.na(asd$ethnicity)] <- "Others"
asd$ethnicity[asd$ethnicity == "others"] <- "Others"

# replace NA's in relation
levels(asd$relation) <- c(levels(asd$relation), "Unknown")
asd$relation[is.na(asd$relation)] <- "Unknown"

# replace NA's in age by mean
asd$age[is.na(asd$age)] <- mean(asd$age, na.rm = TRUE)

# save to data folder 
save(asd, file = paste0(save_dir,"asd.RData"))

#--------------------
# 3. Bank 
#--------------------
bank <- read.csv("bank-additional-full.csv", header = TRUE, sep = ";")

# numeric outcome
bank$outcome <- as.numeric(as.numeric(bank$y) == 2)

# remove original outcome column
bank <- bank[,-which(colnames(bank) %in% c('y'))]

# add train variables
bank <- add_train(bank)

# save to data folder 
save(bank, file = paste0(save_dir,"bank.RData"))

#--------------------------
# 4. Cardiotocography
#--------------------------
cardio <- read.csv("CTG.csv", header = TRUE)
# first line is blank
cardio <- cardio[-1,]
# first 5 columns seem to be garbage, second-to-last is
# an outcome measure
cardio <- cardio[,-(c(1:6,18,(ncol(cardio) - 11):(ncol(cardio)-1)))]

# make outcome predicting NSP = 2
cardio$outcome <- as.numeric(cardio$NSP == 2)

# remove original outcome column
cardio <- cardio[,-which(colnames(cardio) %in% c('NSP'))]

# add train variables
cardio <- add_train(cardio)

# save to data folder 
save(cardio, file = paste0(save_dir,"cardio.RData"))

#--------------------------
# 5. default
#--------------------------
default <- read.csv("default of credit card clients.csv", header = TRUE)
# remove id column
default <- default[,-1]

# make outcome 
default$outcome <- default$default.payment.next.month

# remove original outcome column
default <- default[,-which(colnames(default) %in% c('default.payment.next.month'))]

# add train variables
default <- add_train(default)

# save to data folder 
save(default, file = paste0(save_dir,"default.RData"))

#--------------------------
# 6. drug consumption
#--------------------------
drugs <- read.table("drug_consumption.data", sep = ",")

# get rid of id column
drugs <- drugs[,-1]

# column names
colnames(drugs) <- c("age", "gender","education","country","ethnicity","nscore",
                     "escore","oscore","ascore","cscore","imp","ss",paste0("drugs_",1:19))

# make predicting heroin outcome
drugs$outcome <- as.numeric(as.numeric(drugs[,24]) > 2)

# remove other drugs from variables
drugs <- drugs[,-grep("drugs_", colnames(drugs))]

# add train variables
drugs <- add_train(drugs)

# save to data folder 
save(drugs, file = paste0(save_dir,"drugs.RData"))

#--------------------------
# 7. magic telescope data consumption
#--------------------------
magic <- read.table("magic04.data", sep = ",")

# column names
colnames(magic) <- c("fLength","fWidth","fSize","fConc","fConc1",
                     "fAsym","fM3Long","fM3Trans","fAlpha",
                     'fDist','class')

# add outcome
magic$outcome <- as.numeric(as.numeric(magic$class) == 1)                     

# remove original outcome column
magic <- magic[,-which(colnames(magic) %in% c('class'))]

# add train variables
magic <- add_train(magic)

# save to data folder 
save(magic, file = paste0(save_dir,"magic.RData"))

#--------------------------------
# 8. wine
#--------------------------------
red_wine <- read.csv("winequality-red.csv", header = TRUE, sep = ";")
red_wine$red <- 1
white_wine <- read.csv("winequality-white.csv", header = TRUE, sep = ";")
white_wine$red <- 0
wine <- rbind(red_wine, white_wine)


# add outcome
wine$outcome <- as.numeric(as.numeric(wine$quality) >= 7)                     

# remove original outcome column
wine <- wine[,-which(colnames(wine) %in% c('quality'))]

# add train variables
wine <- add_train(wine)

# save to data folder 
save(wine, file = paste0(save_dir,"wine.RData"))

