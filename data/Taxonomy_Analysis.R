#author: Vivek Agarwal
#email: vivek1989@gmail.com
#version 1
#This code implements the elobw method and STM on lessons learned data from WB ICRs.   

#functions and libraries

library(stm)
library(stringr)
library(readtext)
library("readxl")

# preliminaries
setwd('C:/Users/skans/Documents/GitHub/Advanced Policy Analysis/DeCODE/GDI Papers/GDI Paper 1/GDI Final Paper 1 Data and Script/Datasets/')

###################################################################################################
#####################################   Preparing Dataset   #######################################
###################################################################################################
#DATA_DIR <- system.file("C:/Users/skans/Documents/GitHub/Advanced Policy Analysis/DeCODE/GDI Papers/GDI Paper 1/PDF File Extraction/", package = "readtext")
data <- read.csv(file="STM_data.csv", header=TRUE, sep=",")
# #data <- readtext(paste0(DATA_DIR, "cleaned_cleaned_sentences.txt"),
#          # docvarnames = c("Lessons"),
#          # dvsep = "_", 
#          # encoding = "ISO-8859-1")
# #data <- read.table(file = "", 
#  #                     sep = "\t", header=FALSE,fileEncoding= 'utf-8')
# data <- read.xlsx('R_Data_cleaned_lessons.xlsx', header=TRUE)
# 
# #Remvoing special characters
data$lessons <- iconv(data$lessons, to = "ASCII//TRANSLIT")
###################################################################################################
#######################################   Elbow Method   ##########################################
###################################################################################################
#   
library("RSiteCatalyst")
library("RTextTools")
#   
dtm <- create_matrix(data$lessons,
                     stemWords=TRUE,
                     removeStopwords=FALSE,
                     minWordLength=1,
                     removePunctuation= TRUE)
#   
cost_df <- data.frame()

#run kmeans for all clusters up to 10
for(i in 1:10){
  #Run kmeans for each level of i, allowing up to 100 iterations for convergence
  kmeans<- kmeans(x=dtm, centers=i, iter.max=10)
  
  #Combine cluster number and cost together, write to df
  cost_df<- rbind(cost_df, cbind(i, kmeans$tot.withinss))
}

names(cost_df) <- c("cluster", "cost")
#   
#   
# ###################################################################################################
# #####################################   Preparing Dataset   #######################################
# ###################################################################################################
#   
#Removing short sentences
data$WordLength <- str_count(data$lessons, "\\S+")
data <- data[data$WordLength > 10,]
data$WordLength <- NULL

#Remvoing specific words
data$lessons <- gsub('project|projects|water|education|transport|health|agriculture|agricultural|school|schools|teacher|teachers|nurses|doctor|doctors|rice|maize|crops|crop|seed|seeds|cocoa|tunisia|irrigate|irrigation|china|map', '', data$Text)

#Data prep for STM package
processed <- textProcessor(data$lessons, metadata = data)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta)
# docs <- out$documents
# vocab <- out$vocab
# meta <-out$meta

# Estimate STM
LLPrevFit <- stm(documents = out$documents, vocab = out$vocab, K = 50, max.em.its = 5, data = out$meta, init.type = "Spectral")
# Display Expected Topic Proportions
topicNames<-c("Topic 1","Topic 2","Topic 3","Topic 4","Topic 5","Topic 6",
              "Topic 7","Topic 8","Topic 9","Topic 10","Topic 11","Topic 12",
              "Topic 13","Topic 14","Topic 15")

plot.STM(LLPrevFit,type="summary",custom.labels="",topic.names=topicNames)

#Topics
labelTopics(LLPrevFit)