#############load the required packages################
library(shiny)
library(ggplot2)
require(shinydashboard)
library(dplyr)
library(magrittr)
library(caret)
library(gbm)
library(pROC)
library(ggthemes)
library(ggrepel)
library(forcats)
library(tree)
library(treemapify)

######################### Loading Dataset ##########################
df <- read.csv("data.csv")
colnames(df)[1] <- 'Age'

df$Generation <- ifelse(df$Age<37,"Millenials",
                        ifelse(df$Age>=38 & df$Age<54,"Generation X",
                               ifelse(df$Age>=54 & df$Age<73,"Boomers","Silent"
                               )))
######################### New Dataframe for further preprocessing #############################
model <- df
model[, 'Education'] <-
    factor(
        model[, 'Education'],
        levels = c('1', '2', '3', '4', '5'),
        labels = c('Below College', 'College', 'Bachelor', 'Master', 'Doctor'),
        ordered = TRUE
    )
model[, 'EnvironmentSatisfaction'] <-
    factor(
        model[, 'EnvironmentSatisfaction'],
        levels = c('1', '2', '3', '4'),
        ordered = TRUE,
        labels = c('Low', 'Medium', 'High', 'Very High')
    )
model[, 'JobInvolvement'] <-
    factor(
        model[, 'JobInvolvement'],
        levels = c('1', '2', '3', '4'),
        labels = c('Low', 'Medium', 'High', 'Very High'),
        ordered = TRUE
    )
head(model[, 'JobInvolvement'], n = 2)
model[, 'JobSatisfaction'] <-
    factor(
        model[, 'JobSatisfaction'],
        levels = c('1', '2', '3', '4'),
        labels = c('Low', 'Medium', 'High', 'Very High'),
        ordered = TRUE
    )
model[, 'PerformanceRating'] <-
    factor(
        model[, 'PerformanceRating'],
        levels = c('1', '2', '3', '4'),
        labels = c('Low', 'Good', 'Excellent', 'Outstanding'),
        ordered = TRUE
    )
model[, 'RelationshipSatisfaction'] <-
    factor(
        model[, 'RelationshipSatisfaction'],
        levels = c('1', '2', '3', '4'),
        labels = c('Low', 'Medium', 'High', 'Very High'),
        ordered = TRUE
    )
model[, 'WorkLifeBalance'] <-
    factor(
        model[, 'WorkLifeBalance'],
        levels = c('1', '2', '3', '4'),
        labels = c('Bad', 'Good', 'Better', 'Best'),
        ordered = TRUE
    )

model[, 'JobLevel'] <-
    factor(model[, 'JobLevel'],
           levels = c('1', '2', '3', '4', '5'),
           ordered = TRUE)

model %>% mutate_if(is.integer, as.numeric)
colnames(model)[2] <- 'y'

model <- model[c(
    "y",
    "Age",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EmployeeNumber",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
)]

model[, 'y'] <- (ifelse(model[, 'y'] == 'Yes', 1, 0))
model[, 'OverTime'] <- ifelse(model[, 'OverTime'] == 'Yes', 1, 0)
model[, 'Gender'] <- ifelse(model[, 'Gender'] == 'Male', 1, 0)

######### Variable to be used for training #########
model_train <- model[c(
    "Age",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EmployeeNumber",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
)]

############# One-Hot Encoding on training variables ##############
dummies <- dummyVars(~ ., data = model_train)
ex <- data.frame(predict(dummies, newdata = model_train))
names(ex) <- gsub("\\.", "", names(ex))
d <- cbind(model$y, ex)
names(d)[1] <- "y"
sum(model$y)

########### Find linear combinations and remove #################
comboInfo <- findLinearCombos(d)
comboInfo
# remove columns identified that led to linear combos
d <- d[,-comboInfo$remove]
# remove the "ones" column in the first column
d <- d[, c(2:ncol(d))]
y <- model$y
# Add the target variable back to our data.frame
d <- cbind(y, d)
rm(y, comboInfo)

##########Removing all values which have zero variability##########
nzv <- nearZeroVar(d, saveMetrics = TRUE)
head(nzv)
? nearZeroVar
#d <- d[, c(TRUE,!nzv$zeroVar[2:ncol(d)])]
d <- d[, c(TRUE, !nzv$nzv[2:ncol(d)])]
rm(nzv) #Cleaning R environment

preProcValues <- preProcess(d[, 2:ncol(d)], method = c("range"))
d <- predict(preProcValues, d)
# te set
rm(preProcValues)
sum(d$y)

##### Reading default values to predict for test model ########
df1 <- read.csv("test.csv")
names(df1)[1] <- "Age"
df2 <- read.csv("test.csv")
names(df2)[1] <- "Age"
###################### Partition data into test-train ######################
set.seed(1234) # set a seed so you can replicate your results
inTrain <- createDataPartition(y = d$y,   # outcome variable
                               p = .90,   # % of training data you want
                               list = F)
# create your partitions
train <- d[inTrain, ]  # training data set
test <- d[-inTrain, ]
train[, 'y'] <- as.factor(ifelse(train[, 'y'] == 1, 'Yes', 'No'))
#train[,'y'] <- as.factor(train[,'y'])

####### Specify training control parameters ########
ctrl <- caret::trainControl(
    method = "cv",
    number = 5,
    summaryFunction = twoClassSummary,
    classProbs = TRUE
)


####### Train GBM model on dataset ########## 
gbmfit <- train(
    y ~ .,
    data = train,
    method = "gbm",
    verbose = FALSE,
    metric = "ROC",
    trControl = ctrl
)

gbmtrain <- predict(gbmfit, train, type = 'prob')
gbmpreds <- predict(gbmfit, test, type = 'prob')
gbmvalid <- predict(gbmfit, df1, type = "prob")


##### Evaluate ROC on train dataset #######
gbm.ROC.train <-
    pROC::roc(
        train$y,
        gbmtrain$Yes,
        ci = TRUE,
        ci.alpha = 0.9,
        # arguments for plot
        plot = TRUE,
        auc.polygon = TRUE,
        max.auc.polygon = TRUE,
        show.thres = TRUE,
        print.auc = TRUE
    )
##### Evaluate ROC on test dataset #######
gbm.ROC.test <-
    pROC::roc(
        test$y,
        gbmpreds$Yes,
        ci = TRUE,
        ci.alpha = 0.9,
        # arguments for plot
        plot = TRUE,
        auc.polygon = TRUE,
        max.auc.polygon = TRUE,
        show.thres = TRUE,
        print.auc = TRUE
    )

##################### R-Shiny App UI ###########################

#Dashboard header carrying the title of the dashboard
header <- dashboardHeader(title = "HR Dashboard")

#Sidebar content of the dashboard
sidebar <- dashboardSidebar(sidebarMenu(
    menuItem(
        "About the Organization",
        tabName = "dashboard",
        icon = icon("dashboard")
    ),
    menuItem(
        "Exploratory Data Analysis",    tabName = "EDA",    icon = icon("calendar")
    ),
    menuItem("Predictive App", tabName = "predict", icon = icon("book"))
))

tbs <- tabItems(
    # First tab content
    tabItem(
        tabName = "dashboard",
        fluidRow(
            box(
                title = "Number of Employees by Department"     ,
                status = "primary"        ,
                solidHeader = TRUE        ,
                plotOutput("plot2", height = "300px")
            ),
            
            box(
                title = "Major Job roles inside the Organization"        ,
                status = "primary"        ,
                solidHeader = TRUE        ,
                plotOutput("plot1", height = "300px")
            )
            
            
        ) #End of Fluid Row 1
        ,
        fluidRow(
            box(
                title = "Gender Distribution"        ,
                status = "primary"        ,
                solidHeader = TRUE        ,
                plotOutput("plot4", height = "300px")
            ),
            
            box(
                title = "Average Salary by Gender"        ,
                status = "primary"        ,
                solidHeader = TRUE        ,
                plotOutput("plot3", height = "300px")
            )


            
        ) # En
    ),
    ######################################################
    #### Second tab content #####
    tabItem(
        tabName = "EDA",
        fluidRow(
            box(
                title = "Is income a reason for Employees to leave?"        ,
                status = "primary"        ,
                solidHeader = TRUE        ,
                collapsible = TRUE        ,
                plotOutput("incomeattrition", height = "300px")
            )
            
            ,
            box(
                title = "Attrition by education level"        ,
                status = "primary"        ,
                solidHeader = TRUE        ,
                collapsible = TRUE        ,
                plotOutput("attritionvseducation", height = "300px")
            )
        ) #End of Fluid Row 1
        ,
        fluidRow(
            box(
                title = "Understanding different generations"        ,
                status = "primary"        ,
                solidHeader = TRUE        ,
                collapsible = TRUE        ,
                plotOutput("diffgenerations", height = "300px")
            )
            ,
            box(
                title = "Behavioral Difference between generations"        ,
                status = "primary"        ,
                solidHeader = TRUE        ,
                collapsible = TRUE        ,
                plotOutput("behavedef", height = "300px")
            )
            
        ) # End of fluid row 2
        ,
        fluidRow(
            box(
                title = "Monthly Income vs percent Hike"        ,
                status = "primary"        ,
                solidHeader = TRUE        ,
                collapsible = TRUE        ,
                plotOutput("incomevshike", height = "300px")
            ),
            box(
                title = "Performance Rating vs Monthly Income"   ,
                status = "primary"        ,
                solidHeader = TRUE        ,
                collapsible = TRUE        ,
                plotOutput("PRvsIncome", height = "300px")
            )
        ) # End of fluid Row 3
        ,
        fluidRow(
            box(
                title = "Difference between average salaries per day of people staying vs people leaving"        ,
                status = "primary"        ,
                solidHeader = TRUE        ,
                collapsible = TRUE        ,
                plotOutput("saldiff", height = "300px")
            ),
            box(
                title = "Environment Satisfaction vs Job Position"        ,
                status = "primary"        ,
                solidHeader = TRUE        ,
                collapsible = TRUE        ,
                plotOutput("EnvsJob", height = "300px")
            )
        ) #End of fluid row 4
        ,
        fluidRow(
            box(
                title = "Work Life Balance vs Department"        ,
                status = "primary"        ,
                solidHeader = TRUE        ,
                collapsible = TRUE        ,
                plotOutput("worklifebal", height = "300px")
            )
            ,
            box(
                title = "Stock options vs attrition"        ,
                status = "primary"        ,
                solidHeader = TRUE        ,
                collapsible = TRUE        ,
                plotOutput("stockvsattrition", height = "300px")
            )
        ) #End of fluid row 4
        
    ),
    #######################################################
    
    # Third tab content
    tabItem(tabName = "predict",
            fluidRow(
                box(
                    sliderInput("age", "Age ", 19, 60, 25),
                    numericInput("yearsatcompany", "Years at Company", value=5, min=0, max=40),
                    numericInput("totalworkingyears", "Total Work Ex", value=7, min=0, max=40),
                    selectInput("Overtime", "Pick Overtime",
                                c("Yes" = 1,
                                  "No" = 0), selected = 1),
                    selectInput("StockOptionLevel", "What is the Stock Option Level?",
                                c("0" ,
                                  "1" ,
                                  "2" ,
                                  "3" ), selected = 1),
                    radioButtons(
                        "btravel",
                        "Business Travel",
                        choices = c(
                            "Non Travel" = "A",
                            "Travel Frequently" = "B",
                            "Travel Rarely" = "C"
                        ), selected = "A"
                    ),
                    radioButtons(
                        "EnvironmentSatisfactionL",
                        "Environment Satisfaction Level",
                        choices = c(
                            "Low" = 0,
                            "Medium" = 3,
                            "High" = 6,
                            "Very High" = 9
                        ), selected = 3
                    ),
                    sliderInput("salary", "Monthly Income ", 1000, 50000, 10000),
                    sliderInput("YearsWithCurrManager", "Years With Current Manager", 0, 20, 5),
                    # numer
                    # numericInput("", "Years at company", 0, 40, 25),
                    # selectInput("yearsatcompany", "Years at company",
                    #             c()),
                ),
                # valueBoxOutput("approvalBox",width = 6)
                infoBoxOutput("approvalBox", width = 6),
                infoBoxOutput("SuggestionBox",width = 6)
            ))
)



# combine the two fluid rows to make the body
body <- dashboardBody(tbs,
                      tags$head(
                          tags$link(rel = "stylesheet", type = "text/css", href = "custom.css")
                      ))

#completing the ui part with dashboardPage
ui <-
    dashboardPage(title = 'HR Dashboard', header, sidebar, body, skin =
                      'blue')

# create the server functions for the dashboard
server <- function(input, output) {
    #some data manipulation to derive the values of KPI boxes
    Average_Age <- mean(df$Age)
    # sales.account <- recommendation %>% group_by(Account) %>% summarise(value = sum(Revenue)) %>% filter(value==max(value))
    # prof.prod <- recommendation %>% group_by(Product) %>% summarise(value = sum(Revenue)) %>% filter(value==max(value))
    Median_Income <- median(df$MonthlyIncome)
    Dist_From_Home <- round(mean(df$DistanceFromHome),2)
    Total_Work_Exp <- round(mean(df$TotalWorkingYears),2)
    
    #creating the valueBoxOutput content
    output$plot1 <- renderPlot({
        role.amount <- df %>% select(JobRole) %>% group_by(JobRole) %>% summarize(amount=n()) %>%
            ggplot(aes(area=amount, fill=JobRole, label=JobRole)) +  geom_treemap(alpha =0.7) +
            geom_treemap_text(grow = T, reflow = T, colour = "black", 
                              min.size= 2) +
            #scale_fill_brewer(palette = "YlOrRd") +
            theme(legend.position = "none") +
            labs(
                caption = "The area of each tile represents the number of
employees by type of job role.",
                fill = "JobRole"
            )+
            theme_economist()+  
            theme(legend.position = 'none')
        
        role.amount
    })
    
    output$plot2 <- renderPlot({
        
        
        df %>% group_by(Department, Gender) %>% 
            summarise(amount=n()) %>%
            mutate(pct = amount/sum(amount),lbl = scales::percent(pct))%>% 
            ggplot(aes(x="", y=amount, fill=Department), show.legend=FALSE, width=) + 
            geom_bar(stat="identity", position="dodge", alpha = 0.7) +
            #theme(axis.text.x = element_text(angle = 90), 
            #      plot.title=element_text(hjust=0.5), aspect.ratio=1) + 
            labs() + 
            coord_polar()+
            #scale_fill_manual(values=c("#FE642E", "#0080FF","#00FF40"))+
            theme_economist()+
            theme(legend.position = 'right', legend.text = element_text(size=9), 
                  legend.title = element_text(size=12))
        
        
    })
    
    output$plot3 <- renderPlot({
        gender.income <- df %>% select(Gender, MonthlyIncome) %>% group_by(Gender) %>% summarise(avg_income=round(mean(MonthlyIncome), 2)) %>%
            ggplot(aes(x=Gender, y=avg_income)) + geom_bar(stat="identity", fill="dark grey", width=0.5) + 
            geom_text(aes(x=Gender, y=0.01, label= paste0("$ ", avg_income)),
                      hjust=-2, vjust=0, size=3, 
                      colour="black", fontface="bold",
                      angle=360) + labs(x="Gender",y="Salary") + coord_flip() + 
            theme_minimal() + theme(plot.title=element_text(size=14, hjust=0.5)) +
            theme(plot.background=element_rect(fill="lightblue"))
        
        gender.income
        
    })
    
    output$plot4 <- renderPlot({
        dat_text <- data.frame(
            label = c("Mean = 37.33 \n Years Old", "Mean = 36.65 \n Years Old"),
            Gender   = c("Female", "Male")
        )
        
        
        
        gender.dist <- df %>% select(Gender, Age) %>% filter(Gender == 'Male' | Gender== "Female") %>% 
            filter(!is.na(Age)) %>% group_by(Gender) %>% 
            ggplot(aes(x=Age)) + geom_density(aes(fill=Gender), alpha=0.8, show.legend=FALSE) + facet_wrap(~Gender) + theme_minimal() + 
            geom_vline(aes(xintercept=mean(Age)),
                       color="red", linetype="dashed", size=1) + labs(title="Age Distribution") + 
            theme(plot.title=element_text(hjust=0.5)) + scale_fill_manual(values=c("#F781F3", "azure4")) + 
            geom_text(
                data    = dat_text,
                mapping = aes(x = 45, y = 0.03, label = label),
                hjust   = -0.1,
                vjust   = -1
            )+
            theme(plot.background=element_rect(fill="lightblue"))
        
        gender.dist
        
    })
    
    output$modelinfo <- renderPrint({
        gbmfit$bestTune
    })
    
    
    ###### EDA #######
    output$incomeattrition <- renderPlot({
        df$JobSatisfaction <- as.factor(df$JobSatisfaction)
        
        df %>% select(JobSatisfaction, MonthlyIncome, Attrition) %>% group_by(JobSatisfaction, Attrition) %>%
            summarize(med=median(MonthlyIncome)) %>%
            ggplot(aes(x=fct_reorder(JobSatisfaction, -med), y=med, color=Attrition)) + 
            geom_point(size=3) + 
            geom_segment(aes(x=JobSatisfaction, 
                             xend=JobSatisfaction, 
                             y=0, 
                             yend=med)) + facet_wrap(~Attrition) + 
            labs( 
                 y="Median Income",
                 x="Level of Job Satisfaction") + 
            theme(axis.text.x = element_text(angle=65, vjust=0.6), plot.title=element_text(hjust=0.5), strip.background = element_blank(),
                  strip.text = element_blank()) + 
            coord_flip() + theme_minimal() + scale_color_manual(values=c("#58FA58", "#FA5858")) + 
            geom_text(aes(x=JobSatisfaction, y=0.01, label= paste0("$ ", round(med,2))),
                      hjust=-0.5, vjust=-0.5, size=4, 
                      colour="black", fontface="italic",
                      angle=360)+
            theme_economist()+
            theme(legend.position = 'right')
    })
    
    output$attritionvseducation <- renderPlot({
        df$Educational_Levels <-  ifelse(df$Education == 1, "Without College D.",
                                         ifelse(df$Education == 2 , "College D.",
                                                ifelse(df$Education == 3, "Bachelors D.",
                                                       ifelse(df$Education == 4, "Masters D.", "Phd D."))))
        
        # I want to know in terms of proportions if we are loosing key talent here.
            df %>% select(Educational_Levels, Attrition) %>% group_by(Educational_Levels, Attrition) %>% 
            summarize(n=n()) %>% 
            ggplot(aes(x=fct_reorder(Educational_Levels,n), y=n, fill=Attrition, color=Attrition)) + geom_bar(stat="identity") + facet_wrap(~Attrition) + 
            coord_flip() + scale_fill_manual(values=c("#2EF688", "#F63A2E")) + scale_color_manual(values=c("#09C873","#DD1509")) + 
            geom_label(aes(label=n, fill = Attrition), colour = "white", fontface = "italic") + 
            labs(x="", y="Number of Employees") + theme_wsj() + 
            theme(legend.position="none", plot.title=element_text(hjust=0.5, size=14))+
            theme_economist()+
            theme(legend.position = 'right')
        
        
        
    })
    
    output$diffgenerations <- renderPlot({
        # First we must create categoricals variables based on Age
        df$Generation <- ifelse(df$Age<37,"Millenials",
                                ifelse(df$Age>=38 & df$Age<54,"Generation X",
                                       ifelse(df$Age>=54 & df$Age<73,"Boomers","Silent"
                                       )))
        
        
        # Let's see the distribution by generation now
        df %>% select(Generation, NumCompaniesWorked, Attrition) %>% 
            ggplot() + geom_boxplot(aes(x=reorder(Generation, NumCompaniesWorked, FUN=median), 
                                        y=NumCompaniesWorked, fill=Generation)) + 
            theme_tufte() + facet_wrap(~Attrition) + 
            scale_fill_brewer(palette="RdBu") + coord_flip() + 
            labs(x="", y="Number of Companies Previously Worked") + 
            #theme(legend.position="bottom", legend.background = element_rect(fill="#FFF9F5",
            #                                                                size=0.5, linetype="solid", 
            #                                                               colour ="black")) +
            theme_economist()+
            theme(legend.position = 'right')
        
        
    })
    
    output$behavedef <- renderPlot({
        
        df %>% select(Generation, NumCompaniesWorked, Attrition) %>% group_by(Generation, Attrition) %>%
            summarize(avg=mean(NumCompaniesWorked)) %>% 
            ggplot(aes(x=Generation, y=avg, color=Attrition)) + 
            geom_point(size=3) + theme_tufte() +  # Draw points
            geom_segment(aes(x=Generation, 
                             xend=Generation, 
                             y=min(avg), 
                             yend=max(avg)), 
                         linetype="dashed", 
                         size=0.1,
                         color="white") +  
            labs(
                 y="Average Number of Companies worked for",x="") +  
            coord_flip() + scale_color_manual(values=c("#58FA58", "#FA5858")) + 
            theme_economist()+
            theme(legend.position = 'right')
        
        
    })
    
    output$incomevshike <- renderPlot({
        
        
        options(repr.plot.width=8, repr.plot.height=7) 
        
        per.sal <- df %>% select(Attrition, PercentSalaryHike, MonthlyIncome) %>% 
            ggplot(aes(x=PercentSalaryHike, y=MonthlyIncome)) + geom_jitter(aes(col=Attrition), alpha=0.7) + 
            theme_economist() + theme(legend.position="none") + scale_color_manual(values=c("#58FA58", "#FA5858")) + 
            labs() + theme(plot.title=element_text(hjust=0.5, color="white"), plot.background=element_rect(fill="#0D7680"),
                                                                     axis.text.x=element_text(colour="white"), axis.text.y=element_text(colour="white"),
                                                                     axis.title=element_text(colour="white")) +
            theme_economist()+
            theme(legend.position = 'right')
        per.sal
        
        
    })
    
    output$hist <- renderText({
        df1['OverTime'] <- as.numeric(input$Overtime)
        df1['Age'] <- as.numeric(input$age) / 100
        df1['MonthlyIncome'] <- as.numeric(input$salary) / 20000
        if (input$btravel == 'A') {
            df1['BusinessTravelNonTravel'] <- 1
        } else if (input$btravel == 'B') {
            df1['BusinessTravelTravel_Frequently'] <- 1
        } else if (input$btravel == 'B') {
            df1['BusinessTravelTravel_Rarely'] <- 1
        }
        preds <- predict(gbmfit, df1, type = 'prob')
        result <- as.numeric(preds[[2]])
        result
    })
    
    output$image_display <- renderImage({
        list(
            src="featured.png"
        )
    })
    
    #####THE RED BOX OUTPUT####
    output$approvalBox <- renderInfoBox({
        df1['OverTime'] <- as.numeric(input$Overtime)
        if(as.numeric(input$Overtime)==1){
            df2['OverTime'] <- 0
        } else {df2['OverTime'] <- 0}
        
        df1['Age'] <- as.numeric(input$age) / 100
        df2['Age'] <- as.numeric(input$age) / 100
        
        df1['MonthlyIncome'] <- as.numeric(input$salary) / 20000
        df2['MonthlyIncome'] <- (as.numeric(input$salary)+1500) / 20000
        
        df1['StockOptionLevel'] <- as.numeric(input$StockOptionLevel)/4
        df2['StockOptionLevel'] <- (as.numeric(input$StockOptionLevel)+1)/4
        
        df1['EnvironmentSatisfactionL'] <- as.numeric(input$EnvironmentSatisfactionL)/9
        df2['EnvironmentSatisfactionL'] <- (as.numeric(input$EnvironmentSatisfactionL)+3)/9
        
        df1['YearsWithCurrManager'] <- as.numeric(input$YearsWithCurrManager)/20
        df2['YearsWithCurrManager'] <- as.numeric(input$YearsWithCurrManager)/20
        
        if (input$btravel == 'A') {
            df1['BusinessTravelNonTravel'] <- 1
        } else if (input$btravel == 'B') {
            df1['BusinessTravelTravel_Frequently'] <- 1
        } else if (input$btravel == 'C') {
            df1['BusinessTravelTravel_Rarely'] <- 1
        }
        
        if (input$btravel == 'A') {
            df2['BusinessTravelNonTravel'] <- 1
        } else if (input$btravel == 'B') {
            df2['BusinessTravelNonTravel'] <- 1
        } else if (input$btravel == 'C') {
            df2['BusinessTravelNonTravel'] <- 1
        }
        df1['YearsAtCompany'] <- as.numeric(input$yearsatcompany)/40
        df2['YearsAtCompany'] <- as.numeric(input$yearsatcompany)/40
        
        df1['TotalWorkingYears'] <- as.numeric(input$totalworkingyears)/40
        df2['TotalWorkingYears'] <- as.numeric(input$totalworkingyears)/40
        
        preds <- predict(gbmfit, df1, type = 'prob')
        result <- round(as.numeric(preds[[2]]), 2)
        
        preds2 <- predict(gbmfit, df2, type = 'prob')
        result2 <- round(as.numeric(preds2[[2]]), 2)
        
        if (result >= 0.5) {
            result <- paste(result * 100, '%')
            infoBox(
                result,
                paste(
                    'There is a',
                    result,
                    "probability the employee of age",
                    input$age,
                    "and salary $",
                    input$salary,
                    'will leave the firm'
                ),
                icon = icon("thumbs-down", lib = "glyphicon"),
                color = "red",
                fill = TRUE
            )
            # valueBox(
            #   formatC(result, format="d", big.mark=',')
            #   ,paste('There is a',result,"probability the employee of age",input$age,"and salary $",input$salary, 'will leave the firm')
            #   ,icon = icon("thumbs-down",lib='glyphicon')
            #   ,color = "red")
        } else {
            result <- paste(result * 100, '%')
            valueBox(
                formatC(result, format = "d", big.mark = ',')
                ,
                paste(
                    'There is a',
                    result,
                    "probability the employee of age",
                    input$age,
                    "and salary $",
                    input$salary,
                    'will leave the firm'
                )
                ,
                icon = icon("thumbs-up", lib = 'glyphicon')
                ,
                color = "green"
            )
        }
        
        
        
    })
    
    output$SuggestionBox <- renderInfoBox({
        df1['OverTime'] <- as.numeric(input$Overtime)
        if(as.numeric(input$Overtime)==1){
            df2['OverTime'] <- 0
        } else {df2['OverTime'] <- 0}
        
        df1['Age'] <- as.numeric(input$age) / 100
        df2['Age'] <- as.numeric(input$age) / 100
        
        df1['MonthlyIncome'] <- as.numeric(input$salary) / 20000
        df2['MonthlyIncome'] <- (as.numeric(input$salary)+1500) / 20000
        
        df1['StockOptionLevel'] <- as.numeric(input$StockOptionLevel)/4
        df2['StockOptionLevel'] <- (as.numeric(input$StockOptionLevel)+1)/4
        
        df1['EnvironmentSatisfactionL'] <- as.numeric(input$EnvironmentSatisfactionL)/9
        df2['EnvironmentSatisfactionL'] <- as.numeric(input$EnvironmentSatisfactionL)/9
        
        df1['YearsWithCurrManager'] <- as.numeric(input$YearsWithCurrManager)/20
        df2['YearsWithCurrManager'] <- as.numeric(input$YearsWithCurrManager)/20
        
        if (input$btravel == 'A') {
            df1['BusinessTravelNonTravel'] <- 1
        } else if (input$btravel == 'B') {
            df1['BusinessTravelTravel_Frequently'] <- 1
        } else if (input$btravel == 'C') {
            df1['BusinessTravelTravel_Rarely'] <- 1
        }
        
        if (input$btravel == 'A') {
            df2['BusinessTravelNonTravel'] <- 1
        } else if (input$btravel == 'B') {
            df2['BusinessTravelNonTravel'] <- 1
        } else if (input$btravel == 'C') {
            df2['BusinessTravelNonTravel'] <- 1
        }
        df1['YearsAtCompany'] <- as.numeric(input$yearsatcompany)/40
        df2['YearsAtCompany'] <- as.numeric(input$yearsatcompany)/40
        
        df1['TotalWorkingYears'] <- as.numeric(input$totalworkingyears)/40
        df2['TotalWorkingYears'] <- as.numeric(input$totalworkingyears)/40
        
        preds <- predict(gbmfit, df1, type = 'prob')
        result <- round(as.numeric(preds[[2]]), 2)
        
        preds2 <- predict(gbmfit, df2, type = 'prob')
        result2 <- round(as.numeric(preds2[[2]]), 2)
        
        if (result >= 0.5) {
            result <- paste(result * 100, '%')
            sug <- paste("Suggestions: ")
            infoBox(
                formatC(sug, format = "d", big.mark = ','),
                paste(if(as.numeric(input$Overtime)==1){"Reduce Overtime, "},
                      if(as.numeric(input$StockOptionLevel)!=4){"Increase Stock Option Level by 1, "},
                      if(input$btravel != 'A'){"Reduce Travel, "},
                      "Increase Salary by 1500$. ", 
                      "As a result the reduced probability will be", result2*100, "%"),
                if(as.numeric(input$YearsWithCurrManager)>5){HTML(paste("You can also try to change the manager."))}
                ,
                color = "blue",
                fill = TRUE
            )
            # valueBox(
            #   formatC(result, format="d", big.mark=',')
            #   ,paste('There is a',result,"probability the employee of age",input$age,"and salary $",input$salary, 'will leave the firm')
            #   ,icon = icon("thumbs-down",lib='glyphicon')
            #   ,color = "red")
        } else {
            result <- paste(result * 100, '%')
            sug <- paste("Suggestions: ")
            infoBox(
                formatC(sug, format = "d", big.mark = ','),
                
                paste(
                    "If you ", if(as.numeric(input$Overtime)==1){"Reduce Overtime, "},
                    if(as.numeric(input$StockOptionLevel)!=4){"Increase Stock Option Level by 1, "},
                    if(input$btravel != 'A'){"Reduce Travel, "},
                    "Increase the Salary by 1500$, ",
                    " the reduced probability that the employee will leave is ",result2*100, "%"
                ),
                color = "blue", fill = TRUE
            )
        }
        
        
        
    })   
    output$PRvsIncome <- renderPlot({
            df %>% select(PerformanceRating, MonthlyIncome, Attrition) %>% group_by(factor(PerformanceRating), Attrition) %>% 
            ggplot(aes(x=factor(PerformanceRating), y=MonthlyIncome, fill=Attrition)) + geom_violin() + coord_flip() + facet_wrap(~Attrition) + 
            scale_fill_manual(values=c("#58FA58", "#FA5858")) + theme_economist() + 
            theme(legend.position="bottom", strip.background = element_blank(), strip.text.x = element_blank(), 
                  plot.title=element_text(hjust=0.5, color="white"), plot.background=element_rect(fill="#0D7680"),
                  axis.text.x=element_text(colour="white"), axis.text.y=element_text(colour="white"),
                  axis.title=element_text(colour="white"), 
                  legend.text=element_text(color="white")) + 
            labs(x="Performance Rating",y="Monthly Income") +
            theme_economist()+
            theme(legend.position = 'right')
        
    })
    
    output$saldiff <- renderPlot({
        attrition_daily <- df %>% select(JobRole, Attrition, DailyRate) %>% group_by(JobRole) %>% filter(Attrition == "Yes") %>% 
            summarize(avg_attrition=mean(DailyRate))
        
        
        noattrition_daily <- df %>% select(JobRole, Attrition, DailyRate) %>% group_by(JobRole) %>% filter(Attrition == "No") %>% 
            summarize(avg_noattrition=mean(DailyRate))
        
        colors <- c("#316D15C", "#16D12C", "#B2D116", "#FEBE5D", "#FE9F5D", "#F86E2E", "#F8532E", "#FA451D", "#FA1D1D")
        
        combined_df <- merge(attrition_daily, noattrition_daily)
        colourCount = length(unique(combined_df$JobRole))
        
        percent_diff <- combined_df %>% mutate(pct_diff=round(((avg_noattrition - avg_attrition)/avg_noattrition),2) * 100) %>%
            ggplot(aes(x=reorder(JobRole,pct_diff), y=pct_diff, fill=JobRole)) + geom_bar(stat="identity") + coord_flip() + theme_minimal() +
            #scale_fill_manual(values = colorRampPalette(brewer.pal(9, "Set2"))(colourCount)) + 
            theme(plot.title=element_text(hjust=0.5, size=10), plot.background=element_rect(fill="#FFF1E0"), legend.position="none") + 
            labs(x="JobRole", y="Percent Difference (%)") + 
            geom_label(aes(label=paste0(pct_diff, "%")), colour = "white", fontface = "italic", hjust=0.2)+
            theme_economist()+  
            theme(legend.position = 'none')
        percent_diff
        
    })
    
    output$EnvsJob <- renderPlot({
        # 11. Environment Satisfaction let's use the changes by JobRole
        options(repr.plot.width=8, repr.plot.height=5)
        
        env.attr <- df %>% select(EnvironmentSatisfaction, JobRole, Attrition) %>% group_by(JobRole, Attrition) %>%
            summarize(avg.env=mean(EnvironmentSatisfaction))
        
        ggplot(env.attr, aes(x=JobRole, y=avg.env)) + geom_line(aes(group=Attrition), color="#58ACFA", linetype="dashed") + 
            geom_point(aes(color=Attrition), size=3) +  theme_economist() + theme(plot.title=element_text(hjust=0.5), axis.text.x=element_text(angle=90),
                                                                                  plot.background=element_rect(fill="#FFF1E0")) + 
            labs( y="Average Environment Satisfaction", 
                 x="Job Position") + scale_color_manual(values=c("#58FA58", "#FA5858"))+
            theme_economist()+  
            theme(legend.position = 'none', axis.text.x = element_text(size = 6, angle = 90))
        
        
        
        
    })
    


    
    
    output$worklifebal <- renderPlot({
        
        options(repr.plot.width=8, repr.plot.height=4)
        
        attritions <- df %>% filter(Attrition == "Yes")
        
        attritions$WorkLifeBalance <- as.factor(attritions$WorkLifeBalance)
        
        by.department <- attritions %>% select(Department, WorkLifeBalance) %>% group_by(Department, WorkLifeBalance) %>%
            summarize(count=n()) %>% 
            ggplot(aes(x=fct_reorder(WorkLifeBalance, -count), y=count, fill=Department)) + geom_bar(stat='identity') + facet_wrap(~Department) + 
            theme_economist() + theme(legend.position="bottom", plot.title=element_text(hjust=0.5), plot.background=element_rect(fill="#FFF1E0")) + 
            scale_fill_manual(values=c("#FA5882", "#819FF7", "#FE2E2E")) + 
            geom_label(aes(label=count, fill = Department), colour = "white", fontface = "italic") + 
            labs( x="Work and Life Balance", y="Number of Employees")+
            theme_economist()+
            theme(legend.position = 'none')
        
        by.department
        
    })
    
    output$stockvsattrition <- renderPlot({
        
            df %>% select(StockOptionLevel, Attrition) %>% group_by(StockOptionLevel, Attrition) %>% summarize(n=n())  %>%
            ggplot(aes(x=reorder(StockOptionLevel, -n), y=n, fill=factor(StockOptionLevel))) + geom_bar(stat="identity") + coord_flip() + 
            facet_wrap(~Attrition) + theme_economist() + scale_fill_manual(values=c("#DF0101", "#F5A9A9", "#BEF781", "#04B404")) + 
            guides(fill=guide_legend(title="Stock Option \n Level")) + 
            theme(legend.position="none", plot.background=element_rect(fill="#0D7680"), plot.title=element_text(hjust=0.5, color="white"), 
                  axis.text.x=element_text(colour="white"), axis.text.y=element_text(colour="white"),
                  axis.title=element_text(colour="white"),
                  strip.text.x = element_text(color="white"), 
                  legend.text=element_text(color="white"))  + 
            geom_label(aes(label=n, fill = factor(StockOptionLevel)), colour = "white", fontface = "italic", hjust=0.55) + 
            labs( x="StockOptionLevel", y="number of employees") +
            theme_economist()+
            theme(legend.position = 'none')
        
        
        
    })
    
}


shinyApp(ui, server)
