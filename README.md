# Project Title: Exploring the Intersection of Statistics and Machine Learning through Subreddit Analysis

## Executive Summary
In the context of data science, Statistics and Machine Learning are two intricately linked fields, each mutually enhancing the other with vital concepts and methodologies. This project delved into the nexus of these disciplines through an exhaustive analysis of the subreddit threads r/statistics and r/machinelearning.

The project involved scraping approximately 1000 Reddit posts using the PRAW API, and analyzing them to discern the commonalities and differences between these two subreddits. The project aimed to better understand the interplay between these two fields, their common topics, and the communities that participate in these conversations.

# Methods

The project first used the PRAW API to scrape 1000 posts from r/statistics and r/machine learning. After cleaning the scraped posts, some simple visualization was done using matplotlib and seaborn on the dataset, before finally tokenizing the contents of each post, and then using KMeans clustering to group the posts. Each cluster was then analyzed to identify distributions of r/statistics vs r/machinelearning posts within each cluster.

### Distinct topics identified between r/statistics and r/machinelearning

| **Cluster** | **Value Counts** | **Predominant Topic** | **Common Words**                                                                                |
|-------------|------------------|-----------------------|-------------------------------------------------------------------------------------------------|
| 4           | 0 - 41<br>1 - 1  | r/statistics          | multiple,two,model,binary,random,linear,dependent,independent,regression,variable               |
| 7           | 0 - 27<br>1 - 1  | r/statistics          | calculate,data,ii,matrix,explain,coefficient,type,someone,error,correlation                     |
| 10          | 0 - 41<br>1 - 3  | r/statistics          | time,mann,experiment,perform,compare,study,ratio,appropriate,statistical,test                   |
| 14          | 0 - 34<br>1 - 1  | r/statistics          | repeated,test,calculate,group,small,hypothesis,population,calculation,size,sample               |
| 17          | 0 - 22<br>1 - 1  | r/statistics          | generalisability,one,two,likelihood,calculating,block,surviving,time,equal,probability          |
| 23          | 0 - 22<br>1 - 0  | r/statistics          | tail,compare,unequal,interpret,nonnormal,skewed,two,mean,normal,distribution                    |
| 28          | 0 - 51<br>1 - 3  | r/statistics          | compare,analysis,poisson,binary,probit,model,coefficient,linear,logistic,regression             |
| 29          | 0 - 45<br>1 - 0  | r/statistics          | probability,person,dissertation,anyone,understanding,problem,study,stats,need,help              |
| 30          | 0 - 39<br>1 - 1  | r/statistics          | conduct,component,statistical,multiple,correspondence,post,costeffectiveness,hoc,power,analysis |
| 2           | 0 - 2<br>1 - 26  | r/machinelearning     | field,train,radiance,suggestion,usage,metaanalysis,copyrighted,convolutional,neural,network     |
| 6           | 0 - 2<br>1 - 67  | r/machinelearning     | microsoft,think,building,multiple,regulation,advice,google,generative,voice,ai                  |
| 8           | 0 - 0<br>1 - 70  | r/machinelearning     | tuning,app,ability,fine,like,source,hallucination,training,finetuning,llm                       |
| 9           | 0 - 1<br>1 - 42  | r/machinelearning     | gan,autoencoder,text,prompt,generation,captioning,classifier,segmentation,model,image           |
| 18          | 0 - 2<br>1 - 33  | r/machinelearning     | think,engineer,problem,project,concept,challenge,amazon,learn,hackathon,ml                      |
| 21          | 0 - 0<br>1 - 22  | r/machinelearning     | llm,microsoft,song,cost,brave,cofounder,research,chatgpt,new,gpt4                               |
| 33          | 0 - 0<br>1 - 27  | r/machinelearning     | dashboard,embeddings,else,source,ai,3d,shape,model,api,openai                                   |
| 35          | 0 - 1<br>1 - 23  | r/machinelearning     | synthesizer,singing,eterministic,texttoimage,generative,survey,latent,stable,model,diffusion    |
| 36          | 0 - 0<br>1 - 53  | r/machinelearning     | finetuning,computer,tuning,state,new,reasoning,instruction,large,model,language                 |

*In the value counts column, posts from r/statistics are labelled 0, and posts from r/machinelearning are labelled 1.

### Similar topics between r/statistics and r/machinelearning

 **Cluster** | **Value Counts**   | **Common Words**                                                                            |
|-------------|--------------------|---------------------------------------------------------------------------------------------|
| 0           | 0 - 399<br>1 - 350   | criterion,paper,youtube,content,dataset,course,textbook,learn,book,recommendation |


# Findings:
## Topic Modelling:

In r/statistics, discussions centered around statistical methods and tests, data analysis techniques, study design, and experiment management. There was also a significant amount of content geared towards statistical education and help.

On the other hand, r/machinelearning exhibited discussions around various Machine Learning Models & Techniques, specific AI technologies and companies, applications and projects, and a strong focus on ongoing research in the field of machine learning.


## Community Detection:

Our community detection revealed that r/statistics is frequented by students and learners, educators and professionals, and practicing researchers. Conversely, r/machinelearning harbors communities of machine learning practitioners, researchers and academics, industry professionals, and students and learners.

## Text Classification:

The analysis concluded that just the post contents (i.e., title, content, and top comments), along with the post's tag (extracted from the title), were sufficient to build an accurate text classifier. Traditional models like Logistic Regression and Support Vector Classifier worked exceptionally well (~98% accuracy) with relatively low computational resources required to train the models.

Overall, our exploration into these digital communities helped cast light on their shared threads and distinctive traits alike. It resulted in not only valuable insights into these fields' online discourse but also contributed a functional tool for the ongoing exploration of data science.


