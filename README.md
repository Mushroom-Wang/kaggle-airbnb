<!-- Add banner here -->

# Classification on Airbnb Availability Data

<!-- Add buttons here -->

<!-- Describe your project in brief -->
## Purpose
This project aims to predict whether an Airbnb is avaliable based on a bunch of features.

## Data
The data comes from the [Inside Airbnb data platform](http://insideairbnb.com/get-the-data.html), specifically the Airbnb listings in the region of Asheville, NC. It is cleaned up for this in-class competition. The attributes and the labels are created by some heuristic. 

## Pipeline
First, an exploratory analysis is presented, which contains how I preprocess data and search for the correlation between target variable and features. I also present results on the feature importance and feature selection. Second, I introduce the evaluated models and the motivations of selecting them. One novel part I want to highlight is that I employ a powerful hyperparameter auto-tuner and provide a reusable and easy-to-config implementation. Finally, I report the prediction results.

<!-- The project title should be self explanotory and try not to make it a mouthful. (Although exceptions exist- **awesome-readme-writing-guide-for-open-source-projects** - would have been a cool name)

Add a cover/banner image for your README. **Why?** Because it easily **grabs people's attention** and it **looks cool**(*duh!obviously!*).

The best dimensions for the banner is **1280x650px**. You could also use this for social preview of your repo.

I personally use [**Canva**](https://www.canva.com/) for creating the banner images. All the basic stuff is **free**(*you won't need the pro version in most cases*).

There are endless badges that you could use in your projects. And they do depend on the project. Some of the ones that I commonly use in every projects are given below. 

I use [**Shields IO**](https://shields.io/) for making badges. It is a simple and easy to use tool that you can use for almost all your badge cravings. -->

# Table of contents

<!-- After you have introduced your project, it is a good idea to add a **Table of contents** or **TOC** as **cool** people say it. This would make it easier for people to navigate through your README and find exactly what they are looking for.

Here is a sample TOC(*wow! such cool!*) that is actually the TOC for this README. -->

- [Classification on Airbnb Availability Data](#classification-on-airbnb-availability-data)
- [Table of contents](#table-of-contents)
- [Project Report](#project-report)
- [Installation](#installation)
- [Development](#development)

# Project Report

<!-- Add a demo for your project -->

[Check out my project report](https://drive.google.com/file/d/1BsqSBR9dUWUQgRvWJnMZL9g3-7xIIWnL/view?usp=sharing)

<!-- After you have written about your project, it is a good idea to have a demo/preview(**video/gif/screenshots** are good options) of your project so that people can know what to expect in your project. You could also add the demo in the previous section with the product description.

Here is a random GIF as a placeholder.

![Random GIF](https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif) -->

# Installation
[(Back to top)](#table-of-contents)

<!-- *You might have noticed the **Back to top** button(if not, please notice, it's right there!). This is a good idea because it makes your README **easy to navigate.*** 

The first one should be how to install(how to generally use your project or set-up for editing in their machine).

This should give the users a concrete idea with instructions on how they can use your project repo with all the steps.

Following this steps, **they should be able to run this in their device.**

A method I use is after completing the README, I go through the instructions from scratch and check if it is working. -->

To use this project, first clone the repo on your device using the command below:

```git init```

```git clone https://github.com/Mushroom-Wang/kaggle-airbnb.git```

# Development
[(Back to top)](#table-of-contents)

## Exploratory Analysis
I draw the scatter plots of the target variable vs. each exploratory variable.
![Figure 1: Correlation Matrix](https://raw.githubusercontent.com/Mushroom-Wang/kaggle-airbnb/master/images/correlation-matrix.png)
## Feature Selection
First, I use the Random Forest algorithm to choose the features. Second, I compare the accuracy of the training data when changing the number of features (from 2 to 20) to select. According to Figure 1, choosing more features yields a higher training accuracy. Since selecting 14 to 19 features yields similar classification accuracy, further analysis needed to determine which features are more important.
![Figure 2: Box Plot of RFE Number of Selected Features vs. Classification Accuracy](https://raw.githubusercontent.com/Mushroom-Wang/kaggle-airbnb/master/images/feature-selection.png)
## Feature Importance
The feature importance provides some guidelines for me to drop several features when run the model later.
![Figure 3: Bar Chart of DecisionTreeRegressor Feature Importance Scores](https://raw.githubusercontent.com/Mushroom-Wang/kaggle-airbnb/master/images/feature-importance.png)
# Footer
[(Back to top)](#table-of-contents)

<!-- Let's also add a footer because I love footers and also you **can** use this to convey important info.

Let's make it an image because by now you have realised that multimedia in images == cool(*please notice the subtle programming joke). -->

Leave a star in GitHub if you found this helpful.

<!-- Add the footer here -->

<!-- ![Footer](https://github.com/navendu-pottekkat/awesome-readme/blob/master/fooooooter.png) -->
