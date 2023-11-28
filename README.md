# MLOps-guide
A collection of materials from introductory to advanced. This is roughly the path I’d follow if I were to start my MLOps journey again.

## A collection of materials from introductory to advanced. This is roughly the path I’d follow if I were to start my MLOps journey again.

# ML + engineering fundamentals

While it’s tempting to want to get straight to ChatGPT, it’s important to have a good grasp of machine learning, deep learning, NLP, and reinforcement learning fundamentals.

### 10 free ML courses: make sure to take those classes in order.
[Book] Machine Learning: A Probabilistic Perspective (Kevin P. Murphy). A draft PDF link can be found here.
[Book] Information Theory, Inference, and Learning Algorithms (David MacKay). Free online version here.
[Book] Deep Learning (Ian Goodfellow, Yoshua Bengio, and Aaron Courville). Free online version.
[Book] Introduction to Information Retrieval (Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze). Essential for anyone interested in Natural Language Processing. Free online version.
[Book] Reinforcement Learning: An Introduction (Richard S. Sutton and Andrew G. Barto). Essential for reinforcement learning. Free online version.
[Tutorials] OpenAI’s Spinning up in Deep Reinforcement Learning: A collection of articles that give great intuition for many RL algorithms. Highly recommended for anyone interested in RL.
[Video] Andrej Karpathy’s Zero to Hero series
Tools and concepts I’d prioritize learning
A survivor’s guide to AI courses at Stanford (Updated Feb 2020)

MLOps
What’s MLOps?

Ops in MLOps comes from DevOps, short for Developments and Operations. To operationalize something means to bring it into production, which includes deploying, monitoring, and maintaining it.

# Overview

Overview of ML in production.

[Video] Machine learning production myths (Stanford’s MLSys Seminars)
[Lecture note] Introduction to machine learning in production
Rules of Machine Learning: Best Practices for ML Engineering (Martin Zinkevich, 2019)
What I learned from looking at 200 machine learning tools [Jun 2020]
Machine Learning Tools Landscape v2 (+84 new tools) [Dec 2020]
The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction (Breck et al., 2017)
Building LLM applications for production

# Intermediate

Deep dives into different aspects of ML production.

[Lecture note] Creating training data: sampling, labeling, handling class imbalance, data augmentation
[Lecture note] Feature engineering
[Book excerpt] Data Distribution Shifts and Monitoring
Instrumentation, Observability & Monitoring of Machine Learning Models (Josh Wills, 2019)
RLHF: Reinforcement Learning from Human Feedback

# Advanced

Build the best MLOps platform for your organization!

Real-time machine learning: challenges and solutions
[Lecture note] Data system fundamentals for data scientists
A friendly introduction to machine learning compilers and optimizers
Why data scientists shouldn’t need to know Kubernetes
Self-serve feature platforms: architectures and APIs

# Career

[Free book] Machine Learning Interviews Book
[Twitter thread] The ML interviews process
Career advice for recent Computer Science graduates
Four lessons I learned after my first full-time job after college
7 reasons not to join a startup and 1 reason to
Analysis of compensation, level, and experience details of 19k tech workers
What Glassdoor interview reviews reveal about tech hiring cultures
What we look for in a resume

# Case studies
To get a sense of the challenges of machine learning production, it’s helpful to learn from companies who are doing it.

Using Machine Learning to Predict Value of Homes On Airbnb (Robert Chang, Airbnb Engineering & Data Science, 2017)

In this detailed and well-written blog post, Chang described how Airbnb used machine learning to predict an important business metric: the value of homes on Airbnb. It walks you through the entire workflow: feature engineering, model selection, prototyping, moving prototypes to production. It’s completed with lessons learned, tools used, and code snippets too.

Using Machine Learning to Improve Streaming Quality at Netflix (Chaitanya Ekanadham, Netflix Technology Blog, 2018)

As of 2018, Netflix streams to over 117M members worldwide, half of those living outside the US. This blog post describes some of their technical challenges and how they use machine learning to overcome these challenges, including to predict the network quality, detect device anomaly, and allocate resources for predictive caching.

To understand Netflix’s infrastructure for machine learning, check out Ville Tuulos’s talk Human-Centric Machine Learning Infrastructure @Netflix.

150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com (Bernardi et al., KDD, 2019)

As of 2019, Booking.com has around 150 machine learning models in production. These models solve a wide range of prediction problems (e.g. predicting users’ travel preferences and how many people they travel with) and optimization problems (e.g.optimizing the background images and reviews to show for each user). Adrian Colyer gave a good summary of the six lessons learned here:

Machine learned models deliver strong business value.
Model performance is not the same as business performance.
Be clear about the problem you’re trying to solve.
Prediction serving latency matters.
Get early feedback on model quality.
Test the business impact of your models using randomized controlled trials.
Machine Learning-Powered Search Ranking of Airbnb Experiences (Mihajlo Grbovic, Airbnb Engineering & Data Science, 2019)

This article walks you step by step through a canonical example of the ranking and recommendation problem. The four main steps are system design, personalization, online scoring, and business aspect. The article explains which features to use, how to collect data and label it, why they chose Gradient Boosted Decision Tree, which testing metrics to use, what heuristics to take into account while ranking results, how to do A/B testing during deployment. Another wonderful thing about this post is that it also covers personalization to rank results differently for different users.

From shallow to deep learning in fraud (Hao Yi Ong, Lyft Engineering, 2018)

Fraud detection is one of the earliest use cases of machine learning in the industry. This article explores the evolution of fraud detection algorithms used at Lyft. At first, an algorithm as simple as logistic regression with engineered features was enough to catch most fraud cases. Its simplicity allowed the team to understand the importance of different features. Later, when fraud techniques have become too sophisticated, more complex models are required. This article explores the tradeoff between complexity and interpretability, performance and ease of deployment.

Space, Time and Groceries (Jeremy Stanley, Tech at Instacart, 2017)

Instacart uses machine learning to solve the task of path optimization: how to most efficiently assign tasks for multiple shoppers and find the optimal paths for them. The article explains the entire process of system design, from framing the problem, collecting data, algorithm and metric selection, topped with a tutorial for beautiful visualization.

Creating a Modern OCR Pipeline Using Computer Vision and Deep Learning (Brad Neuberg, Dropbox Engineering, 2017)

An application as simple as a document scanner has two distinct components: optical character recognition and word detector. Each requires its own production pipeline, and the end-to-end system requires additional steps for training and tuning. This article also goes into detail the team’s effort to collect data, which includes building their own data annotation platform.

Scaling Machine Learning at Uber with Michelangelo (Jeremy Hermann and Mike Del Balso, Uber Engineering, 2019)

Uber uses extensive machine learning in their production, and this article gives an impressive overview of their end-to-end workflow, where machine learning is being applied at Uber, and how their teams are organized.

How we grew from 0 to 4 million women on our fashion app, with a vertical machine learning approach (Gabriel Aldamiz, HackerNoon, 2018)

To offer automated outfit advice, Chicisimo tried to qualify people’s fashion taste using machine learning. Due to the ambiguous nature of the task, the biggest challenges are framing the problem and collecting the data for it, both challenges are addressed by the article. It also covers the problem that every consumer app struggles with: user retention.

# Bonus
Some stuff I did that don’t quite fit into any section above, but I want to share anyway :P

[Code] Python-is-cool: Cool Python features that I used to be too afraid to use
[Code] just-pandas-things: Pandas quirks that used to traumatize me
[Code] Coding exercises and solutions for coding interviews
[Video] Switching From a Batch to Streaming Mindset w/ Chip Huyen
[VentureBeat] 4 AI and ML job hunting tips from Chip Huyen
[Booklet] Machine learning systems design (2019): My initial notes on ML systems back. This 8000-word booklet gave ideas for the book Designing Machine Learning Systems in 2022.
