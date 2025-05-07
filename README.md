# COM6012 Assignment Report

**Student ID:** [Your Student ID]  
**Username:** acp24aks  

This report details the results and analysis for the COM6012 assignment, focusing on big data analysis using PySpark. The assignment comprises four questions: log mining, predictive modeling, scalable supervised learning, and recommender systems. All tasks were executed on the Stanage HPC cluster with PySpark 3.5.4 and Python 3.12.

---

## Question 1: Log Mining and Analysis

### Overview
This question analyzed the NASA access log for July 1995 to extract insights about requests from academic institutions in the USA (`.edu`), UK (`.ac.uk`), and Australia (`.edu.au`).

### Task A: Counting Requests from Academic Institutions
- **Objective**: Determine the total number of requests from academic domains in the USA, UK, and Australia.
- **Methodology**: Parsed the log to filter requests by domain and counted occurrences.
- **Results**:
  - **USA**: 218,449 requests
  - **UK**: 25,009 requests
  - **Australia**: 7,004 requests
- **Visualization**: A bar chart illustrates the request counts across the three countries.
  - **Image Placement**: Insert `Q1_figA.jpg` here, after the results.

### Task B: Identifying Unique Institutions and Frequent Hosts
- **Objective**: Count unique academic institutions, list the top 9 most frequent hosts per country, and rank the University of Sheffield among UK institutions.
- **Methodology**: Extracted unique hostnames, sorted by request frequency, and ranked accordingly.
- **Results**:
  - **USA**:
    - Unique institutions: 11,571
    - Top 9 hosts: Refer to `Q1_output.txt` for the list.
  - **UK**:
    - Unique institutions: 1,022
    - Top 9 hosts: Refer to `Q1_output.txt` for the list.
    - University of Sheffield: Ranked 2nd with 623 requests.
  - **Australia**:
    - Unique institutions: 365
    - Top 9 hosts: Refer to `Q1_output.txt` for the list.

### Task C: Visualizing Request Distribution
- **Objective**: Create pie charts showing the distribution of requests for the top 9 hosts and an "Others" category for each country.
- **Methodology**: Aggregated request counts and visualized the top 9 hosts plus a combined "Others" category.
- **Results**:
  - Pie charts generated for USA, UK, and Australia.
  - **Image Placement**:
    - USA: Insert `Q1_figC_USA.jpg` here, after USA results.
    - UK: Insert `Q1_figC_UK.jpg` here, after UK results.
    - Australia: Insert `Q1_figC_AU.jpg` here, after AU results.

### Task D: Heatmap Analysis of the Most Frequent Institution
- **Objective**: Generate heatmaps showing access patterns (day vs. hour) for the most frequent institution in each country.
- **Methodology**: Extracted timestamp data for the top host per country and plotted access frequency.
- **Results**:
  - Heatmaps created for the top institution in each country.
  - **Image Placement**:
    - USA: Insert `Q1_figD_USA.jpg` here, after USA heatmap description.
    - UK: Insert `Q1_figD_UK.jpg` here, after UK heatmap description.
    - Australia: Insert `Q1_figD_AU.jpg` here, after AU heatmap description.

### Task E: Observations and Insights
- **Observations**:
  - The USA led with 218,449 requests, reflecting its larger academic network.
  - The University of Sheffield’s 2nd ranking (623 requests) highlights its prominence in the UK.
  - Heatmaps showed peak access times, varying by country (e.g., daytime spikes).
- **Insights**: These patterns could guide NASA in optimizing server resources and scheduling maintenance.

---

## Question 2: Predictive Modeling on Diabetes Data

### Overview
This question involved preprocessing the Diabetes dataset and building regression and classification models to predict hospital stay duration and readmission.

### Preprocessing
- **Dataset**: Loaded locally due to KaggleHub access issues.
- **Transformations**:
  - One-hot encoded 24 medication features.
  - Converted `readmitted` to binary (1 for >30/<30, 0 for NO).
  - Selected `time_in_hospital` as the numeric target.
  - Split data 80/20, stratified on `readmitted`, with seed 67594.

### Model Training and Evaluation
- **Poisson Regression**:
  - **Target**: `time_in_hospital`
  - **Best Parameter**: `regParam=1.0`
  - **Test RMSE**: 2.950454
  - **Visualization**: Validation curve showing RMSE vs. `regParam`.
    - **Image Placement**: Insert `Q2_fig_poisson_validation.jpg` here, after results.
- **Logistic Regression (L1)**:
  - **Target**: `readmitted`
  - **Best Parameters**: `regParam=0.001`, `elasticNetParam=1.0`
  - **Test Accuracy**: 0.551642
  - **Visualization**: Validation curve showing accuracy vs. parameters.
    - **Image Placement**: Insert `Q2_fig_logistic_l1_validation.jpg` here, after results.
- **Logistic Regression (L2)**:
  - **Target**: `readmitted`
  - **Best Parameter**: `regParam=0.01`
  - **Test Accuracy**: 0.550055
  - **Visualization**: Validation curve showing accuracy vs. `regParam`.
    - **Image Placement**: Insert `Q2_fig_logistic_l2_validation.jpg` here, after results.

---

## Question 3: Scalable Supervised Learning

### Overview
This question classified XOR Arbiter PUFs using scalable machine learning models on varying data fractions.

### Part A: Model Tuning on 1% Sample
- **Models**: Random Forest (RF), Gradient Boosting (GBT), Neural Network (MLP).
- **Tuning Parameters**:
  - RF: `numTrees`, `maxDepth`, `featureSubsetStrategy`
  - GBT: `maxDepth`, `maxIter`, `stepSize`
  - MLP: `maxIter`, `blockSize`, `stepSize`
- **Best Parameters**:
  - RF: `numTrees=10`, `maxDepth=10`, `featureSubsetStrategy='log2'`
  - GBT: `maxDepth=3`, `maxIter=10`, `stepSize=0.01`
  - MLP: `maxIter=200`, `blockSize=64`, `stepSize=0.01`

### Part B: Scaling to Larger Portions
- **Fractions Tested**: 5%, 10%, 20%, 40%, 80%, 100%
- **Results**:
  - All models trained within 30 minutes across all fractions.
  - Performance metrics (accuracy, AUC, runtime) plotted.
  - **Image Placement**:
    - Insert `Q3_accuracy.png` here, after accuracy results.
    - Insert `Q3_auc.png` here, after AUC results.
    - Insert `Q3_runtime.png` here, after runtime results.
- **Maximum Fraction**: All models scaled to 100% of the data successfully.

---

## Question 4: Recommender Systems at Scale

### Overview
This question built a movie recommendation system using ALS and clustered movies based on learned factors.

### Task A: Building the Recommender with ALS
- **Methodology**: 4-fold cross-validation with three ALS settings.
- **Settings**:
  - Setting 1: `rank=10`, `regParam=0.1` (default)
  - Setting 2: `rank=20`, `regParam=0.1`
  - Setting 3: `rank=10`, `regParam=0.5`
- **Results**:
  - Mean RMSE and MAE reported in `Q4_als_results.csv`.
  - Performance plot comparing settings.
    - **Image Placement**: Insert `Q4_als_performance.png` here, after results.

### Task B: Discovering Movie Groups with K-Means
- **Methodology**: Applied K-Means (k=19) to item factors from ALS Setting 1.
- **Results**:
  - Top tags identified for the top three clusters per fold.
  - Tag frequencies in `Q4_tag_frequencies.txt`.
  - Detailed data in `Q4_tag_table.csv`.

### Task C: Insights
- **Observation 1**: Setting 3 underperformed due to a higher `regParam` (0.5), increasing regularization penalties.
- **Observation 2**: Settings 1 and 2 showed similar performance, suggesting robustness of the default rank.

---

## Conclusion
The assignment demonstrated proficiency in PySpark for big data tasks, successfully addressing log mining, predictive modeling, scalable learning, and recommender systems. All required outputs and visualizations were generated as specified.

---

### Image Placement Summary
- **Question 1**:
  - Task A: `Q1_figA.jpg` after request counts.
  - Task C: `Q1_figC_USA.jpg`, `Q1_figC_UK.jpg`, `Q1_figC_AU.jpg` after respective country results.
  - Task D: `Q1_figD_USA.jpg`, `Q1_figD_UK.jpg`, `Q1_figD_AU.jpg` after heatmap descriptions.
- **Question 2**:
  - `Q2_fig_poisson_validation.jpg`, `Q2_fig_logistic_l1_validation.jpg`, `Q2_fig_logistic_l2_validation.jpg` after respective model results.
- **Question 3**:
  - `Q3_accuracy.png`, `Q3_auc.png`, `Q3_runtime.png` after scaling results.
- **Question 4**:
  - `Q4_als_performance.png` after ALS results.
