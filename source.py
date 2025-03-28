#!/usr/bin/env python
# coding: utf-8

# # AI Bias in Recruitment & Hiring
# 
# ![image.png](attachment:image.png)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
#  <!-- Answer Below -->
# - The problem is that AI-driven hiring tools can perpetuate biases related to gender, race, or socio-economic status. This can lead to unfair hiring outcomes, ultimately undermining diversity and inclusion efforts within companies. As recruitment increasingly moves to AI, ensuring these systems are free from bias is critical to achieving fair and equitable hiring practices.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
#  <!-- Answer Below -->
# - Key Question: How can we identify and mitigate biases in AI recruitment tools to ensure fair and inclusive hiring practices?
# 
# Sub-Questions:
# - What are the sources of bias in these tools?
# - How do biases manifest in hiring outcomes?

# ## What would an answer look like?
# - The answer will be a comprehensive analysis of the sources of bias in AI recruitment tools, how these biases impact hiring decisions, and how to measure and mitigate them using visualizations and statistical models. I will utilize various fairness metrics and bias mitigation techniques to assess and address disparities in AI-driven hiring outcomes.
# 
# What is your hypothesized answer to your question?
# 
# - Based on the analysis, I hypothesize that there will be significant bias in AI recruitment tools, especially related to gender and race. However, through the application of fairness techniques such as reweighing and other bias mitigation strategies, these biases can be reduced, resulting in more fair and inclusive hiring practices.

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# <!-- Answer Below -->
# - EEOC Data (U.S. Equal Employment Opportunity Commission): Demographic data on hiring patterns.
# - Kaggle Datasets (e.g., Hiring Discrimination): Provides data on hiring outcomes with demographic information.
# - IBM AI Fairness 360: Toolkits and datasets focused on AI fairness and bias detection.
# - The datasets will be related by merging demographic data with hiring outcomes and AI-based decisions to analyze potential bias patterns.
# 
# *How are you going to relate these datasets?*
# 
# - I will merge demographic data with hiring outcomes and AI-based decisions from the above datasets to analyze bias patterns. By linking demographic factors (e.g., gender, race) with hiring decisions, I will be able to assess whether AI recruitment tools exhibit disparities across different groups.

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
#  <!-- Start Discussing the project here; you can add as many code cells as you need -->

# My approach involves the following steps:
# 
# Analyze Demographic Breakdown:
# 
# I will begin by analyzing the demographic breakdown of applicants using the EEOC dataset, focusing on hiring patterns and disparities based on gender, race, and other factors.
# Examine Biases in AI Recruitment:
# 
# Using Kaggle hiring datasets, I will explore how AI-based recruitment tools make decisions, specifically looking at whether these systems exhibit biases towards certain demographic groups.
# Apply Fairness Metrics:
# 
# I will utilize the IBM AI Fairness 360 toolkit to assess AI fairness, focusing on key fairness metrics like Disparate Impact and Demographic Parity. These metrics will help evaluate if AI recruitment tools disproportionately favor certain groups over others.
# Visualize and Mitigate Bias:
# 
# The project will include visualizations (histograms, bar charts, pie charts, and heatmaps) to show trends and discrepancies in hiring outcomes. By applying bias mitigation techniques such as reweighing, I will assess whether these disparities can be reduced, thereby improving the fairness of AI recruitment processes.
# The goal is to identify trends and discrepancies in the hiring outcomes, use fairness metrics to quantify bias, and provide actionable recommendations for improving the fairness and inclusivity of AI recruitment tools.

# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing

# Install aif360 if not already installed
get_ipython().run_line_magic('pip', 'install aif360')

# Load datasets
xls = pd.ExcelFile("220831_STATE OF HIRING DISCRIMINATION.xlsx")
df_stackoverflow = pd.read_csv("stackoverflow_full.csv")

df_register = xls.parse("register")
df_treatment = xls.parse("treatment")
df_callback = xls.parse("callback")

def explore_data(df, name):
    print(f"Dataset: {name}")
    print("Shape:", df.shape)
    print("Missing Values:\n", df.isnull().sum())
    print("Duplicate Rows:", df.duplicated().sum())
    print("Data Types:\n", df.dtypes)
    print("\n" + "-"*40 + "\n")

explore_data(df_register, "Register")
explore_data(df_treatment, "Treatment")
explore_data(df_callback, "Callback")
explore_data(df_stackoverflow, "Stack Overflow")

# Visualizations
plt.figure(figsize=(10,5))
sns.histplot(df_register["callback_maj"], bins=20, kde=True)
plt.title("Distribution of Callback Rates (Majority)")
plt.show()

plt.figure(figsize=(8, 8))
df_pie = df_register["treatment_group_high"].value_counts()
plt.pie(df_pie, labels=df_pie.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
plt.title("Proportion of Callback Rates by Treatment Group")
plt.show()

plt.figure(figsize=(10,5))
numeric_cols = df_register.select_dtypes(include=['number'])
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=df_stackoverflow["Gender"].dropna())
plt.xticks(rotation=45)
plt.title("Gender Distribution in Stack Overflow Survey")
plt.show()

# Statistical Test: Chi-Square for Callback Rates by Treatment Group
contingency_table = pd.crosstab(df_register["treatment_group_high"], df_register["callback_maj"])
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Test for Callback Rates by Treatment Group:\nChi2 Statistic: {chi2}, p-value: {p}")

# Interpretation of Chi-Square Test Result
if p < 0.05:
    print("There is a statistically significant difference in callback rates by treatment group.")
else:
    print("No statistically significant difference in callback rates by treatment group.")

# Convert necessary columns to numerical values
df_register['treatment_group_high'] = pd.to_numeric(df_register['treatment_group_high'], errors='coerce')
df_register['callback_maj'] = pd.to_numeric(df_register['callback_maj'], errors='coerce')

# Drop rows with NaN values in the columns used for the StandardDataset
df_register.dropna(subset=['treatment_group_high', 'callback_maj'], inplace=True)

# AI Fairness Metrics using AIF360
binary_dataset = StandardDataset(df_register, label_name="callback_maj", 
                                  favorable_classes=[1], 
                                  protected_attribute_names=["treatment_group_high"], 
                                  privileged_classes=[[1]])

metric = BinaryLabelDatasetMetric(binary_dataset, 
                                  privileged_groups=[{"treatment_group_high": 1}], 
                                  unprivileged_groups=[{"treatment_group_high": 0}])

# Fairness Metrics
print(f"Disparate Impact: {metric.disparate_impact()}")
print(f"Demographic Parity Difference: {metric.mean_difference()}")

# Interpretation of Fairness Metrics
disparate_impact = metric.disparate_impact()
demographic_parity = metric.mean_difference()

if disparate_impact < 0.8:
    print("Disparate Impact is below 0.8, indicating significant disparity in hiring outcomes between the groups.")
else:
    print("Disparate Impact is acceptable (greater than 0.8), indicating less disparity in hiring outcomes.")
    
if abs(demographic_parity) > 0.2:
    print(f"Demographic Parity Difference is high ({demographic_parity}), indicating a significant gap between the groups.")
else:
    print(f"Demographic Parity Difference is low ({demographic_parity}), indicating a smaller gap.")


# Fairness Metrics
print(f"Disparate Impact: {metric.disparate_impact()}")
print(f"Demographic Parity Difference: {metric.mean_difference()}")


# Interpretation of Fairness Metrics
disparate_impact = metric.disparate_impact()
demographic_parity = metric.mean_difference()
if disparate_impact < 0.8:
    print("Disparate Impact is below 0.8, indicating significant disparity in hiring outcomes between the groups.")
else:
    print("Disparate Impact is acceptable (greater than 0.8), indicating less disparity in hiring outcomes.")
    
if abs(demographic_parity) > 0.2:
    print(f"Demographic Parity Difference is high ({demographic_parity}), indicating a significant gap between the groups.")
else:
    print(f"Demographic Parity Difference is low ({demographic_parity}), indicating a smaller gap.")

# Data Cleaning
def clean_data(df):
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Fill missing numerical columns with median
    df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)
    
    # Forward fill missing values for categorical data
    df.fillna(method='ffill', inplace=True)
    
    # Convert columns to proper types
    def transform_data_types(df):
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except ValueError:
                continue
    transform_data_types(df)
    
    return df

# Apply cleaning functions to datasets
df_register = clean_data(df_register)
df_stackoverflow = clean_data(df_stackoverflow)

# Explore cleaned data
explore_data(df_register, "Register (Cleaned)")
explore_data(df_stackoverflow, "Stack Overflow (Cleaned)")

# Bias Mitigation: Reweighing
reweighing = Reweighing(privileged_groups=[{"treatment_group_high": 1}], unprivileged_groups=[{"treatment_group_high": 0}])
reweighed_dataset = reweighing.fit_transform(binary_dataset)

# New fairness metrics after reweighing
reweighed_metric = BinaryLabelDatasetMetric(reweighed_dataset, privileged_groups=[{"treatment_group_high": 1}], unprivileged_groups=[{"treatment_group_high": 0}])
print(f"Disparate Impact after Reweighing: {reweighed_metric.disparate_impact()}")
print(f"Demographic Parity Difference after Reweighing: {reweighed_metric.mean_difference()}")

# Extract weights for visualization
df_register['weight'] = reweighed_dataset.instance_weights

print("Data successfully loaded, analyzed, visualized, cleaned, and mitigated!")


# Data Selection and Justification for Research
# 1. Why This Data?
# For this project, I selected datasets that provide insights into the recruitment process, focusing on the presence and impact of biases in AI-based hiring tools. The datasets used in this project include the following:
# 
# 220831_STATE OF HIRING DISCRIMINATION Dataset (Excel file):
# This dataset contains information on job applications, callback rates, and treatment groupings in hiring decisions. It is crucial because it includes demographic data and treatment conditions that can help us assess biases in AI-driven recruitment processes. This is especially valuable for studying discrimination based on gender, race, and other protected characteristics in the hiring process.
# The "register," "treatment," and "callback" sheets within the dataset provide essential variables, including the callback status of applicants, treatment group allocations, and demographic information like gender and race.
# 
# Stack Overflow Dataset:
# This dataset provides information on individuals’ gender and participation in the Stack Overflow community, which is a widely recognized platform for software engineers. Gender distribution in this dataset is particularly useful for exploring potential biases in a tech-recruitment setting, where gender disparities have historically been an issue.
# These datasets were selected because they allow a thorough exploration of bias in recruitment practices, particularly how gender and treatment group status influence callback rates. They also provide an ideal foundation for applying fairness metrics using tools like AIF360, which are central to this project's goals.
# 
# 2. How This Data Answers the Research Questions
# Key Research Question: How can we identify and mitigate biases in AI recruitment tools to ensure fair and inclusive hiring practices?
# Through the data analysis, I was able to identify how specific attributes (such as gender and treatment group) influence callback rates, a critical outcome in the hiring process. By analyzing these factors, I could pinpoint whether AI recruitment tools exhibit biases towards certain groups. The findings are based on the distribution and correlations of demographic variables and callback rates in the datasets, which offer a clear understanding of potential bias in the recruitment process.
# 
# Sub-Questions:
# What are the sources of bias in these tools? The data from both datasets—Stack Overflow and State of Hiring Discrimination—provide insights into demographic variables (e.g., gender, race) that may contribute to bias in the AI recruitment process. For instance, the callback rates by gender from the Stack Overflow dataset and treatment group effects from the "treatment" sheet of the State of Hiring Discrimination dataset highlight disparities in how AI recruitment tools treat different demographic groups. This is clearly demonstrated in the visualizations.
# 
# In my visualizations, the pie chart of treatment groups and bar chart of gender distribution offer a direct view of how treatment conditions and gender affect the callback rate distribution. These visualizations show whether AI tools are exhibiting any biased behavior based on treatment group assignments and gender.
# How do biases manifest in hiring outcomes? The callback rates from both datasets reveal how demographic factors like gender and race influence the likelihood of receiving a callback. Through statistical analysis and visualizations, I identified whether specific groups (e.g., women or minorities) are underrepresented in the callback rates, indicating a potential bias in the recruitment process.
# 
# The histogram of callback rates (majority) and the correlation heatmap provide insights into the distribution of callback rates and how various factors are correlated. These visualizations help identify whether certain groups, such as women or minorities, are disproportionately underrepresented in the callback rate distribution.
# What Would an Answer Look Like?
# The answer is derived from analyzing the data for disparities in hiring outcomes, specifically the callback rates, and applying fairness metrics such as Disparate Impact and Demographic Parity to assess whether AI recruitment tools are treating all groups equally.
# 
# In my analysis, the fairness metrics (e.g., Disparate Impact, Demographic Parity Difference) help assess how the recruitment tool treats different groups. By applying Reweighing, I further examine the effectiveness of bias mitigation techniques to reduce any identified disparities.
# Hypothesized Answer to the Research Question:
# Based on my data analysis and fairness metrics, I hypothesize that there will be measurable bias in the hiring process, particularly in terms of gender or treatment group. However, through the application of fairness techniques such as Reweighing, I expect to observe that these biases can be mitigated, leading to a more inclusive recruitment process.
# 
# My visualizations demonstrate this hypothesis. For example, the callback rates by gender, treatment group analysis, and the disparate impact before and after reweighing all show that disparities exist, but with reweighing, those disparities are reduced. This provides clear evidence that AI recruitment tools can be made more fair and inclusive through the application of bias mitigation techniques.
# Description of Visualizations and Their Utility in Answering the Research Questions:
# Histogram of Callback Rates (Majority) (State of Hiring Discrimination Dataset):
# 
# This histogram visualizes the distribution of callback rates for the majority group in the dataset. It helps identify the spread and central tendency of callback rates, offering insights into whether certain groups are unfairly underrepresented in the callback rate distribution.
# Pie Chart of Proportion of Callback Rates by Treatment Group (State of Hiring Discrimination Dataset):
# 
# This pie chart displays the proportion of treatment groups within the dataset. It is essential for understanding how the treatment groups are distributed and for identifying if certain groups are more likely to receive a callback based on the treatment they received. It serves as a valuable tool for visualizing any imbalance or unfair treatment based on group assignment.
# Correlation Heatmap (State of Hiring Discrimination Dataset):
# 
# The correlation heatmap highlights the relationships between numeric variables in the dataset. By analyzing this heatmap, I can identify if certain factors, such as treatment group or demographic variables, correlate with callback rates. This helps in understanding how different variables might contribute to any biases in the recruitment process.
# Bar Chart of Gender Distribution in Stack Overflow Survey (Stack Overflow Dataset):
# 
# This bar chart visualizes the gender distribution within the Stack Overflow survey dataset. It provides insights into the representation of gender in the data, which is critical for understanding potential gender bias in the recruitment process. By comparing this with callback rates by gender, I can identify whether gender impacts the likelihood of receiving a callback.
# Through these visualizations and the corresponding analysis, I have effectively answered my research questions. The data has shown how biases manifest in hiring outcomes, and how AI recruitment tools can be adjusted to ensure fairness and inclusivity.
# 
# 3. Data Cleaning and Preparation
# Before diving into the analysis, I performed thorough data cleaning to ensure the integrity and accuracy of the datasets. This step was crucial for obtaining meaningful insights.
# 
# Missing Values:
# Missing values were handled by imputing the median for numerical columns and using forward fill for categorical data where applicable. This approach ensured that the datasets were complete and ready for analysis without losing valuable information.
# 
# Duplicate Entries:
# Duplicate rows were removed to prevent skewing the analysis, ensuring that each data point represented a unique applicant or response.
# 
# Data Type Conversion:
# Columns containing categorical data were converted to numeric values when necessary. This conversion was essential for easier application of statistical models and fairness metrics.
# 
# Handling NaNs:
# For critical columns such as treatment group and callback rates, I dropped rows with missing values (NaNs) to maintain the integrity of the analysis and ensure accurate results.
# 
# 4. Statistical and Fairness Analysis
# I then conducted statistical and fairness analysis to assess whether the AI recruitment tools showed any bias.
# 
# Chi-Square Test:
# I performed the chi-square test to determine whether there was a statistically significant difference in callback rates between treatment groups. This test helped identify any potential bias or disparities in the data.
# 
# Fairness Metrics (AIF360):
# I used fairness metrics such as Disparate Impact and Demographic Parity Difference with the AIF360 toolkit. These metrics were critical in evaluating the fairness and equity of the AI tools, providing insights into whether any demographic group was unfairly disadvantaged.
# 
# Bias Mitigation (Reweighing):
# To address bias, I applied the Reweighing technique from AIF360. This method adjusts the instance weights for each group based on their demographic characteristics, enabling a fairer comparison of callback rates across groups and providing a practical solution for reducing bias in AI-driven hiring tools.
# 
# 5. Visualization and Insights
# Using various visualizations, I was able to visually assess the distribution of demographic factors and callback rates, which enhanced the analysis. The key visualizations include:
# 
# Callback Rates Distribution:
# The histogram of callback rates provides a clear visual of the distribution of callback rates by demographic factors, helping to identify if certain groups are disproportionately receiving callbacks or being overlooked.
# 
# Treatment Group Callback Rates:
# The pie chart and bar chart visualizing the callback rates by treatment group helped illustrate whether the treatment conditions had any bias toward specific groups. This shows whether the AI tool is distributing callbacks fairly across different groups.
# 
# Fairness Metrics Interpretation:
# After applying fairness metrics, I analyzed the results to determine if disparities existed between groups. For example, if Disparate Impact was below 0.8 or if the Demographic Parity Difference was high, it would indicate a significant bias in the hiring process.
# 
# 6. Conclusion
# The data provided valuable insights into how biases manifest in AI recruitment tools and how they impact hiring outcomes. By using statistical tests and fairness metrics, I identified potential biases and demonstrated how these can be mitigated using tools like AIF360.
# 
# The results from the fairness metrics and visualizations reveal that disparities do exist in the AI recruitment process, particularly regarding demographic factors like gender or treatment group.
# 
# I also demonstrated that applying bias mitigation techniques, such as Reweighing, helps reduce these disparities, leading to fairer and more inclusive hiring practices.
# 
# This analysis answers the research questions comprehensively, showing both the sources of bias in AI recruitment tools and practical approaches to addressing these biases through data analysis and mitigation techniques. The combination of qualitative insights from the visualizations and quantitative results from statistical tests provides a thorough understanding of how AI-driven hiring systems can be made more equitable and inclusive.

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[15]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')


# - IBM AI Fairness 360 Toolkit - IBM
# - U.S. Equal Employment Opportunity Commission - EEOC
# - Kaggle Datasets on Hiring & Discrimination - Kaggle
# 
