[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/uFG8FNyr)
# Final-Project-Template
 AI Bias in Recruitment & Hiring 
 ![alt text](image.png)

## Project Overview
This project investigates biases in AI-powered recruitment tools and their impact on hiring decisions. It specifically focuses on analyzing how these biases manifest based on factors such as gender, race, and treatment group (socio-economic status). The project utilizes datasets from EEOC, Kaggle, and IBM AI Fairness 360 to explore these biases and the potential strategies for mitigating them to ensure fair and inclusive hiring processes.

How can we identify and mitigate biases in AI recruitment tools to ensure fair and inclusive hiring practices?

By analyzing key demographic variables (e.g., gender, race, and treatment group) and their impact on callback rates, the project highlights potential sources of bias and demonstrates how these biases affect hiring outcomes. Fairness metrics and bias mitigation strategies are also applied to assess and reduce disparities.

## Self Assessment and Reflection

<!-- Edit the following section with your self assessment and reflection -->

### Self Assessment
<!-- Replace the (...) with your score -->

| Category          | Score    |
| ----------------- | -------- |
| **Setup**         | 10 / 10 |
| **Execution**     | 20 / 20 |
| **Documentation** | 10 / 10 |
| **Presentation**  | 30 / 30 |
| **Total**         | 70 / 70 |

### Reflection
<!-- Edit the following section with your reflection .-->

#### What went well?
Successfully identified and gathered relevant datasets that allowed for in-depth analysis of AI bias in hiring.

Created a diverse set of visualizations (histograms, bar charts, pie charts, and heatmaps) that highlight trends and disparities in the data.

Effectively applied fairness metrics such as Disparate Impact and Demographic Parity using IBM AI Fairness 360, providing key insights into the fairness of AI recruitment tools.

Produced a well-structured project report and presentation that communicated the findings clearly.

#### What did not go well?
Faced challenges merging datasets due to differences in format and missing identifiers. Some datasets lacked sufficient 

demographic details, which limited the depth of analysis.

The configuration of certain fairness toolkits required advanced setup, which led to initial delays.

#### What did you learn?
AI-powered hiring tools can unintentionally reinforce biases if the training data is not representative or diverse.

Data preprocessing (e.g., handling missing values, data type conversions) is crucial for accurate bias detection and fairness evaluation.

Ethical AI development requires transparency, inclusion of diverse datasets, and proper regulation to ensure fairness.

Visual storytelling through data visualizations significantly enhances the impact of the project by making complex findings more 
accessible and understandable.

#### What would you do differently next time?
Select datasets with clearer linking variables to streamline data merging processes.

Explore additional AI fairness tools to compare results across different frameworks and assess their effectiveness.

Consider incorporating qualitative research (e.g., case studies, expert interviews) to supplement the quantitative analysis.
Optimize the data processing workflows to increase efficiency and scalability.

---

## Getting Started
### Installing Dependencies

To ensure that you have all the dependencies installed, and that we can have a reproducible environment, we will be using `pipenv` to manage our dependencies. `pipenv` is a tool that allows us to create a virtual environment for our project, and install all the dependencies we need for our project. This ensures that we can have a reproducible environment, and that we can all run the same code.

```bash
pipenv install
```

This sets up a virtual environment for our project, and installs the following dependencies:

- `ipykernel`
- `jupyter`
- `notebook`
- `black`
  Throughout your analysis and development, you will need to install additional packages. You can can install any package you need using `pipenv install <package-name>`. For example, if you need to install `numpy`, you can do so by running:

```bash
pipenv install numpy
```

This will update update the `Pipfile` and `Pipfile.lock` files, and install the package in your virtual environment.

## Helpful Resources:
* [Markdown Syntax Cheatsheet](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
* [Dataset options](https://it4063c.github.io/guides/datasets)
