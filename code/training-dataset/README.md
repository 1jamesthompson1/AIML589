# Building the Value Alignment Training Dataset

This phase is about building a training dataset. This is done by

- Wrangling the WVS data to be in a format that can be used to generate the training dataset
- Creating clusters of respondents
- Creating templates for the question
  - Creating a mapping from the question response to the actual question asked (i.e in the right format).
  - Create system prompt templates for the different question styles
- Generating dataset by merging templates with the clusters responses in one of three ways
  - For each template question response simply use the most common response from the cluster
  - For each template question response sample from the cluster responses
  - For each template question response the answer is the response distribution of the cluster.

## Output



## Input data

The input data you need to have is the World Values Survey Wave 7 data for New Zealand. This is a CSV file that can be downloaded from the [World Values Survey website](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp). The file should be named `WVS_Wave_7_New_Zealand_Csv_v5.1.csv` and placed in the `code/training-dataset/input` directory.