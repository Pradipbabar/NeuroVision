### **What is Statistics?**

**Statistics** is a branch of mathematics that deals with collecting, analyzing, interpreting, presenting, and organizing data. It provides methods for making sense of large amounts of data, understanding trends, patterns, and relationships, and drawing conclusions or making predictions from data.

### **What is Data?**

**Data** refers to individual pieces of factual information recorded and used for analysis. It can come in various forms, including numbers, words, measurements, observations, or descriptions. Data is often classified into two types:
- **Qualitative (Categorical) Data**: Non-numeric data that describes qualities or characteristics (e.g., colors, names).
- **Quantitative (Numerical) Data**: Numeric data that can be measured and counted (e.g., height, weight).

### **Descriptive Statistics**

**Descriptive statistics** are methods used to summarize and describe the characteristics of a dataset. They help in simplifying large amounts of data into more manageable forms. Common descriptive statistics include:
- **Measures of Central Tendency**: These describe the center or average of a dataset.
  - **Mean**: The average value.
  - **Median**: The middle value when the data is ordered.
  - **Mode**: The most frequent value.
  
- **Measures of Dispersion**: These describe the spread or variability of the data.
  - **Range**: The difference between the highest and lowest values.
  - **Variance**: The average of the squared differences from the mean.
  - **Standard Deviation**: The square root of the variance; it measures how spread out the numbers are.

- **Shape of Distribution**:
  - **Skewness**: Describes the asymmetry of the data distribution.
  - **Kurtosis**: Describes the "tailedness" or peaks of the distribution.

### **Inferential Statistics**

**Inferential statistics** allow you to make inferences or generalizations about a population based on a sample of data. The key goal is to draw conclusions about a population without examining every member of that population. This often involves hypothesis testing, confidence intervals, and regression analysis.

- **Hypothesis Testing**: Involves testing an assumption (hypothesis) about a population.
- **Confidence Intervals**: Provide a range of values within which a population parameter is likely to fall.
- **Regression Analysis**: Examines the relationship between two or more variables.

### **Histogram**

A **histogram** is a graphical representation of the distribution of numerical data. It groups data into bins (or intervals) and plots the frequency of data points falling within each bin. It's useful for visualizing the shape of the data distribution, whether it is skewed, symmetric, or has multiple peaks.

- **Example**: If you survey 100 students about their test scores and plot the frequency of scores in bins (e.g., 0-10, 11-20, etc.), a histogram helps you see how scores are distributed.

### **Box Plot (Box-and-Whisker Plot)**

A **box plot** visually shows the distribution of data based on a five-number summary:
- **Minimum**: The smallest data point.
- **First Quartile (Q1)**: The median of the lower half of the dataset.
- **Median (Q2)**: The middle data point.
- **Third Quartile (Q3)**: The median of the upper half of the dataset.
- **Maximum**: The largest data point.

Box plots also show outliers and the spread of the data, which helps identify how much the data varies.

- **Example**: A box plot of heights of students in a class can help identify the central range of heights and any unusually tall or short students (outliers).

### **Bar Chart**

A **bar chart** is a graphical representation of categorical data using bars. The length of each bar corresponds to the frequency or count of items in each category. It’s useful for comparing different groups or categories.

- **Example**: A bar chart showing the number of students in different school grades can help compare the sizes of the grades visually.

### **What is Population?**

In statistics, a **population** refers to the entire group that you want to study or draw conclusions about. It can consist of people, objects, events, or measurements.
- **Example**: If you are studying the heights of adults in a city, the population would be all adults in that city.

### **What is a Variable?**

A **variable** is any characteristic, number, or quantity that can be measured or counted. Variables can be classified as:
- **Independent Variable**: The variable you manipulate or change to see its effect.
- **Dependent Variable**: The variable that changes as a result of the manipulation of the independent variable.

Variables are also categorized as:
- **Quantitative (Numeric)**: Variables that can be measured and have numerical values.
- **Qualitative (Categorical)**: Variables that describe categories or groups.

### **Frequency Distribution**

A **frequency distribution** is a summary of how often each value occurs in a dataset. It can be represented in a table or graphically, such as a histogram.

- **Example**: In a class of 30 students, if 10 students score between 50-60, 8 students score between 60-70, etc., you can create a frequency distribution table or histogram showing these ranges and their corresponding frequencies.


#### **Descriptive Statistics Example**:
Suppose we have exam scores for 10 students: 50, 60, 70, 80, 90, 60, 70, 80, 90, 100.
- **Mean**: (50 + 60 + 70 + 80 + 90 + 60 + 70 + 80 + 90 + 100) / 10 = 75.
- **Median**: Arrange data: 50, 60, 60, 70, 70, 80, 80, 90, 90, 100. Median = (70 + 80) / 2 = 75.
- **Mode**: 60, 70, 80, and 90 occur twice, so we have multiple modes.

#### **Inferential Statistics Example**:
If you want to know the average height of all people in a country, you can’t measure everyone. Instead, you take a sample of 1,000 people, measure their heights, and then use **inferential statistics** to estimate the average height of the entire population.

#### **Frequency Distribution Example**:
For exam scores (50-59: 2 students, 60-69: 3 students, 70-79: 2 students, 80-89: 2 students, 90-100: 1 student), you can construct a frequency table or histogram to visualize how scores are distributed across the ranges.

### **Measure of Dispersion**

The **measure of dispersion** refers to the extent to which data points in a dataset vary or spread out from the center (mean or median). Dispersion provides insights into the variability or consistency of the data. The most common measures of dispersion are:

- **Range**: The difference between the maximum and minimum values in a dataset.

  
- **Variance**: The average of the squared differences between each data point and the mean. It quantifies the overall spread of the data.

  
- **Standard Deviation**: The square root of the variance, giving a measure of the average distance from the mean in the same units as the data.


- **Interquartile Range (IQR)**: The difference between the third quartile (Q3) and the first quartile (Q1). It measures the spread of the middle 50% of the data.


---

### **Percentile and Quartiles**

**Percentiles** and **quartiles** are values that divide a dataset into equal parts. They are used to understand the relative standing of a data point in a dataset.

- **Percentile**: The p-th percentile is the value below which p% of the data fall. For example, the 90th percentile means 90% of the data points are below this value.
  
- **Quartiles**: These are special percentiles that divide the data into four equal parts:
  - **Q1 (First Quartile)**: The 25th percentile (25% of data below it).
  - **Q2 (Second Quartile)**: The median or the 50th percentile (50% of data below it).
  - **Q3 (Third Quartile)**: The 75th percentile (75% of data below it).

#### Example in Python:
To calculate quartiles and percentiles using Python:

```python
import numpy as np

data = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# Calculate Percentiles
percentile_90 = np.percentile(data, 90)
print(f"90th Percentile: {percentile_90}")

# Calculate Quartiles
q1 = np.percentile(data, 25)
q2 = np.percentile(data, 50)
q3 = np.percentile(data, 75)
print(f"Q1: {q1}, Q2 (Median): {q2}, Q3: {q3}")
```

---

### **Finding Outliers Using Python**

**Outliers** are data points that lie far outside the range of the majority of the data. They can be identified using:
- **Z-score**: Measures how many standard deviations a data point is from the mean. Values with a Z-score > 3 or < -3 are considered outliers.
  
- **IQR Rule**: Data points falling below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR are outliers.

#### Example:
```python
import numpy as np

# Calculate IQR
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1

# Calculate Outlier Range
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = [x for x in data if x < lower_bound or x > upper_bound]
print(f"Outliers: {outliers}")
```

---

### **Normal Distribution and Its Empirical Rule**

The **normal distribution** (also known as the Gaussian distribution) is a symmetric, bell-shaped distribution where the mean, median, and mode are all equal. Most data points cluster around the center of the distribution, with fewer points in the tails.

- **Empirical Rule (68-95-99.7 Rule)**: In a normal distribution:
  - 68% of the data lies within 1 standard deviation of the mean.
  - 95% of the data lies within 2 standard deviations.
  - 99.7% of the data lies within 3 standard deviations.

#### Example of plotting normal distribution using Python:
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random data following a normal distribution
data = np.random.normal(0, 1, 1000)

# Plot the histogram
plt.hist(data, bins=30, density=True)
plt.title("Normal Distribution")
plt.show()
```

---

### **Central Limit Theorem (CLT)**

The **Central Limit Theorem (CLT)** states that the sampling distribution of the sample mean approaches a normal distribution as the sample size increases, regardless of the population’s distribution. This is crucial in inferential statistics because it allows us to make predictions about population parameters using sample data.

- **Example**: If you repeatedly sample the heights of people from different regions, the distribution of the sample means will tend to be normally distributed, even if the actual height distribution is skewed.

---

### **Log-Normal Distribution**

A **log-normal distribution** is a probability distribution of a variable whose logarithm is normally distributed. In this distribution, values are positively skewed, and data cannot be negative. It is often used to model data that grows exponentially or multiplicatively.

#### Example:
If incomes or stock prices are analyzed, the log of these values often follows a normal distribution.

```python
# Generate log-normal data
log_normal_data = np.random.lognormal(mean=0, sigma=1, size=1000)

# Plot the histogram
plt.hist(log_normal_data, bins=30, density=True)
plt.title("Log-Normal Distribution")
plt.show()
```

---

### **Power Law Distribution**

A **power law distribution** describes a relationship where large events are rare, but small events are common. Power law distributions are often observed in natural and social phenomena, such as earthquakes, city populations, and wealth distribution.

In a power law distribution, the frequency of an event decreases rapidly with increasing event size.

---

### **Hypothesis Testing**

**Hypothesis testing** is a statistical method used to make decisions about the population based on sample data. It involves:
- **Null Hypothesis (H0)**: Assumes no effect or no difference.
- **Alternative Hypothesis (H1)**: Assumes there is an effect or a difference.
  
The steps include:
1. Formulate H0 and H1.
2. Choose a significance level (commonly α = 0.05).
3. Calculate a test statistic (e.g., t-test, z-test).
4. Compare the p-value to α to decide whether to reject or fail to reject H0.

---

### **Probability Density Function (PDF)**

The **Probability Density Function (PDF)** describes the likelihood of a continuous random variable falling within a particular range of values. The area under the curve of the PDF represents the probability of the variable being within a specific range.

---

### **Probability Mass Function (PMF)**

The **Probability Mass Function (PMF)** is used for discrete random variables. It gives the probability that a discrete variable is exactly equal to a certain value.

#### Example: For a fair six-sided die, the PMF is 1/6 for each face.

---

### **24. Cumulative Distribution Function (CDF)**

The **Cumulative Distribution Function (CDF)** gives the probability that a random variable is less than or equal to a certain value. It accumulates the probability values from the left end of the distribution up to a given point.

#### Example:
```python
import matplotlib.pyplot as plt
import numpy as np

data = np.random.normal(0, 1, 1000)

# Plot the CDF
plt.hist(data, bins=30, density=True, cumulative=True)
plt.title("Cumulative Distribution Function")
plt.show()
```

### **Hypothesis Testing:**

**Hypothesis Testing** is a method used in statistics to determine whether there is enough evidence in a sample of data to infer that a certain condition is true for the entire population. It starts with two hypotheses:

- **Null Hypothesis (H₀)**: This is the assumption that there is no effect or no difference. It assumes the status quo or baseline condition.
  
- **Alternative Hypothesis (H₁ or Ha)**: This is the assumption that there is a significant effect or difference. It represents the conclusion you hope to support.

#### **Key Terms in Hypothesis Testing:**
- **Significance Level (α)**: The probability of rejecting the null hypothesis when it is true. Common choices for α are 0.05 (5%) or 0.01 (1%).
  
- **Test Statistic**: A value calculated from the sample data, used to determine whether to reject the null hypothesis. It can be a Z-statistic, T-statistic, etc.

- **P-value**: The probability of observing a test statistic as extreme as the one calculated, assuming the null hypothesis is true. If the p-value is less than the significance level α, we reject the null hypothesis.

- **Decision Rule**: Based on the p-value or comparing the test statistic to a critical value, decide whether to reject or fail to reject the null hypothesis.

---

### **1. Z-Test:**

A **Z-test** is used when the population variance is known or the sample size is large (typically n > 30). It is used to test whether the sample mean is significantly different from a known population mean.

---

#### **Z-Test Example (Python)**:

Let’s say you want to test if the average height of a group is different from the population mean height of 170 cm. You have a sample of 40 people with an average height of 172 cm and a population standard deviation of 6 cm.

```python
import numpy as np
from scipy import stats

# Known values
population_mean = 170
sample_mean = 172
population_std = 6
sample_size = 40

# Z-Test
z_stat = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))
p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))  # Two-tailed p-value

print(f"Z-Statistic: {z_stat}, P-value: {p_value}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis.")
```

If the p-value is less than 0.05, you would reject the null hypothesis, indicating that the average height is significantly different from 170 cm.

---

### **T-Test:**

A **T-test** is used when the population variance is unknown and the sample size is small (typically n < 30). It is used to test whether the sample mean is significantly different from a known population mean, similar to a Z-test but with different handling for smaller samples.

#### **Types of T-Tests**:
- **One-Sample T-Test**: Compares the sample mean to a known value (e.g., population mean).
- **Two-Sample T-Test**: Compares the means of two independent samples.
- **Paired T-Test**: Compares means from the same group at different times (e.g., before and after treatment).

---

#### **T-Test Example (Python)**:

Suppose you are testing whether the average score of students in a test is significantly different from 75. You have a sample of 10 students with scores [80, 85, 78, 92, 67, 74, 80, 89, 77, 71].

```python
import numpy as np
from scipy import stats

# Sample data
scores = [80, 85, 78, 92, 67, 74, 80, 89, 77, 71]
sample_mean = np.mean(scores)
sample_std = np.std(scores, ddof=1)  # Sample standard deviation
sample_size = len(scores)

# Population mean
population_mean = 75

# T-Test
t_stat, p_value = stats.ttest_1samp(scores, population_mean)

print(f"T-Statistic: {t_stat}, P-value: {p_value}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis.")
```

In this case, if the p-value is less than 0.05, you would reject the null hypothesis, suggesting that the average score is significantly different from 75.

---

### **Two-Sample T-Test (Independent Samples T-Test):**

The **two-sample T-test** compares the means of two independent groups to determine whether there is a significant difference between them. 

#### **Steps in a Two-Sample T-Test**:

1. **State the Hypotheses**:
   - H₀: μ₁ = μ₂ (The means of two groups are equal.)
   - H₁: μ₁ ≠ μ₂ (The means of two groups are not equal.)

2. **Set the Significance Level (α)**.

3. **Determine the Critical Value or P-value**.
4. **Make the Decision**.

#### **Two-Sample T-Test Example (Python)**:

You want to test whether the average scores of two different classes (Class A and Class B) are significantly different. The scores are:

- Class A: [82, 87, 75, 80, 85, 90, 92]
- Class B: [75, 78, 80, 72, 70, 68, 73]

```python
import numpy as np
from scipy import stats

# Class A and Class B scores
class_a = [82, 87, 75, 80, 85, 90, 92]
class_b = [75, 78, 80, 72, 70, 68, 73]

# Two-Sample T-Test
t_stat, p_value = stats.ttest_ind(class_a, class_b)

print(f"T-Statistic: {t_stat}, P-value: {p_value}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis.")
```

### **Type I Error and Type II Error:**

In hypothesis testing, two types of errors can occur:

#### **Type I Error (False Positive)**:
- **Definition**: Occurs when the null hypothesis (H₀) is true, but we incorrectly reject it.
- **Consequences**: We falsely conclude that there is a significant effect or difference when in reality there isn’t.
- **Symbol**: Denoted by **α** (alpha), which is the significance level (e.g., 0.05). If α = 0.05, there is a 5% chance of committing a Type I error.

- **Example**: Imagine testing a drug’s effectiveness. If we conclude that the drug works (rejecting H₀) when it actually doesn’t (H₀ is true), we have committed a Type I error.

#### **Type II Error (False Negative)**:
- **Definition**: Occurs when the null hypothesis (H₀) is false, but we fail to reject it.
- **Consequences**: We incorrectly conclude that there is no effect or difference when there actually is.
- **Symbol**: Denoted by **β** (beta). The probability of avoiding a Type II error is called **power** (1 - β).

- **Example**: Suppose we are testing the same drug, and it does work (H₁ is true), but our test results show no significant effect, leading us to conclude that the drug doesn’t work (failing to reject H₀). This is a Type II error.

#### **Key Differences**:
- **Type I Error**: Detects an effect that isn’t there.
- **Type II Error**: Fails to detect an effect that is actually there.

---

### **Confidence Interval (CI):**

A **Confidence Interval** provides a range of values within which the true population parameter (e.g., mean) is expected to fall, given a certain level of confidence.


---

### **Margin of Error (MoE):**

The **Margin of Error** is the amount added and subtracted from the sample mean to create the confidence interval. It reflects the uncertainty in the estimate.


---

### **Chi-Square Test:**

The **Chi-Square Test** is used to determine whether there is a significant association between two categorical variables or to test the goodness of fit between observed and expected frequencies.

#### **Types of Chi-Square Tests**:
1. **Chi-Square Test for Independence**: Tests if two categorical variables are independent (i.e., unrelated).
2. **Chi-Square Goodness of Fit Test**: Tests if a sample matches the expected distribution.

#### **Chi-Square Test for Independence**:
This test is used to check if there is a significant relationship between two categorical variables in a contingency table.


#### **Steps for the Chi-Square Test for Independence**:
1. **State the Hypotheses**:
   - H₀: The two categorical variables are independent.
   - H₁: The two categorical variables are dependent.

2. **Calculate the Expected Frequencies**:
   \[
   E_{ij} = \frac{(\text{Row Total}) \times (\text{Column Total})}{\text{Grand Total}}
   \]

3. **Calculate the Chi-Square Statistic** using the observed and expected frequencies.

4. **Determine the Critical Value or P-value**: Use a Chi-square distribution table (degrees of freedom = (rows - 1) × (columns - 1)).

5. **Make a Decision**:
   - If the p-value < α or the chi-square statistic > critical value, reject H₀.

#### **Example (Python)**:

Suppose we want to test if there is a relationship between gender (male/female) and preference for two types of products (Product A and Product B). We have the following observed frequencies:

|            | Product A | Product B |
|------------|-----------|-----------|
| Male       | 30        | 10        |
| Female     | 20        | 40        |

```python
import numpy as np
from scipy.stats import chi2_contingency

# Contingency table (observed frequencies)
observed = np.array([[30, 10], [20, 40]])

# Perform the Chi-Square test
chi2_stat, p_value, dof, expected = chi2_contingency(observed)

print(f"Chi-Square Statistic: {chi2_stat}")
print(f"P-value: {p_value}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(expected)

# Decision
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis (variables are dependent).")
else:
    print("Fail to reject the null hypothesis (variables are independent).")
```

If the p-value is less than 0.05, you reject the null hypothesis, suggesting that there is a relationship between gender and product preference.

---

### **Confidence Interval vs. Chi-Square Test**:

- **Confidence Interval**: Provides a range of plausible values for a population parameter (e.g., mean) based on the sample data.
  
- **Chi-Square Test**: Used to test relationships between categorical variables or the goodness of fit for observed vs. expected data.

### **Analysis of Variance (ANOVA):**

**Analysis of Variance (ANOVA)** is a statistical technique used to compare the means of three or more groups to determine if they are significantly different from each other. The main goal of ANOVA is to test for significant differences between group means by analyzing the variances within and between groups. 

ANOVA is useful when you have multiple groups and want to see if any of them differ significantly from the others. It is based on the ratio of the variance between groups to the variance within groups.

### **Types of ANOVA:**
1. **One-Way ANOVA** (Single Factor ANOVA)
2. **Two-Way ANOVA** (Two Factors ANOVA)
3. **Repeated Measures ANOVA**

---

### **1. One-Way ANOVA:**

**One-Way ANOVA** is used to determine whether there are any statistically significant differences between the means of three or more independent (unrelated) groups.

#### **Assumptions of One-Way ANOVA**:
- The populations from which the samples are drawn should be normally distributed.
- Homogeneity of variance (equal variances among groups).
- The observations are independent of each other.

#### **Hypotheses**:
- **Null Hypothesis (H₀)**: All group means are equal.
- **Alternative Hypothesis (H₁)**: At least one group mean is different from the others.


---

#### **Example of One-Way ANOVA** (Python):

Let’s say we want to test if there’s a difference in exam scores between three groups of students who took different types of preparation courses.

| Group 1 | Group 2 | Group 3 |
|---------|---------|---------|
| 85      | 78      | 90      |
| 88      | 82      | 85      |
| 92      | 79      | 89      |

```python
import scipy.stats as stats

# Define the data for each group
group1 = [85, 88, 92]
group2 = [78, 82, 79]
group3 = [90, 85, 89]

# Perform One-Way ANOVA
f_stat, p_value = stats.f_oneway(group1, group2, group3)

print(f"F-statistic: {f_stat}")
print(f"P-value: {p_value}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. At least one group mean is different.")
else:
    print("Fail to reject the null hypothesis. The group means are not significantly different.")
```
If the p-value is less than 0.05, you would conclude that there is a significant difference between the groups.

---

### **2. Two-Way ANOVA**:

**Two-Way ANOVA** is used when you have two independent variables (factors) and you want to understand the interaction between them and their effect on a dependent variable. 

#### **Types of Two-Way ANOVA**:
1. **Without Replication**: When each combination of the factors has only one observation.
2. **With Replication**: When each combination of factors has multiple observations.

#### **Assumptions of Two-Way ANOVA**:
- The populations from which the samples are drawn should be normally distributed.
- Homogeneity of variance.
- The observations are independent.

#### **Hypotheses in Two-Way ANOVA**:
- **Main Effect 1**: The effect of the first factor.
- **Main Effect 2**: The effect of the second factor.
- **Interaction Effect**: Whether the two factors interact with each other.

#### **Example of Two-Way ANOVA (Without Replication)**:
Let’s consider the effect of **Diet** (Factor A) and **Exercise** (Factor B) on weight loss. There are two diets (Diet 1 and Diet 2) and two exercise routines (Exercise 1 and Exercise 2). 

We measure the weight loss of individuals for each combination of diet and exercise:

|               | Exercise 1 | Exercise 2 |
|---------------|------------|------------|
| **Diet 1**    | 3 kg       | 5 kg       |
| **Diet 2**    | 4 kg       | 6 kg       |

For this scenario, Two-Way ANOVA will help us understand:
- Whether **Diet** has a significant effect on weight loss.
- Whether **Exercise** has a significant effect on weight loss.
- Whether there is an **interaction** between Diet and Exercise.

---

### **Repeated Measures ANOVA**:

**Repeated Measures ANOVA** is used when the same subjects are measured multiple times under different conditions. It helps analyze how the dependent variable changes across multiple observations on the same subjects.

For example, a group of patients might receive three different treatments, and their recovery scores are measured after each treatment. Repeated measures ANOVA would compare these scores to determine if the treatments significantly differ.

---

### **ANOVA Test Output**:
After running an ANOVA test, if the p-value is less than the significance level (α), you reject the null hypothesis, indicating that there is at least one group that differs significantly. If the p-value is higher than the significance level, the null hypothesis is not rejected, indicating no significant difference between the group means.

---

### **Post-Hoc Tests**:

If the ANOVA test is significant (i.e., you reject the null hypothesis), it only tells you that there is a difference somewhere between the groups, but it doesn't tell you which groups are different from each other. To find out, you perform **post-hoc tests** like the **Tukey HSD (Honestly Significant Difference)** test to pinpoint where the differences lie.

#### **Tukey HSD Test** Example in Python:

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

# Data for Tukey HSD test
data = {'score': [85, 88, 92, 78, 82, 79, 90, 85, 89],
        'group': ['group1']*3 + ['group2']*3 + ['group3']*3}

df = pd.DataFrame(data)

# Perform Tukey HSD
tukey_result = pairwise_tukeyhsd(df['score'], df['group'], alpha=0.05)
print(tukey_result)
```

### **F-Distribution in Statistics:**

The **F-distribution** is a continuous probability distribution that arises frequently in hypothesis testing, particularly in the analysis of variance (ANOVA) and in variance ratio tests. It is skewed to the right and takes only positive values. The F-distribution is used to compare variances by examining the ratio of two variances, typically when testing for equality of variances in multiple groups.

#### **Key Characteristics of F-Distribution**:
- It is always positive since it is based on a ratio of variances (variances are non-negative).
- It is right-skewed and has a long right tail.
- It depends on two parameters: degrees of freedom for the numerator (df₁) and degrees of freedom for the denominator (df₂).

The F-distribution is used primarily in the context of **ANOVA** and **regression analysis**, where we want to compare the variability between groups or models. 

#### **Formula for F-distribution**:
The F-statistic is computed as the ratio of two sample variances:

Where:
- The **numerator** is the variance between the group means (reflects variability due to treatment).
- The **denominator** is the variance within each group (reflects variability within individual groups).

### **Variance Ratio Test:**

The **variance ratio test** (commonly referred to as an F-test) is a hypothesis test used to determine if two populations have equal variances. It compares the ratio of the two sample variances, and if this ratio is significantly different from 1, we conclude that the variances are not equal.

#### **Hypotheses**:
- **Null Hypothesis (H₀)**: The variances of the two populations are equal. 

- **Alternative Hypothesis (H₁)**: The variances of the two populations are not equal.



---

### **Example of Variance Ratio Test Using F-Distribution**:

Let’s perform an F-test to compare the variances of two groups.

| Group A | Group B |
|---------|---------|
| 25      | 30      |
| 28      | 35      |
| 30      | 33      |
| 35      | 37      |
| 40      | 42      |

We want to test if the variances of Group A and Group B are significantly different.

```python
import numpy as np
import scipy.stats as stats

# Data for two groups
group_a = [25, 28, 30, 35, 40]
group_b = [30, 35, 33, 37, 42]

# Calculate variances
var_a = np.var(group_a, ddof=1)  # Sample variance of Group A
var_b = np.var(group_b, ddof=1)  # Sample variance of Group B

# F-statistic
f_stat = var_a / var_b
print(f"F-statistic: {f_stat}")

# Degrees of freedom
df1 = len(group_a) - 1
df2 = len(group_b) - 1

# Critical F-value (for two-tailed test, alpha = 0.05)
alpha = 0.05
f_critical = stats.f.ppf(1 - alpha/2, df1, df2)

print(f"Critical F-value: {f_critical}")

# Decision
if f_stat > f_critical:
    print("Reject the null hypothesis. Variances are significantly different.")
else:
    print("Fail to reject the null hypothesis. Variances are not significantly different.")
```