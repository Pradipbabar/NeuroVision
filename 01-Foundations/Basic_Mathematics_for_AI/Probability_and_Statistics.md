# Basics of Probability and Statistics

Probability and statistics are foundational to understanding data, making inferences, and making decisions based on data. Here’s a detailed explanation of key concepts in both fields, along with examples.

## 1. Probability

Probability is the measure of the likelihood that an event will occur. It quantifies uncertainty and ranges from 0 (impossible event) to 1 (certain event).

## Basic Concepts

**1.1. Probability of an Event**

**Definition**: The probability of an event \( A \) occurring is given by:
\[ P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of outcomes}} \]

**Example**: If you roll a fair six-sided die, the probability of rolling a 4 is:
\[ P(\text{Rolling a 4}) = \frac{1}{6} \]

**1.2. Complementary Events**

**Definition**: The complement of an event \( A \) is the event that \( A \) does not occur. The probability of the complement is:
\[ P(A^c) = 1 - P(A) \]

**Example**: For the die roll, the probability of not rolling a 4 is:
\[ P(\text{Not Rolling a 4}) = 1 - \frac{1}{6} = \frac{5}{6} \]

**1.3. Conditional Probability**

**Definition**: The probability of an event \( A \) given that event \( B \) has occurred is:
\[ P(A | B) = \frac{P(A \cap B)}{P(B)} \]

**Example**: If we draw a card from a deck and know it is a heart, the probability of it being a queen (conditional probability) is:
\[ P(\text{Queen} | \text{Heart}) = \frac{1}{13} \]

**1.4. Independence**

**Definition**: Two events \( A \) and \( B \) are independent if:
\[ P(A \cap B) = P(A) \cdot P(B) \]

**Example**: If you flip a coin and roll a die, the probability of getting a head and rolling a 3 is:
\[ P(\text{Head} \cap \text{Rolling a 3}) = P(\text{Head}) \cdot P(\text{Rolling a 3}) = \frac{1}{2} \cdot \frac{1}{6} = \frac{1}{12} \]

## 2. Statistics

Statistics involves collecting, analyzing, interpreting, and presenting data. It helps summarize and draw conclusions from data.

## Descriptive Statistics

**2.1. Measures of Central Tendency**

- **Mean**: The average of a set of numbers.
  
  **Example**: For the data set \{3, 5, 7, 9\}, the mean is:
  \[ \text{Mean} = \frac{3 + 5 + 7 + 9}{4} = 6 \]

- **Median**: The middle value when the data is sorted. If the data set has an even number of observations, it is the average of the two middle values.
  
  **Example**: For the data set \{3, 5, 7, 9\}, the median is:
  \[ \text{Median} = \frac{5 + 7}{2} = 6 \]

- **Mode**: The value that occurs most frequently in a data set.
  
  **Example**: For the data set \{3, 5, 5, 7, 9\}, the mode is:
  \[ \text{Mode} = 5 \]

**2.2. Measures of Dispersion**

- **Range**: The difference between the maximum and minimum values.
  
  **Example**: For the data set \{3, 5, 7, 9\}, the range is:
  \[ \text{Range} = 9 - 3 = 6 \]

- **Variance**: The average of the squared differences from the mean.
  
  **Example**: For the data set \{3, 5, 7, 9\} with mean 6:
  \[ \text{Variance} = \frac{(3-6)^2 + (5-6)^2 + (7-6)^2 + (9-6)^2}{4} = \frac{9 + 1 + 1 + 9}{4} = 5 \]

- **Standard Deviation**: The square root of the variance.
  
  **Example**: For the above variance of 5:
  \[ \text{Standard Deviation} = \sqrt{5} \approx 2.24 \]

## Inferential Statistics

**2.3. Hypothesis Testing**

**Definition**: A method for testing a claim or hypothesis about a population based on sample data.

- **Null Hypothesis (\( H_0 \))**: A statement that there is no effect or no difference.
  
- **Alternative Hypothesis (\( H_1 \))**: A statement that there is an effect or a difference.

**Example**: Suppose we want to test if a new teaching method improves test scores. The null hypothesis might be that the new method does not change the test scores, while the alternative hypothesis is that it does.

**2.4. Confidence Intervals**

**Definition**: A range of values that is likely to contain the population parameter with a specified level of confidence.

**Example**: If we estimate the average height of a population with a 95% confidence interval of \[ (160, 170) \], we are 95% confident that the true average height is between 160 cm and 170 cm.

## Regression and Correlation

**2.5. Regression Analysis**

**Definition**: A statistical technique used to understand the relationship between a dependent variable and one or more independent variables.

**Example**: In linear regression, you might model the relationship between house prices (dependent variable) and features such as size and location (independent variables).

**2.6. Correlation**

**Definition**: Measures the strength and direction of the linear relationship between two variables.

**Example**: If we find a high positive correlation between hours studied and test scores, it suggests that as study hours increase, test scores also tend to increase.

## Summary

- **Probability**: Focuses on measuring the likelihood of events, using concepts such as probability of events, conditional probability, and independence.
- **Statistics**: Involves summarizing and analyzing data through descriptive statistics (mean, median, mode, variance, standard deviation) and inferential statistics (hypothesis testing, confidence intervals, regression, correlation).
