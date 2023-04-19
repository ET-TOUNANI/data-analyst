# Simple linear regression

## Introduction

In this project, we will implement a simple linear regression model to predict the yield of a plant based on the temperature.

## Preprocess the data

<div class="cell code" execution_count="132">

```python
# Library importation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

</div>

<div class="cell code" execution_count="133">

```python
# Data importation
data=pd.read_csv('data/data.csv')
```

</div>

<div class="cell code" execution_count="134">

```python
# Calculate the mean
def getMean(vrbl):
    return 1/len(vrbl)*sum(vrbl)
# we can use numpy to calculate avg :
    # np.mean(....)
```

</div>

<div class="cell code" execution_count="135">

```python
# Calculate variance
def getVar(vrbl):
    return 1/len(vrbl)*sum(vrbl*vrbl)-(getMean(vrbl)*getMean(vrbl))
# we can use numpy to calculate variance :
    # np.var(....)
```

</div>

<div class="cell code" execution_count="136">

```python
# Get xi , yi
xi=data.iloc[:,0]
yi=data.iloc[:,1]
```

</div>

<div class="cell code" execution_count="137">

```python
# Calculate Covariance
def getCov(xi,yi):
    return 1/len(xi)*sum(xi*yi)-getMean(xi)*getMean(yi)
# we can use numpy to calculate Covariance :
    # np.cov(xi, yi, ddof=0)[0,1]
```

</div>

<div class="cell code" execution_count="138">

```python
# Calculate the linear regression coefficient :  y = alpha * x + beta
def getAlpha(xi,yi):
    return getCov(xi,yi)/getVar(xi)
# we can use numpy to calculate alpha :
    # np.polyfit(xi, yi, deg=1)[0]
```

</div>

<div class="cell code" execution_count="139">

```python
# Calculate the slope (or beta or la pente)
def getPente(xi, yi):
    return getMean(yi)-getAlpha(xi, yi)*getMean(xi)
# we can use numpy to calculate alpha :
    # np.polyfit(xi, yi, deg=1)[1]
```

</div>

<div class="cell code" execution_count="140">

```python
# Calculate the predicted values of y
def getPredictedValue(xi,yi):
    return getAlpha(xi, yi)*xi+getPente(xi, yi)


# we can use numpy to calculate alpha :
    # calculate coefficients of linear regression model
        #coefficients = np.polyfit(x, y, 1)
    # calculate predicted y-values based on linear regression model
        #y_pred = np.polyval(coefficients, x)
```

</div>

<div class="cell code" execution_count="141">

```python
# Draw using plot
def getDrawPlt(xi,yi,clr):
    plt.plot(xi,yi,'.')
    plt.plot(xi, getPredictedValue(xi, yi), color=clr)
    plt.xlabel("temperature")
    plt.ylabel("rendemment")
    plt.show()
```

</div>

<div class="cell code" execution_count="142">

```python
#  Calculate the correlation coefficient
def getCoefCorrelation(xi,yi):
    return getCov(xi,yi)/(np.sqrt(getVar(xi))*np.sqrt(getVar(yi)))

# we can use numpy to calculate alpha :
    # corr_matrix = np.corrcoef(xi, yi) # calculate correlation matrix
    # corr_coef = corr_matrix[0, 1]     # get coef from corr_matrix
```

</div>

<div class="cell code" execution_count="143">

```python
#  Calculate the determination coefficient (R-squared)
def getCoefDetermination(xi,yi):
    return getCoefCorrelation(xi,yi)**2
```

</div>

<div class="cell code" execution_count="144">

```python
# Interpretation of R-squared
getCoefDetermination(xi,yi)
# Interpretation of R-squared = 0.996 in this case :
    # 0.996 means that 99.6% of the variation
    # in the dependent variable (yi) can be explained by the independent variable (xi)
    # in the linear regression model. This indicates a strong positive linear relationship between the two variables.
```

<div class="output execute_result" execution_count="144">

    0.9962609376200801

</div>

</div>

<div class="cell code" execution_count="145">

```python
# Adding points to the existing DATA
point1=pd.Series([105,180])
point2 = pd.Series([70, 70])
x2=pd.concat([xi, point1],ignore_index=True)
y2=pd.concat([yi, point2], ignore_index=True)
```

</div>

<div class="cell code" execution_count="146">

```python
# Data visualization
getDrawPlt(xi,yi,'r')
getDrawPlt(x2,y2,'g')
```

<div class="output display_data">

![](36a32cbbc175d2fd78cc31218ede6591ee734280.png)

</div>

<div class="output display_data">

![](b24bda1c09f159745635ff321d3f10df79145421.png)

</div>

</div>

<div class="cell code" execution_count="147">

```python
# Calculate the sum of squared errors between the observed y-values (yi) and the predicted y-values
def getError(xi,yi):
    return sum(np.square(yi-getPredictedValue(xi,yi)))
```

</div>

<div class="cell code" execution_count="148">

```python
print(getError(xi,yi))
```

<div class="output stream stdout">

    7.224242424242385

</div>

</div>

    <table>
    <tr>
    <td><img src="output/o1.png"></td>
    <td><img src="output/o2.png"></td>
    </tr>
    </table>

