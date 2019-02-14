# model_selection
##### A set of Python 3 tools to compare the performance and metrics of different ML models
<br/>

## Features
* Preprocessing functions to read in CSV data and manipulate it
* Functions that return a number of essential metrics on model performance
* Generates visualizations to better understand the model

<br/>

## Setup

To use these model selection tools, you'll need to:

* Clone this repository:

      $ git clone https://github.com/nousot/model_selection.git
      $ cd model_selection
      
* Copy `analysis.py` to your project directory, install packages:

      $ cp analysis.py ../path/to/project
      $ pip3 install numpy matplotlib matplotlib_venn seaborn

<br/>
## Using the functions

#### 1) performance
```python
    import analysis
    analysis.performance(estimated, actual, visualize=True, verbose=True):
```

**Arguments:** estimated: array of estimated output probabilities, actual: array of actual output classifications
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Optional: visualize (Bool), verbose (Bool)

**Returns:** returns accuracy, optionally prints other metrics and a performance visualization

![performance function](https://github.com/nousot/model_selection/blob/master/img/performance.png "")

<br/>
#### 2) distribution_metric
```python
    import analysis
    analysis.distribution_metric(estimated, actual, precision=2, visualize=True, verbose=True):
```

**Arguments:** estimated: array of estimated output probabilities, actual: array of actual output classifications
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Optional: precision (int), visualize (Bool), verbose (Bool)

**Returns:** prints the calculated percentage of predictions outside of 1 standard deviation from the mean number of predictions at each probability, optionally draws a visualization

![distribution_metric function](https://github.com/nousot/model_selection/blob/master/img/distribution.png "")

<br/>
## Authors
* **Matt Moderwell** - *Initial work* - [mmoderwell.com](https://mmoderwell.com)
* **Vanessa Tang** - *Initial work*

Also see the list of [contributors](https://github.com/mmoderwell/api_monitoring/contributors) who participated in this project.