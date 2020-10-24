# Time-Series-Forecasting

## Background knowledge:

Australian Energy Market Operator (AEMO) is a not-for-profit organisation partly owned by federal and state governments, with members from electricity and gas generation, transmission, distribution, retail and resources businesses across Australia.

According to AMEO 2019 annual report, their target is **to achieve short-term and long-term electricity demand forecast accuracy within 3% and 5% of actual peak demand**, respectively.

This repo is used this as the demo for Data Analysis and Machine Learning Task on general forecasting modelling.

## Data source:

![AEMO Data](./docs/AemoDataDownload.png)

Data Source: https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data

The price and demand data sets are downloaded from 2018-01 to 2019-12 of Victoria (VIC) with the following data format.

| REGION | SETTLEMENTDATE      | TOTALDEMAND | RRP   | PERIODTYPE |
| ------ | ------------------- | ----------- | ----- | ---------- |
| ...    | ...                 | ...         | ...   | ...        |
| VIC1   | 2018/01/01 00:30:00 | 4251.18     | 92.46 | TRADE      |
| VIC1   | 2018/01/01 01:00:00 | 4092.53     | 87.62 | TRADE      |
| VIC1   | 2018/01/01 01:30:00 | 3958.95     | 73.08 | TRADE      |
| ...    | ...                 | ...         | ...   | ...        |

## Demo in Tutorial.

**Objective**: built a electricity demand predictive model with AEMO data set.

Only used the historical demand record to make demand prediction in this demo, which is using the last 6 hours demand (12 records) to predict the next 0.5 hour demand (1 record), for example, use the `2018/01/01 00:30:00 - 2018/01/01 06:00:00` demand records to predict the demand at `2018/01/01 06:30:00`.

More details in the notebook.

## Challenge to be solved

Use the provided data to build a electricity demand predictive model with the following conditions:

1. Add the `price` record (RRP) as a new input variable for the model.
2. Document your analysis and modelling codes in jupyter notebook.
3. Design the testing functions of your model, for example:

   - data-processing function.
   - model prediction functions.
     - make short-term prediction: 2 hours (4 timestamps ahead).
     - make long-term prediction: 6 hours (12 timestamps ahead).
   - report `mean-absolute-error` and `mean-absolute-error-rate`.

4. Apply cross validation method or optimization method to improve model's performance.
5. Push your repo to your personal GitLab and then fork to Aurecon Data Science GitLab before **Wednesday morning 9:00 am AEST, 27-May-2020**.
6. Test data for model evaluation will be released on **Wednesday morning 9:00 am AEST, 27-May-2020**.
7. Run your pre-trained model with test data in a new jupyter notebook and push it with your results to Gitlab.

## Results

The following notebooks are the solutions of this Challenge.
It built the MLP (Tensorflow) model on the data between 2018 and 2019 and made electricity demand prediction of the first five months of 2020.

These notebooks are designed as examples to encourage more and more people getting into data analysis and machine learning practice. They are not designed for real applications.

In the `src` folder, you will find four notebooks.

- [Demo (MLP.TF) 01 Build Model.ipynb]
- [Demo (MLP.TF) 02 Apply Model.ipynb]
- [Demo (MLP.TF) 03 Short and Long Term Prediction.ipynb]
- [Demo (MLP.TF) 04 Short and Long Term Prediction with Price.ipynb]

All the models used a simple MLP structure: input-100-output. The following table shows the performance of the models.

| MLP name           | input/hidden-node/output | Training MAE | Validation MAE | Test MAE | Training MAPE | Validation MAPE | Test MAPE |
| ------------------ | ------------------------ | ------------ | -------------- | -------- | ------------- | --------------- | --------- |
| Basic model        | 12-100-1                 | 58.8217      | 58.2836        | 57.6027  | 1.1927 %      | 1.2007 %        | 1.2263 %  |
| Short-Term         | 12-100-4                 | 130.1236     | 136.7397       | 138.6194 | 2.6166 %      | 2.8024 %        | 2.9346 %  |
| Short-Term (price) | 24-100-4                 | 128.8209     | 133.8860       | 137.3324 | 2.5915 %      | 2.7498 %        | 2.9130 %  |
| Long-Term          | 24-100-12                | 210.7373     | 246.7550       | 258.6256 | 4.2567 %      | 5.1659 %        | 5.6055 %  |
| Long-Term (price)  | 48-100-12                | 211.7583     | 244.6311       | 259.8957 | 4.2765 %      | 5.1094 %        | 5.6301 %  |

It is clear to see the testing MAPE of short term prediction of our basic MLP model is around 2.9 %, and the long term prediction is about 5.6 %. More conclusions can be drawn from here, but remember this is just a practice.

## More reading materials

- [Learning Git](https://git-scm.com/doc)
- [Learning Pandas](https://pandas.pydata.org/pandas-docs/version/0.15/tutorials.html).
- [Jupyter notebook](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/examples_index.html).
- [scikit-learn](https://scikit-learn.org/stable/tutorial/index.html)
- [tensorflow](https://www.tensorflow.org/tutorials)

## File Structure

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
