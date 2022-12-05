# Metal-Organic Frameworks Feature Regression

This project is about a regression model that predicts the properties of metal-organic frameworks (MOF).

The model uses geometric features of MOFs to predict the bulk and shear properties.

You can automatically tune the hyperparameters to get a better regression model by running the codes.

<br/>


## Dataset
The datasets are in the `/data/` folder with form of `csv` file. The figure below is examples of 5 MOF datas.
![img](/images/data.png)

<br/>


## Determining Regression Model
By running `ipynb_files/MOF_lazy.ipynb` with [LazyPredict](https://github.com/shankarpandala/lazypredict/tree/master), you can determine the best regression model for the datasets. The figures below is top 3 models of the result for each of bulk and shear data.
* Best models for bulk data
    ![img](/images/lazy1.png)
* Best models for shear data
    ![img](/images/lazy2.png)

## Functions
There are some useful functions in `/util/` folder.

All of the examples can be found in `/ipynb_files/MOF_regressor.ipynb` file.

<br/>


### `import_data.py/import_data(`*`filename`*`)` function
Imports a dataset in the `/data/`*`filename`* route and splits it into geometric features, bulk data, and shear data.

* **Arguments**: 
    * ***`filename`***: Filename of target dataset.

* **Returns**: 
    * **Tuple**: `(geometric features, bulk data, shear data)`

* **Example**:
    ```python
    data_x, bulk_y, shear_y = import_data('toacco_geo_chem_mit_order.csv')
    ```

<br/>


### `mof_util.py/test_regressor(`*`model, data_x, bulk_y, shear_y, testsize=0.2, verbose=1`*`)` function

Tests a regression model with geometric feature data x and each of bulk/shear property data y.

It iterates with various testsizes. (e.g. `testsize = [0.2, 0.4, 0.6, 0.8]`)

* **Arguments**: 
    * ***`model`***: Regression model input.
    * ***`data_x`***: Geometric feature x data for the regression model test.
    * ***`bulk_y`***: Target bulk property y data for the regression model test.
    * ***`shear_y`***: Target shear property y data for the regression model test.
    * ***`testsize`***: Minimum interval of the iteration. (Default = 0.2)
    * ***`verbose`***: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = plot only, 2 = full log for each iterations. (Default = 1)

* **Returns**: 
    * Prints test $R^2$ plot and best result.

* **Example**:

    Input:
    ```python
    from sklearn.ensemble import ExtraTreesRegressor
    extra_tree = ExtraTreesRegressor()
    test_regressor(extra_tree, data_x, bulk_y, shear_y, testsize=0.1, verbose=1)
    ```
    Output:
    ```
    ==============================
    ExtraTreesRegressor
    ```
    ![img](/images/output_ex1.png)
    ```
    ------------------------------
    Best bulk test score: 85.48% at test_size = 0.4
    Best shear test score: 78.35% at test_size = 0.2
    ==============================
    ```
<br/>


### `mof_util.py/tune_regressor(`*`model, data_x, bulk_y, shear_y, testsize=0.2, verbose=1, param_grid`*`=param_grid)` function

Tunes model hyperparameters to get the best regression model by using [GridSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).


* **Arguments**: 
    * ***`model`***: Regression model input.
    * ***`data_x`***: Geometric feature x data for the regression model test.
    * ***`bulk_y`***: Target bulk property y data for the regression model test.
    * ***`shear_y`***: Target shear property y data for the regression model test.
    * ***`testsize`***: Minimum interval of the iteration. (Default = 0.2)
    * ***`verbose`***: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = plot only, 2 = full log for each iterations. (Default = 1)
    * ***`param_grid`***: Hyperparameter grid for GridSearchCV. 
    
        (Default = `{
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [None, 6, 9, 12, 15, 18],
    'min_samples_split': [0.01, 0.05, 0.1],
    'max_features': ['sqrt', 'log2', None]
}`)

* **Returns**: 
    * **Tuple**: `(Best model for bulk data, Best model for shear data)`



* **Example**:

    Input:
    ```python
    from sklearn.ensemble import ExtraTreesRegressor
    extra_tree = ExtraTreesRegressor()
    extra_bulk, extra_shear = tune_regressor(extra_tree, data_x, bulk_y, shear_y, testsize=0.1, verbose=1)
    ```
    Output:
    ```
    Bulk regression best parameters:
    {'max_depth': 18, 'max_features': None, 'min_samples_split': 0.01, 'n_estimators': 100}
    Shear regression best parameters:
    {'max_depth': 18, 'max_features': None, 'min_samples_split': 0.01, 'n_estimators': 300}

    ==============================
    ExtraTreesRegressor
    ```
    ![img](/images/output_ex2.png)
    ```
    ------------------------------
    Best bulk test score: 79.16% at test_size = 0.6
    Best shear test score: 75.27% at test_size = 0.9
    ==============================
    ```
    ```
    ==============================
    ExtraTreesRegressor
    ```
    ![img](/images/output_ex3.png)
    ```
    ------------------------------
    Best bulk test score: 79.1% at test_size = 0.5
    Best shear test score: 74.96% at test_size = 0.3
    ==============================
    ```