from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def test_regressor(model, data_x, bulk_y, shear_y, testsize=0.2, verbose=1):
    print('='*30)
    print(str(type(model)).replace("'", ".").split('.')[-2])

    n = int((100) // (testsize*100)) - 1
    N = data_x.shape[0]

    bulk_train = []
    bulk_test = []
    shear_train = []
    shear_test = []
    bulk_test_best = (0, 0)
    shear_test_best = (0, 0)

    for i in range(n):
        test = round((i+1)*testsize, 2)
        train = round(1 - test, 2)
        
        x_train1, x_test1, y_train1, y_test1 = train_test_split(data_x, bulk_y, test_size=testsize, random_state=0)
        reg1 = model.fit(x_train1, y_train1)
        
        x_train2, x_test2, y_train2, y_test2 = train_test_split(data_x, shear_y, test_size=testsize, random_state=0)
        reg2 = model.fit(x_train2, y_train2)

        bulk_train.append((round(reg1.score(x_train1, y_train1)*100, 2), test))
        bulk_test.append((round(reg1.score(x_test1, y_test1)*100, 2), test))
        shear_train.append((round(reg2.score(x_train2, y_train2)*100, 2), test))
        shear_test.append((round(reg2.score(x_test2, y_test2)*100, 2), test))

        bulk_test_best = bulk_test[-1] if bulk_test_best[0] < bulk_test[-1][0] else bulk_test_best
        shear_test_best = shear_test[-1] if shear_test_best[0] < shear_test[-1][0] else shear_test_best

        if verbose == 2:
            print('-'*30)
            print(f'Train({train}) : Test({test}) = {N*train:.0f} : {N*test:.0f}\n')
            print(f'Bulk train score: {bulk_train[i][0]}%\nShear train score: {shear_train[i][0]}%')
            print(f'Bulk test score: {bulk_test[i][0]}%\nShear test score: {shear_test[i][0]}%')
        elif verbose == 0 or verbose == 1:
            pass
        else:
            raise Exception('Invalid verbose argument: 0 for silent, 1 for plot only, 2 for full log')
    
    if verbose != 0:
        bulk_train = np.array(bulk_train)
        shear_train = np.array(shear_train)
        bulk_test = np.array(bulk_test)
        shear_test = np.array(shear_test)
        plt.figure()
        plt.plot(bulk_train[:,1], bulk_train[:,0], '-o', label='Bulk training')
        plt.plot(shear_train[:,1], shear_train[:,0], '-o', label='Shear training')
        plt.plot(bulk_test[:,1], bulk_test[:,0], '-o', label='Bulk test')
        plt.plot(shear_test[:,1], shear_test[:,0], '-o', label='Shear test')
        plt.legend()
        plt.ylim([0,110])
        plt.title('Training and Test Scores')
        plt.xlabel('Test size')
        plt.ylabel('Score')
        plt.show()

    print('-'*30)
    print(f'Best bulk test score: {bulk_test_best[0]}% at test_size = {bulk_test_best[1]}')
    print(f'Best shear test score: {shear_test_best[0]}% at test_size = {shear_test_best[1]}')

    print('='*30)

param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [None, 6, 9, 12, 15, 18],
    'min_samples_split': [0.01, 0.05, 0.1],
    'max_features': ['sqrt', 'log2', None],
}

kf = KFold(random_state=30,
           n_splits=10,
           shuffle=True,
          )

def tune_parameter(model, data_x, bulk_y, shear_y, testsize=0.2):
    grid_search = GridSearchCV(estimator=model, 
                            param_grid=param_grid, 
                            cv=kf, 
                            n_jobs=-1, 
                            verbose=0
                            )

    x_train1, x_test1, y_train1, y_test1 = train_test_split(data_x, bulk_y, test_size=testsize, random_state=0)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(data_x, shear_y, test_size=testsize, random_state=0)

    grid_search.fit(x_train1, y_train2)
    best1 = grid_search.best_params_

    grid_search.fit(x_train2, y_train2)
    best2 = grid_search.best_params_

    print(f'Bulk regression best parameters:\n{best1}')
    print(f'Shear regression best parameters:\n{best2}')

    return best1, best2

def best_regressor(model, best1, best2):
    best_model1 = type(model)(max_depth=best1['max_depth'], 
                              max_features=best1['max_features'], 
                              min_samples_split=best1['min_samples_split'], 
                              n_estimators=best1['n_estimators'])

    best_model2 = type(model)(max_depth=best2['max_depth'], 
                              max_features=best2['max_features'], 
                              min_samples_split=best2['min_samples_split'], 
                              n_estimators=best2['n_estimators'])

    return best_model1, best_model2

def tune_regressor(model, data_x, bulk_y, shear_y, verbose=1, testsize=0.2):
    best1, best2 = tune_parameter(model, data_x, bulk_y, shear_y)
    best_bulk, best_shear = best_regressor(model, best1, best2)
    test_regressor(best_bulk, data_x, bulk_y, shear_y, verbose=verbose, testsize=testsize)
    test_regressor(best_shear, data_x, bulk_y, shear_y, verbose=verbose, testsize=testsize)