# Takeaways Third Chapter (Español - Inglés)

1. Se puede hacer tuneado automático de parámetros con **GridSearchCV.** Recibe un param_grid que es un diccionario y testea todas las combinaciones paramétricas.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__learning_rate': (0.01, 0.1, 1, 10),
    'classifier__max_leaf_nodes': (3, 10, 30)}
model_grid_search = GridSearchCV(model, param_grid=param_grid,
                                 n_jobs=2, cv=2)
model_grid_search.fit(data_train, target_train) 

Luego, nos quedaremos con el mejor modelo y podremos saber su parametría:

model_grid_search.predict(data_test.iloc[0:5])
print(f"The best set of parameters is: "
      f"{model_grid_search.best_params_}")
```

> Podemos quedarnos con todos los resultados en un dataframe:

```python
 cv_results = pd.DataFrame(model_grid_search.cv_results_).sort_values(
    "mean_test_score", ascending=False)
```

> Se puede hacer mucha magia de visualización para tenerlo en un Heatmap.

**Searching for more than two hyperparamters is too costly;**
**A grid-search does not necessarily find an optimal solution.**

2. **The RandomizedSearchCV** class allows for such stochastic search. It is used similarly to the GridSearchCV but the sampling distributions need to be specified instead of the parameter values. For instance, we will draw candidates using a log-uniform distribution because the parameters we are interested in take positive values with a natural log scaling (.1 is as close to 1 as 10 is).
Random search (with RandomizedSearchCV) is typically beneficial compared to grid search (with GridSearchCV) to optimize 3 or more hyperparameters.

Frente a GridSearch es conveniente porque en esta última es relativamente sencillo que pasemos por la zona que no tiene los mejores parámetros. Sin embargo, en Randomized al usar una distribución es más posible encontrar esa región. 

Random search (with RandomizedSearchCV) is typically beneficial compared to grid search (with GridSearchCV) to optimize 3 or more hyperparameters.

3. Cómo funciona Grid Search en K-fold Cross Valid con n_splits == 5??
Para cada split de cross valid se entrena un modelo en datos de train en 3/5 y evalúa esos hyperparams en un conjunto 1/5 a veces llamado validación y se escogen los mejores hyperparams. Finalmente, con esos hyperparams se entrena un modelo en esos (3+1)/5  y se evalúa en el conjunto de test 1/5. 

Sin embargo, esto solo es una previa Xvalid. Es habitual hacer la llamada xvalid anidada, que dice que una vez hecha la xvalid previa para elegir params se debe hacer otra con el mejor modelo para evaluar su capacidad de generalización.


Wrap-up
Hyperparameters have an impact on the model's performance and should be wisely chosen;

The search for the best hyperparameters can be automated with a grid-search approach or a randomized search approach;

A grid-search is expensive and does not scale when the number of hyperparameters to optimize increase. Besides, the combination are sampled only on a regular grid.

A randomized-search allows a search with a fixed budget even with an increasing number of hyperparameters. Besides, the combination are sampled on a non-regular grid.