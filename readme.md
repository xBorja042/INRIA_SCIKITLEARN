M12
1.- STANDARIZAR ES BIEN. NOS AYUDA A BAJAR EL NÚMERO DE ITERACIONES EN LOG REG Y OBTENDREMOS UNA MEJOR GENERALIZACION. 
2.- EL USO DE PIPELINES ES BIEN:
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=500)
)
3.- COMO COMPARAR PERFORMANCES (BIEN):
from sklearn.model_selection import cross_validate
cv_results = cross_validate(model, X, y)
cv_results  NOS DEVUELVE UN DICCIONARIO CON EL TIEMPO DE ENTRENAMIENTO, Y EL TRAIN-TEST SCORE. SE VE QUE CUANDO SE ESTANDARIZA SE TARDA MENOS EN ENTRENAR Y SE SACAN MEJORES RESULTADOS.
NO DEJA EL MODELO ENTRENADO, PERO ESTE COMPORTAMIENTO SE PUEDE CONTROLAR.


M13
1.- HAY MUCHAS FORMAS DE CATEGORIZAR LAS VARIABLES. ORNIDAL ENCODER FUNCIONA IGUAL QUE LABEL ENCODER. COGE LAS COSAS POR ORDEN ALFABETICO Y LAS ORDENA METIENDO UN FACTOR DE ORDEN FEO QUE METERÁ BIAS. EN CASO DE MODELOS LINEALES ESTO SERÁ PEOR, PARA ÁRBOLES NI TAN MAL. MEJOR EN MODELOS LINEALES USAR UN ONE HOT ENCODING, AUNQUE ENSANCHARÁ MUCHO LA MATRIZ DE FEATURES.
2.- EN ONEHOT ENCODING CADA FEATURE VA A TENER UNA DIMENSION IGUAL AL NÚMERO DE VALORES ÚNICOS EN ESA FEATURE. CUANDO TENGAMOS UN NÚMERO ALTO DE VALORES POSIBLES POR FEATURE (CARDINALIY) VA A HACER QUE AUMENTEN MUCHO LOS COSTES COMPUTACIONALES.
3.- CUANDO UN EJEMPLO DE CATEG APARECE POCO VA A DAR PROBLEMAS AL HACER CROSS VALIDATION (ENTRE OTROS). PARA ESTO EXISTE EL PARAM "handle_unknown". 
4.- COMO TODAS LAS VARIABLES ESTÁN ESCALADAS EN 0 Y 1 EN CASO DE ONEHOT NO HACE FALTA REESCALAR. ASÍ PODEMOS SUBIR EL NÚMERO DE ITERACIONES EN LOGREG
5.- HABRÁ VECES QUE DEBERÁ PETAR LA MOVIDA PERO EL CROSSVALID NO LO VA A ENSEÑR Y PODRÁ DEVOLVER NANS. ESTO PUEDE PASAR PORQUE AL IR A CODIFICAR UN VALOR PARA EL TEST SET NO LO HA VISTO EN EL TRAIN Y NO VA A SABER CODIFICARLO, POR LO QUE DEVOLVERÁ UN NAN. 
6.- QUÉ PASA SI TIENES COLUMNAS A LAS QUE QUIERES APLICAR UNA TRANSFORMACION U OTRA. SE PUEDE SELECCIONAR A QUÉ COLUMNAS APLICAR UNAS TRANSFORMACIONES:
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns)])

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))

from sklearn import set_config
set_config(display='diagram')
model

7.- HAY MODELOS MÁS POTENTES QUE LOS LINEARS MODELS. COMO LOS GRADIENT-BOOSTED TREES. LAS VENTAJAS DE LOS MODELOS LINEALES ES QUE SUELEN SER SENCILLOS COMPUTACIONALMENTE. LOS GBTREES NO PRECISAN ESCALADO DE LAS FEATURES Y SE PUEDE USAR UN CODIFICADO ORDNIAL DE LAS FEATURES, NO ES NCESARIO ONE HOT ENCODING.
from sklearn.ensemble import HistGradientBoostingClassifier

This estimator is much faster than GradientBoostingClassifier for big datasets (n_samples >= 10 000).

This estimator has native support for missing values (NaNs). During training, the tree grower learns at each split point whether samples with missing values should go to the left or right child, based on the potential gain. When predicting, samples with missing values are assigned to the left or right child consequently. If no missing values were encountered for a given feature during training, then samples with missing values are mapped to whichever child has the most samples.