# Laboratorio 8 – Máquinas Vectoriales de Soporte
## SmartStay Advisors: Clasificación y Predicción de Precios en Airbnb

**CC3074 – Minería de Datos | Universidad del Valle de Guatemala | Semestre I – 2026**

---

## Preparación del conjunto de datos para modelos de márgenes

El conjunto de datos original de Airbnb contiene 171,748 registros y 80 variables. Tras eliminar filas con valores nulos en variables críticas (`review_scores_rating`, `bathrooms`, `bedrooms`, `beds`, `reviews_per_month`) y filtrar precios fuera del rango \$1–\$1,000, se obtienen 60,619 registros utilizables.

De ese universo se extrae una muestra estratificada de 15,000 observaciones con seed=42 y partición 70/30 (10,500 entrenamiento / 4,500 prueba), idéntica a la utilizada en todas las entregas anteriores para garantizar comparaciones válidas.

Las máquinas vectoriales de soporte requieren dos transformaciones que no son opcionales sino estructurales al algoritmo:

1. **Escalado obligatorio (StandardScaler):** SVM maximiza el margen entre clases midiendo distancias euclidianas en el espacio de características. Sin escalar, variables de gran magnitud como `price_num` ($8–$1,000) dominarían completamente sobre variables acotadas como `bathrooms` (0–10), haciendo que el hiperplano óptimo ignore estas últimas. Con `StandardScaler` cada variable queda con media=0 y desviación estándar=1.

2. **Eliminación de valores nulos:** A diferencia de árboles o Random Forest, SVM no tiene mecanismo interno para manejar NaN. Cualquier valor faltante interrumpe el cálculo del kernel.

Las variables categóricas (`room_type`, `host_is_superhost`, `instant_bookable`) se convierten a indicadores binarios (0/1) antes del escalado.

La variable respuesta para clasificación se construye mediante percentiles P33 y P66 del precio:

| Categoría | Rango de precio | Registros en muestra |
|-----------|----------------|----------------------|
| `barata`  | ≤ \$133        | ~33.1% |
| `media`   | \$133 – \$237  | ~33.2% |
| `cara`    | > \$237        | ~33.8% |

La distribución balanceada de tres clases (~33% cada una) es deliberada: garantiza que el modelo no tenga sesgo hacia ninguna categoría y que las métricas de accuracy sean comparables directamente entre clases.

---

## Exploración del espacio de hiperparámetros en SVM de clasificación

Se entrenaron 27 modelos SVM de clasificación combinando tres kernels (lineal, RBF, polinomial), tres valores de C (0.1, 1, 10) y tres valores de gamma ('scale', 0.001, 0.01). Los resultados completos, ordenados por accuracy en test:

| Modelo | Train Acc | Test Acc | Overfit | Tiempo (s) |
|--------|-----------|----------|---------|------------|
| SVM_rbf_C10_Gscale | 0.7834 | **0.6762** | 0.1072 | 18.0 |
| SVM_rbf_C1_Gscale | 0.7054 | 0.6658 | 0.0397 | 16.4 |
| SVM_rbf_C10_G0_01 | 0.6923 | 0.6656 | 0.0267 | 17.3 |
| SVM_poly_C10_Gscale | 0.7419 | 0.6562 | 0.0857 | 26.3 |
| SVM_rbf_C1_G0_01 | 0.6583 | 0.6487 | 0.0096 | 15.6 |
| SVM_poly_C1_Gscale | 0.6799 | 0.6389 | 0.0410 | 16.8 |
| SVM_linear_C1_* | 0.6322 | 0.6338 | -0.0016 | ~24 |
| SVM_linear_C10_* | 0.6326 | 0.6336 | -0.0010 | ~78 |
| SVM_linear_C0.1_* | 0.6307 | 0.6316 | -0.0009 | ~14 |
| SVM_poly_C0.1_Gscale | 0.5648 | 0.5411 | 0.0237 | 17.8 |
| SVM_poly_C*_G0_001 | ~0.34 | ~0.32 | ~0.02 | ~20 |

**Kernel RBF** es consistentemente el más efectivo. Con gamma='scale' y C=10 alcanza 67.62% de accuracy en test. El kernel lineal estabiliza alrededor de 63%, independientemente de C, porque el problema no es linealmente separable. El kernel polinomial con gamma pequeño (0.001) colapsa a ~32%, indicando que esa combinación no genera ningún margen útil.

El efecto de C es observable: al subir de 0.1 a 10 en RBF con gamma='scale', el accuracy en test sube de 63.6% a 67.6%, pero el overfitting también sube de 1.0% a 10.7%. C controla la tolerancia al error de entrenamiento; valores altos permiten que el modelo memorice mejor el training set a costa de mayor varianza.

---

## Diagnóstico por clase: dónde se equivoca el mejor modelo SVM

El modelo `SVM_rbf_C10_Gscale` muestra el siguiente reporte por clase en el conjunto de prueba (4,500 registros):

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| barata | 0.740 | 0.732 | **0.736** | 1,438 |
| media | 0.753 | 0.702 | **0.727** | 1,578 |
| cara | 0.549 | 0.595 | **0.571** | 1,484 |
| **macro avg** | 0.681 | 0.676 | 0.678 | 4,500 |

La clase `cara` (precio > \$237) concentra los mayores errores: F1 de 0.571, 18 puntos por debajo de las otras dos. Esto tiene una explicación estructural: las propiedades de precio alto comparten muchas características con las de precio medio (mismas calificaciones, similar capacidad), y los factores que realmente diferencian una propiedad cara (ubicación premium, amenidades especiales) no están completamente capturados en las 21 variables disponibles.

Los errores barata↔cara (salto de dos categorías) son los más graves para el negocio de SmartStay, ya que implican recomendaciones significativamente fuera del presupuesto del cliente.

---

## Matrices de confusión: SVM vs. Random Forest

Las matrices de confusión del mejor SVM y el mejor baseline (Random Forest) revelan patrones distintos de error. Ambas matrices están guardadas en `confusion_matrices.png`.

El SVM distribuye los errores de forma más equilibrada entre clases adyacentes (barata↔media, media↔cara), mientras que Random Forest, aunque con mayor accuracy global, comete errores más concentrados en la clase `media`, que es inherentemente la más difícil de separar al estar en el centro del rango de precios.

---

## Comparación de todos los modelos de clasificación

Evaluando todos los algoritmos sobre el mismo conjunto de prueba (4,500 registros, mismos datos de entrenamiento):

| Modelo | Train Acc | Test Acc | Overfitting | Tiempo |
|--------|-----------|----------|-------------|--------|
| **RandomForest** | 1.0000 | **0.7104** | 0.2896 | 1.5s |
| SVM_rbf_C10_Gscale | 0.7834 | 0.6762 | 0.1072 | 18.0s |
| SVM_rbf_C1_Gscale | 0.7054 | 0.6658 | 0.0397 | 16.4s |
| SVM_rbf_C10_G0_01 | 0.6923 | 0.6656 | 0.0267 | 17.3s |
| SVM_linear_C1 | 0.6322 | 0.6338 | -0.0016 | 24s |
| LogisticRegression | 0.6267 | 0.6336 | -0.0069 | 0.2s |
| KNN_10 | 0.6964 | 0.6322 | 0.0642 | <0.1s |
| DecisionTree | 1.0000 | 0.6202 | 0.3798 | 0.4s |
| KNN_5 | 0.7341 | 0.6124 | 0.1217 | <0.1s |
| GaussianNB | 0.5830 | 0.5840 | -0.0010 | <0.1s |

**Random Forest obtiene el mayor accuracy en test (71.04%)** pero a un costo de ajuste extremo: train accuracy de 100% con 28.96 puntos de diferencia respecto al test. Es el modelo más sobreajustado de todos.

**Decision Tree sin poda** replica el mismo patrón pero con peor generalización (62.02% en test), siendo el más sobreajustado de todos con 37.98 puntos de brecha.

**El mejor SVM (rbf, C=10, gamma=scale)** obtiene 67.62% en test con solo 10.72 puntos de overfitting — significativamente más estable que RF y DT.

**Logistic Regression y SVM lineal** convergen a ~63.4% de accuracy. Es coherente: ambos buscan un hiperplano de separación lineal en el espacio de features, por lo que tienen capacidad expresiva similar.

**GaussianNB (58.4%)** es el de menor rendimiento, confirmando que la distribución gaussiana independiente por feature no captura bien las relaciones multivariadas de este conjunto de datos.

---

## Análisis de sobreajuste en clasificación

Para determinar si un modelo está sobreajustado en clasificación se comparan:
- **Train Accuracy vs Test Accuracy:** una brecha > 10 puntos porcentuales es señal de sobreajuste.
- **F1-score por clase:** si una clase cae drásticamente en test respecto al train, el modelo memorizó patrones específicos de esa clase.
- **Accuracy NO es suficiente:** un modelo puede tener accuracy aceptable pero F1 muy bajo en clases minoritarias o difíciles.

| Modelo | Diagnóstico |
|--------|-------------|
| DecisionTree | Sobreajuste severo (brecha 37.98%) — memoriza el training |
| RandomForest | Sobreajuste severo (brecha 28.96%) — ensemble amplifica memorización |
| KNN_5 | Sobreajuste moderado (brecha 12.17%) |
| SVM_rbf_C10 | Sobreajuste leve (brecha 10.72%) |
| KNN_10 | Ajuste razonable (brecha 6.42%) |
| SVM_rbf_C1 | Bien ajustado (brecha 3.97%) |
| SVM_linear | Sin sobreajuste (brecha < 0.2%) |
| LogisticRegression | Sin sobreajuste (brecha < 0.7%) |
| GaussianNB | Sin sobreajuste (brecha < 0.1%) |

Para manejar el sobreajuste:
- En **árboles de decisión:** podar (max_depth, min_samples_split).
- En **Random Forest:** reducir n_estimators o aplicar max_features más restrictivo.
- En **SVM:** bajar C (mayor margen, más tolerancia al error de training) o reducir gamma (kernel más suave).
- En **KNN:** aumentar k (suaviza la frontera de decisión).

---

## Regresión SVR: estimación directa del precio

Se entrenaron 9 modelos SVR (3 kernels × 3 valores de C con gamma='scale') más un modelo tuneado con GridSearchCV (36 combinaciones, 3-fold CV).

| Modelo | R² Test | RMSE Test | Overfitting |
|--------|---------|-----------|-------------|
| **SVR_tuned (rbf, C=50, ε=20)** | **0.4935** | **\$123.94** | 0.0485 |
| SVR_rbf_C10 | 0.4283 | \$131.67 | — |
| SVR_linear_C10 | 0.3751 | \$137.66 | — |
| SVR_rbf_C1 | 0.2883 | \$146.92 | — |
| SVR_linear_C1 | 0.3722 | \$137.99 | — |
| SVR_poly_C10 | 0.2858 | \$147.17 | — |
| SVR_poly_C1 | 0.1499 | \$160.57 | — |
| SVR_rbf_C0.1 | 0.0348 | \$171.10 | — |
| SVR_poly_C0.1 | 0.0159 | \$172.76 | — |

El GridSearchCV encontró como mejores hiperparámetros: `kernel=rbf`, `C=50`, `epsilon=20`, `gamma=scale`. El parámetro epsilon (tubo de insensibilidad) grande (20) indica que el modelo tolera errores de hasta \$20 sin penalización, lo que favorece la generalización en un conjunto con alta dispersión de precios.

El SVR tuneado explica el 49.35% de la varianza del precio con un error promedio de \$123.94 (RMSE) y \$76.93 (MAE). El overfitting es mínimo (4.85 puntos entre train R²=0.5420 y test R²=0.4935), lo que lo convierte en el modelo de regresión más estable.

---

## Comparación de modelos de regresión: SVR vs. entregas anteriores

| Modelo | R² Test | RMSE Test | MAE Test | Overfitting |
|--------|---------|-----------|----------|-------------|
| **RandomForest** | **0.5977** | **\$110.45** | **\$70.08** | 0.3486 |
| SVR_tuned (rbf) | 0.4935 | \$123.94 | \$76.93 | **0.0485** |
| KNN_5 | 0.4419 | \$130.10 | — | — |
| LinearRegression | 0.4191 | \$132.73 | — | — |
| DecisionTree | 0.1908 | \$156.66 | — | 0.8092 |

**Random Forest es el mejor en métricas puras** (R²=0.5977, RMSE=\$110.45) pero con sobreajuste severo: train R²=0.9464 vs test R²=0.5977, una brecha de 34.86 puntos. En producción, este comportamiento implica que el modelo funciona muy bien en datos que ya vio pero puede degradarse significativamente en propiedades con características distintas al training set.

**SVR tuneado es el más generalizable** (overfitting de solo 4.85%) con métricas de prueba razonables. Para SmartStay, donde constantemente se analizan propiedades nuevas que el modelo no ha visto, la estabilidad del SVR es una ventaja operativa real.

**Regresión Lineal** (R²=0.4191) establece que las relaciones lineales explican el 41.9% de la varianza. El SVR mejora esto en 7.4 puntos usando un kernel RBF que captura relaciones no lineales entre precio y características.

**Decision Tree de regresión** sin poda es el peor modelo: R²=0.1908 en test con overfitting extremo (train R²≈1.0), confirmando que memorizó el training sin aprender ningún patrón generalizable.

---

## Síntesis comparativa y recomendaciones para SmartStay

**Para clasificar precios (barata / media / cara):**

El mejor modelo en accuracy absoluto es **Random Forest (71.04%)**, pero su sobreajuste severo (37.98 puntos de brecha en train-test) lo hace menos confiable en propiedades nuevas. El **SVM con kernel RBF, C=10 y gamma=scale (67.62%)** ofrece el mejor balance entre rendimiento y generalización, con solo 10.72 puntos de overfitting. Si la prioridad es estabilidad en producción, SVM_rbf_C1_Gscale (66.58%, overfitting 3.97%) es incluso más robusto.

La clase `cara` es la más difícil de predecir correctamente (F1=0.571 en el mejor SVM). Las variables disponibles no capturan completamente qué hace que una propiedad sea premium. Para mejorar esta clase sería necesario incorporar variables de ubicación más finas (barrio específico, distancia a puntos de interés) o texto de las descripciones.

**Para predecir el precio numérico:**

**Random Forest** es el más preciso (RMSE=\$110.45) pero con severo sobreajuste. El **SVR tuneado** (RMSE=\$123.94, overfitting=4.85%) es el más estable. Para una plataforma que analiza propiedades continuamente, se recomienda SVR tuneado como modelo de producción, con reentrenamiento periódico.

Todos los modelos muestran mayor error en propiedades de precio alto (>$500), donde la varianza intrínseca es mayor y los datos son más escasos. Esto es esperado dado que los precios siguen una distribución sesgada a la derecha.

---

*Resultados obtenidos sobre 15,000 muestras, seed=42, partición 70/30, con StandardScaler. Todos los modelos entrenados sobre el mismo conjunto de datos.*
