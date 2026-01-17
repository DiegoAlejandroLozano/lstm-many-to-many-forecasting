# 🧠 Forecasting Multivariado con LSTM  
### Predicción Multi-Horizonte de Consumo Eléctrico (Resolución 15 minutos)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-LSTM-orange?style=flat)
![License](https://img.shields.io/badge/License-MIT-green)

Este proyecto implementa un **pipeline profesional, modular y reproducible** para la **predicción multivariada del consumo eléctrico**, utilizando una arquitectura **LSTM many-to-many** optimizada para *forecasting multi-horizonte* en intervalos de **15 minutos**.

Incluye:

- Preprocesamiento avanzado para series temporales  
- Ventanas deslizantes multi-horizonte (multi-step ahead)  
- LSTM multicapas con inicialización de pesos profesional  
- Entrenamiento robusto con validación y checkpointing  
- Evaluación por horizonte y global en **escala original**  
- Transformación logarítmica reversible `log1p → scaler → inverse_transform → expm1`  

El repositorio está pensado para **mostrar habilidades profesionales** en:

- Data Science  
- Machine Learning Engineering  
- Modelado de series temporales  
- Deep Learning aplicado a problemas reales (energía)

---

# 🚀 Características del Proyecto

### 🔹 Pipeline completo para forecasting multi-horizonte

- Arquitectura **LSTM many-to-many truncada**:
  - Entrada: ventana de longitud fija `seq_len`
  - Salida: `horizon` pasos futuros  
- Predicción simultánea de **múltiples usuarios** (multi-target)  
- Ventanas deslizantes `(seq_len → horizon)` generadas de forma eficiente  
- División temporal estricta: `train / valid / test` sin *data leakage*  
- Compatibilidad con GPU (`cuda`) o CPU

### 🔹 Preprocesamiento y escalado

- Transformación opcional `log1p` sobre las series objetivo (usuarios)  
- Escalado con `StandardScaler` **solo en las columnas objetivo**  
- Inversión de escala coherente en la evaluación:
  - `y_scaled → inverse_transform → expm1 → y_real`  

### 🔹 Evaluación avanzada

- **Métricas por horizonte**:
  - MAE, MSE, RMSE para cada `t+1`, `t+2`, …  
  - Promedio sobre todas las series objetivo por cada horizonte  
- **Métrica global agregada**:
  - Aplanando todos los horizontes y todas las series objetivo  
  - Permite medir el rendimiento global del modelo  
- Evaluación en **escala original**, no en datos escalados

### 🔹 Diseño modular (estilo producción)

- `DataLoading` — preprocesamiento, splits temporales, ventanas y DataLoaders  
- `Modelo` — arquitectura LSTM multihorizonte (many-to-many truncada)  
- `train_regression` — bucle de entrenamiento con validación y checkpoint  
- `Evaluator` — métricas profesionales + formateo de resultados  

---

# 📊 Resultados del Modelo (Test – escala original)

Configuración principal de experimento:

- `seq_len = 12` (ventana de 12 pasos → 3 horas en intervalos de 15 minutos)  
- `horizon = 2` (predicción a 15 y 30 minutos)  
- `NUM_USUARIOS = 3` (tres series objetivo de consumo)  
- LSTM:
  - `hidden_size = 64`
  - `num_layers = 2`
  - `bidirectional = False`
  - `dropout = 0.0`
- Optimizador: **Adam**, `lr = 1e-3`  
- Función de pérdida: **MSELoss**  
- Gradiente con *clipping* (`max_norm = 1.0`)

---

### ⭐ Métricas Promedio por Horizonte

| Métrica | Horizonte 1 *(t+1)* | Horizonte 2 *(t+2)* |
|--------|---------------------|---------------------|
| **MAE** | 1.0812 | 1.0697 |
| **MSE** | 4.3926 | 4.3104 |
| **RMSE** | 1.8201 | 1.8055 |

---

### ⭐ Métricas Globales

| Métrica | Global |
|--------|--------|
| **MAE** | 1.0755 |
| **MSE** | 4.3515 |
| **RMSE** | 2.0860 |

---

# 📌 Interpretación

- El **error absoluto promedio** es ≈ **1.07 unidades**.  
  Como el dataset ElectricityLoadDiagrams está medido en **kW** (potencia instantánea),  
  las predicciones del modelo también se interpretan en **kW**.

- Un error de ≈1 kW es **bajo en términos relativos** para series con valores que pueden oscilar entre 0–70 kW (usuarios domésticos) o más para usuarios comerciales.

- El desempeño se mantiene **estable entre horizontes de 15 y 30 minutos**, sin degradación significativa.

- Un RMSE global ≈ **2.086 kW** es competitivo para forecasting de corto plazo, considerando:
  - Serie multivariada  
  - Predicción simultánea de 3 usuarios  
  - Forecasting de dos pasos futuros  
  - Transformación logarítmica reversible 

**Aplicaciones típicas:**

- Gestión y planificación energética en el corto plazo  
- Balanceo de carga en redes de distribución  
- Sistemas de alerta temprana por picos de consumo  
- Optimización de microgrids y recursos distribuidos  
- Soporte a la toma de decisiones en empresas eléctricas

---

# 🧩 Arquitectura del Proyecto

Estructura principal del repositorio:

```bash
.
├── main.py                     # Script principal: orquesta el pipeline completo
├── README.md
│
├── src/
│   ├── data_loading.py         # Clase DataLoading: preprocesado, ventanas, DataLoaders
│   ├── model.py                # Clase Modelo: arquitectura LSTM many-to-many truncada
│   ├── train.py                # Función train_regression: entrenamiento + validación
│   └── evaluate.py             # Clase Evaluator: métricas por horizonte y globales
│
├── data/
│   └── 02_datos_procesados/
│       └── datos.csv           # Dataset procesado de consumo eléctrico (no incluido)
│
├── models/
│   └── best_model.pth          # Mejor modelo guardado (checkpoint)
│
└── reports/
    └── curvas_entrenamiento.png  # Curva Train/Val RMSE por época
