# main.py

"""
Script principal del proyecto de predicción multihorizonte con redes LSTM.

Este módulo coordina el flujo completo:
1. Configuración de reproducibilidad.
2. Carga y preprocesamiento de datos.
3. Construcción del modelo.
4. Entrenamiento supervisado con validación.
5. Visualización y guardado de curvas de entrenamiento.
6. Carga del mejor modelo encontrado.
7. Evaluación final en el conjunto de test (escala original).

El archivo reúne todos los componentes del pipeline:
- DataLoading  → preparación de datos y generación de secuencias.
- Modelo       → arquitectura LSTM many-to-many.
- train_regression → ciclo completo de entrenamiento.
- Evaluator    → métricas por horizonte y globales.

Su propósito es ejecutar end-to-end la experimentación principal del proyecto.
"""

import os, torch, random

from torch      import nn
from matplotlib import pyplot as plt

import numpy as np

from src.data_loading import DataLoading
from src.model        import Modelo
from src.train        import train_regression
from src.evaluate     import Evaluator

# ---------- Reproducibilidad ----------
def set_seed(seed:int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True)
# -------------------------------------

def main():
    """
    Ejecuta todo el pipeline de entrenamiento y evaluación del modelo LSTM
    para predicción multi-horizonte de consumo eléctrico.

    ### Pasos principales:

    1) **Reproducibilidad y selección de dispositivo**
       - Configura todas las semillas.
       - Selecciona automáticamente GPU si está disponible.

    2) **Carga y preparación de datos**
       - Usa `DataLoading.build()` para generar:
         - DataLoaders de train/val/test,
         - Forma de entrada del modelo (`input_shape`),
         - Escalador y configuraciones de log-transform.

    3) **Construcción del modelo**
       - Instancia la arquitectura LSTM definida en `src.model.Modelo`.

    4) **Entrenamiento supervisado**
       - Ejecuta `train_regression()` con validación por época.
       - Guarda el mejor modelo basado en RMSE de validación.
       - Registra el histórico de métricas.

    5) **Visualización de curvas de entrenamiento**
       - Genera y guarda una figura con RMSE de entrenamiento y validación.

    6) **Carga del mejor modelo**
       - Recupera el estado guardado durante el entrenamiento.

    7) **Evaluación final en TEST**
       - Usa el módulo `Evaluator` para calcular:
         - Métricas por horizonte (por-step),
         - Métrica global agregada,
         - Métricas en **escala original** (desescaladas).

    ### Notas:
    - Este módulo está diseñado como punto de entrada único del proyecto.
    - Su organización permite ejecutar fácilmente experimentos controlados.
    - El pipeline es compatible con múltiples variantes del modelo
      y diferentes configuraciones de secuencias/horizonte.
    """

    set_seed(seed=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Datos
    dm = DataLoading(
        data_path  = "data/02_datos_procesados/datos.csv",
        seq_len    = 12,
        batch_size = 64,
        horizon    = 2,
        use_log    = True
    )
    train_dl, val_dl, test_dl, input_shape = dm.build()

    # 2) Modelo
    modelo = Modelo(
        input_shape = input_shape,
        horizon     = dm.horizon,
        target_dim  = DataLoading.NUM_USUARIOS,
        hidden_size = 64,
        num_layers  = 2,
    )

    # 3) Entrenamiento
    model_path = "models/best_model.pth"
    modelo_entrenado, historia = train_regression(
        modelo        = modelo,
        train_dl      = train_dl,
        val_dl        = val_dl,
        num_epochs    = 50,
        learning_rate = 1e-3,
        device        = device,
        model_path    = model_path,
        grad_clip     = 1.0
    )

    # 4) Curvas
    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(9,4))
    plt.plot(historia["train_rmse"], label="Train RMSE")
    plt.plot(historia["val_rmse"], label="Val RMSE")
    plt.xlabel("Época")
    plt.ylabel("RMSE (Datos escalados)")
    plt.title("Curvas de entrenamiento")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/curvas_entrenamiento.png", dpi=200)

    # 5) Cargar el mejor modelo (opcional, ya se hace en entrenamiento)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        modelo_entrenado.load_state_dict(state)

    # 6) Evaluar en TEST (escala original)
    evaluator = Evaluator(
        modelo  = modelo_entrenado, 
        horizon = dm.horizon,
        scaler  = dm.scaler, 
        use_log = dm.use_log, 
        device  = device
    )
    evaluator.evaluar(dl_test=test_dl)

if __name__ == "__main__":
    main()