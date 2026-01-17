#src/train.py
"""
Módulo de entrenamiento para modelos de regresión con redes neuronales recurrentes (RNN)
utilizando PyTorch.

Este archivo proporciona una función principal de entrenamiento (`train_regression`)
y una función auxiliar para evaluación por época (`_eval_epoch`), permitiendo llevar
a cabo un ciclo de entrenamiento completo con validación, clipping de gradientes,
selección de optimizador y almacenamiento del mejor modelo según el RMSE en validación.

La estructura está diseñada para ser utilizada con modelos de predicción de series
temporales, incluyendo arquitecturas LSTM many-to-many o many-to-one, aunque funciona
con cualquier modelo compatible con entrada y salida tipo tensores.
"""

import math
from pathlib import Path
from typing  import Dict, List, Tuple, Optional

import torch
import torch.nn    as nn
import torch.optim as optim
from torch.utils.data import DataLoader

@torch.no_grad()
def _eval_epoch(model:nn.Module, dl_val:DataLoader, device:str, fn_loss:nn.Module) -> Tuple[float, float]:
    """
    Evalúa el modelo en un único epoch utilizando el conjunto de validación.

    ### Funcionalidad:
    - Cambia el modelo a modo `eval()` para desactivar dropout y batchnorm.
    - Calcula la pérdida total sobre todo el DataLoader de validación.
    - Acumula la pérdida ponderada por tamaño de batch.
    - Retorna MSE y RMSE finales del epoch.

    ### Parámetros:
    - **model** (*nn.Module*): Modelo a evaluar.
    - **dl_val** (*DataLoader*): DataLoader del conjunto de validación.
    - **device** (*str*): Dispositivo a usar ("cpu" o "cuda").
    - **fn_loss** (*nn.Module*): Función de pérdida, típicamente `nn.MSELoss()`.

    ### Retorno:
    - (*Tuple[float, float]*):
        - **mse**: Error cuadrático medio promedio.
        - **rmse**: Raíz del error cuadrático medio.

    ### Notas:
    - Se usa el decorador `@torch.no_grad()` para desactivar el cálculo de gradientes.
    - La pérdida se pondera por el tamaño del batch para obtener un promedio exacto.
    """
    model.eval()
    loss_sum, count = 0.0, 0
    for X, Y in dl_val:
        X, Y      = X.float().to(device), Y.float().to(device)
        Y_pred    = model(X)
        loss      = fn_loss(Y_pred, Y)
        bs        = X.size(0)
        loss_sum += loss.item() * bs
        count    += bs
    mse  = loss_sum / max(1, count)
    rmse = math.sqrt(mse)
    return mse, rmse

def train_regression(
    modelo        : nn.Module,
    train_dl      : DataLoader,
    val_dl        : DataLoader,
    num_epochs    : int   = 50,
    learning_rate : float = 1e-3,
    device        : str   = "cpu",
    print_every   : int   = 1,
    model_path    : str   = "models/best_model.pth",
    *,
    grad_clip      : Optional[float] = 1.0,
    optimizer_name : str             = "adam",
    weight_decay   : float           = 0.0
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Ejecuta el ciclo completo de entrenamiento para un modelo de regresión
    basado en series temporales.

    ### Funcionalidad:
    - Configura el optimizador (Adam o SGD).
    - Entrena durante múltiples épocas con forward/backward y actualización de pesos.
    - Aplica clipping de gradientes para mejorar estabilidad.
    - Evalúa en el conjunto de validación al final de cada época.
    - Guarda automáticamente el mejor modelo según RMSE de validación.
    - Retorna la historia de entrenamiento (MSE/RMSE por época).

    ### Parámetros:
    - **modelo** (*nn.Module*): Modelo PyTorch a entrenar.
    - **train_dl** (*DataLoader*): DataLoader del conjunto de entrenamiento.
    - **val_dl** (*DataLoader*): DataLoader del conjunto de validación.
    - **num_epochs** (*int*): Número total de épocas de entrenamiento.
    - **learning_rate** (*float*): Tasa de aprendizaje.
    - **device** (*str*): Dispositivo ("cpu" o "cuda").
    - **print_every** (*int*): Frecuencia de impresión de métricas (cada N epochs).
    - **model_path** (*str*): Ruta donde se guardará el mejor modelo.
    - **grad_clip** (*float*, opcional): Norm límite para el clipping de gradientes.
    - **optimizer_name** (*str*): "adam" o "sgd".
    - **weight_decay** (*float*): L2 regularization.

    ### Retorno:
    - (*Tuple[nn.Module, Dict[str, List[float]]]*):
        - **modelo**: Modelo final (con mejor estado cargado).
        - **historia**: Histórico con listas de:
            - train_mse  
            - train_rmse  
            - val_mse  
            - val_rmse  

    ### Notas:
    - La pérdida utilizada es MSE (`nn.MSELoss()`), adecuada para regresión multivariada.
    - El modelo se guarda solo cuando mejora el RMSE de validación.
    - El clipping de gradientes ayuda a estabilizar el entrenamiento de LSTM.
    - Funciona con arquitecturas many-to-one, many-to-many o seq2seq, siempre que la
      salida del modelo sea compatible con MSELoss.
    """    
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    modelo.to(device)
    fn_loss = nn.MSELoss() # Función de pérdida

    if optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(modelo.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(modelo.parameters(), lr=learning_rate, weight_decay=weight_decay)

    historia = {
        "train_mse"  : [],
        "train_rmse" : [],
        "val_mse"    : [],
        "val_rmse"   : []
    }
    best_val_rmse, best_state = float("inf"), None

    for epoch in range(1, num_epochs+1):
        # --- train ---
        modelo.train()
        loss_sum, count = 0.0, 0
        for X, Y in train_dl:
            X, Y = X.float().to(device), Y.float().to(device)
            optimizer.zero_grad()
            Y_pred = modelo(X)
            loss   = fn_loss(Y_pred, Y)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=grad_clip)
            optimizer.step()

            bs = X.size(0)
            loss_sum += loss.item() * bs
            count    += bs
        train_mse  = loss_sum / max(1, count)
        train_rmse = math.sqrt(train_mse)

        # --- val ---
        val_mse, val_rmse = _eval_epoch(
            model   = modelo,
            dl_val  = val_dl,
            device  = device,
            fn_loss = fn_loss
        )

        historia["train_mse"].append(train_mse)
        historia["train_rmse"].append(train_rmse)
        historia["val_mse"].append(val_mse)
        historia["val_rmse"].append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state    = {k:v.detach().cpu() for k,v in modelo.state_dict().items()}
            torch.save(best_state, model_path)

        if (epoch%print_every==0) or epoch in (1, num_epochs):
            print(
                f"Ep {epoch:03d}/{num_epochs:03d} | "
                f"Train RMSE={train_rmse:.4f} | Val RMSE={val_rmse:.4f}"
            )
    
    if best_state is not None:
        modelo.load_state_dict(best_state)

    return modelo, historia