# src/model.py
# -*- coding; utf-8 -*-

import torch
import torch.nn as nn
from typing import Tuple

class Modelo(nn.Module):
    """
    Modelo LSTM para predicción multivariada con estructura many-to-many truncada.

    ### Funcionalidad:
    - **Procesa secuencias temporales** de longitud fija (`seq_len`) con múltiples variables predictoras.
    - **Predice múltiples pasos futuros** (`horizon`) para varias variables de salida (`target_dim`).
    - **Utiliza una arquitectura LSTM** configurable:
        - número de capas (`num_layers`)
        - tamaño del estado oculto (`hidden_size`)
        - opción de usar bidireccionalidad
        - dropout entre capas

    ### Parámetros:
    - **input_shape** (*Tuple[int, int]*):  
      Tamaño de la entrada al modelo, en formato `(seq_len, n_features)`.
    - **horizon** (*int*):  
      Número de pasos futuros que el modelo debe predecir.
    - **target_dim** (*int*):  
      Número de variables a predecir en cada paso del horizonte.
    - **hidden_size** (*int*):  
      Dimensión del estado oculto de cada capa LSTM.
    - **num_layers** (*int*):  
      Número de capas LSTM apiladas.
    - **dropout** (*float*):  
      Tasa de dropout entre capas (solo si `num_layers > 1`).
    - **bidirectional** (*bool*):  
      Si `True`, utiliza LSTM bidireccional.

    ### Retorno del modelo:
    - (*Tensor*): Tensores de salida de tamaño `(batch_size, horizon, target_dim)`.

    ### Excepciones:
    - **ValueError**:  
      Si `horizon <= 0` o `horizon > seq_len`.
    """

    def __init__(
        self,
        input_shape   : Tuple[int, int], # (seq_len, n_features)
        horizon       : int,
        target_dim    : int,
        hidden_size   : int,
        num_layers    : int,
        dropout       : float = 0.0,
        bidirectional : bool  = False
    ):
        """
        Inicializa el modelo LSTM para predicción multivariada con horizonte múltiple.

        ### Funcionalidad:
        - Configura la arquitectura LSTM según los hiperparámetros proporcionados.
        - Valida que el `horizon` sea consistente con el tamaño de la secuencia de entrada.
        - Define:
            - la capa LSTM principal
            - el tamaño real del estado oculto (bidireccional o no)
            - la capa lineal de salida para proyectar cada estado oculto en `target_dim`
        - Al final invoca la inicialización personalizada de parámetros.

        ### Parámetros:
        - **input_shape** (*Tuple[int, int]*):  
          Forma de las entradas al modelo `(seq_len, n_features)`.
        - **horizon** (*int*):  
          Número de pasos futuros que el modelo debe predecir.  
          Debe cumplir: `1 <= horizon <= seq_len`.
        - **target_dim** (*int*):  
          Número de variables objetivo que se predicen en cada paso del horizonte.
        - **hidden_size** (*int*):  
          Tamaño del estado oculto por capa del LSTM.
        - **num_layers** (*int*):  
          Número de capas LSTM apiladas.
        - **dropout** (*float*):  
          Dropout aplicado entre capas LSTM (solo si `num_layers > 1`).
        - **bidirectional** (*bool*):  
          Si es `True`, activa un LSTM bidireccional, duplicando el `state_dim`.

        ### Excepciones:
        - **ValueError**:  
          Se lanza cuando `horizon <= 0` o cuando `horizon > seq_len`.

        """
        super().__init__()

        if horizon <= 0:
            raise ValueError("Horizon debe ser mayor que 0.")
        elif horizon > input_shape[0]:
            raise ValueError(f"Horizon={horizon} no puede ser mayor que seq_len={input_shape[0]}")
        
        _, n_feature       = input_shape
        self.horizon       = horizon
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size    = n_feature,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = self.bidirectional
        )

        # Tamaño del vector de estados ocultos
        state_dim = hidden_size * (2 if self.bidirectional else 1)

        # Capa de salida de la predicción
        self.head = nn.Linear(in_features=state_dim, out_features=target_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Inicializa los parámetros del LSTM y de la capa lineal de salida.

        ### Funcionalidad:
        - Aplica **Xavier Uniform** para los pesos de entrada del LSTM.
        - Aplica **Orthogonal** para los pesos recurrentes del LSTM.
        - Inicializa:
            - **biases en cero**
            - **forget gate bias en 1**, lo cual mejora la estabilidad del entrenamiento.
            - En modo bidireccional, también inicializa el forget gate de la dirección inversa.
        - La capa lineal (`self.head`):
            - pesos inicializados con **Xavier Uniform**
            - bias inicializado en cero.

        ### Retorno:
        - (*None*)
        """
        # Inicialización recomendada para LSTM/Linear (Xavier/Orthogonal)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                hidden = self.hidden_size
                param.data[hidden:2*hidden] = 1.0

                # Forget gate (backward), si existe
                if self.bidirectional:
                    param.data[5*hidden : 6*hidden] = 1.0

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Ejecuta el paso forward del modelo sobre un batch de secuencias.

        ### Entrada:
        - **x** (*Tensor*):  
          Tensor de entrada con forma `(batch_size, seq_len, n_features)`.

        ### Proceso:
        - La secuencia completa se envía al LSTM.
        - Del tensor de salidas del LSTM (`out`) se extraen únicamente
          los últimos `horizon` estados ocultos.
        - Cada uno de esos estados se pasa por la capa lineal para producir
          las predicciones finales.

        ### Salida:
        - (*Tensor*):  
          Predicciones del modelo con forma `(batch_size, horizon, target_dim)`.

        ### Notas:
        - Implementa una arquitectura **many-to-many truncada**:  
          usa toda la secuencia como entrada, pero solo produce predicciones
          en los últimos `horizon` pasos.
        """
        out, _        = self.lstm(x)              # out: (B, seq_len, state_dim)
        last_horizon  = out[:, -self.horizon:, :] # último "horizon" pasos
        y_pred        = self.head(last_horizon)   # (B, horizon, target_dim)
        return y_pred

# ==== Prueba ====
def main():
    # Número de pasos de tiempo y de características
    seq_len    = 3
    horizon    = 2
    n_features = 5

    # Crear el modelo
    modelo = Modelo(
        input_shape = (seq_len, n_features),
        horizon     = horizon,
        target_dim  = 3,
        hidden_size = 64,
        num_layers  = 2,
        dropout     = 0.0
    )

    # Simular un batch con cuatro muestras
    X = torch.randn(4, seq_len, n_features)

    # Pasar por el modelo
    y_pred = modelo(X)

    # Mostrar forma y valores
    print(f"Entrada X shape: {X.shape}")
    print(f"Salida y_pred shape: {y_pred.shape}")
    print(f"Predicciones:\n{y_pred}")

if __name__ == "__main__":
    main()