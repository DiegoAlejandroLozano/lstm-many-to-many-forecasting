# src/data_loading.py

import math

from pathlib import Path
from typing  import Tuple, Optional, List

import numpy  as np
import pandas as pd
import torch

from torch.utils.data      import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class DataLoading:

    """
    Clase encargada de cargar, transformar y preparar un dataset multivariado
    para tareas de series temporales con modelos RNN (LSTM/GRU) en PyTorch.

    ### Funcionalidad:
    - **Lectura y ordenamiento temporal** del archivo CSV.
    - **Selección dinámica de columnas**: variables cíclicas + usuarios MT_xxx.
    - **Transformaciones opcionales logarítmicas** para estabilizar varianza.
    - **Escalado automático** (StandardScaler) solo de columnas objetivo.
    - **División temporal** en conjuntos de entrenamiento, validación y prueba.
    - **Generación de ventanas deslizantes** (X, Y) para forecasting multihorizonte.
    - **Conversión a tensores** y creación de DataLoaders listos para entrenar RNN.

    ### Parámetros:
    - **data_path** (*str*): Ruta del archivo CSV con las variables cíclicas y usuarios.
    - **batch_size** (*int*): Tamaño del batch para los DataLoaders.
    - **seq_len** (*int*): Longitud de la secuencia de entrada (timestep pasados).
    - **horizon** (*int*): Número de pasos futuros a predecir.
    - **valid_ratio** (*float*): Proporción del dataset asignada a validación.
    - **test_ratio** (*float*): Proporción del dataset asignada a prueba.
    - **shuffle_train** (*bool*): Indica si se debe mezclar el dataset de entrenamiento.
    - **num_workers** (*int*): Número de procesos para DataLoader.
    - **use_log** (*bool*): Si True, aplica transformación log1p a las columnas objetivo.

    ### Atributos:
    - **scaler** (*StandardScaler*): Escalador usado para normalizar solo los usuarios.
    - **input_shape** (*Tuple[int, int]*): Forma esperada por el modelo RNN: (seq_len, n_features).
    """

    NUM_USUARIOS = 3
    
    def __init__(
        self, 
        data_path     : str,
        batch_size    : int   = 64,
        seq_len       : int   = 12,
        horizon       : int   = 2,
        valid_ratio   : float = 0.15,
        test_ratio    : float = 0.15,
        shuffle_train : bool  = True,
        num_workers   : int   = 0,
        use_log       : bool  = False
    ):
        """
        Inicializa la clase encargada de la carga, transformación y preparación
        del dataset para modelos de series temporales basados en RNN en PyTorch.

        ### Funcionalidad:
        - Define parámetros globales para lectura, escalado y creación de ventanas.
        - Configura tamaños de secuencia, horizonte y división temporal.
        - Permite activar el uso opcional de logaritmos para estabilizar valores.
        - Prepara la ruta del dataset y los parámetros del DataLoader.
        - Inicializa el escalador (StandardScaler) que se entrenará posteriormente.

        ### Parámetros:
        - **data_path** (*str*): Ruta del archivo CSV procesado.
        - **batch_size** (*int*): Tamaño de cada batch en los DataLoaders.
        - **seq_len** (*int*): Cantidad de pasos pasados usados como entrada.
        - **horizon** (*int*): Número de pasos futuros a predecir por la RNN.
        - **valid_ratio** (*float*): Proporción de datos destinada a validación.
        - **test_ratio** (*float*): Proporción destinada a prueba.
        - **shuffle_train** (*bool*): Si True, mezcla los batches del conjunto de entrenamiento.
        - **num_workers** (*int*): Número de procesos usados por los DataLoaders.
        - **use_log** (*bool*): Si True, aplica transformación `np.log1p` a las columnas de usuarios.

        ### Atributos inicializados:
        - **self.data_path** (*Path*): Ruta del archivo en formato Pathlib.
        - **self.scaler** (*StandardScaler | None*): Inicialmente vacío; entrenado en `build()`.
        - **self.input_shape** (*Tuple[int,int]*): Forma utilizada por el modelo RNN (definida en `build()`).
        """
        self.data_path     = Path(data_path)
        self.batch_size    = batch_size
        self.seq_len       = seq_len
        self.horizon       = horizon
        self.valid_ratio   = valid_ratio
        self.test_ratio    = test_ratio
        self.shuffle_train = shuffle_train
        self.num_workers   = num_workers
        self.use_log       = use_log

        self.scaler : Optional[StandardScaler] = None

    def build(self):
        """
        Construye todo el pipeline de preparación del dataset.

        ### Funcionalidad:
        - Lee el archivo CSV.
        - Selecciona las columnas a utilizar (variables cíclicas + usuarios).
        - Convierte los valores a float32 para compatibilidad con PyTorch.
        - Divide el dataset en train/valid/test respetando el orden temporal.
        - Aplica opcionalmente log1p en las columnas objetivo.
        - Entrena un StandardScaler únicamente sobre usuarios (train).
        - Escala train/valid/test usando el scaler entrenado.
        - Genera ventanas deslizantes (X, Y) para forecasting.
        - Convierte X e Y a tensores PyTorch.
        - Crea DataLoaders listos para usar en entrenamiento de RNN.

        ### Retorno:
        - (**train_dl**): DataLoader de entrenamiento.
        - (**val_dl**): DataLoader de validación.
        - (**test_dl**): DataLoader de prueba.
        - (**input_shape**): Tupla con la forma del tensor de entrada para el modelo.
        """
        
        if not self.data_path.exists():
            raise ValueError(f"No se encontró: {self.data_path.resolve()}")
        
        # Generador reproducible
        g = torch.Generator()
        g.manual_seed(42)

        # Lectura de los datos
        df                               = self._read_csv(path=self.data_path)
        columnas_seleccionadas           = self._build_feature_columns("dia_año_sin", "dia_año_cos", df=df, num_usuarios=self.NUM_USUARIOS)
        series                           = df[columnas_seleccionadas].values.astype(np.float32)
        self.input_shape:Tuple[int, int] = (self.seq_len, series.shape[1]) # (seq_len, n_features)
        tr, va, te                       = self._time_split(series=series, valid_ratio=self.valid_ratio, test_ratio=self.test_ratio)

        if self.use_log:
            tr[:, -self.NUM_USUARIOS:] = np.log1p(tr[:, -self.NUM_USUARIOS:])
            va[:, -self.NUM_USUARIOS:] = np.log1p(va[:, -self.NUM_USUARIOS:])
            te[:, -self.NUM_USUARIOS:] = np.log1p(te[:, -self.NUM_USUARIOS:])

        # Se escalan únicamente los valores de las columnas de los usuarios.
        # Las variables cíclicas no se escalan porque ya fueron transformadas
        # mediante funciones seno y coseno, quedando dentro del círculo unitario. 
        self.scaler = StandardScaler().fit(tr[:, -self.NUM_USUARIOS:])

        tr_scaled = tr.copy()
        va_scaled = va.copy()
        te_scaled = te.copy()

        tr_scaled[:, -self.NUM_USUARIOS:] = self.scaler.transform(tr[:, -self.NUM_USUARIOS:])
        va_scaled[:, -self.NUM_USUARIOS:] = self.scaler.transform(va[:, -self.NUM_USUARIOS:])
        te_scaled[:, -self.NUM_USUARIOS:] = self.scaler.transform(te[:, -self.NUM_USUARIOS:])

        # Ventanas
        X_tr, Y_tr = self._make_windows(arr=tr_scaled, seq_len=self.seq_len, horizon=self.horizon, num_usuarios=self.NUM_USUARIOS)
        X_va, Y_va = self._make_windows(arr=va_scaled, seq_len=self.seq_len, horizon=self.horizon, num_usuarios=self.NUM_USUARIOS)
        X_te, Y_te = self._make_windows(arr=te_scaled, seq_len=self.seq_len, horizon=self.horizon, num_usuarios=self.NUM_USUARIOS)

        # Tensores para RNN
        X_tr = torch.from_numpy(X_tr)
        X_va = torch.from_numpy(X_va)
        X_te = torch.from_numpy(X_te)
        Y_tr = torch.from_numpy(Y_tr)
        Y_va = torch.from_numpy(Y_va)
        Y_te = torch.from_numpy(Y_te)

        train_dl = DataLoader(
            dataset     = TensorDataset(X_tr, Y_tr),
            batch_size  = self.batch_size,
            shuffle     = self.shuffle_train,
            num_workers = self.num_workers,
            generator   = g
        )

        val_dl = DataLoader(
            dataset     = TensorDataset(X_va, Y_va),
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers
        )

        test_dl = DataLoader(
            dataset     = TensorDataset(X_te, Y_te),
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers
        )

        return train_dl, val_dl, test_dl, self.input_shape
 
    # =============== Utils ===============
    @staticmethod
    def _read_csv(path:Path) -> pd.DataFrame:
        """
        Lee el archivo CSV, convierte la columna 'Fecha' a formato datetime y
        ordena el contenido cronológicamente.

        ### Parámetros:
        - **path** (*Path*): Ruta del archivo CSV.

        ### Retorno:
        - (*pd.DataFrame*): DataFrame ordenado y con la columna Fecha formateada.
        """
        df          = pd.read_csv(path, sep=";")
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        df          = df.sort_values("Fecha").reset_index(drop=True)
        return df
    
    @staticmethod
    def _build_feature_columns(*variables_secuenciales:str, df:pd.DataFrame, num_usuarios:int)->List[str]:
        """
        Construye la lista final de columnas a utilizar como features.

        ### Funcionalidad:
        - Agrega las columnas secuenciales opcionales (variables cíclicas).
        - Detecta automáticamente la posición de 'MT_001'.
        - Agrega las columnas correspondientes a cada usuario MT_xxx.

        ### Parámetros:
        - **variables_secuenciales** (*str*): Nombres de columnas cíclicas a usar como features.
        - **df** (*pd.DataFrame*): DataFrame fuente.
        - **num_usuarios** (*int*): Cantidad de columnas MT_xxx a incluir.

        ### Retorno:
        - (*List[str]*): Lista ordenada de nombres de columnas seleccionadas.
        """
        start             = df.columns.get_loc("MT_001") 
        columnas          = [col for col in variables_secuenciales]
        columnas_usuarios = df.columns[start:(start+num_usuarios)].to_list() 
        columnas.extend(columnas_usuarios)
        return columnas
    
    @staticmethod
    def _time_split(series:np.ndarray, valid_ratio:float, test_ratio:float):
        """
        Realiza un split temporal del dataset en train, validation y test.

        ### Funcionalidad:
        - Mantiene el orden cronológico.
        - Calcula tamaños según proporciones indicadas.
        - Garantiza que cada split tenga al menos un elemento.

        ### Parámetros:
        - **series** (*np.ndarray*): Arreglo con todas las features seleccionadas.
        - **valid_ratio** (*float*): Proporción para validación.
        - **test_ratio** (*float*): Proporción para prueba.

        ### Retorno:
        - (*train, valid, test*): Tres arreglos numpy para cada split.
        """
        n       = len(series)
        n_test  = int(math.floor(n*test_ratio))
        n_valid = int(math.floor(n*valid_ratio))
        n_train = n - n_valid - n_test
        if min(n_train, n_valid, n_test) <= 0:
            raise ValueError("Ratios inválidos para la longitud de la serie.")
        return series[:n_train], series[n_train:n_train+n_valid], series[n_train+n_valid:]
    
    @staticmethod
    def _make_windows(arr:np.ndarray, seq_len:int, horizon:int, num_usuarios:int):
        """
        Genera ventanas deslizantes (X, Y) para modelos de forecasting multiserie.

        ### Funcionalidad:
        - Crea ventanas de entrada X con longitud `seq_len`.
        - Crea ventanas de salida Y con horizonte `horizon`.
        - La matriz X incluye **todos los features**.
        - La matriz Y incluye **solo las columnas objetivo** (usuarios MT_xxx).

        ### Parámetros:
        - **arr** (*np.ndarray*): Serie completa ya escalada.
        - **seq_len** (*int*): Longitud de la secuencia de entrada.
        - **horizon** (*int*): Cantidad de timesteps a predecir.
        - **num_usuarios** (*int*): Número de columnas de salida (MT_xxx).

        ### Retorno:
        - (**X**, **Y**):  
          - X → (num_muestras, seq_len, n_features)  
          - Y → (num_muestras, horizon, num_usuarios)
        """
        n_secuencias = len(arr) - seq_len - horizon + 1
        if n_secuencias <= 0:
            raise ValueError(f"Serie muy corta para seq_len={seq_len} y horizon={horizon}")

        X = np.zeros((n_secuencias, seq_len, arr.shape[1]), dtype=np.float32)   
        Y = np.zeros((n_secuencias, horizon, num_usuarios), dtype=np.float32)

        for i in range(n_secuencias):
            X[i] = arr[i:i+seq_len, :]
            Y[i] = arr[i+seq_len:(i+seq_len+horizon), -num_usuarios:]

        return X, Y


# =============== Prueba de la clase ===============
def main():
    dataLoading = DataLoading(
        data_path  = "../data/02_datos_procesados/datos.csv"
    )
    dataLoading.build()

if __name__ == "__main__":
    main()