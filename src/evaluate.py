# src/evaluate.py

import torch

import numpy  as np
import pandas as pd

from typing           import Dict, Any
from torch.utils.data import DataLoader
from torch            import nn
from sklearn.metrics  import mean_absolute_error, mean_squared_error

class Evaluator:
    """
    Clase encargada de evaluar un modelo RNN/LSTM en un escenario many-to-many
    con múltiples horizontes y múltiples variables objetivo por horizonte.

    Esta clase recibe las predicciones del modelo y calcula:
    - Métricas por horizonte (MAE, MSE, RMSE)
    - Métrica global agregada (todas las predicciones aplanadas)
    Además, invierte el escalado aplicado a las series originales antes
    de calcular las métricas.

    ### Parámetros:
    - **modelo** (*nn.Module*):
        Modelo ya entrenado que se desea evaluar.
    - **horizon** (*int*):
        Número de pasos futuros que predice el modelo. Es la dimensión
        de salida en el eje temporal de Y.
    - **scaler**:
        Objeto de escalado (StandardScaler) utilizado durante el preprocesamiento.
        Es obligatorio para poder invertir el escalado antes de calcular métricas.
    - **use_log** (*bool*, opcional):
        Indica si las series fueron transformadas con log1p. Si es True,
        durante la inversión se aplica expm1().
    - **device** (*str*, opcional):
        Dispositivo donde se ejecuta el modelo: `"cpu"` o `"cuda"`.

    ### Funcionalidad general:
    - Reúne todas las predicciones del dataloader de test.
    - Convierte los tensores a NumPy.
    - Invierte el escalado.
    - Calcula métricas por horizonte.
    - Calcula métricas globales.
    - Imprime resultados en formato tabular.
    """
    def __init__(self, modelo:nn.Module, horizon:int, *, scaler, use_log:bool=False, device:str="cpu"):  
        """
        Inicializa el evaluador configurando el modelo, scaler, horizonte
        y el dispositivo donde se ejecutará la evaluación.

        ### Parámetros:
        - **modelo** (*nn.Module*): Modelo de PyTorch ya entrenado.
        - **horizon** (*int*): Número de pasos a futuro que produce el modelo.
        - **scaler** (*StandardScaler*): Objeto usado para invertir el escalado.
        - **use_log** (*bool*): Indica si se aplicó log1p antes del escalado.
        - **device** (*str*): `"cpu"` o `"cuda"`.

        ### Excepciones:
        - **ValueError**: Si el scaler es None.
        """      
        if scaler is None:
            raise ValueError("Se requiere 'scaler' para invertir el escalado")        
        self.modelo  = modelo.to(device)
        self.horizon = horizon
        self.scaler  = scaler
        self.use_log = use_log
        self.device  = device

    @staticmethod
    def _to_np(x:torch.Tensor)->np.ndarray:
        """
        Convierte un tensor de PyTorch en un arreglo NumPy, garantizando
        que esté en CPU y sin gradientes.

        ### Parámetros:
        - **x** (*torch.Tensor*): Tensor a convertir.

        ### Retorno:
        - (*np.ndarray*): Versión NumPy del tensor.
        """
        return x.detach().cpu().numpy()
    
    @staticmethod
    def _calcular_metricas(y_true:np.ndarray, y_pred:np.ndarray)->Dict[str, Any]:
        """
        Calcula las métricas clásicas de regresión:
        MAE, MSE y RMSE.

        ### Parámetros:
        - **y_true** (*np.ndarray*): Valores reales.
        - **y_pred** (*np.ndarray*): Predicciones ya invertidas (escala original).

        ### Retorno:
        - (*dict*): Diccionario con las métricas calculadas.
        """
        mae  = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        mse  = mean_squared_error(y_true=y_true, y_pred=y_pred)
        rmse = float(np.sqrt(mse))
        return {"MAE":mae, "MSE":mse, "RMSE":rmse}  
    
    def _inverse_scaler(self, z:np.ndarray)->np.ndarray:
        """
        Invierte el escalado aplicado a las columnas objetivo.
        Si `use_log` es True, también aplica expm1().

        ### Parámetros:
        - **z** (*np.ndarray*): Valores escalados con StandardScaler.

        ### Retorno:
        - (*np.ndarray*): Valores desescalados en su escala original.
        """
        inv = self.scaler.inverse_transform(z)
        if self.use_log:
            inv = np.expm1(inv)
        return inv
    
    def _evaluar_por_time_step(self, y_true_sc:np.ndarray, y_pred_sc:np.ndarray) -> pd.DataFrame:
        """
        Evalúa el rendimiento del modelo **por horizonte**, es decir,
        calcula las métricas separadamente para cada paso futuro:

        Para cada horizonte `h`:
        1. Invierte el escalado de todas las series.
        2. Calcula métricas serie por serie.
        3. Promedia las métricas entre series.
        4. Reporta un diccionario global por horizonte.

        ### Parámetros:
        - **y_true_sc** (*np.ndarray*):
            Arreglo de etiquetas con escalado, de forma (N, horizon, target_dim).
        - **y_pred_sc** (*np.ndarray*):
            Arreglo de predicciones escaladas, misma forma que y_true_sc.

        ### Retorno:
        - (*pd.DataFrame*):
            DataFrame donde:
            - **columnas** → cada horizonte (p. ej. "horizonte_1", "horizonte_2")
            - **filas**    → métricas: MAE, MSE, RMSE
        """
        metricas_horizonte = {} # Almacena las métricas por horizonte
        # Recorrido por horizonte
        for hori in range(y_true_sc.shape[1]):
            y_true_time_step = self._inverse_scaler(z=y_true_sc[:, hori, :])
            y_pred_time_step = self._inverse_scaler(z=y_pred_sc[:, hori, :])
            metricas_series  = {} # Almacena las métricas por serie
            # Recorrido por serie
            for serie in range(y_true_sc.shape[2]):
                metricas_series[f"serie_{serie+1}"] = self._calcular_metricas(
                    y_true=y_true_time_step[:, serie],
                    y_pred=y_pred_time_step[:, serie]
                )
            # Recorrido por el diccionario de métricas por serie
            primeros_resultados = next(iter(metricas_series.values()))
            metricas_promedio   = {clave:0 for clave in primeros_resultados.keys()}
            for metrica in metricas_promedio.keys():
                valores = []
                for serie in metricas_series.keys():
                    valores.append(metricas_series[serie][metrica])
                metricas_promedio[metrica] = float(np.mean(valores))
            metricas_horizonte[f"horizonte_{hori+1}"] = metricas_promedio
        return pd.DataFrame(metricas_horizonte)

    def _evaluacion_global(self, y_true_sc:np.ndarray, y_pred_sc:np.ndarray)->pd.Series:
        """
        Calcula una **métrica global agregada** considerando todos los valores
        de todos los horizontes y todas las series como una única gran serie.

        Pasos:
        1. Reescala todos los horizontes completos.
        2. Aplana y_true e y_pred.
        3. Calcula métricas sobre toda la serie resultante.

        ### Parámetros:
        - **y_true_sc** (*np.ndarray*): Verdaderos (escalados).
        - **y_pred_sc** (*np.ndarray*): Predicciones (escaladas).

        ### Retorno:
        - (*pd.Series*): Serie con las métricas globales.
        """
        y_true         = self._inverse_scaler(y_true_sc.reshape(-1, y_true_sc.shape[-1]))
        y_pred         = self._inverse_scaler(y_pred_sc.reshape(-1, y_pred_sc.shape[-1]))        
        y_true_flatten = y_true.flatten()
        y_pred_flatten = y_pred.flatten()
        metricas       = self._calcular_metricas(y_true=y_true_flatten, y_pred=y_pred_flatten)
        return pd.Series(metricas)
    
    def _imprimir_resultados(self, metricas_por_horizonte:pd.DataFrame, metricas_global:pd.Series)->None:
        """
        Imprime en consola las métricas de evaluación de manera formateada,
        separando claramente las métricas por horizonte y las métricas globales.

        ### Funcionalidad:
        - Redondea las métricas a un número manejable de decimales para mejorar la lectura.
        - Muestra una tabla con las métricas promedio por horizonte.
        - Muestra una tabla con las métricas globales agregadas.
        - Utiliza separadores y encabezados para hacer la salida más clara y profesional.

        ### Parámetros:
        - **metricas_por_horizonte** (*pd.DataFrame*):
            DataFrame donde cada columna representa un horizonte (p.ej. `horizonte_1`,
            `horizonte_2`, etc.) y cada fila una métrica (MAE, MSE, RMSE).
        - **metricas_global** (*pd.Series*):
            Serie con las métricas globales calculadas sobre todas las muestras,
            todos los horizontes y todas las series objetivo.

        ### Retorno:
        - (*None*): No retorna ningún valor. Su efecto principal es la impresión
        formateada de los resultados en la consola.
        """
        # Redondear un poco para que no se vea tan ruidoso
        metricas_por_horizonte = metricas_por_horizonte.round(4)
        metricas_global        = metricas_global.round(4)
        # Encabezado métricas por horizonte
        print("\n" + "="*40)
        print("MÉTRICAS PROMEDIO POR HORIZONTE".center(40))
        print("="*40)
        print(metricas_por_horizonte.to_string())

        print("\n"+"="*40)
        print("MÉTRICAS GLOBALES".center(40))
        print("="*40)
        print(metricas_global.to_frame(name="Global").to_string())
    
    def evaluar(self, dl_test:DataLoader):
        """
        Ejecuta la evaluación completa del modelo usando el dataloader de test.

        Flujo:
        1. Se desactiva el gradiente (`no_grad()`).
        2. Se recorren los batches del test.
        3. Se recolectan predicciones y valores reales.
        4. Se concatenan en arreglos NumPy 3D:
           (N_total, horizon, target_dim)
        5. Se evalúa por horizonte.
        6. Se evalúa la métrica global.

        ### Parámetros:
        - **dl_test** (*DataLoader*): Dataloader con los datos de test.

        ### Retorno:
        - **None**. Imprime resultados por consola.
        """
        self.modelo.eval()
        preds_sc, true_sc = [], []
        with torch.no_grad():
            for X, Y in dl_test:
                X, Y   = X.to(self.device), Y.to(self.device)
                Y_pred = self.modelo(X) 
                preds_sc.append(self._to_np(x=Y_pred))
                true_sc.append(self._to_np(x=Y)) 
        y_pred_sc = np.concatenate(preds_sc, axis=0)
        y_true_sc = np.concatenate(true_sc, axis=0)
        metricas_por_horizonte_df = self._evaluar_por_time_step(y_true_sc=y_true_sc, y_pred_sc=y_pred_sc)
        metricas_global_df        =self._evaluacion_global(y_true_sc=y_true_sc, y_pred_sc=y_pred_sc)   
        self._imprimir_resultados(metricas_por_horizonte_df, metricas_global_df)    
        
# ==== Prueba de la clase ===

def main():
    from data_loading import DataLoading
    from model        import Modelo

    datos = DataLoading(
        data_path  = "../data/02_datos_procesados/datos.csv",
        batch_size = 64,
        seq_len    = 3,
        horizon    = 2
    )

    _, _, test_dl, input_shape = datos.build()

    modelo = Modelo(
        input_shape = input_shape, 
        horizon     = datos.horizon,
        target_dim  = DataLoading.NUM_USUARIOS,
        hidden_size = 64,
        num_layers  = 2
    )

    evaluador = Evaluator(
        modelo  = modelo,
        horizon = datos.horizon,
        scaler  = datos.scaler,
        use_log = datos.use_log
    )

    evaluador.evaluar(dl_test=test_dl)

if __name__ == "__main__":
    main()