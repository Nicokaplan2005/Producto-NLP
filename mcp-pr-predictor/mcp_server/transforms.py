"""
Transformaciones serializables para sklearn pipelines.
Definidas aquí (no como lambdas) para que joblib pueda picklearlas/cargarlas
desde cualquier contexto.
"""
import pandas as pd


def to_str_array(arr):
    """Convierte un array a strings, reemplazando NaN por __MISSING__."""
    return (pd.DataFrame(arr)
              .astype(str)
              .replace("nan", "__MISSING__")
              .values)
