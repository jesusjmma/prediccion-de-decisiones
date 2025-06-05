#!/usr/bin/env python3
"""
test_times.py
Lee todos los CSV del directorio Config().LOCAL_PATH y guarda en ./times.csv:
  - Diferencia entre "Tiempo de la pulsación" y "Tiempo de aparición de la letra observada"
  - Diferencia entre "Tiempo de aparición de la letra observada" y "Tiempo de inicio"
Ordenados de menor a mayor según la diferencia entre "Tiempo de la pulsación" y "Tiempo de inicio".
Incluye columnas "ID del participante", "Trial" y "Respuesta".
"""
import os
import glob
import pandas as pd
from scripts.config import Config


def procesar_archivos_csv():
    ruta = Config().LOCAL_PATH
    patron = os.path.join(ruta, '*.csv')
    archivos = glob.glob(patron)

    if not archivos:
        print(f"No se encontraron archivos CSV en {ruta}")
        return

    resultados = []

    for archivo in archivos:
        try:
            df = pd.read_csv(archivo)
        except Exception as e:
            print(f"Error leyendo {archivo}: {e}")
            continue

        columnas_necesarias = [
            'ID del participante',
            'Trial',
            'Respuesta',
            'Tiempo de la pulsación',
            'Tiempo de aparición de la letra observada',
            'Tiempo de inicio'
        ]
        if any(c not in df.columns for c in columnas_necesarias):
            print(f"Columnas faltantes en {archivo}: {[c for c in columnas_necesarias if c not in df.columns]}")
            continue

        for col in ['Tiempo de la pulsación', 'Tiempo de aparición de la letra observada', 'Tiempo de inicio']:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        for _, fila in df.iterrows():
            id_part = fila['ID del participante']
            trial = fila['Trial']
            respuesta = fila['Respuesta']
            t_puls = fila['Tiempo de la pulsación']
            t_letra = fila['Tiempo de aparición de la letra observada']
            t_inicio = fila['Tiempo de inicio']

            try:
                delta_p_l = (t_puls - t_letra).total_seconds()
            except Exception:
                delta_p_l = None
            try:
                delta_l_i = (t_letra - t_inicio).total_seconds()
            except Exception:
                delta_l_i = None
            try:
                delta_p_i = (t_puls - t_inicio).total_seconds()
            except Exception:
                delta_p_i = None

            resultados.append({
                'ID participante': id_part,
                'Trial': trial,
                'Respuesta': respuesta,
                'Δ Pulsación-Letra (s)': delta_p_l,
                'Δ Letra-Inicio (s)': delta_l_i,
                'Δ Pulsación-Inicio (s)': delta_p_i
            })

    if resultados:
        df_res = pd.DataFrame(resultados)
        df_sorted = df_res.sort_values(by='Δ Pulsación-Inicio (s)', na_position='last')
        # Guardar en CSV
        output_path = os.path.join('.', 'times.csv')
        df_sorted.to_csv(output_path, index=False)
        print(f"Datos guardados en {output_path}")
    else:
        print("No se generaron resultados.")

if __name__ == '__main__':
    procesar_archivos_csv()