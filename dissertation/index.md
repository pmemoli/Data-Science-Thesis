## Abstract

1 parrafo motivacion
1 parrafo del pq del problema
1-2 parrafo de como lo resolviste (puedo mencionar ligeramente los detalles tecnicos)

Para esto desarrollamos una tecnica de evaluacion en base a reportes tecnicos de los llms y metricas de incertidumbre. 

El desafio es encontrar una metrica que nos de una intuicion respecto a la capacidad del modelo para saber si funcoino o no. 

El trabajo es encontrar señales sobre las que puedo utilizar directamente o entrenar un modelo sobre esas señales.

Tecnica para encontrar de forma eficiente faults en el modelo.

1 parrafo de resultados

## Introduccion

Lo mismo pero mas desarrollado (Motivacion y el pq del problema)

## Marco teorico

. Breve introduccion a modelos de lenguaje y transformers
. Hablar de las metricas de alucinacion y estado del arte

## Parte tecnica

Historia: 

1. Buscamos features a nivel token que permitan discriminar bien.
2. Evaluamos distintas formas de agregar (descriptores de distribucion y atencion).
3. Entrenamos un modelo simple.
4. Resultados sobre distintos datasets.

XD hablar con luciano bien

### Evaluando metricas de evaluacion

Lo dejo para el final
.Resultados (1 modelo y 1 dataset [TEST, GSM8K])

### [El nombre de la metrica]

.Breve overview
.Desarrolle este algoritmo y esta metrica
.Resultados: evaluar y comparar en GSM8K [TEST], MATH, 2+ y maybe otros modelos 
    (phi, qwen y oss)

## Conclusion y futuras direcciones
