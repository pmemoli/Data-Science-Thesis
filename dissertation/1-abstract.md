## Abstract

1 parrafo motivacion
1 parrafo del pq del problema
1-2 parrafo de como lo resolviste
1 parrafo de resultados

## Abstract

Actualmente, el motor de mejora de los LLMs es la calidad de los datos de entrenamiento. Agregar más parámetros es cada vez más costoso, y ya no se corresponde directamente con mejor performance. Saber qué tipo de datos necesita el modelo para mejorar su performance es un problema no-trivial, que actualmente se aborda mediante benchmarks desarrollados manualmente.

Esta tesis propone una métrica basada exclusivamente en función de los estados internos del modelo -hidden states, matrices de atención, etc-, que permita discriminar automáticamente entre dominios que el modelo tiene buen o mal performance.

Para desarrollar la métrica, utilizamos GSM8K y Phi 3.5 Mini (3.8B) como dataset y modelo respectivamente. Al ser un modelo autoregresivo que genera un token a la vez, propusimos un conjunto de métricas a nivel token individual. Posteriormente, agregamos estas métricas mediante promedios y cuantiles sobre la secuencia completa generada, evaluando mediante AUROC cuáles son más efectivas para discriminar respuestas incorrectas. Los experimentos identificaron a la entropía de shannon como el feature más discriminativo.

En una segunda etapa, exploramos formas más sofisticadas de agregar la entropía de cada token. Adaptamos el paper *Quantifying Attention Flow in Transformers* para modelos autoregresivos, obteniendo diversos métodos para cuantificar la influencia que tuvo cada token generado, según la atención recibida. Utilizando esta influencia realizamos un promedio ponderado de la entropía, seleccionando la técnica con mejor AUROC sobre GSM8K.

[RESULTADOS SOBRE TODOS LOS DATASETS Y MODELOS (WIP)]
