from sklearn.metrics import roc_auc_score
import torch
import os
import numpy as np

suite = "phi3-gpqa-test"
tensor_path = f"src/data/runs/{suite}"
tensor_files = os.listdir(tensor_path)

EPS = 1e-10


def se_mean(probabilities):
    """entropia de shannon media"""
    se_values = -probabilities * torch.log(probabilities + EPS)
    return se_values.mean()


def se_max(probabilities):
    """entropia de shannon max"""
    se_values = -probabilities * torch.log(probabilities + EPS)
    return se_values.max()


def nll_mean(probabilities):
    """Media de Log-Probabilidad Negativa"""
    nll_values = -torch.log(probabilities + EPS)
    return nll_values.mean()


def nll_max(probabilities):
    """Media de Log-Probabilidad Negativa"""
    nll_values = -torch.log(probabilities + EPS)
    return nll_values.max()


def lntp(probabilities):
    """Probabilidad Normalizada por Longitud"""
    log_mean = torch.log(probabilities + EPS).mean()
    return torch.exp(log_mean)


def mtp(probabilities):
    """Probabilidad Mínima de Token"""
    return torch.min(probabilities)


def perplexity(probabilities):
    """Perplejidad"""
    return torch.exp(nll_mean(probabilities))


def response_improbability(probabilities):
    """Response Improbability"""
    log_joint_prob = torch.sum(torch.log(probabilities + EPS))
    return 1.0 - torch.exp(log_joint_prob)


# Diccionario mapeando nombres a funciones
METRICS_MAP = {
    "se_mean": se_mean,
    "se_max": se_max,
    "nll_mean": nll_mean,
    "nll_max": nll_max,
    "lntp": lntp,
    "mtp": mtp,
    "perplexity": perplexity,
    "response_improbability": response_improbability,
}

# --- 3. PROCESAMIENTO DE DATOS ---
# Almacenamiento de resultados: {metric_name: {'pos': [], 'neg': []}}
results = {name: {"pos": [], "neg": []} for name in METRICS_MAP.keys()}

print(f"Procesando {len(tensor_files)} archivos...")
for i, file in enumerate(tensor_files):
    if i % 100 == 0:
        print(f"{i}/{len(tensor_files)}")

    full_path = f"{tensor_path}/{file}"
    tensor = torch.load(full_path)

    for tensor_item in tensor:
        success = tensor_item["success"]
        prompt_length = tensor_item["prompt_length"]
        generated = tensor_item["generation"]
        full_probs = tensor_item["full_layer_selected_prob"].to(torch.float32)
        probabilities = full_probs[-1, 0]

        # Calcular cada métrica
        for metric_name, metric_func in METRICS_MAP.items():
            val = metric_func(probabilities).item()
            if success:
                results[metric_name]["pos"].append(val)
            else:
                results[metric_name]["neg"].append(val)

# --- 4. CÁLCULO DE AUROC Y VALORES MEDIOS ---
print("\nResultados AUROC (Clase Positiva = Fallo/Alucinación):")
print("-" * 50)

metric_results = {}

for metric_name in results.keys():
    pos_scores = results[metric_name]["pos"]  # Éxitos
    neg_scores = results[metric_name]["neg"]  # Fallos (Alucinaciones)

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        print(f"{metric_name}: No hay suficientes datos de ambas clases.")
        continue

    # Calcular AUROC
    y_true = [0] * len(pos_scores) + [1] * len(neg_scores)
    y_scores = pos_scores + neg_scores
    auroc = roc_auc_score(y_true, y_scores)

    # Calcular valor medio de todos los scores
    mean_value = np.mean(y_scores)

    print(f"{metric_name:<25}: AUROC = {auroc:.4f}, Mean = {mean_value:.4f}")

    metric_results[metric_name] = {
        "auroc": auroc,
        "mean": float(mean_value),
    }

print("\nDiccionario de resultados:")
print(metric_results)
