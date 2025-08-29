# White box metrics are exclusively based on token distribution, and can be computed in a single pass.

# Last layer logprob metrics
def shannon_entropy(hidden_states, lm_head):
    # (1/N) * Sum_i H(T_i)
    pass

def predictive_entropy(hidden_states, lm_head):
    # -(1/N) * Sum_i p(selected_i) * log p(selected_i)
    pass

def negative_log_likelihood(hidden_states, lm_head):
    # -(1/N) * Sum_i log p(selected_i)
    pass

# Layer distribution variation metrics
def mean_kl_divergence(hidden_states, lm_head, quarters_from_end=1):
    # (1/L) * Sum_i KL(p_i || p_L)
    pass

def var_kl_divergence(hidden_states, lm_head, quarters_from_end=1, median=False):
    # Var(KL(p_1 || p_L), ..., KL(p_L-1 || p_L)) (and median)
    pass

def early_late_kl_divergence(hidden_states, lm_head, quarters_from_end=1):
    # KL(p_early || p_late) (p_early and late averages of some quarters)
    pass

def var_shannon(hidden_states, lm_head, quarters_from_end=1, median=False):
    # Var(H(p_1), ..., H(p_L)) (and median)
    pass

# Early exit metrics
def state_exit_threshold(hidden_states, lm_head, exit_layer):
    pass

def softmax_exit_threshold(hidden_states, lm_head, exit_layer):
    pass

def state_exit_layer(hidden_states, lm_head, threshold):
    pass

def softmax_exit_layer(hidden_states, lm_head, threshold):
    pass

