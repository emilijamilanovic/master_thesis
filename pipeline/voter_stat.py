from scipy.stats import binom

def find_min_runs(p_correct=0.5924, p_noise=4.2e-5, conf_thresh=0.90, error_thresh=0.01, max_n=1000):
    '''
    p_correct = probability the LLM includes a truly relevant chunk in a single run
    p_noise = probability the LLM includes a truly irrelevant (noise) chunk in a single run 
    '''
    for n in range(1, max_n + 1):
        k_majority = n // 2 + 1
        p_correct_majority = 1 - binom.cdf(k_majority - 1, n, p_correct)
        p_noise_majority = 1 - binom.cdf(k_majority - 1, n, p_noise)
        
        if p_correct_majority >= conf_thresh and p_noise_majority <= error_thresh:
            return n, p_correct_majority, p_noise_majority
    
    return None, None, None  

if __name__ == "__main__":
    n, p_c, p_n = find_min_runs()
    if n:
        print(f"Minimum runs needed: {n}")
        print(f"Probability correct chunk wins: {p_c:.4f}")
        print(f"Probability noise chunk wins:   {p_n:.6f}")
    else:
        print("No suitable value of n found within given range.")