from scipy.stats import binom

# Defaults are the pooled estimates from the 500-run Pandoc experiment with
# ground truth 391-419 (from the original 500-run experiment, whose runs are not in this repository).
#
# NOTE on p_noise: with GT 391-419 the Llama runs produced ZERO false positives
# (0 out of 500 x 143 opportunities), so the plug-in estimate is exactly 0.
# A zero rate cannot be used directly -- it makes any single inclusion
# infinitely strong evidence. The default below is the "rule of three" 95%
# upper bound, 3/(N*U) = 3/71500, which is the conservative standard estimate
# for an unobserved event. Revisit if a smoothing convention is adopted
# pipeline-wide.
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