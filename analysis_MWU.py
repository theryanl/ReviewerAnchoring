# This file contains code to run the main analyses, which include 4 statistical tests:
#   1. One-sided permutation test with Mann Whitney U statistic between overall initial and control scores (shows that manipulation is correct)
#   2. [...] between overall revised and control scores (tests if reviewers anchor)
#   3. [...] between evaluation initial and control scores (same but for most affected category)
#   4. [...] between evaluation revised and control scores (same but for most affected category)

# Test
# We use the permutation test with the Mann Whitney U statistic. 
# The permutation test is slightly better for lower sample sizes than the actual test itself using the statistic. 
# Furthermore, the permutation test does not fall victim to reduced variance, power, or type-1 error from tie-breaking methods.

# Statistic
# We use the Mann Whitney U statistic for sample 1; the statistic for sample 2 is size1 * size2 - statistic1, 
# which is just # total comparisons - statistic. Since this is completely captured in the statistic for sample 1
# (as across permutations, the number of total comparisons is the same), we do not consider the statistic for sample 2. 

# Tie Correction
# For the Mann Whitney U tests, we do not use the asymptotic method and continuity correction for tie-breaking,
# as it implicitly includes a normality assumption. This also does not affect the U statistic. 
# We also do not use the tie correction factor for the Mann Whitney U test at scipy.stats.tiecorrect,
# as when we apply this to the concatenated {sample 1, sample 2}, there are the same amount of ties across permutations, 
# resulting in the same correction factor across permutations despite there potentially being different amounts of ties
# actually being compared (tied value distrbution across groups)
# Instead, since the permutation test is immune to the variance weaknesses of the individual Mann Whitney statistic & test, 
# we just use the permutation test with the standard mid-rank method to represent ties, and do not tiebreak. 
# This is fine because ties don't need to be broken - p-value comes from variance, but variance does not rely on U statistic
# in the permutation test framework. 

import os, pandas, argparse, time, numpy as np
from scipy import stats

def get_scores(path):
# input is the path to the folder containing {initial_scores, revised_scores, control_scores}.csv
# outputs a three-item list containing the dataframes of initial_scores, revised_scores, control_scores.
    dfs = []
    for file in ["initial", "revised", "control"]:
        filepath = path + os.sep + file + ".csv"

        if not os.path.exists(filepath):
            raise ValueError(f'csv {filepath} does not exist')
    
        df = pandas.read_csv(filepath)
        dfs.append(df)
    return dfs

def col_to_numpy(df):
# input is a dataframe with one column
# output is the contents of that column in a 1d numpy array
    to_numpy = df.to_numpy()
    if len(to_numpy.shape) == 1:
        return to_numpy
    return np.squeeze(to_numpy)

def bootstrap_ci(a, b, observed_statistic, n_bootstrap=100000):
    n_a = len(a)
    n_b = len(b)

    bootstrap_statistics = []

    for _ in range(n_bootstrap):
        bootstrap_a = np.random.choice(a, n_a, replace=True)
        bootstrap_b = np.random.choice(b, n_b, replace=True)
        bootstrap_statistic = stats.mannwhitneyu(bootstrap_a, bootstrap_b).statistic
        bootstrap_statistics.append(bootstrap_statistic)

    lower = np.percentile(bootstrap_statistics, 2.5)
    upper = np.percentile(bootstrap_statistics, 97.5)

    return lower, upper

def mann_whitney_test(a, b):
    # Calculate the p-value using the permutation test with the Mann-Whitney U statistic
    n_permutations = 100000
    combined = np.concatenate((a, b))
    n_a = len(a)
    n_b = len(b)
    observed_statistic = stats.mannwhitneyu(a, b).statistic
    total_statistic = n_a * n_b

    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        permuted_a = combined[:n_a]
        permuted_b = combined[n_a:]
        permuted_statistic = stats.mannwhitneyu(permuted_a, permuted_b).statistic
        if permuted_statistic >= observed_statistic:
            count += 1

    p_value = count / n_permutations

    ci = bootstrap_ci(a, b, observed_statistic)

    return observed_statistic, total_statistic, p_value, ci


def main(args):
    np.random.seed(0) # for reproducibility
    initial_scores_df, revised_scores_df, control_scores_df = get_scores(args.data_path)

    overall_initial_scores = col_to_numpy(initial_scores_df["Overall"])
    overall_revised_scores = col_to_numpy(revised_scores_df["Overall"])
    overall_control_scores = col_to_numpy(control_scores_df["Overall"])
    evaluation_initial_scores = col_to_numpy(initial_scores_df["Evaluation"])
    evaluation_revised_scores = col_to_numpy(revised_scores_df["Evaluation"])
    evaluation_control_scores = col_to_numpy(control_scores_df["Evaluation"])

    statistic1, total_statistic1, p_value1, ci1 = mann_whitney_test(overall_control_scores, overall_initial_scores)
    statistic2, total_statistic2, p_value2, ci2 = mann_whitney_test(overall_control_scores, overall_revised_scores)
    statistic3, total_statistic3, p_value3, ci3 = mann_whitney_test(evaluation_control_scores, evaluation_initial_scores)
    statistic4, total_statistic4, p_value4, ci4 = mann_whitney_test(evaluation_control_scores, evaluation_revised_scores)

    # means of the scores
    print("Mean scores:")
    print("overall initial", np.mean(overall_initial_scores))
    print("overall revised", np.mean(overall_revised_scores))
    print("overall control", np.mean(overall_control_scores))
    print("evaluation initial", np.mean(evaluation_initial_scores))
    print("evaluation revised", np.mean(evaluation_revised_scores))
    print("evaluation control", np.mean(evaluation_control_scores))

    # mann whitney test results
    print("\nStatistics:")
    print("overall initial control", statistic1, total_statistic1, p_value1, ci1)
    print("overall revised control", statistic2, total_statistic2, p_value2, ci2)
    print("evaluation initial control", statistic3, total_statistic3, p_value3, ci3)
    print("evaluation revised control", statistic4, total_statistic4, p_value4, ci4)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    # parser.add_argument('--mwu_alt', type=str, default='less', help='The alternative argument in the scipy Mann Whitney U test. Options are less(default), greater, two-sided.')
    # alternative greater/less/two-sided only affects p value
    # parser.add_argument('--mwu_method', type=str, default='asymptotic', help='The method argument in the scipy Mann Whitney U test. Options are asymptotic(default), exact, auto')
    # method exact vs. asymptotic only affects p value
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    start_time = time.time()
    main(args)
    print(f'completed: {time.time() - start_time}s')