# This file contains code to run the supplemental analyses, which include the calculation of:
#   1. Junior vs senior reviewer scores Mann Whitney U statistic and 95% CI
#   2. High confidence vs low confidence reviewer scores Mann Whitney U statistic and 95% CI
#   3. Main institution vs other institutions reviewer scores Mann Whitney U statistic and 95% CI
#   4. Significance category revised and control scores
#   5. Novelty category revised and control scores
#   6. Soundness category revised and control scores
#   7. Clarity category revised and control scores

# Statistic
# We use the Mann Whitney U statistic for sample 1; the statistic for sample 2 is size1 * size2 - statistic1, 
# which is just # total comparisons - statistic. Since this is completely captured in the statistic for sample 1
# (as across permutations, the number of total comparisons is the same), we do not consider the statistic for sample 2. 

# Tie Correction
# For the Mann Whitney U statistic calculation, since we are not using p-values, we do not worry about the variance 
# warping imposed by tie correction through mid-ranks. We bootstrap the confidence intervals instead, which takes into
# account the warping. 

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

def mann_whitney_statistic(a, b):
    # Calculate the Mann-Whitney U statistic
    n_a = len(a)
    n_b = len(b)
    observed_statistic = stats.mannwhitneyu(a, b).statistic
    total_statistic = n_a * n_b

    ci = bootstrap_ci(a, b, observed_statistic)

    return observed_statistic, total_statistic, ci

def compare_seniority(initial_scores_df, revised_scores_df, control_scores_df):
    seniority_buckets = np.unique(col_to_numpy(initial_scores_df["Year"]))
    participant_year_counts = []

    junior_initial_scores = np.array([])
    junior_revised_scores = np.array([])
    junior_control_scores = np.array([])
    senior_initial_scores = np.array([])
    senior_revised_scores = np.array([])
    senior_control_scores = np.array([])

    for bucket in seniority_buckets:
        print('bucket', bucket)
        initial_bucket_overall_scores = col_to_numpy(initial_scores_df[initial_scores_df["Year"] == bucket]["Overall"])
        revised_bucket_overall_scores = col_to_numpy(revised_scores_df[revised_scores_df["Year"] == bucket]["Overall"])
        control_bucket_overall_scores = col_to_numpy(control_scores_df[control_scores_df["Year"] == bucket]["Overall"])
    
        participant_year_counts.append(len(initial_bucket_overall_scores) + len(control_bucket_overall_scores))
        # print(bucket, sum(participant_year_counts))
        # third year -- boundary

        if len(initial_bucket_overall_scores.shape) > 0:
            if len(participant_year_counts) <= 3: # junior
                junior_initial_scores = np.append(junior_initial_scores, initial_bucket_overall_scores)
                junior_revised_scores = np.append(junior_revised_scores, revised_bucket_overall_scores)
            else:
                senior_initial_scores = np.append(senior_initial_scores, initial_bucket_overall_scores)
                senior_revised_scores = np.append(senior_revised_scores, revised_bucket_overall_scores)

        if len(control_bucket_overall_scores.shape) > 0:
            if len(participant_year_counts) <= 3: # junior
                junior_control_scores = np.append(junior_control_scores, control_bucket_overall_scores)
            else:
                senior_control_scores = np.append(senior_control_scores, control_bucket_overall_scores)
    
    print(f'number of junior experimental participants: {len(junior_initial_scores)}')
    print(f'number of senior experimental participants: {len(senior_initial_scores)}')
    print(f'number of junior control participants: {len(junior_control_scores)}')
    print(f'number of senior control participants: {len(senior_control_scores)}')
    print(f'mean junior initial overall score: {np.mean(junior_initial_scores)}')
    print(f'mean junior revised overall score: {np.mean(junior_revised_scores)}')
    print(f'mean junior control overall score: {np.mean(junior_control_scores)}')
    print(f'mean senior initial overall score: {np.mean(senior_initial_scores)}')
    print(f'mean senior revised overall score: {np.mean(senior_revised_scores)}')
    print(f'mean senior control overall score: {np.mean(senior_control_scores)}')

    # mann whitney statistics
    statistic1, total_statistic1, ci1 = mann_whitney_statistic(junior_control_scores, junior_revised_scores)
    statistic2, total_statistic2, ci2 = mann_whitney_statistic(senior_control_scores, senior_revised_scores)
    print(f'junior control vs junior revised: {statistic1}, {total_statistic1}, {ci1}')
    print(f'senior control vs senior revised: {statistic2}, {total_statistic2}, {ci2}')

def get_hc_overall_scores(initial_scores_df, revised_scores_df, control_scores_df):
# from the score dfs, pick out the high-confidence answers and return their overall scores.
    hc_initial_scores_df = initial_scores_df[initial_scores_df["Confidence"] >= 3]
    hc_revised_scores_df = revised_scores_df[revised_scores_df["Confidence"] >= 3]
    hc_control_scores_df = control_scores_df[control_scores_df["Confidence"] >= 3]

    print(f'{hc_initial_scores_df.shape[0]} of {initial_scores_df.shape[0]} experimental group participants were confident.')
    print(f'{hc_control_scores_df.shape[0]} of {control_scores_df.shape[0]} control group participants were confident.')

    hc_initial_overall_scores = col_to_numpy(hc_initial_scores_df["Overall"])
    hc_revised_overall_scores = col_to_numpy(hc_revised_scores_df["Overall"])
    hc_control_overall_scores = col_to_numpy(hc_control_scores_df["Overall"])

    return hc_initial_overall_scores, hc_revised_overall_scores, hc_control_overall_scores

def get_lc_overall_scores(initial_scores_df, revised_scores_df, control_scores_df):
# from the score dfs, pick out the high-confidence answers and return their overall scores.
    lc_initial_scores_df = initial_scores_df[initial_scores_df["Confidence"] < 3]
    lc_revised_scores_df = revised_scores_df[revised_scores_df["Confidence"] < 3]
    lc_control_scores_df = control_scores_df[control_scores_df["Confidence"] < 3]

    print(f'{lc_initial_scores_df.shape[0]} of {initial_scores_df.shape[0]} experimental group participants were unconfident.')
    print(f'{lc_control_scores_df.shape[0]} of {control_scores_df.shape[0]} control group participants were unconfident.')

    lc_initial_overall_scores = col_to_numpy(lc_initial_scores_df["Overall"])
    lc_revised_overall_scores = col_to_numpy(lc_revised_scores_df["Overall"])
    lc_control_overall_scores = col_to_numpy(lc_control_scores_df["Overall"])

    return lc_initial_overall_scores, lc_revised_overall_scores, lc_control_overall_scores

def compare_confidence(initial_scores_df, revised_scores_df, control_scores_df):
    hc_initial_scores, hc_revised_scores, hc_control_scores = get_hc_overall_scores(initial_scores_df, revised_scores_df, control_scores_df)
    lc_initial_scores, lc_revised_scores, lc_control_scores = get_lc_overall_scores(initial_scores_df, revised_scores_df, control_scores_df)

    print(f'number of high confidence experimental participants: {len(hc_initial_scores)}')
    print(f'number of low confidence experimental participants: {len(lc_initial_scores)}')
    print(f'number of high confidence control participants: {len(hc_control_scores)}')
    print(f'number of low confidence control participants: {len(lc_control_scores)}')
    print(f'mean high confidence initial overall score: {np.mean(hc_initial_scores)}')
    print(f'mean high confidence revised overall score: {np.mean(hc_revised_scores)}')
    print(f'mean high confidence control overall score: {np.mean(hc_control_scores)}')
    print(f'mean low confidence initial overall score: {np.mean(lc_initial_scores)}')
    print(f'mean low confidence revised overall score: {np.mean(lc_revised_scores)}')
    print(f'mean low confidence control overall score: {np.mean(lc_control_scores)}')

    # mann whitney statistics
    statistic1, total_statistic1, ci1 = mann_whitney_statistic(hc_control_scores, hc_revised_scores)
    statistic2, total_statistic2, ci2 = mann_whitney_statistic(lc_control_scores, lc_revised_scores)

    print(f'high confidence control vs high confidence revised: {statistic1}, {total_statistic1}, {ci1}')
    print(f'low confidence control vs low confidence revised: {statistic2}, {total_statistic2}, {ci2}')

def separate_scores_by_institution(initial_scores_df, revised_scores_df, control_scores_df):
# takes in three dfs: initial, revised, control
# returns six dfs: CMU initial, revised, control, nonCMU initial, revised, control
    cmu_initial_scores_df = initial_scores_df[initial_scores_df["Institution"] == "CMU"]
    cmu_revised_scores_df = revised_scores_df[revised_scores_df["Institution"] == "CMU"]
    cmu_control_scores_df = control_scores_df[control_scores_df["Institution"] == "CMU"]

    print(f'{cmu_initial_scores_df.shape[0]} of {initial_scores_df.shape[0]} experimental group participants were from CMU.')
    print(f'{cmu_control_scores_df.shape[0]} of {control_scores_df.shape[0]} control group participants were from CMU.')

    noncmu_initial_scores_df = initial_scores_df[initial_scores_df["Institution"] == "Non-CMU"]
    noncmu_revised_scores_df = revised_scores_df[revised_scores_df["Institution"] == "Non-CMU"]
    noncmu_control_scores_df = control_scores_df[control_scores_df["Institution"] == "Non-CMU"]

    return cmu_initial_scores_df, cmu_revised_scores_df, cmu_control_scores_df, noncmu_initial_scores_df, noncmu_revised_scores_df, noncmu_control_scores_df

def compare_institutions(initial_scores_df, revised_scores_df, control_scores_df):
# Compares the scores across institutions, with CMU being one group and non-CMU being the other
    cmu_initial_scores_df, cmu_revised_scores_df, cmu_control_scores_df, \
    noncmu_initial_scores_df, noncmu_revised_scores_df, noncmu_control_scores_df = \
        separate_scores_by_institution(initial_scores_df, revised_scores_df, control_scores_df)
    
    cmu_initial_scores_overall = np.mean(cmu_initial_scores_df["Overall"])
    cmu_revised_scores_overall = np.mean(cmu_revised_scores_df["Overall"])
    cmu_control_scores_overall = np.mean(cmu_control_scores_df["Overall"])
    noncmu_initial_scores_overall = np.mean(noncmu_initial_scores_df["Overall"])
    noncmu_revised_scores_overall = np.mean(noncmu_revised_scores_df["Overall"])
    noncmu_control_scores_overall = np.mean(noncmu_control_scores_df["Overall"])

    print(f'cmu initial scores: {cmu_initial_scores_overall}')
    print(f'cmu revised scores: {cmu_revised_scores_overall}')
    print(f'cmu control scores: {cmu_control_scores_overall}')
    print(f'noncmu initial scores: {noncmu_initial_scores_overall}')
    print(f'noncmu revised scores: {noncmu_revised_scores_overall}')
    print(f'noncmu control scores: {noncmu_control_scores_overall}')

    # mann whitney statistics
    statistic1, total_statistic1, ci1 = mann_whitney_statistic(cmu_control_scores_df["Overall"], cmu_revised_scores_df["Overall"])
    statistic2, total_statistic2, ci2 = mann_whitney_statistic(noncmu_control_scores_df["Overall"], noncmu_revised_scores_df["Overall"])

    print(f'cmu control vs cmu revised: {statistic1}, {total_statistic1}, {ci1}')
    print(f'noncmu control vs noncmu revised: {statistic2}, {total_statistic2}, {ci2}')

def main(args):
    np.random.seed(0) # for reproducibility
    initial_scores_df, revised_scores_df, control_scores_df = get_scores(args.data_path)

    # 1. Junior vs senior reviewer scores Mann Whitney U statistic and 95% CI
    compare_seniority(initial_scores_df, revised_scores_df, control_scores_df)

    # 2. High confidence vs low confidence reviewer scores Mann Whitney U statistic and 95% CI
    compare_confidence(initial_scores_df, revised_scores_df, control_scores_df)

    # 3. Main institution vs other institutions reviewer scores Mann Whitney U statistic and 95% CI
    compare_institutions(initial_scores_df, revised_scores_df, control_scores_df)

    # 4-7. Significance, Novelty, Soundness, Clarity category revised and control scores: Mann Whitney U statistic and 95% CI
    supplemental_categories = ["Significance", "Novelty", "Soundness", "Clarity"]

    for category in supplemental_categories:
        initial_scores_category = col_to_numpy(initial_scores_df[category])
        revised_scores_category = col_to_numpy(revised_scores_df[category])
        control_scores_category = col_to_numpy(control_scores_df[category])

        # means
        print(f'{category} initial: {np.mean(initial_scores_category)}')
        print(f'{category} revised: {np.mean(revised_scores_category)}')
        print(f'{category} control: {np.mean(control_scores_category)}')

        # mann whitney statistics
        statistic1, total_statistic1, ci1 = mann_whitney_statistic(control_scores_category, revised_scores_category)
        print(f'{category} control vs revised: {statistic1}, {total_statistic1}, {ci1}')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    start_time = time.time()
    main(args)
    print(f'completed: {time.time() - start_time}s')