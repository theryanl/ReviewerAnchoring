import os, pandas, argparse, scipy, time, numpy as np
from functools import reduce

numerical_column_names = ["Significance", "Novelty", "Soundness", "Evaluation", "Clarity", "Overall", "Confidence"]

def get_csv_last(path):
# input is the path to a csv file
# outputs the last row of the csv file as a dataframe
    # print(f'getting the csv for {path}')
    if not os.path.exists(path):
        raise ValueError(f'csv {path} does not exist')
    all_scores_df = pandas.read_csv(path)
    return all_scores_df.tail(1)

def preprocess_df(df):
# replaces column names with new_col_names
# replaces score data with integers
    old_col_names = df.columns
    new_col_names = ["Timestamp", "Email", "Significance", "Significance_text", "Novelty", "Novelty_text", 
                     "Soundness", "Soundness_text", "Evaluation", "Evaluation_text", "Clarity", "Clarity_text", 
                     "Overall", "Confidence", "Hyperlink_text", "Animated_figures_text", "Institution", "Year"]

    column_dict = {}
    for i in range(len(old_col_names)):
        column_dict[old_col_names[i]] = new_col_names[i]
    
    df = df.rename(columns=column_dict)
    for col_name in numerical_column_names:
        df[col_name] = df[col_name].apply(lambda x: 10 if (x[1] == "0") else int(x[0]))

    return df

def get_scores(path, num_participants):
# input is the path to the folder containing all the subject data
# output is a tuple (control scores, initial scores, revised scores), where
# each is a dataframe containing the raw data from each group.
    participant_folders = sorted(os.listdir(path))
    if participant_folders[0] =='.DS_Store':
        participant_folders = participant_folders[1:]

    control_scores_list = []
    initial_scores_list = []
    revised_scores_list = []

    for i in range(num_participants):
        foldername = path + os.sep + participant_folders[i]
        # if i < 5: print(foldername) # testing code
        if 'before.csv' in os.listdir(foldername):
            # experimental group
            before_csv_path = foldername + os.sep + 'before.csv'
            after_csv_path = foldername + os.sep + 'after.csv'
            initial_scores = get_csv_last(before_csv_path)
            revised_scores = get_csv_last(after_csv_path)
            initial_scores_list.append(initial_scores)
            revised_scores_list.append(revised_scores)
        else:
            # control group
            control_csv_path = foldername + os.sep + 'Reviewer Form for Paper Review Study.csv'
            control_scores = get_csv_last(control_csv_path)
            control_scores_list.append(control_scores)
    
    control_scores_df = preprocess_df(pandas.concat(control_scores_list))
    initial_scores_df = preprocess_df(pandas.concat(initial_scores_list))
    revised_scores_df = preprocess_df(pandas.concat(revised_scores_list))
    return (initial_scores_df, revised_scores_df, control_scores_df)

def col_to_numpy(df):
# input is a dataframe with one column
# output is the contents of that column in a 1d numpy array
    to_numpy = df.to_numpy()
    if len(to_numpy.shape) == 1:
        return to_numpy
    return np.squeeze(to_numpy)

def permutation_test(scores1, scores2, num_permutations):
# input is two numpy 1d arrays, and number of trials (optional) for the permutation test
# output is the effect size and p-value of the permutation test
    scores1_length = scores1.shape[0]
    effect_size = np.mean(scores1) - np.mean(scores2)

    combined_scores = np.concatenate((scores1, scores2))
    count_above_effect_size = 0
    for i in range(num_permutations):
        np.random.shuffle(combined_scores)
        random_scores1, random_scores2 = combined_scores[:scores1_length], combined_scores[scores1_length:]
        if (np.mean(random_scores1) - np.mean(random_scores2) >= effect_size):
            count_above_effect_size += 1
    
    return effect_size, count_above_effect_size / num_permutations

def run_tests(initial_scores, revised_scores, control_scores, descriptor, args):
    if args.check_variances:
        print(f'Initial scores variance: {np.var(initial_scores)}, Revised scores variance: {np.var(revised_scores)}, Control scores variance: {np.var(control_scores)}')

    if args.test_figure_effectiveness:
        # compare between the experimental group's initial scores and control group scores
        # demonstrates the effectiveness of the stimuli
        figure_change_means_diff = np.mean(control_scores) - np.mean(initial_scores)
        print(f'The change in figure caused a change in mean {descriptor} score of {figure_change_means_diff}.')

    if args.compare_before_after:
        # compare between the experimental group's initial and revised overall scores
        before_after_means_diff = np.mean(revised_scores) - np.mean(initial_scores)
        print(f'The experimental group revised {descriptor} scores to be {before_after_means_diff} above their initial scores on average.')
    
    if args.compare_between_groups:
        # compare between the experimental group's revised scores and the control group's overall scores
        # statistic, pvalue = permutation_test(control_scores, revised_scores, args.num_permutations)
        permutation_test_result = scipy.stats.permutation_test(
                                  [control_scores, revised_scores], 
                                  (lambda x, y : np.mean(x) - np.mean(y)), 
                                  permutation_type='independent', 
                                  n_resamples = args.num_permutations, 
                                  alternative = 'greater')
        statistic, pvalue = permutation_test_result.statistic, permutation_test_result.pvalue
        print(f'The control group {descriptor} scores were {statistic} above the experimental group revised scores on average (p={pvalue}).')

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

def check_reduced_scores_after_rebuttal(initial_scores, revised_scores):
    for col_name in numerical_column_names:
        initial_score_col = col_to_numpy(initial_scores[col_name])
        revised_score_col = col_to_numpy(revised_scores[col_name])
        for i in range(initial_score_col.shape[0]):
            if initial_score_col[i] > revised_score_col[i]:
                print(f'{i}th participant in experimental group reduced their {col_name} score after rebuttal from {initial_score_col[i]} to {revised_score_col[i]}')

def compare_categorical_scores(initial_scores_df, revised_scores_df, control_scores_df, args):
# input is initial, revised, and control dfs

    results_dict = {}
    for col_name in numerical_column_names:
        col_dict = {}
        initial_col = col_to_numpy(initial_scores_df[col_name])
        revised_col = col_to_numpy(revised_scores_df[col_name])
        control_col = col_to_numpy(control_scores_df[col_name])
        col_dict["initial_mean"] = np.mean(initial_col)
        col_dict["revised_mean"] = np.mean(revised_col)
        col_dict["control_mean"] = np.mean(control_col)

        if args.test_figure_effectiveness:
            col_dict["figure_effectiveness"] = col_dict["control_mean"] - col_dict["initial_mean"]
        if args.compare_before_after:
            col_dict["before_after"] = col_dict["revised_mean"] - col_dict["initial_mean"]
        if args.compare_between_groups:
            col_dict["between_groups"] = col_dict["control_mean"] - col_dict["revised_mean"]

        print(f'{col_name} mean initial score: {col_dict["initial_mean"]}, revised score: {col_dict["revised_mean"]}, control score: {col_dict["control_mean"]}')
        print(f'figure effectiveness: {col_dict["figure_effectiveness"]}, revised increase: {col_dict["before_after"]}, increase reduced: {col_dict["between_groups"]}\n')

        results_dict[col_name] = col_dict
    return results_dict

def separate_scores_by_institution(initial_scores_df, revised_scores_df, control_scores_df):
# takes in three dfs: initial, revised, control
# returns six dfs: CMU initial, revised, control, nonCMU initial, revised, control
    noncmu_institutions_list = ["UC Berkeley", "University of Pittsburgh", "University of Southern California", "University of Chicago", 
                                "UCB", "Stanford University", "Duke University, Computer Science Department", "University of Pennsylvania", 
                                "PITT", "RMU"]
    cmu_initial_scores_df = initial_scores_df[~initial_scores_df["Institution"].isin(noncmu_institutions_list)]
    cmu_revised_scores_df = revised_scores_df[~revised_scores_df["Institution"].isin(noncmu_institutions_list)]
    cmu_control_scores_df = control_scores_df[~control_scores_df["Institution"].isin(noncmu_institutions_list)]

    print(f'{cmu_initial_scores_df.shape[0]} of {initial_scores_df.shape[0]} experimental group participants were from CMU.')
    print(f'{cmu_control_scores_df.shape[0]} of {control_scores_df.shape[0]} control group participants were from CMU.')

    noncmu_initial_scores_df = initial_scores_df[initial_scores_df["Institution"].isin(noncmu_institutions_list)]
    noncmu_revised_scores_df = revised_scores_df[revised_scores_df["Institution"].isin(noncmu_institutions_list)]
    noncmu_control_scores_df = control_scores_df[control_scores_df["Institution"].isin(noncmu_institutions_list)]

    return cmu_initial_scores_df, cmu_revised_scores_df, cmu_control_scores_df, noncmu_initial_scores_df, noncmu_revised_scores_df, noncmu_control_scores_df

def compare_across_institutions(initial_scores_df, revised_scores_df, control_scores_df):
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

    # comparsion of confidence scores
    cmu_initial_scores_confidence = np.mean(cmu_initial_scores_df["Confidence"])
    cmu_revised_scores_confidence = np.mean(cmu_revised_scores_df["Confidence"])
    cmu_control_scores_confidence = np.mean(cmu_control_scores_df["Confidence"])

    noncmu_initial_scores_confidence = np.mean(noncmu_initial_scores_df["Confidence"])
    noncmu_revised_scores_confidence = np.mean(noncmu_revised_scores_df["Confidence"])
    noncmu_control_scores_confidence = np.mean(noncmu_control_scores_df["Confidence"])

    print(f'cmu initial confidence: {cmu_initial_scores_confidence}')
    print(f'cmu revised confidence: {cmu_revised_scores_confidence}')
    print(f'cmu control confidence: {cmu_control_scores_confidence}')

    print(f'noncmu initial confidence: {noncmu_initial_scores_confidence}')
    print(f'noncmu revised confidence: {noncmu_revised_scores_confidence}')
    print(f'noncmu control confidence: {noncmu_control_scores_confidence}')

def find_change_locations(initial_scores_df, revised_scores_df, control_scores_df):
# takes in three dfs: initial, revised, control
# checks where participants changed/did not change their scores
    initial_overall_scores = col_to_numpy(initial_scores_df["Overall"])
    revised_overall_scores = col_to_numpy(revised_scores_df["Overall"])

    indices_where_overall_scores_changed = np.where(initial_overall_scores != revised_overall_scores)[0]
    score_change_max = np.max(revised_overall_scores - initial_overall_scores)
    score_change_min = np.min(revised_overall_scores - initial_overall_scores)
    print(f'number of participants who changed their overall scores: {len(indices_where_overall_scores_changed)}')
    print(f'max overall score change: {score_change_max}, min overall score change: {score_change_min}')

    initial_significance_scores = col_to_numpy(initial_scores_df["Significance"])
    revised_significance_scores = col_to_numpy(revised_scores_df["Significance"])
    indices_where_significance_scores_changed = np.where(initial_significance_scores != revised_significance_scores)[0]
    print(f'number of participants who changed their significance scores: {len(indices_where_significance_scores_changed)}')

    initial_novelty_scores = col_to_numpy(initial_scores_df["Novelty"])
    revised_novelty_scores = col_to_numpy(revised_scores_df["Novelty"])
    indices_where_novelty_scores_changed = np.where(initial_novelty_scores != revised_novelty_scores)[0]
    print(f'number of participants who changed their novelty scores: {len(indices_where_novelty_scores_changed)}')

    initial_soundness_scores = col_to_numpy(initial_scores_df["Soundness"])
    revised_soundness_scores = col_to_numpy(revised_scores_df["Soundness"])
    indices_where_soundness_scores_changed = np.where(initial_soundness_scores != revised_soundness_scores)[0]
    print(f'number of participants who changed their soundness scores: {len(indices_where_soundness_scores_changed)}')

    initial_evaluation_scores = col_to_numpy(initial_scores_df["Evaluation"])
    revised_evaluation_scores = col_to_numpy(revised_scores_df["Evaluation"])
    indices_where_evaluation_scores_changed = np.where(initial_evaluation_scores != revised_evaluation_scores)[0]
    print(f'number of participants who changed their evaluation scores: {len(indices_where_evaluation_scores_changed)}')

    initial_clarity_scores = col_to_numpy(initial_scores_df["Clarity"])
    revised_clarity_scores = col_to_numpy(revised_scores_df["Clarity"])
    indices_where_clarity_scores_changed = np.where(initial_clarity_scores != revised_clarity_scores)[0]
    print(f'number of participants who changed their clarity scores: {len(indices_where_clarity_scores_changed)}')

    # random clarity statistical significance test
    control_clarity_scores = col_to_numpy(control_scores_df["Clarity"])
    statistic, pvalue = permutation_test(revised_clarity_scores, control_clarity_scores, 100000)
    print(f'clarity pvalue: {pvalue}, statistic: {statistic}')

    indices_where_categorical_scores_changed = reduce(np.union1d, 
            (indices_where_significance_scores_changed, indices_where_novelty_scores_changed, 
            indices_where_soundness_scores_changed, indices_where_evaluation_scores_changed, 
            indices_where_clarity_scores_changed))
    print(f'number of participants who changed any of their categorical scores: {len(indices_where_categorical_scores_changed)}')

    num_participants_who_changed_categorical_scores_but_not_overall_scores = len(np.setdiff1d(indices_where_categorical_scores_changed, indices_where_overall_scores_changed))
    print(f'number of participants who changed any of their categorical scores but not their overall scores: {num_participants_who_changed_categorical_scores_but_not_overall_scores}')
    num_participants_who_changed_overall_scores_but_not_categorical_scores = len(np.setdiff1d(indices_where_overall_scores_changed, indices_where_categorical_scores_changed))
    print(f'number of participants who changed their overall scores but not any of their categorical scores: {num_participants_who_changed_overall_scores_but_not_categorical_scores}')

    num_participants_who_changed_both_overall_and_categorical_scores = len(np.intersect1d(indices_where_overall_scores_changed, indices_where_categorical_scores_changed))
    print(f'number of participants who changed both their overall and categorical scores: {num_participants_who_changed_both_overall_and_categorical_scores}')
    num_participants_who_changed_neither_overall_nor_categorical_scores = len(np.setdiff1d(np.arange(len(initial_overall_scores)), np.union1d(indices_where_overall_scores_changed, indices_where_categorical_scores_changed)))
    print(f'number of participants who changed neither their overall nor categorical scores: {num_participants_who_changed_neither_overall_nor_categorical_scores}')

def compare_seniority(initial_scores_df, revised_scores_df, control_scores_df):
    seniority_buckets = np.unique(col_to_numpy(initial_scores_df["Year"]))
    half = control_scores_df.shape[0] // 2   # 27
    exp_count = 0
    ctrl_count = 0
    initial_overall_score_half1 = []
    initial_overall_score_half2 = []
    revised_overall_score_half1 = []
    revised_overall_score_half2 = []
    control_overall_score_half1 = []
    control_overall_score_half2 = []

    for bucket in seniority_buckets:
        initial_bucket_overall_scores = col_to_numpy(initial_scores_df[initial_scores_df["Year"] == bucket]["Overall"])
        revised_bucket_overall_scores = col_to_numpy(revised_scores_df[revised_scores_df["Year"] == bucket]["Overall"])
        control_bucket_overall_scores = col_to_numpy(control_scores_df[control_scores_df["Year"] == bucket]["Overall"])
        print(f'experimental - bucket: {bucket}, length: {len(initial_bucket_overall_scores)}')
        print(f'control - bucket: {bucket}, length: {len(control_bucket_overall_scores)}')

        if len(initial_bucket_overall_scores.shape) > 0:
            for i in range(initial_bucket_overall_scores.shape[0]):
                initial_overall_score = initial_bucket_overall_scores[i]
                revised_overall_score = revised_bucket_overall_scores[i]
                if exp_count < half:
                    initial_overall_score_half1.append(initial_overall_score)
                    revised_overall_score_half1.append(revised_overall_score)
                else:
                    initial_overall_score_half2.append(initial_overall_score)
                    revised_overall_score_half2.append(revised_overall_score)

                exp_count += 1

        if len(control_bucket_overall_scores.shape) > 0:
            for i in range(control_bucket_overall_scores.shape[0]):
                control_overall_score = control_bucket_overall_scores[i]
                if ctrl_count < half:
                    control_overall_score_half1.append(control_overall_score)
                else:
                    control_overall_score_half2.append(control_overall_score)

                ctrl_count += 1
    
    # sanity check
    assert len(initial_overall_score_half1) == len(revised_overall_score_half1) == half
    assert len(initial_overall_score_half2) == len(revised_overall_score_half2) == half
    assert len(control_overall_score_half1) == len(control_overall_score_half2) == half

    print(f'mean junior initial overall score: {np.mean(initial_overall_score_half1)}')
    print(f'mean junior revised overall score: {np.mean(revised_overall_score_half1)}')
    print(f'mean junior control overall score: {np.mean(control_overall_score_half1)}')

    print(f'mean senior initial overall score: {np.mean(initial_overall_score_half2)}')
    print(f'mean senior revised overall score: {np.mean(revised_overall_score_half2)}')
    print(f'mean senior control overall score: {np.mean(control_overall_score_half2)}')

def export_data(initial_scores_df, revised_scores_df, control_scores_df):
    # remove "Timestamp" and "Email" columns from dataframes
    initial_scores_df = initial_scores_df.drop(columns=["Timestamp", "Email"])
    revised_scores_df = revised_scores_df.drop(columns=["Timestamp", "Email"])
    control_scores_df = control_scores_df.drop(columns=["Timestamp", "Email"])

    noncmu_institutions_list = ["UC Berkeley", "University of Pittsburgh", "University of Southern California", "University of Chicago", 
                                "UCB", "Stanford University", "Duke University, Computer Science Department", "University of Pennsylvania", 
                                "PITT", "RMU"]
    initial_institutions_list = initial_scores_df["Institution"].unique()
    control_institutions_list = control_scores_df["Institution"].unique()
    institutions_list = np.union1d(initial_institutions_list, control_institutions_list)
    cmu_institutions_list = np.setdiff1d(institutions_list, noncmu_institutions_list)

    # unify Carnegie Mellon University entries in "Institution" column
    for institution in cmu_institutions_list:
        initial_scores_df["Institution"] = initial_scores_df["Institution"].replace(institution, "Carnegie Mellon University")
        revised_scores_df["Institution"] = revised_scores_df["Institution"].replace(institution, "Carnegie Mellon University")
        control_scores_df["Institution"] = control_scores_df["Institution"].replace(institution, "Carnegie Mellon University")

    # replace PITT with University of Pittsburgh
    initial_scores_df["Institution"] = initial_scores_df["Institution"].replace("PITT", "University of Pittsburgh")
    revised_scores_df["Institution"] = revised_scores_df["Institution"].replace("PITT", "University of Pittsburgh")
    control_scores_df["Institution"] = control_scores_df["Institution"].replace("PITT", "University of Pittsburgh")

    initial_scores_df.to_csv('initial_scores.csv', index=False)
    revised_scores_df.to_csv('revised_scores.csv', index=False)
    control_scores_df.to_csv('control_scores.csv', index=False)

def main(args):
    np.random.seed(0) # for reproducibility
    study_data_folder = args.data_path
    initial_scores_df, revised_scores_df, control_scores_df = get_scores(study_data_folder, args.num_participants)

    if args.check_reduced_scores_after_rebuttal:
        check_reduced_scores_after_rebuttal(initial_scores_df, revised_scores_df)
        # only two, both for confidence, one from 2 -> 1 and one from 4 -> 3, does not cross threshold of 3.

    if args.run_all_scores:
        compare_categorical_scores(initial_scores_df, revised_scores_df, control_scores_df, args)

    elif args.run_overall_scores:
        initial_overall_scores = col_to_numpy(initial_scores_df["Overall"])
        revised_overall_scores = col_to_numpy(revised_scores_df["Overall"])
        control_overall_scores = col_to_numpy(control_scores_df["Overall"])
        run_tests(initial_overall_scores, revised_overall_scores, control_overall_scores, "overall", args)

    if args.run_confidence_scores: # testing if higher confidence results mirror this result
        # define high-confidence as >= 3
        hc_initial_overall_scores, hc_revised_overall_scores, hc_control_overall_scores = get_hc_overall_scores(initial_scores_df, revised_scores_df, control_scores_df)
        run_tests(hc_initial_overall_scores, hc_revised_overall_scores, hc_control_overall_scores, "hc overall", args)

        lc_initial_overall_scores, lc_revised_overall_scores, lc_control_overall_scores = get_lc_overall_scores(initial_scores_df, revised_scores_df, control_scores_df)
        run_tests(lc_initial_overall_scores, lc_revised_overall_scores, lc_control_overall_scores, "lc overall", args)

    if args.run_across_institutions:
        compare_across_institutions(initial_scores_df, revised_scores_df, control_scores_df)
    
    if args.run_change_locations:
        find_change_locations(initial_scores_df, revised_scores_df, control_scores_df)
    
    if args.run_seniority_comparison:
        compare_seniority(initial_scores_df, revised_scores_df, control_scores_df)
    
    if args.export_data:
        export_data(initial_scores_df, revised_scores_df, control_scores_df)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='study_data')
    parser.add_argument('--num_participants', type=int, default=108)
    parser.add_argument('--num_permutations', type=int, default=100000)

    parser.add_argument('--check_reduced_scores_after_rebuttal', type=bool, default=False)
    parser.add_argument('--run_all_scores', type=bool, default=False)
    parser.add_argument('--run_overall_scores', type=bool, default=False)
    parser.add_argument('--run_confidence_scores', type=bool, default=False)
    parser.add_argument('--run_across_institutions', type=bool, default=False)
    parser.add_argument('--run_change_locations', type=bool, default=False)
    parser.add_argument('--run_seniority_comparison', type=bool, default=False)
    parser.add_argument('--export_data', type=bool, default=True)

    parser.add_argument('--test_figure_effectiveness', type=bool, default=True)
    parser.add_argument('--compare_before_after', type=bool, default=True)
    parser.add_argument('--compare_between_groups', type=bool, default=True)
    parser.add_argument('--check_variances', type=bool, default=False)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    start_time = time.time()
    main(args)
    print(f'Total runtime for {args.num_permutations} permutations: {time.time() - start_time}')
    