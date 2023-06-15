# ReviewerAnchoring
Code, Data and Interfaces for Reviewer Anchoring Experiment

Code:
The code for the main tests for significance using permutation test with Mann Whitney U statistic is in analysis_MWU.py. The code for the supplemental analysis with Mann Whitney U statistic and bootstrapped CIs is in analysis_MWU_supp.py. The code for the old analysis (statistic = difference in means) is in analysis.py.

Data:
Anonymized data collected from the experiment are in data/. Control group review scores, experimental group initial review scores, and experimental group final review scores are in control.csv, initial.csv, and revised.csv. Each file contains numerical scores for {Significance, Novelty, Soundness, Evaluation, Clarity} on a 1-4 scale, Overall scores on a 1-10 scale, Confidence on a 1-5 scale, Insitution (CMU or Non-CMU), and Year (Phd xth year or Post-PhD).

Interfaces:
The paper provided to participants to review is fake_paper/index.html. The animated/frozen GIF figures are provided in fake_paper/images.

The reviewer form provided in this directory is a printout version of the Google Form participants filled in online. The full google form is available at https://forms.gle/LRQmz4mcaQohZ89v6.
