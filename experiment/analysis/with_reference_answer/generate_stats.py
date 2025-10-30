import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import os

df = pd.read_csv(os.path.join(os.getcwd(), "evaluation", "all_responses_grading_separated_vllm_with_reference.csv"))
df_human = pd.read_csv(os.path.join(os.getcwd(), "evaluation", "human_evaluation.csv"))

# Remove all columns starting with feedback
df = df[[col for col in df.columns if not col.startswith('feedback')]]

# Remove all rows that contain the value 'Error' in any column
df = df[~df.apply(lambda row: row.astype(str).str.contains('Error').any(), axis=1)]

# Convert score criteria columns to int
criteria_columns = [f'score_criteria{i}' for i in range(1, 8)]
for col in criteria_columns:
    df[col] = df[col].astype(int)
    df_human[col] = df_human[col].astype(int)

# Add compound and std score
df['compound_score'] = df[criteria_columns].sum(axis=1)
df['std_score'] = df[criteria_columns].std(axis=1)
df_human['compound_score'] = df_human[criteria_columns].sum(axis=1)
df_human['std_score'] = df_human[criteria_columns].std(axis=1)

# Save the cleaned dataframe
df.to_csv('all_responses_grading_separated_cleaned.csv', index=False)

# Compute IQR for each group and add as a column
def add_iqr(grouped_df, original_df, score_col, group_cols):
    iqr_list = []
    for _, row in grouped_df.iterrows():
        group_filter = (original_df[group_cols[0]] == row[group_cols[0]]) & (original_df[group_cols[1]] == row[group_cols[1]])
        group_scores = original_df.loc[group_filter, score_col]
        q3 = group_scores.quantile(0.75)
        q1 = group_scores.quantile(0.25)
        iqr = q3 - q1
        iqr_list.append(iqr)
    grouped_df['iqr'] = iqr_list
    return grouped_df

# Group by intervention and score criterias
grouped_intervention_score_criteria1 = df.groupby(['intervention', 'phase'])['score_criteria1'].agg(['sum','median', 'std', 'count']).reset_index()
grouped_intervention_score_criteria1['mean'] = df.groupby(['intervention', 'phase'])['score_criteria1'].mean().values
grouped_intervention_score_criteria1 = add_iqr(grouped_intervention_score_criteria1, df, 'score_criteria1', ['intervention', 'phase'])

grouped_intervention_score_criteria2 = df.groupby(['intervention', 'phase'])['score_criteria2'].agg(['sum','median', 'std', 'count']).reset_index()
grouped_intervention_score_criteria2['mean'] = df.groupby(['intervention', 'phase'])['score_criteria2'].mean().values
grouped_intervention_score_criteria2 = add_iqr(grouped_intervention_score_criteria2, df, 'score_criteria2', ['intervention', 'phase'])

grouped_intervention_score_criteria3 = df.groupby(['intervention', 'phase'])['score_criteria3'].agg(['sum','median', 'std', 'count']).reset_index()
grouped_intervention_score_criteria3['mean'] = df.groupby(['intervention', 'phase'])['score_criteria3'].mean().values
grouped_intervention_score_criteria3 = add_iqr(grouped_intervention_score_criteria3, df, 'score_criteria3', ['intervention', 'phase'])

grouped_intervention_score_criteria4 = df.groupby(['intervention', 'phase'])['score_criteria4'].agg(['sum','median', 'std', 'count']).reset_index()
grouped_intervention_score_criteria4['mean'] = df.groupby(['intervention', 'phase'])['score_criteria4'].mean().values
grouped_intervention_score_criteria4 = add_iqr(grouped_intervention_score_criteria4, df, 'score_criteria4', ['intervention', 'phase'])

grouped_intervention_score_criteria5 = df.groupby(['intervention', 'phase'])['score_criteria5'].agg(['sum','median', 'std', 'count']).reset_index()
grouped_intervention_score_criteria5['mean'] = df.groupby(['intervention', 'phase'])['score_criteria5'].mean().values
grouped_intervention_score_criteria5 = add_iqr(grouped_intervention_score_criteria5, df, 'score_criteria5', ['intervention', 'phase'])

grouped_intervention_score_criteria6 = df.groupby(['intervention', 'phase'])['score_criteria6'].agg(['sum','median', 'std', 'count']).reset_index()
grouped_intervention_score_criteria6['mean'] = df.groupby(['intervention', 'phase'])['score_criteria6'].mean().values
grouped_intervention_score_criteria6 = add_iqr(grouped_intervention_score_criteria6, df, 'score_criteria6', ['intervention', 'phase'])

grouped_intervention_score_criteria7 = df.groupby(['intervention', 'phase'])['score_criteria7'].agg(['sum','median', 'std', 'count']).reset_index()
grouped_intervention_score_criteria7['mean'] = df.groupby(['intervention', 'phase'])['score_criteria7'].mean().values
grouped_intervention_score_criteria7 = add_iqr(grouped_intervention_score_criteria7, df, 'score_criteria7', ['intervention', 'phase'])

grouped_intervention_compound_score = df.groupby(['intervention', 'phase'])['compound_score'].agg(['sum','median', 'std', 'count']).reset_index()
grouped_intervention_compound_score['mean'] = df.groupby(['intervention', 'phase'])['compound_score'].mean().values
grouped_intervention_compound_score = add_iqr(grouped_intervention_compound_score, df, 'compound_score', ['intervention', 'phase'])

grouped_model_compound_score = df.groupby(['model-name', 'phase'])['compound_score'].agg(['sum','median', 'std', 'count']).reset_index()
grouped_model_compound_score['mean'] = df.groupby(['model-name', 'phase'])['compound_score'].mean().values
grouped_model_compound_score['intervention'] = df.groupby(['model-name', 'phase'])['intervention'].first().values
grouped_model_compound_score = add_iqr(grouped_model_compound_score, df, 'compound_score', ['model-name', 'phase'])


# Save the grouped dataframes
grouped_intervention_score_criteria1.to_csv('all_responses_grading_separated_grouped_intervention_score_criteria1.csv', index=False)
grouped_intervention_score_criteria2.to_csv('all_responses_grading_separated_grouped_intervention_score_criteria2.csv', index=False)
grouped_intervention_score_criteria3.to_csv('all_responses_grading_separated_grouped_intervention_score_criteria3.csv', index=False)
grouped_intervention_score_criteria4.to_csv('all_responses_grading_separated_grouped_intervention_score_criteria4.csv', index=False)
grouped_intervention_score_criteria5.to_csv('all_responses_grading_separated_grouped_intervention_score_criteria5.csv', index=False)
grouped_intervention_score_criteria6.to_csv('all_responses_grading_separated_grouped_intervention_score_criteria6.csv', index=False)
grouped_intervention_score_criteria7.to_csv('all_responses_grading_separated_grouped_intervention_score_criteria7.csv', index=False)
grouped_intervention_compound_score.to_csv('all_responses_grading_separated_grouped_intervention_mean.csv', index=False)
grouped_model_compound_score.to_csv('all_responses_grading_separated_grouped_model_mean.csv', index=False)

# Filter for phase 2
df_p2 = df[df['phase'] == 2]

# Calculate Cronbach's alpha for the seven criteria
def cronbach_alpha(df, columns):
    k = len(columns)
    item_scores = df[columns]
    variances = item_scores.var(axis=0, ddof=1)
    total_score = item_scores.sum(axis=1)
    total_variance = total_score.var(ddof=1)
    alpha = (k / (k - 1)) * (1 - variances.sum() / total_variance)
    return alpha

alpha = cronbach_alpha(df, criteria_columns)
print(f"Cronbach's alpha for the seven criteria: {alpha:.4f}")
# Save Cronbach's alpha to a csv file
pd.DataFrame([{
    'cronbach_alpha': alpha
}]).to_csv('cronbach_alpha_seven_criteria.csv', index=False)

# Shapiro-Wilk test on mean compound score and score criterias
shapiro_wilk_results = []
for criteria in criteria_columns + ['compound_score']:
    shapiro_stat, shapiro_p = stats.shapiro(df_p2[criteria])
    shapiro_wilk_results.append({
        'criteria': criteria,
        'W': shapiro_stat,
        'p_value': shapiro_p
    })
pd.DataFrame(shapiro_wilk_results).to_csv('shapiro_wilk_mean_compound_score.csv', index=False)

# Mann-Whitney U test
mannwhitney_results = []
for criteria in criteria_columns + ['compound_score']:
    group1 = df_p2[df_p2['intervention'] == True][criteria]
    group2 = df_p2[df_p2['intervention'] == False][criteria]
    u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    mannwhitney_results.append({
        'criteria': criteria,
        'u_stat': u_stat,
        'p_value': p_value
    })
pd.DataFrame(mannwhitney_results).to_csv('mannwhitney_u_test_results.csv', index=False)


# Cohen's d for compound score
group1 = df_p2[df_p2['intervention'] == True]['compound_score']
group2 = df_p2[df_p2['intervention'] == False]['compound_score']
n1, n2 = len(group1), len(group2)
s1, s2 = group1.std(), group2.std()
s_pool = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
cohens_d = (group1.mean() - group2.mean()) / s_pool**0.5
pd.DataFrame([{
    'criteria': 'compound_score',
    'cohens_d': cohens_d
}]).to_csv('cohens_d_compound_score.csv', index=False)


# Rank-Biserial Correlation
rbc_results = []
for criteria in criteria_columns + ['compound_score']:
    group1 = df_p2[df_p2['intervention'] == False][criteria]
    group2 = df_p2[df_p2['intervention'] == True][criteria]
    u_stat, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    n1, n2 = len(group1), len(group2)
    rbc = 1 - (2 * u_stat) / (n1 * n2)
    rbc_results.append({
        'criteria': criteria,
        'rank_biserial_correlation': rbc
    })
pd.DataFrame(rbc_results).to_csv('rank_biserial_correlation_results.csv', index=False)

# Compare the human evaluation with the assesstment of M-prometheus (df_human vs the same entries in df)
# Identify the entries by model-name, phase and attempt. Apply pearson correlation and spearman correlation
comparison_results = []
for criteria in criteria_columns + ['compound_score']:
    merged = pd.merge(df_human[['model-name', 'phase', 'attempt', criteria]], df[['model-name', 'phase', 'attempt', criteria]], on=['model-name', 'phase', 'attempt'], suffixes=('_human', '_model'))
    pearson_corr, pearson_p = stats.pearsonr(merged[f'{criteria}_human'], merged[f'{criteria}_model'])
    spearman_corr, spearman_p = stats.spearmanr(merged[f'{criteria}_human'], merged[f'{criteria}_model'])
    comparison_results.append({
        'criteria': criteria,
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p
    })
pd.DataFrame(comparison_results).to_csv('human_model_comparison_results.csv', index=False)

# Calculate agreement percentage and Cohen's kappa for all criteria and compound_score
criteria_to_compare = criteria_columns + ['compound_score']
agreement_stats = []

for criteria in criteria_to_compare:
    comparison_winner_results = {}
    agreement_counter = 0
    for i in range(len(df_human)):
        for j in range(len(df_human)):
            if i == j:
                continue
            else:
                human_response_1_score = df_human.iloc[i][criteria]
                human_response_2_score = df_human.iloc[j][criteria]
                model_response_1_score = df[
                    (df['model-name'] == df_human.iloc[i]['model-name']) &
                    (df['phase'] == df_human.iloc[i]['phase']) & 
                    (df['attempt'] == df_human.iloc[i]['attempt'])
                ][criteria].values[0]
                model_response_2_score = df[
                    (df['model-name'] == df_human.iloc[j]['model-name']) &
                    (df['phase'] == df_human.iloc[j]['phase']) & 
                    (df['attempt'] == df_human.iloc[j]['attempt'])
                ][criteria].values[0]

                if human_response_1_score > human_response_2_score:
                    human_response_winner = i
                elif human_response_1_score < human_response_2_score:
                    human_response_winner = j
                else:
                    human_response_winner = 'tie'
                
                if model_response_1_score > model_response_2_score:
                    model_response_winner = i
                elif model_response_1_score < model_response_2_score:
                    model_response_winner = j
                else:
                    model_response_winner = 'tie'
                
                if human_response_winner == model_response_winner:
                    agreement_counter += 1

                comparison_winner_results[f'{i}-{j}'] = {
                    "human_response_winner": human_response_winner,
                    "model_response_winner": model_response_winner
                }

    agreement_percentage = agreement_counter / len(comparison_winner_results) * 100
    human_winners = [v['human_response_winner'] for v in comparison_winner_results.values()]
    model_winners = [v['model_response_winner'] for v in comparison_winner_results.values()]
    kappa = cohen_kappa_score(human_winners, model_winners)
    agreement_stats.append({
        'criteria': criteria,
        'agreement_percentage': agreement_percentage,
        'cohens_kappa': kappa
    })
    print(f"{criteria}: Agreement Percentage = {agreement_percentage:.2f}%, Cohen's Kappa = {kappa:.4f}")

# Save results to CSV
pd.DataFrame(agreement_stats).to_csv('pairwise_agreement_stats.csv', index=False)
