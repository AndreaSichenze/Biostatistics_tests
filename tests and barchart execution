import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import datetime

# Normality and significance thresholds
normality_threshold = 50
p_value_threshold = 0.005

# Normality test (Shapiro-Wilk and Kolmogorov-Smirnov)
def normality_test(df, target_variable=None):
    n_results = {}
    normal_vars = []
    non_normal_vars = []
    target_normal = None

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            variable = df[column].dropna()
            sample_size = variable.shape[0]

            if sample_size < normality_threshold:
                stat, p_value = stats.shapiro(variable)
                method = "Shapiro-Wilk"
            else:
                variable_normalized = (variable - variable.mean()) / variable.std()
                stat, p_value = stats.kstest(variable_normalized, 'norm')
                method = "Kolmogorov-Smirnov"

            is_normal = p_value >= p_value_threshold
            n_results[column] = {
                "Sample Size": sample_size,
                "Test": method,
                "Statistic": stat,
                "p-value": p_value,
                "Normal Distribution": "Yes" if is_normal else "No"
            }

            if column == target_variable:
                target_normal = is_normal
            elif is_normal:
                normal_vars.append(column)
            else:
                non_normal_vars.append(column)

    return n_results, normal_vars, non_normal_vars, target_normal

# Chi-squared test for categorical variables
def chisquare_test(df, var1, var2):
    contingency_table = pd.crosstab(df[var1], df[var2])
    stat, p, dof, expected = chi2_contingency(contingency_table)
    return p

# Mann-Whitney U test for non-normal numeric variables
def MWU(df, target_variable, grouping_variable):
    unique_values = df[grouping_variable].unique()
    if len(unique_values) >= 2:
        group1 = df[df[grouping_variable] == unique_values[0]][target_variable]
        group2 = df[df[grouping_variable] == unique_values[1]][target_variable]
    else:
        print(f"Warning: Grouping variable '{grouping_variable}' has less than 2 unique values. Skipping Mann-Whitney U test.")
        return {"U_statistic": None, "p_value": None, "Significant": None}

    stat, p_value = mannwhitneyu(group1, group2)
    result = {
        "U_statistic": stat,
        "p_value": p_value,
        "Significant": "Yes" if p_value < p_value_threshold else "No"
    }
    return result

# Run statistical tests
def run_tests(df, target_variable, categorical_vars, normal_vars, non_normal_vars, target_normal):
    results = {}

    # Chi-squared test for categorical variables
    for var in categorical_vars:
        p_value = chisquare_test(df, target_variable, var)
        print(f"Chi-squared test for '{target_variable}' vs '{var}': p-value = {p_value}")
        results[f'{var} (chi)'] = {'p_value': p_value, 'test': 'Chi-squared'}

    # Test for numeric variables
    for var in normal_vars + non_normal_vars:
        if target_normal and var in normal_vars:
            # Both normal: Student's t
            stat, p_value = ttest_ind(
                df[df[target_variable] == 0][var].dropna(),
                df[df[target_variable] == 1][var].dropna()
            )
            print(f"Student’s t-test for '{target_variable}' vs '{var}': p-value = {p_value}")
            results[f'{var} (t)'] = {'p_value': p_value, 'test': 'T-test'}
        else:
            # Else: Mann-Whitney U
            result = MWU(df, target_variable, var)
            p_value = result['p_value']
            print(f"Mann-Whitney U test for '{target_variable}' vs '{var}': p-value = {p_value}")
            results[f'{var} (MW)'] = {'p_value': p_value, 'test': 'Mann-Whitney U'}

    return results


# Plot Barchart
def plot_results(results, target_variable, save_plot_pdf=False):
    p_values = {var: result['p_value'] for var, result in results.items()}
    sorted_p_values = dict(sorted(p_values.items(), key=lambda item: item[1], reverse=True))

    labels = list(sorted_p_values.keys())
    values = list(sorted_p_values.values())
    colors = ['skyblue' if p > p_value_threshold else 'red' for p in values]

    plt.figure(figsize=(8, 5))
    plt.barh(range(len(labels)), values, color=colors)
    plt.title(r'$\it{p}$-value in relationship with ' + target_variable)
    plt.xlabel('P-value')

    for i, (label, value) in enumerate(zip(labels, values)):
        if value < p_value_threshold:
            updated_label = f'{label} ***'
            plt.text(-0.02, i, updated_label, color='blue', va='center', ha='right', fontsize=10)
        else:
            plt.text(-0.02, i, label, color='black', va='center', ha='right', fontsize=10)

        if value < p_value_threshold:
            sci_value = f'{value:.2e}'
            mantissa, exponent = sci_value.split('e')
            formatted_p = f'{float(mantissa):.2f}*10^({int(exponent)})'
            plt.annotate(
                formatted_p,
                xy=(value, i),
                xytext=(value + 0.01, i),
                fontsize=10,
                va='center'
            )

    plt.axvline(x=p_value_threshold, color='red', linestyle='--', label=f'α = {p_value_threshold}')
    plt.grid(True)
    plt.yticks(range(len(labels)), [''] * len(labels))
    plt.legend()
    plt.tight_layout()

    plt.text(-0.15, len(labels) + 0.2, 'Variables', fontsize=12, fontweight='bold', ha='center')

    if save_plot_pdf:
        plt.savefig(f"{target_variable}_p_values_plot.pdf")

    plt.show()

# Execution - Neuroblastoma
inputdatafilename = '10_7717_peerj_5665_dataYM2018_neuroblastoma1.csv'
df = pd.read_csv(inputdatafilename)
target_variable = 'outcome'
categorical_vars = ['sex', 'site', 'stage', 'risk', 'autologous_stem_cell_transplantation', 'radiation', 'degree_of_differentiation', 'UH_or_FH', 'MYCN_status ', 'surgical_methods']
numeric_vars = ['age', 'time_months']

print(f"Data Analysis of {inputdatafilename} Dataset \n")

n_results, normal_vars, non_normal_vars, target_normal = normality_test(df[numeric_vars + [target_variable]], target_variable)
results = run_tests(df, target_variable, categorical_vars, normal_vars, non_normal_vars, target_normal)
plot_results(results, target_variable, save_plot_pdf=True)
