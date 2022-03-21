import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import tree
from sklearn.tree import export_graphviz
from subprocess import call 
from IPython.display import Image
from types import MappingProxyType

# USED TO FORMAT FIGURES
PRETTY_NAMES = MappingProxyType(
            {
                'population_total':'Total population', 
                'percent_racial_minority': 'Percentage racial minority\n(all persons except white, non-Hispanic)', 
                'percent_households_more_people_than_rooms':'Percent of households with more people than rooms', 
                'median_non_home_dwell_time_percent_increase_jan_to_april': 'Percent increase in median non-home dwell time*', 
                'percent_freq_or_always_mask_use': 'Percent of survey respondents who reported\nfrequently or always wearing a mask in public', 
                'telework_score':'Estimate of the percentage of the population that can telework', 
                'start_date':'Start date of the policy environment', 
                'percent_republican_2016_pres':'Percent of Republican votes in the 2016 presidential election', 
                'percent_over_65':'Percent of population over 65 years old', 
                'percentage_completely_home_percent_increase_jan_to_april':'Percent increase in individuals staying at home*', 
                'percent_below_poverty_line':'Percent of population below the poverty line', 
                'median_home_dwell_time_percent_increase_jan_to_april':'Percent increase in home dwell time*', 
                'population_density':'Population density', 
                'avg_new_cases':'Average new reported COVID-19 cases\nper 100,000 people'
            })

def plot_feature_importance_bar_chart(feature_importance, PRETTY_NAMES, figure_name):
    sns.set(font_scale = 1.25, style = 'whitegrid')
    plt.figure(figsize = (8,5))
    ax = sns.barplot(x = 'importance', y = 'feature', data = feature_importance.replace(PRETTY_NAMES), palette = 'Greens_r')
   
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    
    plt.savefig('reports/figures/' + figure_name, bbox_inches = 'tight')
    
    
# NOTE on calling code 
#  df_anc_pred = df_anc_pred.set_index('fips').merge(features_w_anc[feats_to_incl], left_index = True, right_index = True)
#  df_anc_pred.rename(columns = PRETTY_NAMES, inplace = True)

def plot_real_predicted_scatter(preds_and_feats, figure_name):
    for feat in preds_and_feats.columns:
        plt.rc('font', size=12) 
        plt.style.use('seaborn-whitegrid')
        
        if feat == 'Start date of the policy environment':
            preds_and_feats[feat] = [np.datetime64(int(x), 'ns') for x in preds_and_feats[feat]]
            
        plt.scatter(preds_and_feats[feat], preds_and_feats.y_test, c = 'b', alpha = 0.8, marker = '.', label = 'Real')
        plt.scatter(preds_and_feats[feat], preds_and_feats.RRF_pred, c = 'r', alpha = 0.8, marker = '.', label = 'Predicted')
        
        title_str = 'Real and Predicted Average New COVID-19 Cases in the Test Dataset ($n$  = {})'
        plt.title(title_str.format(len(preds_and_feat.index())))
        
        plt.xlabel(feat)
        plt.ylabel('Average New COVID-19 Cases per 100k People')
        
        plt.grid(color = '#D3D3D3', linestyle = 'solid')
        plt.legend(loc = 'upper right', frameon = True)
        

        plt.savefig('reports/figures/' + figure_name + '_' + feat.replace('\n', ' ').replace('*', '') + '_scatter.png')
        plt.show()
        
def plot_correlation_between_features(features_dataset, png_filename, r_filename, p_filename):
    '''Assumes feats_to_incl (list) and PRETTY_NAMES (dict of str:str pairs) are global variables that respectively specify the 
    features of interest and the nicely formatted names corresponding to those features 
    
    Args:
        features_dataset (pandas df): dataset with feature values for each instance 
        png_filename (str): file to save the heatmap image to 
        r_filename (str): file to save the pearson correlation coefficients to 
        p_filename (str): file to save the corresponding p-values to
    '''
    sns.set(font_scale=.75)
    fig, ax = plt.subplots()
    
    # feats_heatmap = [x for x in feats_to_incl if x != "avg_new_cases"]
    pretty_df = features_dataset[feats_heatmap].rename(columns = PRETTY_NAMES)
    sns.heatmap(pretty_df.corr().round(2), annot = True, cmap = 'RdBu_r', center = 0.0, vmin = -1.0, vmax = 1.0)
    plt.xticks(rotation = 45, horizontalalignment = 'right')
    plt.yticks(rotation = 0, horizontalalignment = 'right')
    
    # save the correlation coefficients to a CSV file
    pretty_df.corr().to_csv(r_filename)
    
    # save the corresponding p_values to another CSV file
    calculate_pvalues(pretty_df).to_csv(p_filename)
    
    # plt.figure(figsize=(6,5))
    plt.title('Pearson Correlation between Variables\nin Study Sample (n = {})'.format(len(pretty_df.index)), fontsize = 14)
    ax.tick_params(labelsize=11)
    
    # save picture to PNG file
    # plt.yticks(rotation = 0, verticalalignment="center")
    plt.savefig(png_filename, bbox_inches="tight")
    
    
def plot_correlation_coefficient(figure_name, features_w_anc):
    sns.set(font_scale = 1.7, style = 'whitegrid')
    plt.figure(figsize = (8,7))
    corr_col = 'Correlation coefficient between feature\nand average new reported COVID-19 cases per 100,000 people'
    case_col = 'Average new reported COVID-19 cases\nper 100,000 people'
    feat_corr_df = pd.DataFrame(columns = ['Feature', 'Correlation with average new cases'])
    corr_data = features_w_anc.rename(columns = PRETTY_NAMES).corr().loc[case_col]
    feat_corr_df['Feature'] = corr_data.index
    feat_corr_df[corr_col] = corr_data.tolist()
    feat_corr_df['Hue'] = np.where(feat_corr_df[corr_col] > 0, 
                                   'Postive correlation', 
                                   'Negative correlation')
    feat_corr_df = feat_corr_df[feat_corr_df['Feature'] != case_col]
    feat_corr_df.replace(PRETTY_NAMES, 
                         inplace = True)
    feat_corr_df = feat_corr_df[feat_corr_df['Feature'] != 'avg_new_cases']
    sns.set_palette(palette=['#6F1A07', '#1D4E89'])
    with sns.axes_style("whitegrid"):
        ax = sns.barplot(x = corr_col, y = 'Feature', 
                         hue = 'Hue', data = feat_corr_df)
        ax.get_legend().set_visible(False)
    # ax.set_title('Importance of features for estimating the percent days with decreasing new cases per 100k')
    plt.savefig(figure_name, bbox_inches = 'tight')
    return feat_corr_df

def plot_tree_estimator(regr_model, feature_list, estimator_n = 5):
        
    estimator = regr_model.estimators_[estimator_n]
    export_graphviz(estimator, out_file = 'reports/figures/tree.dot', 
                    feature_names = feature_list,
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)
    
    call(['dot', '-Tpng', 'reports/figures/tree.dot', '-o', 'reports/figures/tree.png', '-Gdpi=600'], shell = True)