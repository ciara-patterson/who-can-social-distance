import pandas as pd 
import numpy as np
import requests
from datetime import datetime

import sys
sys.path.append('../..')
sys.path.append('..')

class CountiesDataset:
    '''
    The US county dataset from which the predictive model can be built. 
    It contains both features and the average number of COVID-19 cases 
    during policies that met the inclusion criteria (value that we are predicting).

    Attributes
        dataset: A DataFrame where each county is a row and each column is a feature
            or the average number of cases. The index is the county FIPS codes.
    
    '''

    def __init__(self, load = True) -> None:
        '''
        Initiates a simple county dataset. To save time, once the dataset is created once,
        it can be loaded from a CSV (load = True).
        '''
        if load:
            file_path = 'data/processed/county_dataset.csv'
            self.dataset = pd.read_csv(file_path, index_col = 0)
            self.dataset['start_date'] = self.dataset['start_date'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d'))
        else:
            self.dataset = self.build_dataset()
            self.dataset.to_csv('data/processed/features_dataset.csv')
         
    def build_dataset(self):
        cases = get_nyt_case_data() 

        # get control data 
        control_data = get_policy_control_data()
        start_end_df = get_start_end_date(control_data)

        # get only case information for counties that are in both datasets
        shared_fips = cases.merge(start_end_df, left_index = True, right_index = True, how = 'inner')
        cases = cases[(cases.index.isin(shared_fips.index))]

        # calculate additional case information 
        cases = add_new_cases_deaths(cases)
        cases['date'] = cases['date'].astype('datetime64')

        # get case results while policy environment was in effect
        case_results = get_results(start_end_df, cases)

        # get basic features for these counties 
        features = CountyFeatures(fips_codes = cases.index.unique.tolist())
        features.build_features_dataset()
        
        self.dataset = case_results.merge(features, left_index = True, right_index = True)



class CountyFeatures:
    ''' 
    The US County dataset with the features that are used in the final model 
    (% racial minority, % reported frequent or always mask use, etc.)

    Attributes:
        fips_codes (list): County FIPS codes to be included in the dataset 
        features_dataset (DataFrame): DataFrame where each row is a county 
            and each feature is a column. fips_codes is the index.
    '''

    def __init__(self, fips_codes, starting_dataset = None):
        self.fips_codes = fips_codes
        
        # initialize the dataset's features with either a starting dataset or empty dataframe
        if starting_dataset:
            self.features_dataset = starting_dataset
        else:
            self.features_dataset = pd.DataFrame(index = fips_codes)

    def build_features_dataset(self):
        '''Build a features dataset with information on each FIPS code.'''
        
        # get the same heatmap for all U.S. counties to see if certain correlations hold elsewhere
        county_df = self.features_dataset
        
        # telework scores are generated using O*NET likelihood code
        # county_df['county_fips'] = self.fips_codes
        # county_df.set_index('county_fips', inplace = True)
        
        # get telework data
        telework_data = self.__get_us_telework_data()
        county_df = county_df.merge(telework_data, left_index = True, right_index = True, how = 'inner')
        
        # read in NY times mask survey data for all US counties
        mask_data = self.__get_mask_survey()
        # mask_data = self.__get_freq_always(mask_data)
        county_df = county_df.merge(mask_data['percent_freq_or_always_mask_use'], how = 'inner', left_index = True, right_index = True)
        
        # get 2016 election data for all US counties
        elec_2016_data = self.__get_election_data()
        county_df = county_df.merge(elec_2016_data['percent_republican_2016_pres'], how = 'inner', left_index = True, right_index = True)
        
        # get CDC social vulnerability data for all US counties
        svi_data = self.__get_cdc_svi_data()
        county_df = county_df.merge(svi_data, how = 'inner', left_index = True, right_index = True)

        # where available, add information on changes in mobility from SafeGraph
        sg_data = self.__get_safegraph_data()
        county_df = county_df.merge(sg_data, left_index = True, right_index = True, how = 'left')

        self.features_dataset = county_df
        return self.features_dataset

    def output_features_dataset_csv(self, filepath):
        self.features_dataset.to_csv(filepath)

    def __get_safegraph_data(self):
        '''Note that safegraph data is only available for all counties'''
        safegraph_data_path = "data/processed/safegraph_data_eligible_counties.csv"
        sg_data = pd.read_csv(safegraph_data_path)
        sg_data.set_index('fips', inplace = True)
        return sg_data['median_non_home_dwell_time_percent_increase_jan_to_april']

    def __get_us_telework_data(self):
        telework_data_path = 'data/processed/US_telework_estimates_counties.csv'
        telework_scores = pd.read_csv(telework_data_path)
        return telework_scores.set_index('GEO_ID')['telework_score']
    
    def __get_mask_survey(self):
        '''Extracts the mask survey data from the NYT Github and loads into a Pandas DataFrame
        
        Returns:
            mask_df (DataFrame): DataFrame with the mask survey data by county
        '''
        #fetch the raw survey responses from the New York Times github
        data_filename = 'mask-use-by-county.csv' #only CSV file with recorded start and end dates
        csv_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/mask-use/' + data_filename
        
        req = requests.get(csv_url)
        url_content = req.content
        
        #write to a csv file
        csv_file = open('data/raw' + data_filename, 'wb')
        csv_file.write(url_content)
        csv_file.close()
        
        # read in csv file into dataframe
        mask_df = pd.read_csv('data/raw' + data_filename) 
        
        #create boolean series identifying relevant county info
        counties_bool = mask_df.COUNTYFP.isin(self.fips_codes)

        #return the relevant parts of the dataframe
        mask_df = mask_df[counties_bool]
        mask_df.set_index('COUNTYFP', inplace = True)
        mask_df['percent_freq_or_always_mask_use'] = mask_df['FREQUENTLY'] + mask_df['ALWAYS']
        return mask_df

    def __get_election_data(self):
        '''
        Reads data from a CSV with 2000-2016 county-level presidential election data and
        returns a DataFrame with the FIPS and the % of Republican votes
       
        Returns:
            (DataFrame): DataFrame with a county FIPS code and the % of votes for the Republican candidate in 2016
        
        '''
        all_elecs = pd.read_csv("data/raw/county_pres_2000_2016.csv")

        elec_2016 = all_elecs[all_elecs['year'] == 2016]
        rep_elec_2016 = elec_2016[elec_2016['party'] == 'republican']
        rep_elec_2016['percent_republican_2016_pres'] = rep_elec_2016['candidate_votes'] / rep_elec_2016['total_votes']
        
        counties_bool = rep_elec_2016.county_fips.isin(self.fips_codes)
        rep_elec_2016 = rep_elec_2016[counties_bool]

        rep_elec_2016 = rep_elec_2016[['county_fips', 'percent_republican_2016_pres']]
        rep_elec_2016['county_fips'] = rep_elec_2016['county_fips'].astype('int')
        rep_elec_2016.set_index('county_fips', inplace = True)

        return rep_elec_2016
        

    def __get_cdc_svi_data(self):
        '''Read in CDC social vulnerability data by county
        
        Args:
            county_fips_codes (list): list of FIPS code for counties in the dataset
                this argument is useful when examining counties in a certain state or region rather than the entire US
        
        Returns:
            (DataFrame): DataFrame with county-level CDC SVI data
        
        '''
        svi = pd.read_csv("data/raw/SVI2018_US_COUNTY.csv")
        
        counties_bool = svi.FIPS.isin(self.fips_codes)
        svi = svi[counties_bool]
        
        svi['population_density'] = svi['E_TOTPOP'] / svi['AREA_SQMI']

        svi = svi[['FIPS', 'E_TOTPOP', 'EP_UNINSUR', 'EP_POV', 'EP_CROWD', 'EP_MINRTY', 'EP_NOHSDP', 'EP_AGE65', 'population_density']]
    
        rename_dict = {'EP_UNINSUR':'percent_uninsured', 
                    'E_TOTPOP':'population_total',
                    'EP_POV':'percent_below_poverty_line', 
                    'EP_CROWD':'percent_households_more_people_than_rooms', 
                    'EP_MINRTY':'percent_racial_minority', 
                    'EP_NOHSDP':'percent_without_high_school_diploma',
                    'EP_AGE65':'percent_over_65'}
        
        formatted_svi = svi.rename(columns = rename_dict)
        formatted_svi.set_index('FIPS', inplace = True)

        formatted_svi = formatted_svi.replace(-999, np.nan)
        
        return formatted_svi


def get_nyt_case_data():
    # load case data from the NYT times 
    cases = get_county_case_dataset()
    cases['date'] = cases['date'].astype('datetime64')
    cases = cases.dropna(subset = ['fips']) # drop data without identifying info 

    # add the date to the county case dataset 
    cases = add_date_county_case_dataset(cases)
    cases['fips'] = cases['fips'].astype('int64')
    cases.set_index('fips', inplace = True)

    # append population estimates to the COVID-19 caseload data 
    pop_df = get_population_totals()
    cases = cases.merge(pop_df['population'].astype('float64'), left_index = True, right_index = True)

    return cases
    
def get_county_case_dataset():
    '''Downloads the COVID-19 county data from the New York Times Github Repository
    
    Returns:
        df (dataframe): NYT Covid-19 data
    '''
#     data_filename = 'us-counties.csv' #only CSV file with recorded start and end dates
    # download the NYT covid-19 data frome github
    nyt_covid19_file = "us-counties.csv"
    csv_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/' + nyt_covid19_file
    req = requests.get(csv_url)
    url_content = req.content

    #write to a csv file
    csv_file = open('data/raw/' + nyt_covid19_file, 'wb')
    csv_file.write(url_content)
    csv_file.close()

    df = pd.read_csv('data/raw/' + nyt_covid19_file) #read in csv file into dataframe
    return df

def create_date_table(start='2020-01-21'):
    '''Creates an empty date table from the "start" of the pandemic until today. This is used for creating the daily COVID-19 data
    
    Args:
        start (datetime str): date of first recorded case in the US
    '''
    now = datetime.now()
    end = now.strftime('%Y-%m-%d')
    df = pd.DataFrame({"date": pd.date_range(start, end)})
    return df

def add_date_county_case_dataset(cases):
    '''Adds data for each county from the first confirmed cases in the US until today.
    
    In the original NYT data, sometimes if no new cases are observed for the county, the date is not included.
    However, for analysis purposes, we want to know if some counties saw no new COVID-19 cases on certain dates.
    
    Args: 
        case (DataFrame):

    Returns:

    '''

    new_dfs = []
    for fips_code in cases['fips'].unique():
        dates = create_date_table()
        county_df = cases[cases['fips'] == fips_code]
        county_dates = county_df.merge(dates, on = 'date', how = 'outer')

        #forward fill case date for dates with no new cases
        cols = list(county_dates.columns)
        cols.remove('date')
        county_dates.loc[:,cols] = county_dates.sort_values(by = 'date').loc[:,cols].ffill(axis=0)

        #dates before counties first confirmed case will be empty
        #fill empty rows with county info, 0 cases, 0 deaths
        county_name = county_df['county'].unique()[0]
        state_name = county_df['state'].unique()[0]
        values = {'fips':fips_code, 'county':county_name, 'state':state_name, 'cases':0.0, 'deaths':0.0}
        county_dates.fillna(value=values, inplace = True)
        
        new_dfs.append(county_dates)
    return pd.concat(new_dfs)

def get_population_totals():
    '''Returns a dataset with 2019 ACS 5 year population estimates. Most current population available for 
    all counties in the dataset.'''

#     filename = 'all_state_data/ACS_2019_5yr_population_estimates_US.csv'
    pop_file_path = 'data/edited/ACS_2019_5yr_population_estimates_US.csv'
    pop_df = pd.read_csv(pop_file_path)

    # remove row with column headers
    pop_df = pop_df[pop_df['GEO_ID'] != 'id']

    # reformat GEOIDs to include fips code only
    pop_df['GEO_ID'] = [int(x.replace('0500000US', '')) for x in pop_df['GEO_ID']]

    # rename columns for clarity
    pop_df.rename(columns = {'GEO_ID':'fips', 'B01003_001E':'population'}, inplace = True)

    return pop_df[['fips', 'population']].set_index('fips')

def get_data_by_date_nyt(df, col_var, coldiff, windowvar = 7):
    '''Adds rolling averages and new case counts to the NYT data - from Alaina's NYT data notebook '''

    # given a data frame, sort by a col_var and get the difference
    df.sort_values(col_var, inplace = True)
    col_name = 'new_' + coldiff 
    df[col_name] = df[coldiff] - df[coldiff].shift() 
    df[col_name].iloc[0] = df[coldiff].iloc[0]
    
    # get rolling average based on a timewindow. use the 'new'
    # variable column to calculate the average
    col_name2 = 'moving_'+ coldiff
    df[col_name2] = round(df[col_name].rolling(window=windowvar).mean(),2)
   
    return df

def add_new_cases_deaths(nyt):
    '''
    '''
    all_counties = nyt.index.unique()
    df_w_all_data = pd.DataFrame()

    for fips_code in all_counties:

        if np.isnan(fips_code):
            continue

        df_sub = nyt.loc[fips_code]
        df_sub1 = get_data_by_date_nyt(df_sub, 'date', 'cases')
        new_df = get_data_by_date_nyt(df_sub1, 'date', 'deaths')

        # get cases & deaths per 100k
        new_df['100k_cases_new'] = round((new_df['new_cases']/new_df['population']) * 100000.0,2)
        new_df['100k_deaths_new'] = round((new_df['new_deaths']/new_df['population']) * 100000.0,2)

        # get the 7-day rolling average for cases & deaths
        new_df['100k_cases_moving'] = round(new_df['100k_cases_new'].rolling(window=7).mean(),2)
        new_df['100k_deaths_moving'] = round(new_df['100k_deaths_new'].rolling(window=7).mean(),2)

        # get cummulative cases per 100k
        new_df['100k_cases_cummulative'] = round((new_df['cases']/new_df['population']) * 100000.0,2)
        new_df['100k_deaths_cummulative'] = round((new_df['deaths']/new_df['population']) * 100000.0,2)

        # get moving average for cummulative cases
        new_df['cases_moving_avg'] = round(new_df['cases'].rolling(window=7).mean(),2)

        #get the % change in new cases:
        new_df['change_rate'] = round(new_df['cases'].pct_change(fill_method='bfill')*100.0,2)

        #get the % change in the 7 day moving average of new cases:
        new_df['change_rate7'] = round(new_df['cases_moving_avg'].pct_change(fill_method='bfill')*100.0,2)
        new_df['daily_change_diff'] = new_df['change_rate7'] - new_df['change_rate7'].shift() 

        df_w_all_data = df_w_all_data.append(new_df)
    return df_w_all_data

def get_end_date(weeks):
    '''return the first of the eligible weeks that is more than a week away from the next week
        ensures that gaps between when the county met the policy criteria are not included'''
    lag_time = np.timedelta64(14,'D')
    weeks = [pd.to_datetime(x, infer_datetime_format=True) for x in weeks]

    for i in range(0, len(weeks)):
        dif = weeks[i] - weeks[i-1]
        if dif > np.timedelta64(7,'D'):
            return weeks[i-1] + np.timedelta64(6,'D') + lag_time #get last day of the week + lag

    #if the loop completes without entering the if statement return the last date in the list
    return weeks[-1] + np.timedelta64(6,'D') + lag_time #get last day of the week + lag

def get_start_date_fall(weeks):
    '''return the first of the eligible weeks that is more than a week away from the next week
        ensures that gaps between when the county met the policy criteria are not included. This will return 
        the start date of the fall policy environment and will only work at counties that have 2 periods.'''
    weeks = [pd.to_datetime(x, infer_datetime_format=True) for x in weeks]
    for i in range(0, len(weeks)):
        dif = weeks[i] - weeks[i-1]
        if dif > np.timedelta64(7,'D'):
            return weeks[i] #get last day of the week + lag
    
def get_start_end_date(policy_data, geo_col = 'County', weeks_col = 'Eligible Weeks (Sunday)', prioritize_fall = False):
    '''In the compare policies Jupyter Notebook, data is collected regarding when the relevant policy environment
    is in place. That data given in the weeks_col is used to get a single start and end date for the policy environment.
    
    Weeks may not be consecutive. If that is the case, then the end date is in the end of the last consecutive week.'''
    start_dates = []
    end_dates = []
    num_days = [] #useful for later calculations

    for index, row in policy_data.iterrows():
        if prioritize_fall:
            start_date = get_start_date_fall(row[weeks_col])
            end_date = pd.to_datetime(row[weeks_col][-1], infer_datetime_format=True)
        else:
            start_date = pd.to_datetime(row[weeks_col][0], infer_datetime_format=True)
            end_date = get_end_date(row[weeks_col])
        start_dates.append(start_date)
        end_dates.append(end_date)
        delta = end_date - start_date
        num_days.append(delta.days)

    fips_codes = policy_data[geo_col].tolist()
    data = zip(fips_codes, start_dates, end_dates, num_days)
    df = pd.DataFrame(data, columns = ['fips', 'start_date', 'end_date', 'num_days'])
    return df.set_index('fips')

def get_policy_control_data():

    # counties with similar policies
    control_data = pd.read_csv("data/interim/counties_policy_environment_data.csv")
    control_data['Eligible Weeks (Sunday)'] = [x.split('\n') for x in control_data['Eligible Weeks (Sunday)']]
    return control_data

def get_metric_on_date(cases, metric, start_date = None, date_df = None, date_col = 'start_date'):
    '''Returns the metric/column value inputted on a given start date'''
    values = []
    for fips_code in cases.index.unique():
        date = date_df.loc[fips_code][date_col]

        # output and ignore any fips codes with faulty data (cause key errors)
        try:
            county_cases = cases.loc[fips_code]
        except KeyError:
            print(fips_code)
            continue
        metric_on_start = county_cases[county_cases['date'] == date][metric]

        #if look_for_non_null is True
        #modify the date until you get a non-null number
        if metric_on_start.empty:
            values.append(np.nan)
        else:
            values.append(metric_on_start.item())

    data = zip(cases.index.unique(), values)
    df = pd.DataFrame(data, columns = ['fips', 'metric'])
    return df.set_index('fips')


def get_results(start_end_df, cases):
    # calculate average new cases using the start and end date of the policy environment
    results = start_end_df.copy()

    #get cumulative cases at the start and end of the policy
    starting_cases = get_metric_on_date(cases, '100k_cases_cummulative', date_df = start_end_df, date_col = 'start_date')
    print(starting_cases.shape)
    ending_cases = get_metric_on_date(cases, '100k_cases_cummulative', date_df = start_end_df, date_col = 'end_date')
    print(ending_cases.shape)
    
    #if ending_cases are null, then the 2 weeks after the policy's effective end date is in the future
    #that data will not be used since we cannot evaluate the impact of the policy yet
    ending_cases = ending_cases.dropna()

    #rename cols before merge
    starting_cases.rename(columns = {'metric':'starting_cases_100k'}, inplace = True)
    ending_cases.rename(columns = {'metric':'ending_cases_100k'}, inplace = True)
    case_results = starting_cases.merge(ending_cases, left_index = True, right_index = True)

#     return case_results
    print(case_results.shape)
    results = start_end_df.merge(case_results, left_index = True, right_index = True)
    print(results.shape)

    results['net_cases'] = results['ending_cases_100k'] - results['starting_cases_100k']
    results['avg_new_cases'] = np.divide(results['net_cases'], results['num_days'])
    return results