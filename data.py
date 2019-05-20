import datadotworld as dw
import numpy as np
import pandas as pd

#############################################################
                ## HELPER FUNCTIONS ##
#############################################################

def weekday_name(integer):
    """Convert integer from 0-6 to return corresponding weekday name.

    Function takes an integer from dayofweek value for timestamp and
    returns the name of the day of the week. For example: a day of
    week value of 0 returns 'Monday'

    Parameters
    ----------
    integer:    int  (valid values range 0 to 6)

    Returns
    -------
    Str:        Returns the name of the day of the week.

    Example
    -------
    >>> weekday_name(3)  # returns Thursday
    """
    day_names = ("Monday","Tuesday","Wednesday","Thursday",
                 "Friday","Saturday","Sunday")
    return day_names[integer]


def month_name(integer):
    """Convert integer from timestamp month to return corresponding
    month name.

    Function takes integer from month value for timestamp and returns
    corresponding name of month in calendar year (zero indexed) as a
    string.  For example, a timestamp with datetime.month value of 3
    returns April.

    Parameters
    ----------
    integer:  int (timestamp.month)

    Returns
    -------
    Str:    Name of month at correspnding place in list of months
            zero indexed and in caldenar year order.

    Example
    -------
    >>> month_name(12)  # returns December
    """
    month_names = ['January','February','March','April','May','June',
                   'July','August','September','October','November','December']
    return month_names[integer-1]

def get_fiscalyear(column, fiscalyear_start=7):
    """Create conversion column that reads dates and returns fiscal year.

    Function takes a datetime series object or column and produces a
    list with the corresponding fiscal year as a four digit year for
    each date in the original series.

    The function's default value is based on the Maryland Govt fiscal
    year which runs from July 1st (month 7) to June 30th.  It returns
    a list that is the same size as the original column making it easy
    to simply use the return from the function call as a new column
    for the same dataframe, adding data for fiscal year. The
    fiscalyear_start parameter allows for generation of fiscal year data
    for various months.

    Parameters
    -----------
    column:             pandas Series
                    a column of data type datetime.

    fiscalyear_start:   int
                    a number representing the numerical value for month of
                    the year to be used as the start of the fiscal year.

    Returns
    --------
    Pandas Series: Series or column containing data for 4 digit number representing fiscal year.

    Examples
    --------
    >>> get_fiscalyear(df['request_date']) # assumes July fiscal year start

    >>> get_fiscalyear(df['request_date'], fiscalyear_start=3) # assumes March fiscal year start

    >>> get_fiscalyear(df['request_date'], 10) # assumes October fiscal year start
    """

    fiscal_year = [date.year + 1 if date.month >= fiscalyear_start
                   else date.year for date in column]

    return fiscal_year


#############################################################
                ## REQUEST DATA ##
#############################################################

def get_data(dataset_name, dataframe_name):
    """Request data from datadotworld API, and returns pandas dataframe.

    Additional information on the datadotworld api can be found at the
    following site: https://apidocs.data.world/api

    Parameters
    ----------
    dataset_name:     str
                    name assigned to the desired dataset stored with
                    datadotworld service.

    dataframe_name:   str
                    name of the key associated with the datadotworld
                    dataset which stores objects within the dataset within
                    a dictionary of dataframes in key value pair.

    Returns
    -------
    Pandas Dataframe

    Examples
    --------
    >>> get_data(dataset_name='census2020', dataframe_name='Kansas')

    >>> get_data('performance_indicators', 'public_safety')
    """
    dataworld_obj = dw.load_dataset(dataset_name)
    dataframe = dataworld_obj.dataframes[dataframe_name]

    return dataframe


def clean_data(dframe):
    '''

    '''
    target_columns = (['wo_id','date_completed','prob_type','bl_id','completed_by',
                        'date_requested','time_completed','time_start','time_end'])

    if isinstance(dframe, pd.core.frame.DataFrame):
        try:
            dframe = dframe[target_columns][(dframe['prob_type'] != 'TEST(DO NOT USE)')]
            dframe['date_completed'] = pd.to_datetime(dframe['date_completed'])
            dframe['date_requested'] = pd.to_datetime(dframe['date_requested'])
            dframe.set_index('date_requested', inplace=True)
            dframe['duration'] = dframe['date_completed'] - dframe.index
            dframe['fiscal_year_requested'] = get_fiscalyear(dframe.index)
            dframe['fiscal_year_completed'] = get_fiscalyear(dframe['date_completed'])
            dframe.sort_index(inplace=True)

            status = 'Pass'
        except Exception as e:
            status = 'Fail'
            # log event if failed
    else:
        print('Function requires pandas dataframe object but received type: {}.'
             .format(type(dframe)))

    return dframe


def make_map_dframe(dframe, excelfile, skipnrows=6):
    """Generate dataframe for mapping charts with archibus data.

    Parameters
    ----------
    dframe:     pandas dataframe
            the dataframe returned from the clean_data function
            in the data.py module

    excelfile:  file object
            excel file with the latitude and longitude data required
            to identify the location of markers for archibus data.

    skipnrows:  int
            the number of rows to skip when reading in excel sheet
            for conversion to pandas dataframe.

    Returns
    -------
    Pandas Dataframe

    Example
    -------
    >>> make_map_dframe(dframe=df, excelfile='file.xlsx', skipnrows=5)

    """

    df = pd.read_excel(excelfile, skiprows=skipnrows)
    df.columns = ['bl_id','name','addr','site_id','latitude','longitude']
    geo_dict = {}
    for bld in df['bl_id'].unique():
        geo_dict[bld] = {'latitude': df.loc[df['bl_id'] == bld]['latitude'].values[0],
                        'longitude': df.loc[df['bl_id'] == bld]['longitude'].values[0],
                        'bld_name': df.loc[df['bl_id'] == bld]['name'].values[0]}

    dframe['latitude'] = dframe['bl_id'].apply(lambda x: geo_dict[x]['latitude'])
    dframe['longitude'] = dframe['bl_id'].apply(lambda x: geo_dict[x]['longitude'])
    dframe['bld_name'] = dframe['bl_id'].apply(lambda x: geo_dict[x]['bld_name'])

    return dframe

#############################################################
                ## MODULE VARIABLES ##
#############################################################

data = get_data(
    dataset_name='dgs-kpis/fmd-maintenance',
    dataframe_name='archibus_maintenance_data')

dframe = clean_data(data)

map_dframe = make_map_dframe(
    dframe=dframe, excelfile='data/building_lat_longs.xlsx')


preventative_types = ['PREVENTIVE MAINT','HVAC|PM']
corrective_types = ['BOILER','CHILLERS','COOLING TOWERS','HVAC',
                   'HVAC INFRASTRUCTURE','HVAC|REPAIR']

hvac_types = ['PREVENTIVE MAINT','HVAC|PM','BOILER','CHILLERS',
              'COOLING TOWERS','HVAC','HVAC INFRASTRUCTURE',
              'HVAC|REPAIR' ]
