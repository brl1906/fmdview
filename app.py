""" Interactive dashboard for performance overview of Facility Mantenance
Activity. A Story Behind The Curve Narrative in 10 Charts.

author: Babila Lima
"""
from configparser import ConfigParser
import datetime as dt
import os
import subprocess
import warnings

import dash
import dash_core_components as dcc
import dash_html_components as html
import datadotworld as dw
import numpy as np
import pandas as pd
import plotly.graph_objs as go

warnings.filterwarnings('ignore')

# subprocess.call(["dw", "configure"])

# load mapbox token
parser = ConfigParser()
parser.read('config.ini')
fmd_kpi_mapbox_token = parser.get('MapboxToken','token')


app = dash.Dash('FMDView')
app.config.supress_callback_exceptions = True
app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})  # noqa: E501

if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'  # noqa: E501
    })

server = app.server

# HANDLE DATA FOR DASHBOARD
def fiscal_year(df):
    """
    Function takes a dataframe with a DateTimeIndex and
    produces list with the corresponding fiscal year as a
    four digit year for each date on the index of the dataframe.

    The function is based on the Maryland Govt fiscal year which
    runs from July 1st to June 30th.  It returns a list that is the
    same size as the original dataframe and allows the function call
    to be passed as a new column for fiscal year.
    """
    fiscal_year = np.where(df.index.month >= 7,df.index.year+1, df.index.year)
    return fiscal_year

# load dataset from data.world service
dataset = dw.load_dataset('dgs-kpis/fmd-maintenance')
archibus_data = dataset.dataframes['archibus_maintenance_data']

def get_dataframe(data_source):
    '''

    '''
    target_columns = (['wo_id','date_completed','prob_type','bl_id','completed_by',
                        'date_requested','time_completed','time_start','time_end'])
    if isinstance(data_source, str):
        df = pd.read_excel(io=data_source)
        df = df[target_columns][(df['prob_type'] != 'TEST(DO NOT USE)')]
        df['date_requested'] = pd.to_datetime(df['date_requested'])
        df.set_index('date_requested', inplace=True)
        df['duration'] = df['date_completed'] - df.index
        df['fiscal_year_requested'] = fiscal_year(df)
        df['fiscal_year_completed'] =  [date.year + 1 if date.month >= 7 else date.year for date in df['date_completed']]
        df.sort_index(inplace=True)

    elif isinstance(data_source, dw.models.dataset.LocalDataset):
        df = data_source.dataframes['archibus_maintenance_data']
        df = df[target_columns][(df['prob_type'] != 'TEST(DO NOT USE)')]
        df['date_completed'] = pd.to_datetime(df['date_completed'])
        df['date_requested'] = pd.to_datetime(df['date_requested'])
        df.set_index('date_requested', inplace=True)
        df['duration'] = df['date_completed'] - df.index
        df['fiscal_year_requested'] = np.where(df.index.month >= 7,df.index.year+1, df.index.year)
        df['fiscal_year_completed'] =  [date.year + 1 if date.month >= 7 else date.year for date in df['date_completed']]
        df.sort_index(inplace=True)

    else:
        print("""
Function expects type io string or datadotworld.models.dataset.LocalDataset
but got {}""".format(type(data_source)))

    return df

df = get_dataframe(dataset)

def create_map_plotting_df(dataframe, file, nrows2skip):
    """
    """
    lat_long_dataframe = pd.read_excel(file, skiprows=nrows2skip)
    lat_long_dataframe.columns = ['bl_id','name','addr','site_id','latitude','longitude']

    geo_dict = {}
    for bld in lat_long_dataframe['bl_id'].unique():
        geo_dict[bld] = {'latitude': lat_long_dataframe.loc[lat_long_dataframe['bl_id'] == bld]['latitude'].values[0],
                        'longitude': lat_long_dataframe.loc[lat_long_dataframe['bl_id'] == bld]['longitude'].values[0],
                        'bld_name': lat_long_dataframe.loc[lat_long_dataframe['bl_id'] == bld]['name'].values[0]}

    dataframe['latitude'] = dataframe['bl_id'].apply(lambda x: geo_dict[x]['latitude'])
    dataframe['longitude'] = dataframe['bl_id'].apply(lambda x: geo_dict[x]['longitude'])
    dataframe['bld_name'] = dataframe['bl_id'].apply(lambda x: geo_dict[x]['bld_name'])

    return dataframe

lat_lon_df = create_map_plotting_df(dataframe=df, file='data/building_lat_longs.xlsx',
                   nrows2skip=6)

# CREATE CHARTS FOR DASHBOARD
def map_workorder_volume_distribution(fy=''):
    """
    """
    # dictionary of dataframes with lat & lon filtered by fiscal year for dynamic graphing
    mapping_pivot_dfs_dict = {}
    for yr in df['fiscal_year_requested'].unique():
        mapping_pivot_dfs_dict[yr] = pd.pivot_table(lat_lon_df.loc[lat_lon_df['fiscal_year_requested'] == yr],
                                                    index=['bl_id','bld_name'],values=['latitude','longitude','wo_id'],
                                                    aggfunc={'latitude':np.mean, 'longitude':np.mean, 'wo_id':'count'})

    # create map traces for the number of work order requests by building location
    multi_traces_all_fiscalyears = []
    single_trace_selected_year = []
    colors = ['skyblue','lightpink','lavender','palegoldenrod','burlywood','darkmagenta']

    # create map traces for plotting all fiscal years
    for (key,value),color in zip(mapping_pivot_dfs_dict.items(), colors):
        multi_traces_all_fiscalyears.append(
            go.Scattermapbox(
                lat = [lat for lat in value['latitude'].values],
                lon = [lon for lon in value['longitude'].values],
                mode = 'markers',
                name = 'FY{}'.format(key),
                marker = {'color': color,
                         'size': [count / 10 for count in value['wo_id'].values]},
                text = ['<b>FY{}</b>:<br>{}<br>{:,} requests'.format(key,name[1], count)  for
                        name, count in zip(value.index, value['wo_id'].values)],
                hoverinfo = 'text'
                            )
                                         )

    # handle conditional setting of map data for selected fiscal year
    if fy:
        data = single_trace_selected_year
        title_text = '<b>FY{}</b> Maintenance Activity Distribution & Volume'.format(fy)

        # create map trace for plotting selected fiscal year
        single_trace_selected_year.append(
            go.Scattermapbox(
                lat = [lat for lat in mapping_pivot_dfs_dict[fy]['latitude'].values],
                lon = [lon for lon in mapping_pivot_dfs_dict[fy]['longitude'].values],
                mode = 'markers',
                name = 'FY{}'.format(fy),
                marker = {'color': '#D4395B',
                         'size': [count / 10 for count in mapping_pivot_dfs_dict[fy]['wo_id'].values]},
                text = ['<b>FY{}</b>:<br>{}<br>{:,} requests'.format(fy ,name[1], count)  for
                            name, count in zip(mapping_pivot_dfs_dict[fy].index,
                                               mapping_pivot_dfs_dict[fy]['wo_id'].values)],
                hoverinfo = 'text',
                opacity = .7
                            )
                                        )
    else:
        data = multi_traces_all_fiscalyears
        title_text = 'Maintenance Activity Distribution & Volume'



    layout = go.Layout(
        title = title_text,
        autosize = True,
        font = {'color': '#CCCCCC'},
        titlefont = {'color': '#CCCCCC'},
        hovermode = 'closest',
        margin = {'t':35, 'b':5,
                 'l':5, 'r':5},
        # legend styling to show viewer most recent fiscal year first
        legend = {
           'traceorder':'reversed'
                },

        mapbox = {'accesstoken': fmd_kpi_mapbox_token,
                 'bearing': 0,
                  'pitch': 0,
                  'zoom': 10,
                  'style': 'dark',
                 'center': {'lat': value['latitude'].mean(),
                           'lon': value['longitude'].mean()}
                 },
        paper_bgcolor = '#303939',
                    )

    fig = {'data':data, 'layout':layout}
    return fig

# for 2d historgram chart filtering
current_year = [year for year in df['fiscal_year_requested'].unique()][-1]
last_year = [year for year in df['fiscal_year_requested'].unique()][-2]

def make_2dhist_figure_for_fiscal_year(fy=''):
    """
    """

    def month_name(integer):
        """function takes integer from month value for timestamp
        and returns the name of the month as a string.
        Example: a timestamp with datetime.month value of 0 returns 'January'
        """
        month_names = ['January','February','March','April','May','June',
                        'July','August','September','October','November','December']
        return month_names[integer-1]


     # filter out still open work orders
    if fy == '':
        filtered_df = df[(df['date_completed'].notnull())]
        title = 'FY{}-FY{} Duration & Volume<br>Distribution Density by Type<br><i>(bubblesize: % durations > type avg)</i>'.format(
        min(df['fiscal_year_requested']),max(df['fiscal_year_requested']))
    else:
        filtered_df = df[(df['date_completed'].notnull()) &
                    (df['fiscal_year_requested'] == fy)]
        title = 'FY{} Duration & Volume<br>Distribution Density by Type<br><i>(bubblesize: % durations > type avg)</i>'.format(fy)

    filtered_df['completed_month_name'] = [month_name(date) for date in filtered_df['date_completed'].dt.month]
    problems = filtered_df['prob_type'].value_counts().index.tolist()
    prob_type_counts, prob_type_avg_duration = [],[]

    for prob in problems:
        prob_type_counts.append(filtered_df[filtered_df['prob_type'] == prob]['wo_id'].count())
        prob_type_avg_duration.append(filtered_df[filtered_df['prob_type'] == prob]['duration'].mean().days)

    # create list for sizing bubbles on chart based on pct of the work orders
    # in that problem type that exceed the average duration for that type
    pct_workorders_exceeding_mean_duration_for_type = {}
    for prob in problems:
        avg = filtered_df[filtered_df['prob_type'] == prob]['duration'].mean().days
        number_exceding_mean_duration = filtered_df[(filtered_df['prob_type'] == prob) &
                         (filtered_df['duration'].dt.days > avg)]['duration'].count()
        count_ = filtered_df[(filtered_df['prob_type'] == prob)]['wo_id'].count()
        pct_workorders_exceeding_mean_duration_for_type[prob] = number_exceding_mean_duration / count_ * 100

    x = prob_type_counts
    y = prob_type_avg_duration
    data = [
    go.Histogram2dContour(
        x = prob_type_counts,
        y = prob_type_avg_duration,
        name = '',
        colorscale = 'Blues',
        reversescale = True,
        xaxis = 'x',
        yaxis = 'y'
    ),

    go.Scatter(
        x = prob_type_counts,
        y = prob_type_avg_duration,
        xaxis = 'x',
        yaxis = 'y',
        name = 'problem type',
        mode = 'markers',
        text = ['{}:<br>{:.0f}% > avg duration'.format(key,val) for key,val in pct_workorders_exceeding_mean_duration_for_type.items()],
        hoverinfo = 'text',
        marker = dict(
            color = '#D4395B',
            size = [val / 3.5 for key,val in pct_workorders_exceeding_mean_duration_for_type.items()]),
        opacity = .7
    ),
    go.Histogram(
        y = prob_type_avg_duration,
        xaxis = 'x2',
        nbinsy = 25,
        name = 'avg days',
        marker = dict(
            color = '#CCCCCC')
                ),
    go.Histogram(
        x = prob_type_counts,
        yaxis = 'y2',
        name = 'type volume',
        nbinsx = 25,
        marker = dict(
            color = '#CCCCCC')
                )
            ]

    layout = go.Layout(
        title = title,
        autosize = True,
        xaxis = dict(
            title = 'request volume (by problem type)',
            zeroline = False,
            domain = [0,0.85],
            showgrid = False
        ),
        yaxis = dict(
            title = 'avg duration (days)',
            zeroline = False,
            domain = [0,0.85],
            showgrid = False
        ),
        xaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            showgrid = False
        ),
        yaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            showgrid = False
        ),

        bargap = .01,
        hovermode = 'closest',
        showlegend = False,
        margin = {'l': 75, 't':80,
                 'b': 80,'r': 80},
        font = {'color': '#CCCCCC'},
        titlefont = {'color': '#CCCCCC',
                    'size': 14},

        plot_bgcolor = '#303939',
        paper_bgcolor = '#303939',
                    )
    fig = {'data':data, 'layout':layout}
    return fig


def request_completion_gap_line_chart(frequency='Monthly'):
    """
    Function creates line chart showing volume of work order
    requests and work orders completed for various resampling
    frequencies as passed to the function.
    Function takes  Annually, Monthly, Weekly and Daily resampling
    frequencies.  Default is Monthly.
    """

    # use dictionary to pass the single letter frequency resampling values
    # as function values from easier to understand key passed to used in dropdown
    frequency_dict = {'Annually': 'A', 'Monthly':'M',
                     'Weekly': 'W', 'Daily':'D'}

    # handle creation of hover text dynamically based on frequency argument
    if frequency == 'Monthly':
        request_trace_text = ['<b>{} {}</b><br>requested: {:,}'.format(
            date.strftime('%b'),date.strftime('%Y'),val) for date,val in zip(
            df.resample(frequency_dict[frequency])['wo_id'].count().index,
            df.resample(frequency_dict[frequency])['wo_id'].count())]

        completed_trace_text = ['<b>{} {}</b><br>completed: {:,}'.format(
            date.strftime('%b'),date.strftime('%Y'),val) for date,val in zip(
            df.resample(frequency_dict[frequency])['date_completed'].count().index,
            df.resample(frequency_dict[frequency])['date_completed'].count())]

    elif frequency == 'Annually':
        request_trace_text = ['<b>{}</b><br>requested: {:,}'.format(
            date.strftime('%Y'),val) for date,val in zip(
            df.resample(frequency_dict[frequency])['wo_id'].count().index,
            df.resample(frequency_dict[frequency])['wo_id'].count())]

        completed_trace_text = ['<b>{}</b><br>completed: {:,}'.format(
            date.strftime('%Y'),val) for date,val in zip(
            df.resample(frequency_dict[frequency])['date_completed'].count().index,
            df.resample(frequency_dict[frequency])['date_completed'].count())]

    elif frequency == 'Weekly':
        request_trace_text = ['<b>Week {}</b><br>{}<br>requested: {:,}'.format(
            date.week, date.year, val) for date,val in zip(
            df.resample(frequency_dict[frequency])['wo_id'].count().index,
            df.resample(frequency_dict[frequency])['wo_id'].count())]

        completed_trace_text = ['<b>Week {}</b><br>{}<br>completed: <b>{:,}</b>'.format(
            date.week, date.year, val) for date,val in zip(
            df.resample(frequency_dict[frequency])['date_completed'].count().index,
            df.resample(frequency_dict[frequency])['date_completed'].count())]

    elif frequency == 'Daily':
        request_trace_text = ['{}<br>requested: {}'.format(
            date.strftime('%b-%d-%Y'),val) for date,val in zip(
            df.resample(frequency_dict[frequency])['wo_id'].count().index,
            df.resample(frequency_dict[frequency])['wo_id'].count())]

        completed_trace_text = ['{}<br>completed: {}'.format(
            date.strftime('%b-%d-%Y'),val) for date,val in zip(
            df.resample(frequency_dict[frequency])['date_completed'].count().index,
            df.resample(frequency_dict[frequency])['date_completed'].count())]


    request_trace = go.Scatter(
    x = df.resample(frequency_dict[frequency])['wo_id'].count().index,
    y = df.resample(frequency_dict[frequency])['wo_id'].count(),
    name = 'opened',
    line = {'color': '#D4395B',
           'width': 5},
    opacity = .8,
    text = request_trace_text,
    hoverinfo = 'text'
                    )

    completed_trace = go.Scatter(
        x = df.resample(frequency_dict[frequency])['date_completed'].count().index,
        y = df.resample(frequency_dict[frequency])['date_completed'].count(),
        name = 'completed',
        line = {'color': '#ABCDAB',
               'width': 4,},
        text = completed_trace_text,
        hoverinfo = 'text'
                        )

    layout = go.Layout(
        title = 'Work Order Request|Completion Gap<br>Frequency: <b><i>{}</i></b>'.format(frequency),
        autosize = True,
        legend = {'orientation': 'h'},
        font = {'color': '#CCCCCC'},
        titlefont = {'color': '#CCCCCC'},
        hovermode = 'closest',
        margin = {'r':35, 'b':10,
                     'l': 50, 't': 35},
        paper_bgcolor = '#303939',
        plot_bgcolor = '#303939'
                        )
    fig = {'data':[request_trace,completed_trace], 'layout':layout}
    return fig


def make_daily_count_close_heatmap():
    """
    """

    def weekday_name(integer):
        """function takes integer from dayofweek value for timestamp and returns
        the name of the day of the week.
        example: a day of week value of 0 returns 'Monday'
        """

        day_names = ("Monday","Tuesday","Wednesday","Thursday",
                         "Friday","Saturday","Sunday")
        return day_names[integer]

    # filter out still open work orders
    filtered_df = df[df['date_completed'].notnull()]
    filtered_df['completed_day_name'] = filtered_df['date_completed'].apply(
        lambda x: weekday_name(x.dayofweek))

    z_dict = {}
    x = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    for year in filtered_df['fiscal_year_completed'].unique():
        z_dict[year] = []
        for day in x:
            z_dict[year].append(
                filtered_df[(filtered_df['fiscal_year_completed'] == year) &
                              (filtered_df['completed_day_name'] == day)]['wo_id'].count()
                                )

    trace = go.Heatmap(
        z = [value for value in z_dict.values()],
        y = ['FY {}'.format(int(year)) for year in z_dict.keys()],
        x = x,
        colorscale = 'Portland')

    layout = go.Layout(
        hovermode = 'closest',
        font = {'color': '#CCCCCC'},
        titlefont = {'color': '#CCCCCC',
                    'size': 14},
        title = 'Work Orders Closed Daily',
        margin = {'r':55, 'b':70,
                 'l': 70, 't': 45},
        plot_bgcolor = '#303939',
        paper_bgcolor = '#303939'
                        )

    return {'data':[trace], 'layout':layout}

def make_daily_percent_close_heatmap():
    """
    """
    def weekday_name(integer):
        """function takes integer from dayofweek value for timestamp and returns
        the name of the day of the week.
        example: a day of week value of 0 returns 'Monday'
        """
        day_names = ("Monday","Tuesday","Wednesday","Thursday",
                         "Friday","Saturday","Sunday")
        return day_names[integer]

    # pct of of total work orders closed daily aggregated
    # at day level (by fiscal year)
    filtered_df = df[df['date_completed'].notnull()]
    filtered_df['completed_day_name'] = filtered_df['date_completed'].apply(
    lambda x: weekday_name(x.dayofweek))
    z_dict = {}
    x = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    for year in filtered_df['fiscal_year_completed'].unique():
        z_dict[year] = []
        for day in x:
            z_dict[year].append(
                (filtered_df[(filtered_df['fiscal_year_completed'] == year) &
                             (filtered_df['completed_day_name'] == day)]['wo_id'].count() /
                 filtered_df[filtered_df['fiscal_year_completed'] == year]['wo_id'].count() * 100)
                                )

    trace = go.Heatmap(
        z = [value for value in z_dict.values()],
        y = ['FY {}'.format(int(year)) for year in z_dict.keys()],
        x = x,
        colorscale = 'Cividis')

    layout = go.Layout(
    hovermode = 'closest',
    font = {'color': '#CCCCCC'},
    titlefont = {'color': '#CCCCCC',
                'size': 14},
    title = 'Percentage of Work Orders Closed Daily',
    margin = {'r':15, 'b':70,
             'l': 70, 't': 45},
    plot_bgcolor = '#303939',
    paper_bgcolor = '#303939'
                    )

    return {'data':[trace],'layout':layout}

def make_monthly_count_close_heatmap():
    """
    """

    # distribution of number of work orders closed (agg by month) per fiscal year
    def month_name(integer):
        """function takes integer from month value for timestamp
        and returns the name of the month as a string.
        Example: a timestamp with datetime.month value of 0 returns 'January'
        """
        month_names = ['January','February','March','April','May','June',
                        'July','August','September','October','November','December']
        return month_names[integer-1]

    # filter out still open work orders
    filtered_df = df[df['date_completed'].notnull()]
    filtered_df['completed_month_name'] = [month_name(date) for date in filtered_df['date_completed'].dt.month]

    z_dict = {}
    x = ['January','February','March','April','May','June',
         'July','August','September','October','November','December']

    for year in filtered_df['fiscal_year_completed'].unique():
        z_dict[year] = []
        for month in x:
            z_dict[year].append(
                filtered_df[(filtered_df['fiscal_year_completed'] == year) &
                              (filtered_df['completed_month_name'] == month)]['wo_id'].count()
                                )

    trace = go.Heatmap(
        z = [value for value in z_dict.values()],
        y = ['FY {}'.format(int(year)) for year in z_dict.keys()],
        x = x,
        colorscale = 'Portland')

    layout = go.Layout(
        hovermode = 'closest',
        font = {'color': '#CCCCCC'},
        titlefont = {'color': '#CCCCCC',
                    'size': 14},
        title = 'Work Orders Closed Monthly',
        margin = {'r':15, 'b':99,
                 'l': 70, 't': 45},
        plot_bgcolor = '#303939',
        paper_bgcolor = '#303939'
                        )
    return {'data':[trace],'layout':layout}

def make_monthly_percent_close_heatmap():
    """
    """
    # percentage of work orders closed (agg by month) per fiscal year
    def month_name(integer):
        """function takes integer from month value for timestamp
        and returns the name of the month as a string.
        Example: a timestamp with datetime.month value of 0 returns 'January'
        """
        month_names = ['January','February','March','April','May','June',
                        'July','August','September','October','November','December']
        return month_names[integer-1]

    # filter out still open work orders
    filtered_df = df[df['date_completed'].notnull()]
    filtered_df['completed_month_name'] = [month_name(date) for date in filtered_df['date_completed'].dt.month]

    z_dict = {}
    x = ['January','February','March','April','May','June',
         'July','August','September','October','November','December']

    for year in filtered_df['fiscal_year_completed'].unique():
        z_dict[year] = []
        for month in x:
            z_dict[year].append(
                (filtered_df[(filtered_df['fiscal_year_completed'] == year) &
                             (filtered_df['completed_month_name'] == month)]['wo_id'].count() /
                 filtered_df[filtered_df['fiscal_year_completed'] == year]['wo_id'].count() * 100)
                                )

    trace = go.Heatmap(
        z = [value for value in z_dict.values()],
        y = ['FY {}'.format(int(year)) for year in z_dict.keys()],
        x = x,
        colorscale = 'Cividis')

    layout = go.Layout(
    hovermode = 'closest',
    font = {'color': '#CCCCCC'},
    titlefont = {'color': '#CCCCCC',
                'size': 14},
    title = 'Percentage of Work Orders Closed Monthly',
    margin = {'r':15, 'b':99,
             'l': 70, 't': 45},
    plot_bgcolor = '#303939',
    paper_bgcolor = '#303939'
                        )

    return {'data':[trace],'layout':layout}

def make_on_time_kpi_barchart_opened_closed_same_year():
    """
    """
    fy_list, fy_dfs = [],[]
    for year in df['fiscal_year_requested'].unique():
        fy_dfs.append(df.loc[(df['fiscal_year_requested']==year)])
        fy_list.append(year)
    fy_dfs = dict(zip(fy_list,fy_dfs))

    # get total number of work orders requested in each fiscal year
    work_order_volume = []
    for key, value in fy_dfs.items():
        work_order_volume.append(value.wo_id.count())
    work_order_volume = dict(zip(fy_list,work_order_volume))

    closed_workorders_dfs = []
    for key, value in fy_dfs.items():
        closed_workorders_dfs.append(value[value['date_completed'].notnull()])
    closed_workorders_dfs = dict(zip(fy_dfs.keys(), closed_workorders_dfs))

    # add data for work order volume by problem type per fiscal year
    # add average duration for each problem type per fiscal year
    # add indicator for whether work order was on time or not

    for key, value in closed_workorders_dfs.items():
        value['count'] = value.groupby(['prob_type'])['prob_type'].transform('count')
        value['avg_duration'] = (value.groupby(['prob_type'])['duration'].\
                            transform('sum') / value['count'])
        value['on_time'] = np.where(value.duration <= value.avg_duration,'hit','miss')


    # bar charts of pct closed on time vs volume
    kpi_values_dict = {}
    for year in closed_workorders_dfs.keys():
        kpi_values_dict[year] = closed_workorders_dfs[year]['on_time'].\
                                value_counts()[0] / fy_dfs[year]['wo_id'].count() * 100

    y1_trace = go.Bar(
        x = [year for year in kpi_values_dict.keys()],
        y = [value for value in kpi_values_dict.values()],
        name = 'on-time %',
        text = ['<b>FY{}</b><br>ON TIME: {:.0f}%'.format(yr,val) for yr, val in zip(
            [year for year in kpi_values_dict.keys()],
            [value for value in kpi_values_dict.values()])],
        marker = {'color': '#3c5a89',
                 'line': {'color': '#3c5a89',
                         'width': 1}},
        hoverinfo = 'text'
                        )

    y2_trace = go.Bar(
        x = [year for year in work_order_volume.keys()],
        y = [value for value in work_order_volume.values()],
        name = 'requests',
        marker = {'color': '#656969',
                 'line': {'color': '#CCCCCC',
                         'width': 1}},
        text = ['<b>FY{}</b><br>{:,} requests'.format(yr,val) for yr, val in zip(
            [year for year in work_order_volume.keys()],
            [value for value in work_order_volume.values()])],
        hoverinfo = 'text',
        opacity = .2,
        yaxis = 'y2')

    layout = go.Layout(
        hovermode = 'closest',
        legend = {'orientation': 'h'},
        title = 'On Time Overview by Fiscal Year',
        font = {'color': '#CCCCCC'},
        titlefont = {'color': '#CCCCCC'},
        yaxis = dict(title = '% work orders closed on time',
                    showgrid = False,
                    titlefont = {'color':'#3c5a89'},
                    tickfont = {'color': '#3c5a89'}),

        yaxis2=dict(title='number requests',
                        showgrid = False,
                        titlefont=dict(color='#656969'),
                        tickfont=dict(color='#656969'),
                        overlaying='y',
                        side='right'),
        margin = {'r':70, 'b':20,
                  'l': 70, 't': 45},
        plot_bgcolor = '#303939',
        paper_bgcolor = '#303939'
                        )
    return {'data':[y1_trace, y2_trace], 'layout': layout}


def make_pm_to_cm_stacked_bar_chart():
    """
    """
    # define filters for calculating PM : CM indicator
    corrective_maintenance = ['BOILER','CHILLERS','COOLING TOWERS','HVAC',
                                    'HVAC INFRASTRUCTURE','HVAC|REPAIR']
    preventative_maintenance = ['PREVENTIVE MAINT','HVAC|PM']
    hvac_problem_types = corrective_maintenance + preventative_maintenance

    fy_list, fy_dfs = [],[]
    for year in df['fiscal_year_requested'].unique():
        fy_dfs.append(df.loc[(df['fiscal_year_requested']==year)])
        fy_list.append(year)
    fy_dfs = dict(zip(fy_list,fy_dfs))

    # get total number of work orders requested in each fiscal year
    work_order_volume = []
    for key, value in fy_dfs.items():
        work_order_volume.append(value.wo_id.count())
    work_order_volume = dict(zip(fy_list,work_order_volume))

    kpi_values_dict = {}
    for key, value in fy_dfs.items():
            kpi_values_dict[key] = (
            value[value['prob_type'].isin(preventative_maintenance)]['prob_type'].
            value_counts().sum() /

            value[value['prob_type'].isin(corrective_maintenance)]['prob_type'].
            value_counts().sum() * 100 )

    trace1 = go.Bar(
    x = [year for year in kpi_values_dict.keys()],
    y = [value for value in kpi_values_dict.values()],
    marker = {'color' : '#3c5a89',
             'line': {'color': '#3c5a89',
                     'width':1}},
    name = 'pm / cm',
    text = ['<b>FY{}</b><br>PM:CM {:.0f}%'.format(yr,val)  for yr, val in zip([year for year in kpi_values_dict.keys()],
                                                                     [value for value in kpi_values_dict.values()])],
    hoverinfo = 'text'
                    )

    trace2 = go.Bar(
        x = [year for year in work_order_volume.keys()],
        y = [value for value in work_order_volume.values()],
        marker = {'color': '#656969',
                 'line': {'color': '#CCCCCC',
                         'width': 1}},
        opacity = .2,
        yaxis = 'y2',
        name = 'total work orders',
        text = ['<b>FY{}</b>:<br>{:,} requests'.format(yr,val) for yr, val in zip(
            [year for year in work_order_volume.keys()],
            [value for value in work_order_volume.values()])],
        hoverinfo = 'text'
                    )

    layout = go.Layout(
        hovermode = 'closest',
        legend = {'orientation' : 'h'},
        font = {'color': '#CCCCCC'},
        titlefont = {'color': '#CCCCCC',
                    'size': 14},
        title =' HVAC PM:CM KPI FY{} to FY{}'.format(min([year for year in kpi_values_dict.keys()]),max([year for year in kpi_values_dict.keys()])),
        yaxis=dict(title='pct %',
                   showgrid = False,
                   titlefont = {'color':'#3c5a89'},
                   tickfont = {'color': '#3c5a89'},
                  ),

        yaxis2=dict(title = 'number requests',
                    showgrid = False,
                    titlefont = {'color': '#656969'},
                    tickfont =  {'color': '#656969'},
                    overlaying='y',
                    side='right'
                    ),
        plot_bgcolor = '#303939',
        paper_bgcolor = '#303939',
                        )


    return {'data':[trace1,trace2], 'layout':layout}


def make_avg_durations_by_top_volume_chart():
    """
    """

    # create dictionary of filtered dataframes for top problem types for each year
    dataframe_filter_lists, filtered_dataframes = [],{}
    yearly_avg_durations = {}
    for i, num in enumerate([3,5,10,15]):
        # create filter as list object and use loop counter to acess index of list filter
        # to create dataframe with the top X filter
        dataframe_filter_lists.append(df['prob_type'].value_counts().head(num).index)
        filtered_dataframes['top {}'.format(str(num))] = df[(df['prob_type'].isin(dataframe_filter_lists[i]))]

        for year in filtered_dataframes['top {}'.format(str(num))]['fiscal_year_requested'].unique():
            if year == 2019:
                pass
            else:
                yearly_avg_durations['top: {} year: {}'.format(num,year)] = (
                    filtered_dataframes['top {}'.format(str(num))][(filtered_dataframes['top {}'.format(str(num))]['fiscal_year_requested'] == year) &
                                                                  (filtered_dataframes['top {}'.format(str(num))]['fiscal_year_completed'] == year)]['duration'].mean())


    # for comparison add key value for all problem types (not just top 3,5,10,15 as filtered above)
    for year in df['fiscal_year_requested'].unique():
        if year == 2019:
            pass
        else:
            yearly_avg_durations['all {}'.format(year)] = (df[(df['fiscal_year_requested'] == year) &
                                             (df['fiscal_year_completed'] == year)]['duration'].mean())

    # trace for all work orders for each fiscal year
    trace_all = go.Bar(
        x = [text.split()[-1] for text in yearly_avg_durations.keys()],
        y = [value.days for key,value in yearly_avg_durations.items() if 'all' in key],
        name = 'all<br>work orders',
        marker = {'color': '#3c5a89'},
        text = ['All:<br> {} days'.format(value.days) for key,value in yearly_avg_durations.items() if 'all' in key],
                    )
    # traces for top 3, top 5, top 10, top 15 work orders per fiscal year
    trace_topFilters = []
    for num, color in zip([3,5,10,15],['#D3DEE4','#849DAB','#589ABF','#04A1F8']):
        trace_topFilters.append(go.Bar(
            x = [text.split()[-1] for text in yearly_avg_durations.keys()],
            y = [value.days for key,value in yearly_avg_durations.items() if 'top: {} year'.format(num) in key ],
            name = 'Top {}<br>work orders'.format(num),
            marker = {'color': color},
            text = ['FY{}<br>Top {}:<br> {} days'.format(key.split()[-1],num,value.days) for key,value in yearly_avg_durations.items() if 'top: {} year'.format(num) in key],
            hoverinfo = 'text',
                        ))
    # traces for annual work order reqeust volume
    trace_requestVolume = go.Scatter(
        x = [year for year in df.groupby('fiscal_year_requested')['wo_id'].count().index],
        y = [value for value in df.groupby('fiscal_year_requested')['wo_id'].count()],
        mode = 'lines',
        name = 'requests',
        text = [('FY{}:<br>{:,} requests'.format(year,val)) for val,year in zip([value for value in df.groupby('fiscal_year_requested')['wo_id'].count()],
                   [value for value in df.groupby('fiscal_year_requested')['wo_id'].count().index])],
        hoverinfo = 'text',
        line = {'color': '#C70039',
               'width': 3},
        yaxis = 'y2')

    layout = go.Layout(
        legend = {'orientation': 'h'},
        hovermode = 'closest',
        font = {'color': '#CCCCCC'},
        titlefont = {'color': '#CCCCCC'},
        title = 'Average Work Order Durations<br><i>requested & completed within same fiscal year</i>',
        yaxis = {'title': 'days',
                'showgrid': False,
                'titlefont': {'color': '#3c5a89'},
                'tickfont': {'color': '#3c5a89'}},

        yaxis2 = {'title': 'work order volume',
                 'showgrid': False,
                 'titlefont': {'color': '#C70039'},
                 'tickfont': {'color': '#C70039'},
                 'overlaying': 'y',
                 'side': 'right'},

        plot_bgcolor = '#303939',
        paper_bgcolor = '#303939'
                        )

    traces = trace_topFilters
    for trace in [trace_all, trace_requestVolume]:
        traces.append(trace)

    return {'data':traces, 'layout':layout}


### APP LAYOUT SECTION ###
# dictionary for creating dropdown filter for resampling frequency chart
frequency_dict = {'Annually': 'A', 'Monthly':'M', 'Weekly': 'W', 'Daily':'D'}


app.layout = html.Div(
    children = [

        # Page Container Element
        html.Div([

# *** START HEADER FILTER SECTION***
        html.Div(
            [
                html.H1(
                    'FMDView',
                    className = 'twelve columns',
                    style = {'text-align': 'left',
                            'color': '#A499AB',
                            'font-size':60}
                        ),
            ],
            className = 'row',
            style = {'margin-top':5}
                ),

      html.Div(
            [
                html.Div(
                    [
                        html.P('''Filter by Fiscal Year to Explore
                        Work Order Distribution & Density:'''),

                        dcc.Dropdown(
                            id='2d-histogram-fiscal-year-input',
                            options = [{'label': fy, 'value': fy} for
                                      fy in df['fiscal_year_requested'].unique()],
                            multi=False,
                            value=''
                                    ),


                    ],
                    className='two columns'
                ),

                html.Div(
                    [
                        html.P('''Explore Work Requests & Completion
                        Gap by Frequency:'''),

                        dcc.Dropdown(
                            id='line-chart-frequency-input',
                            options = [{'label': key, 'value': key} for
                                       key in frequency_dict.keys()],
                            multi=False,
                            value=''
                                    ),
                    ],
                    className='two columns'
                ),

                html.Div(
                    [
                        html.H3('''An exploratory narrative of performance &
                        activity in 10 Interactive Charts.''')
                    ],
                    className = 'eight columns',
                    style = {'text-align': 'center',
                            'color': '#A499AB',
                            'font-size': 40}
                        )
            ],
          className='row',
          style = {'margin-top':10}

                ),


# *** END HEADER FILTER SECTION***

# ***FIRST ROW OF CHARTS**
    html.Div(
        [
            html.Div(
                [
                    dcc.Graph(
                        id = '2d-histogram-duration-volume-by-type',
                        #figure = make_2dhist_figure_for_fiscal_year(fy=current_year),

                        config = {
                            'displaylogo':False,
                            'showLink':False,
                            'displayModeBar': 'hover',
                            'modeBarButtons': [['zoom2d', 'lasso2d'],['toImage', 'resetScale2d']]
                                }
                            ),
                ],
                className = 'four columns',
                style = {'margin-top': '20'}
                    ),

            html.Div(
                [
                    dcc.Graph(
                        id = 'scatter-mapbox-workorder-building-volume-distribution',
                        #figure = map_workorder_volume_distribution(),
                        config = {
                            'displaylogo':False,
                            'showLink':False,
                            'displayModeBar': 'hover',
                            'modeBarButtons': [['toImage']]
                                }
                            ),
                ],
                className = 'eight columns',
                style = {'margin-top': '20'}
                    ),
        ],
        className = 'row',
            ),


# ***SECOND ROW OF CHARTS*** -- Completion Distrubution Frequency & Daily Heatmaps + Stacked Pct Chart
    html.Div(
        [
            html.Div(
                [
                    dcc.Graph(
                        id = 'heatmap-weekday-count-closed',
                        figure = make_daily_count_close_heatmap(),
                        config = {
                            'displaylogo':False,
                            'showLink': False,
                            'modeBarButtons': [['zoom2d'],['toImage','resetScale2d']]
                                }
                            ),
                ],
                className = 'four columns',
                style = {'margin-top': '10'}
                    ),

            html.Div(
                [
                    dcc.Graph(
                        id = 'heatmap-weekday-percentage-closed',
                        figure = make_daily_percent_close_heatmap(),
                        config = {
                            'displaylogo':False,
                            'showLink': False,
                            'modeBarButtons': [['zoom2d'],['toImage','resetScale2d']]
                                }
                            ),
                ],
                className = 'four columns',
                style = {'margin-top': '10'}
                    ),

            html.Div(
                [
                    dcc.Graph(
                        id = 'line-chart-request-completion-gap',
                        figure = request_completion_gap_line_chart(),
                        config = {
                            'displaylogo': False,
                            'showLink': False,
                            'modeBarButtons': [["hoverClosestGl2d", 'hoverCompareCartesian'],
                                               ['zoom2d'], ['toImage', 'resetScale2d']]
                                }
                                ),
                ],
                className = 'four columns',
                style = {'margin-top': '10'}
                    )
        ],
        className = 'row',
            ),

# ***THIRD ROW OF CHARTS*** -- Completion Distrubution Frequency & Monthly Heatmaps
    html.Div(
        [
            html.Div(
                [
                    dcc.Graph(
                        id = 'heatmap-monthly-count-closed',
                        figure = make_monthly_count_close_heatmap(),
                        config = {
                            'displaylogo':False,
                            'showLink': False,
                            'modeBarButtons': [['zoom2d'],['toImage','resetScale2d']]
                                }
                            ),
                ],
                className = 'four columns',
                style = {'margin-top': '5'}
                    ),

            html.Div(
                [
                    dcc.Graph(
                        id = 'heatmap-monthly-percentage-closed',
                        figure = make_monthly_percent_close_heatmap(),
                        config = {
                            'displaylogo':False,
                            'showLink': False,
                            'modeBarButtons': [['zoom2d'],['toImage','resetScale2d']]
                                }
                            ),
                ],
                className = 'four columns',
                style = {'margin-top': '5'}
                    ),

            html.Div(
                [
                     dcc.Graph(
                        id = 'stacked-bar-pct-closed-on-time-within-same-year',
                        figure = make_on_time_kpi_barchart_opened_closed_same_year(),
                        config = {
                            'displaylogo': False,
                            'showLink': False,
                            'modeBarButtons': [["hoverClosestGl2d", 'hoverCompareCartesian'], ['zoom2d',
                                                'select2d'], ['toImage', 'resetScale2d']]
                                }
                                ),
                ],
                className = 'four columns',
                style = {'margin-top': '5'}
                    ),
        ],
        className = 'row',
            ),
# Fourth Row
    html.Div(
        [
            html.Div(
                [
                    dcc.Graph(
                        id = 'barchart-avg-durations-top-3.5.10.15-volume',
                        figure = make_avg_durations_by_top_volume_chart(),
                        config = {
                            'displaylogo': False,
                            'showLink': False,
                            'modeBarButtons': [["hoverClosestGl2d", 'hoverCompareCartesian'], ['zoom2d',
                                                'select2d'], ['toImage', 'resetScale2d']]
                                }
                            ),
                ],
                className = 'eight columns',
                style = {'margin-top': '5'}
                    ),

            html.Div(
                [
                    dcc.Graph(
                        id = 'stacked-bar-pm-to-cm',
                        figure = make_pm_to_cm_stacked_bar_chart(),
                        config = {
                            'displaylogo': False,
                            'showLink': False,
                            'modeBarButtons': [["hoverClosestGl2d", 'hoverCompareCartesian'], ['zoom2d',
                                                'select2d'], ['toImage', 'resetScale2d']]
                                }
                            ),
                ],
                className = 'four columns',
                style = {'margin-top': '5'}
                    ),
        ],
        className = 'row',
            ),
                ],
            style = {'backgroundColor':'#EAE8EB'} # set website background
                )
                ]
                     )

### CALLBACKS FOR DASHBOARD INTERACTIVITY & FILTERING ###
@app.callback(
    dash.dependencies.Output(component_id='2d-histogram-duration-volume-by-type',
                             component_property='figure'),
    [dash.dependencies.Input(component_id='2d-histogram-fiscal-year-input',
                             component_property='value')])
def update_figure(selected_year):
    """ Call the special function for creating
    this chart which returns the 2 dimensional histogram
    with desired settings and returns chart with selected
    fiscal year or All years if no filter passed."""
    return make_2dhist_figure_for_fiscal_year(fy=selected_year)


@app.callback(
    dash.dependencies.Output(component_id='scatter-mapbox-workorder-building-volume-distribution',
                             component_property='figure'),
    [dash.dependencies.Input(component_id='2d-histogram-fiscal-year-input',
                            component_property='value')])
def update_map(selected_year):
    """ Call the special function for creating this map
    with settings for center lat long, marker color to match the
    single year filter selection with the 2d histogram and plots
    all data for each fiscal year and provides an intereactive legend
    for each year if no filter fiscal year is selected."""
    return map_workorder_volume_distribution(fy=selected_year)


@app.callback(
    dash.dependencies.Output(component_id='line-chart-request-completion-gap',
                             component_property='figure'),
    [dash.dependencies.Input(component_id='line-chart-frequency-input',
                             component_property='value')])
def update_line_chart(selected_frequency):
    """Call function handling the line chart with frequencies
    as a dictionary of values and keys from Annually to Daily."""
    return request_completion_gap_line_chart(frequency=selected_frequency)


if __name__ == '__main__':
    app.run_server(debug=True)
