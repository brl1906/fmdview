from configparser import ConfigParser
from datetime import datetime
import sys
import warnings

import data
from data import dframe, map_dframe, hvac_types, corrective_types
from data import preventative_types, weekday_name, get_fiscalyear, month_name
import plotly.graph_objs as go
import numpy as np
from sklearn import preprocessing

warnings.filterwarnings(action='ignore')

### Chart types are organized categorically and alphabetically For example,
### chart objects are grouped by class or type and each chart type appears
### in sections with documented headers for: Bar charts, Box & Violin plots,
### Financial charts, Heatmaps, Histograms, Line charts, Maps, Polar plots,
### Pie charts, Sankey plots, Scatter plots

class Chart():
    """Establishes standard bar chart object for FMD dashboard.
    
    Class sets uniform stylings for plot, background, chart 
    font and title font.  It has 2 attributes that mirror the 
    basic implementation required to create a Plotly chart:
    data and a layout.
    
    Parameters
    ----------
    traces:   list
            
    
    layout:   dict
            Dictionary object corresponding to parameters
            in Plotly documentation for chart layout. 
    
    """
    
    def __init__(self, traces, layout):
        self.traces = traces
        self.layout = layout
        self.layout['font']['color'] = '#CCCCCC'
        self.layout['titlefont']['color'] = '#CCCCCC'
        self.layout['plot_bgcolor'] = '#303939'
        self.layout['paper_bgcolor'] = '#303939'
        self.layout['hovermode'] = 'closest'
        ## Plotly prefers list objects passed as data  
        self.fig = {'data':traces, 'layout':layout}


################################################################################
################################################################################

                                ## BAR CHARTS ##

################################################################################
################################################################################

# ---- pct work orders closed on time ---- #
def pct_closed_ontime(dframe):
    """
    """
    fiscalyears, fy_dfs = [],[]
    for year in dframe['fiscal_year_requested'].unique():
        fy_dfs.append(dframe.loc[(dframe['fiscal_year_requested']==year)])
        fiscalyears.append(year)
    ## convert fy_dfs from list to dict object as key & value for year and df
    fy_dfs = dict(zip(fiscalyears,fy_dfs))

    ## get total number of work orders requested each fiscal year
    workorder_volume = []
    for key, value in fy_dfs.items():
        workorder_volume.append(value['wo_id'].count())
    workorder_volume = dict(zip(fiscalyears, workorder_volume))

    closed_workorders_dfs = []
    for key, value in fy_dfs.items():
        closed_workorders_dfs.append(value[value['date_completed'].notnull()])
    closed_workorders_dfs = dict(zip(fy_dfs.keys(), closed_workorders_dfs))


    ## Implement methodology for calculating work orders on time:
    ## 1. Get work order request volume by problem type for each fiscal year
    ## 2. Identify the mean duration for each request type within each year
    ## 3. Identify the instances and number of ocurrences where the duration of
    ##    an individual work order had a duration less than or equal to the
    ##    duration of the mean for its type for that fiscal year.

    for key, value in closed_workorders_dfs.items():
        value['count'] = value.groupby(['prob_type'])['prob_type'].transform('count')
        value['avg_duration'] = (value.groupby(['prob_type'])['duration'].\
                            transform('sum') / value['count'])
        value['on_time'] = np.where(value.duration <= value.avg_duration,'hit','miss')

    # bar charts of pct closed on time vs volume
    kpi_values_dict = {}
    for year in closed_workorders_dfs.keys():
        kpi_values_dict[year] = (closed_workorders_dfs[year]['on_time'].
                                 value_counts()[0] / fy_dfs[year]['wo_id'].
                                 count() * 100)

    traces = []
    values_text = ['<b>FY{}</b><br>ON TIME: {:.0f}%'.
                   format(yr,val) for yr, val in
                   zip([year for year in kpi_values_dict.keys()],
                       [value for value in kpi_values_dict.values()])]

    requests_text = ['<b>FY{}</b><br>{:,} requests'
                     .format(yr,val) for yr, val in
                     zip([year for year in workorder_volume.keys()],
                         [value for value in workorder_volume.values()])]

    for dict_object,name,text,marker_color,line_color,opacity,yaxis in zip(
        [kpi_values_dict, workorder_volume],['% on time','requests'],
        [values_text, requests_text],['#3c5a89','#656969'],
        ['#3c5a89','#CCCCCC'],[0.9, 0.2],['y1','y2']):
        traces.append(
            go.Bar(
                x = [year for year in dict_object.keys()],
                y = [value for value in dict_object.values()],
                name = name,
                text = text,
                marker = {'color': marker_color,
                         'line': {'color': line_color,
                                 'width': 1}},
                hoverinfo = 'text',
                opacity = opacity,
                yaxis = yaxis)
        )

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

    fig = {'data':traces, 'layout': layout}
    return fig


# ---- PM to CM KPI (preventative vs. corrective) ---- #
def pm2cm_kpi(dframe,pm_types,cm_types,chart_label='',hvac_types=[]):
    """Returns chart visualizing the changes in the PM to CM Key
    Performance Indicator.

    The PM to CM KPI is a measure of the ratio of the volume of
    preventative maintenance performed in a given fiscal year
    against the volume of corrective maintenance performed in the
    year.

    Parameters
    ----------
    dframe:   Pandas Dataframe
            Dataframe returned from data.py module

    pm_types:    List
            List of strings of valid work order problem types used
            for preventative maintenance work across building assets.

    cm_types:    List
            List of strings of valid work order problem types used
            for corrective maintenance work across building assets.

    chart_label: Str  (optional)
            Name of problem type or category of work used to show or
            create a filtered view of the PM:CM measure for a specific
            problem type as a subcategory.  For example, passing 'HVAC'
            will show HVAC as the first wor in the chart title.

    hvac_types:  List  (optional)
            List of strings of valid work order problem types in
            use for HVAC work.

    Returns
    -------
    Dict:   Plotly figure dictionary with keys: 'data' and 'layout'

    Example
    -------
    >>> pm2cm_kpi(dframe=df, pm_types=['window-pm','floor-pm'],
                  cm_types=['ROOF','CEILING','COOLING TOWERS',
                            'DOORS','TANKS'])

    >>> pm2cm_kpi(dframe, preventative, corrective, 'HVAC',
                  hvac_types=['WINDOW UNIT','FILTER','HVAC'])

    """

    fiscalyears, fy_dfs = [],[]
    for year in dframe['fiscal_year_requested'].unique():
        fy_dfs.append(dframe.loc[(dframe['fiscal_year_requested']==year)])
        fiscalyears.append(year)
    fy_dfs = dict(zip(fiscalyears,fy_dfs))

    ## get total number of requests for each fiscal year
    workorder_volume = []
    for key, value in fy_dfs.items():
        workorder_volume.append(value['wo_id'].count())
    workorder_volume = dict(zip(fiscalyears, workorder_volume))

    kpi_values_dict = {}
    for key, value in fy_dfs.items():
            kpi_values_dict[key] = (
                value[value['prob_type'].isin(pm_types)]['prob_type'].
                value_counts().sum() /

                value[value['prob_type'].isin(cm_types)]['prob_type'].
                value_counts().sum() * 100 )

    traces = []
    pm2cm_text = ['<b>FY{}</b><br>PM:CM {:.0f}%'.
                  format(yr,val) for yr, val in
                  zip([year for year in kpi_values_dict.keys()],
                      [value for value in kpi_values_dict.values()])]

    requests_text = ['<b>FY{}</b>:<br>{:,} requests'.
                     format(yr,val) for yr, val in
                     zip([year for year in workorder_volume.keys()],
                         [value for value in workorder_volume.values()])]

    for dict_object,name,text,marker_color,line_color,opacity,trace_yaxis in zip(
        [kpi_values_dict, workorder_volume],['pm:cm','all requests'],
        [pm2cm_text, requests_text],['#3c5a89','#656969'],
        ['#3c5a89','#CCCCCC'],[0.9, 0.2],['y1','y2']):
        traces.append(
            go.Bar(
                x = [year for year in dict_object.keys()],
                y = [value for value in dict_object.values()],
                name = name,
                text = text,
                marker = {'color': marker_color,
                         'line': {'color': line_color,
                                 'width': 1}},
                hoverinfo = 'text',
                opacity = opacity,
                yaxis = trace_yaxis)
        )


    layout = go.Layout(
        hovermode = 'closest',
        legend = {'orientation' : 'h'},
        font = {'color': '#CCCCCC'},
        titlefont = {'color': '#CCCCCC',
                    'size': 14},
        title = ('{} PM:CM KPI FY{} to FY{}'.
                 format(chart_label, min([year for year in kpi_values_dict.keys()]),
                        max([year for year in kpi_values_dict.keys()]))),

        yaxis=dict(title='pct %',
                   showgrid = False,
                   titlefont = {'color':'#3c5a89'},
                   tickfont = {'color': '#3c5a89'}),
        yaxis2=dict(title = 'work requests',
                    showgrid = False,
                    titlefont = {'color': '#656969'},
                    tickfont =  {'color': '#656969'},
                    overlaying='y',
                    side='right'),
        plot_bgcolor = '#303939',
        paper_bgcolor = '#303939',
                        )

    fig = {'data':traces, 'layout':layout}
    return fig

################################################################################
################################################################################

                        ## BOX & VIOLIN PLOTS ##

################################################################################
################################################################################




################################################################################
################################################################################

                            ## FINANCIAL CHARTS ##

################################################################################
################################################################################





################################################################################
################################################################################

                                ## HEATMAPS ##

################################################################################
################################################################################

# ---- count and percentage of requests closed by weekday ---- #
def weekday_completion(dframe, z_values='count',colorscale='Portland'):
    """Return heatmap vizualizing the count or percentage of work orders
    closed each weekday for each fiscal year.

    Parameters
    ----------
    dframe:   Pandas Dataframe
            Expects the dataframe of archibus maintenance work order data
            returned from data.py module

    z_values: Str (optional)
            Sets the calculation and dislpay method for the z values associated
            with the heatmap. Available options include: 'count' & 'percentage'.
            The default value will return how many work orders were completed on
            each weekday within a fiscal year while 'percentage' will return the
            percentage of work orders closed on a weekday out of all work
            ordrers closed that year.    For example-- 804 work orders were
            completed on Monday in 2018 vs 12% of all work orders completed in
            2018 were completed on Mondays.

    colorscale: Str (default value: 'Portland')
            The colorscale style for the heatmap. Availalbe colorscale options
            can be found in the Plotly or Matplotlib documentation.

    Returns
    -------
    Dict:  Returns a dictionary with keys 'data' and 'layout' corresponding to
           a Plotly figure object.  {'data':traces, 'layout':layout}

    Example
    -------
    >>> weekday_completion(dframe=dframe)

    >>> weekday_completion(dframe, z_values='percentage')
    """

    # filter out still open work orders
    df = dframe[dframe['date_completed'].notnull()]
    df['completed_day_name'] = (df['date_completed'].
                                apply(lambda x: weekday_name(x.dayofweek)))

    z_dict = {}
    x = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    ## Switch data returned for chart based on z_values parameter
    ## to control for chart views and data based on parameters occur here
    valid_zparams = ['count','percentage']
    if z_values not in valid_zparams:
        print(
        """Error: Valid values for z_values parameter include: "count" &
"percentage". Will generate UnboundLocalError as variables in layout were not
initialized because proper parameter was not passed.""")

    else:
        if z_values == 'count':
            chart_title = 'Work Orders Closed Daily'
            for year in df['fiscal_year_completed'].unique():
                z_dict[year] = []
                for day in x:
                    z_dict[year].append(
                        df[(df['fiscal_year_completed'] == year) &
                           (df['completed_day_name'] == day)]['wo_id'].count())

        else:
            chart_title = 'Percentage of Work Orders Closed Daily'
            for year in df['fiscal_year_completed'].unique():
                z_dict[year] = []
                for day in x:
                    z_dict[year].append(
                    df[(df['fiscal_year_completed'] == year) &
                      (df['completed_day_name'] == day)]['wo_id'].count() /
                    df[df['fiscal_year_completed'] == year]['wo_id'].count() * 100)


    trace = go.Heatmap(
        z = [value for value in z_dict.values()],
        y = ['FY {}'.format(int(year)) for year in z_dict.keys()],
        x = x,
        colorscale = colorscale)

    layout = go.Layout(
        hovermode = 'closest',
        font = {'color': '#CCCCCC'},
        titlefont = {'color': '#CCCCCC',
                    'size': 14},
        title = chart_title,
        margin = {'r':55, 'b':70,
                 'l': 70, 't': 45},
        plot_bgcolor = '#303939',
        paper_bgcolor = '#303939'
                        )
    fig = {'data':[trace], 'layout':layout}
    return fig

# ---- count and percentage of requests closed by month ---- #
def month_completion(dframe, z_values='count',colorscale='Portland'):
    """Return heatmap vizualizing the count or percentage of work orders
    closed each month for each fiscal year.

    Parameters
    ----------
    dframe:   Pandas Dataframe
            Expects the dataframe of archibus maintenance work order data
            returned from data.py module

    z_values: Str (optional)
            Sets the calculation and dislpay method for the z values associated
            with the heatmap. Available options include: 'count' & 'percentage'.
            The default value will return how many work orders were completed on
            each weekday within a fiscal year while 'percentage' will return the
            percentage of work orders closed in each month out of all work
            ordrers closed that year.    For example-- 804 work orders were
            completed in March of 2018 vs 12% of all work orders completed in
            2018 were completed in March.

    colorscale: Str (default value: 'Portland')
            The colorscale style for the heatmap. Availalbe colorscale options
            can be found in the Plotly or Matplotlib documentation.

    Returns
    -------
    Dict:  Returns a dictionary with keys 'data' and 'layout' corresponding to
           a Plotly figure object.  {'data':traces, 'layout':layout}

    Example
    -------
    >>> weekday_completion(dframe=dframe)

    >>> weekday_completion(dframe, z_values='percentage')
    """

    # filter out still open work orders
    df = dframe[dframe['date_completed'].notnull()]    
    df['completed_month_name'] = (df['date_completed'].
                                  apply(lambda x: month_name(x.month)))

    z_dict = {}
    x = ['January','February','March','April','May','June','July','August',
         'September','October','November','December']

    ## Switch data returned for chart based on z_values parameter
    ## to control for chart views and data based on parameters occur here
    valid_zparams = ['count','percentage']
    if z_values not in valid_zparams:
        print(
        """Error: Valid values for z_values parameter include: "count" &
"percentage". Will generate UnboundLocalError as variables in layout were not
initialized because proper parameter was not passed.""")

    else:
        if z_values == 'count':
            chart_title = 'Work Orders Closed Monthly'
            for year in df['fiscal_year_completed'].unique():
                z_dict[year] = []
                for month in x:
                    z_dict[year].append(
                        df[(df['fiscal_year_completed'] == year) &
                           (df['completed_month_name'] == month)]['wo_id'].count())
        
        elif z_values == 'percentage':
            chart_title = 'Percentage of Work Orders Closed Monthly'
            for year in df['fiscal_year_completed'].unique():
                z_dict[year] = []
                for month in x:
                    z_dict[year].append(
                    df[(df['fiscal_year_completed'] == year) &
                      (df['completed_month_name'] == month)]['wo_id'].count() /
                    df[df['fiscal_year_completed'] == year]['wo_id'].count() * 100)
                    
        else:
            #### ADD WARNING HERE ABOUT VALID Z_VALUES PARAMETERS ###
            pass 
        
    trace = go.Heatmap(
        z = [value for value in z_dict.values()],
        y = ['FY {}'.format(int(year)) for year in z_dict.keys()],
        x = x,
        colorscale = colorscale)

    layout = go.Layout(
        hovermode = 'closest',
        font = {'color': '#CCCCCC'},
        titlefont = {'color': '#CCCCCC',
                    'size': 14},
        title = chart_title,
        margin = {'r':55, 'b':70,
                 'l': 70, 't': 45},
        plot_bgcolor = '#303939',
        paper_bgcolor = '#303939'
                        )
    fig = {'data':[trace], 'layout':layout}
    return fig

################################################################################
################################################################################

                            ## HISTOGRAMS ##

################################################################################
################################################################################





################################################################################
################################################################################

                            ## LINE CHARTS ##

################################################################################
################################################################################
# ---- opened vs completion gap (backlog) ---- #     
## TODO: add docstring
## TODO: style hovertext with break lines and bold
## TODO: move labels horizontal below chart

def open_vs_completed(dframe, frequency='M', id_column='wo_id', completion_column='date_completed'):
    """
    """
    
    resampled_data = {}
    traces = []
    
    for name, column, color in zip(['opened','completed'],[id_column, completion_column],['#D4395B','#ABCDAB']):
        
        frequency_dict = {'A':'Annually','M':'Monthly','W': 'Weekly','D':'Daily'}
        resampled_data[name] = dframe.resample(frequency)[column].count()
        
        traces.append(go.Scatter(
            x = resampled_data[name].index,
            y = resampled_data[name].values,
            name = name, 
            line = {'color':color,
                   'width': 5},
            opacity = .8, 
            text = ['<b>{} {}</b><br>{}: {:,}'.format(date.strftime('%Y'), date.strftime('%b'), name, val)
                    for date, val in resampled_data[name].items()],
            hoverinfo = 'text'))
        
        traces.append(go.Scatter(
            x = dframe.resample('W')[column].count().rolling(window=4).mean().index,
            y = dframe.resample('W')[column].count().rolling(window=4).mean().values,
            name = '4week SMA',
            line = {'color': color,
                   'width':2},
            
            hoverinfo = 'name+x+y'))
        
    layout = go.Layout(
        title = 'Work Order Request|Completion Gap<br>Frequency: <b><i>{}</i></b>'.format(frequency_dict[frequency]),
        autosize = True,
        legend = {'orientation': 'h'},
        font = {'color': '#CCCCCC'},
        titlefont = {'color': '#CCCCCC'},
        hovermode = 'closest',
        margin = {'r':35, 'b':10,
                     'l': 50, 't': 35},
        paper_bgcolor = '#303939',
        plot_bgcolor = '#303939',
        xaxis = dict(
        rangeselector = dict(
            buttons = list([
                dict(count = 6,
                     label = '6m',
                     step = 'month',
                     stepmode = 'backward'),
                dict(count = 1,
                    label = 'YTD',
                    step = 'year',
                    stepmode = 'todate'),
                dict(count = 1,
                    label = '1y',
                    step = 'year',
                    stepmode = 'backward'),
                dict(step = 'all')
            ])),
        rangeslider = {'visible':True},
        type = 'date'))
    fig = {'data': traces, 'layout':layout}
        
    return fig





################################################################################
################################################################################

                                ## MAPS ##

################################################################################
################################################################################






################################################################################
################################################################################

                            ## POLAR PLOTS ##

################################################################################
################################################################################





################################################################################
################################################################################

                            ## PIE CHARTS ##

################################################################################
################################################################################






################################################################################
################################################################################

                            ## SANKEY PLOTS ##

################################################################################
################################################################################






################################################################################
################################################################################

                            ## SCATTER PLOTS ##

################################################################################
################################################################################
def duration_vs_volume(dframe, fiscalyear_column, year):
    """Return scatter plot on relationship between duration and volume for each 
    problem type.
    
    Parameters
    ----------
    dframe:            Pandas Dataframe
            Dataframe returned from data.py module
            
    fiscalyear_column: Pandas Series
            Column containing 4 digit int type data on fiscal year of a request. For 
            example: [2001,2018,2011,2015]

    fiscalyear:        Int
            Four digit number to indicate the desired fiscal year. 
            
    Returns
    -------
    Dict:   Plotly figure dictionary with keys: 'data' and 'layout'

    Example
    -------
    >>> duration_volume_scatter(dframe, 'fiscal_year_requested', 2017)        
    """
    
    remove_openorders = data.remove_open_workorders(dframe)
    dataframe = data.filter_fiscalyear(dframe=remove_openorders,
                                       column=fiscalyear_column, fiscalyear=year)
    workorder_volume, avg_duration, pct_ontime = data.ontime(dataframe)
    
    traces = [
        go.Scatter(
            x = workorder_volume,
            y = avg_duration,
            xaxis = 'x',
            yaxis = 'y',
            name = 'problem type',
            mode = 'markers',
            text = ['{}:<br><b>ontime: </b>{:.0f}%<br><b>volume:</b> {:,}'.format(
                name, pct, volume) for name, pct, volume in zip(
                pct_ontime.keys(), pct_ontime.values, workorder_volume.values)],
            hoverinfo = 'text',
            marker = dict(
                color = '#D4395B',
                size = preprocessing.normalize([pct_ontime])[0] * 100),
            opacity = .7),

        go.Histogram(
            y = avg_duration,
            xaxis = 'x2',
            nbinsy = 25,
            name = 'avg days',
            marker = dict(
                color = '#CCCCCC')),
        
        go.Histogram(
            x = workorder_volume,
            yaxis = 'y2',
            name = 'type volume',
            nbinsx = 45,
            marker = dict(
                color = '#CCCCCC'))
    ]

    layout = go.Layout(
            title = '<b>FY{}</b> Workorder Avg Duration & Volume<br>(size = pct ontime)'.format(year),
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
            paper_bgcolor = '#303939')
    fig = {'data':traces, 'layout':layout}
    
    return fig