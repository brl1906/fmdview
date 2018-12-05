# FMDView
FMDView is an interactive dashboard to support management, analysis for new program development or process improvement opportunities and outcome budget development.   

## Output
![dashboard](https://blima06.pythonanywhere.com/)


## Built Primarily With
Python 3.6, Plotly, Dash, Pandas, PythonAnywhere

## Use Me
1. clone the project
2. set up a virtual environment, activate it and install the dependencies with pip ```pip install -r requirements.txt```
3. get a free [mapbox](https://www.mapbox.com/help/how-access-tokens-work/) access token
4. create a configuration file named 'config.ini' and set the token value for the MapboxToken section to your mapbox access token

9. run the program from the virtual environment with the command ```python app.py```

## TODO:
* [ ] resolve rendering on map chart  
* [ ] add map in chart connected to 2d histogram that displays volume and location of requests by the top 3, top 5 filter selection on 2d histogram chart.
* [ ] build test pipeline with data.world api
