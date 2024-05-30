# climate-dashboard

This project is part of the class "Intro to Data Science with Python" made available by Harvard through EdX. 

The dashboard allows users to visualize temperature data at global, country, and city levels and analyze historical temperature trends and natural disaster events.

## Features
Global Temperature Analysis: Visualize the average, maximum, and minimum land temperatures globally using line charts with polynomial regression trend lines.
Temperature by Country: Select multiple countries to view their historical temperature trends with options to display yearly average temperatures.
World Map View: Display a choropleth map showing the average temperature of countries over a selected range of years.
Temperature Increase View: Visualize the temperature increase from 1850 to 2014 for different countries using a choropleth map.
Natural Disaster Analysis: Explore natural disaster occurrences by type and by country, including top 10 countries for natural disaster occurrences and trends over time.

## Datasets
The datasets used for the projects are the following, available for download on Kaggle:
1. https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data
2. https://www.kaggle.com/datasets/brsdincer/all-natural-disasters-19002021-eosdis

### Data Sources

1. Global Temperature Data: Sourced from the GlobalTemperatures.csv file.
2. Temperature by Country Data: Sourced from the GlobalLandTemperaturesByCountry.csv file.
3. Temperature by City Data: Sourced from the GlobalLandTemperaturesByMajorCity.csv file.
4. Natural Disaster Data: Sourced from the 1900_2021_DISASTERS.xlsx file.

## Usage
*Global Temperature Analysis*:
Navigate to the "Global Temperature Analysis" page to visualize global temperature trends. The page includes the following visualizations:

*Average Land Temperature
Maximum Land Temperature
Minimum Land Temperature*
Each visualization includes a trend line obtained using polynomial regression.

*Temperature by Country*:
Navigate to the "Temperature by Country" page to select multiple countries and view their temperature history. You can choose to display yearly average temperatures or monthly temperatures.

*World Map View*:
Navigate to the "World Map View" page to visualize the average temperature of countries on a world map for a selected range of years.

*Temperature Increase View*:
Navigate to the "Temperature Increase View" page to see the increase in temperature for different countries from 1850 to 2014.

*Natural Disaster Analysis*:
Navigate to the "Natural Disaster Analysis" page to explore natural disaster occurrences

*Top 10 Countries for Natural Disasters*: A bar chart showing the top 10 countries with the highest number of natural disasters.
Natural Disasters by Type Over Time: A line chart showing the trend of different types of natural disasters over time.
Natural Disasters by Country: Select multiple countries to view the trend of natural disasters over time.

## Custom CSS
The dashboard uses a custom font from Google Fonts. The CSS code is included within the Streamlit application to apply the font across the dashboard.
```
st.write("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@200&display=swap');
html, body, [class*="css"]  {
   font-family: 'Lexend', sans-serif;
}
</style>
""", unsafe_allow_html=True)
```
