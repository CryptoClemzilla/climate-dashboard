import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import altair as alt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

iso_codes = {
    'Åland': 'ALA',
    'Afghanistan': 'AFG',
    'Africa': '',  # No ISO code provided
    'Albania': 'ALB',
    'Algeria': 'DZA',
    'American Samoa': 'ASM',
    'Andorra': 'AND',
    'Angola': 'AGO',
    'Anguilla': 'AIA',
    'Antigua And Barbuda': 'ATG',
    'Argentina': 'ARG',
    'Armenia': 'ARM',
    'Aruba': 'ABW',
    'Asia': '',  # No ISO code provided
    'Australia': 'AUS',
    'Austria': 'AUT',
    'Azerbaijan': 'AZE',
    'Bahamas': 'BHS',
    'Bahrain': 'BHR',
    'Baker Island': '',  # No ISO code provided
    'Bangladesh': 'BGD',
    'Barbados': 'BRB',
    'Belarus': 'BLR',
    'Belgium': 'BEL',
    'Belize': 'BLZ',
    'Benin': 'BEN',
    'Bhutan': 'BTN',
    'Bolivia': 'BOL',
    'Bonaire, Saint Eustatius And Saba': 'BES',
    'Bosnia And Herzegovina': 'BIH',
    'Botswana': 'BWA',
    'Brazil': 'BRA',
    'British Virgin Islands': 'VGB',
    'Bulgaria': 'BGR',
    'Burkina Faso': 'BFA',
    'Burma': '',  # No ISO code provided
    'Burundi': 'BDI',
    "Côte D'Ivoire": 'CIV',
    'Cambodia': 'KHM',
    'Cameroon': 'CMR',
    'Canada': 'CAN',
    'Cape Verde': 'CPV',
    'Cayman Islands': 'CYM',
    'Central African Republic': 'CAF',
    'Chad': 'TCD',
    'Chile': 'CHL',
    'China': 'CHN',
    'Christmas Island': 'CXR',
    'Colombia': 'COL',
    'Comoros': 'COM',
    'Congo (Democratic Republic Of The)': 'COD',
    'Congo': 'COG',
    'Costa Rica': 'CRI',
    'Croatia': 'HRV',
    'Cuba': 'CUB',
    'Curaçao': 'CUW',
    'Cyprus': 'CYP',
    'Czech Republic': 'CZE',
    'Denmark (Europe)': 'DNK',
    'Denmark': 'DNK',
    'Djibouti': 'DJI',
    'Dominica': 'DMA',
    'Dominican Republic': 'DOM',
    'Ecuador': 'ECU',
    'Egypt': 'EGY',
    'El Salvador': 'SLV',
    'Equatorial Guinea': 'GNQ',
    'Eritrea': 'ERI',
    'Estonia': 'EST',
    'Ethiopia': 'ETH',
    'Europe': '',  # No ISO code provided
    'Falkland Islands (Islas Malvinas)': 'FLK',
    'Faroe Islands': 'FRO',
    'Federated States Of Micronesia': 'FSM',
    'Fiji': 'FJI',
    'Finland': 'FIN',
    'France (Europe)': 'FRA',
    'France': 'FRA',
    'French Guiana': 'GUF',
    'French Polynesia': 'PYF',
    'French Southern And Antarctic Lands': 'ATF',
    'Gabon': 'GAB',
    'Gambia': 'GMB',
    'Gaza Strip': '',  # No ISO code provided
    'Georgia': 'GEO',
    'Germany': 'DEU',
    'Ghana': 'GHA',
    'Greece': 'GRC',
    'Greenland': 'GRL',
    'Grenada': 'GRD',
    'Guadeloupe': 'GLP',
    'Guam': 'GUM',
    'Guatemala': 'GTM',
    'Guernsey': 'GGY',
    'Guinea Bissau': 'GNB',
    'Guinea': 'GIN',
    'Guyana': 'GUY',
    'Haiti': 'HTI',
    'Heard Island And Mcdonald Islands': 'HMD',
    'Honduras': 'HND',
    'Hong Kong': 'HKG',
    'Hungary': 'HUN',
    'Iceland': 'ISL',
    'India': 'IND',
    'Indonesia': 'IDN',
    'Iran': 'IRN',
    'Iraq': 'IRQ',
    'Ireland': 'IRL',
    'Isle Of Man': 'IMN',
    'Israel': 'ISR',
    'Italy': 'ITA',
    'Jamaica': 'JAM',
    'Japan': 'JPN',
    'Jersey': 'JEY',
    'Jordan': 'JOR',
    'Kazakhstan': 'KAZ',
    'Kenya': 'KEN',
    'Kingman Reef': '',  # No ISO code provided
    'Kiribati': 'KIR',
    'Kuwait': 'KWT',
    'Kyrgyzstan': 'KGZ',
    'Laos': 'LAO',
    'Latvia': 'LVA',
    'Lebanon': 'LBN',
    'Lesotho': 'LSO',
    'Liberia': 'LBR',
    'Libya': 'LBY',
    'Liechtenstein': 'LIE',
    'Lithuania': 'LTU',
    'Luxembourg': 'LUX',
    'Macau': 'MAC',
    'Macedonia': 'MKD',
    'Madagascar': 'MDG',
    'Malawi': 'MWI',
    'Malaysia': 'MYS',
    'Mali': 'MLI',
    'Malta': 'MLT',
    'Martinique': 'MTQ',
    'Mauritania': 'MRT',
    'Mauritius': 'MUS',
    'Mayotte': 'MYT',
    'Mexico': 'MEX',
    'Moldova': 'MDA',
    'Monaco': 'MCO',
    'Mongolia': 'MNG',
    'Montenegro': 'MNE',
    'Montserrat': 'MSR',
    'Morocco': 'MAR',
    'Mozambique': 'MOZ',
    'Namibia': 'NAM',
    'Nepal': 'NPL',
    'Netherlands (Europe)': 'NLD',
    'Netherlands': 'NLD',
    'New Caledonia': 'NCL',
    'New Zealand': 'NZL',
    'Nicaragua': 'NIC',
    'Niger': 'NER',
    'Nigeria': 'NGA',
    'Niue': 'NIU',
    'North America': '',  # No ISO code provided
    'North Korea': 'PRK',
    'Northern Mariana Islands': 'MNP',
    'Norway': 'NOR',
    'Oceania': '',  # No ISO code provided
    'Oman': 'OMN',
    'Pakistan': 'PAK',
    'Palau': 'PLW',
    'Palestina': '',  # No ISO code provided
    'Palmyra Atoll': '',  # No ISO code provided
    'Panama': 'PAN',
    'Papua New Guinea': 'PNG',
    'Paraguay': 'PRY',
    'Peru': 'PER',
    'Philippines': 'PHL',
    'Poland': 'POL',
    'Portugal': 'PRT',
    'Puerto Rico': 'PRI',
    'Qatar': 'QAT',
    'Reunion': 'REU',
    'Romania': 'ROU',
    'Russia': 'RUS',
    'Rwanda': 'RWA',
    'Saint Barthélemy': 'BLM',
    'Saint Kitts And Nevis': 'KNA',
    'Saint Lucia': 'LCA',
    'Saint Martin': 'MAF',
    'Saint Pierre And Miquelon': 'SPM',
    'Saint Vincent And The Grenadines': 'VCT',
    'Samoa': 'WSM',
    'San Marino': 'SMR',
    'Sao Tome And Principe': 'STP',
    'Saudi Arabia': 'SAU',
    'Senegal': 'SEN',
    'Serbia': 'SRB',
    'Seychelles': 'SYC',
    'Sierra Leone': 'SLE',
    'Singapore': 'SGP',
    'Sint Maarten': 'SXM',
    'Slovakia': 'SVK',
    'Slovenia': 'SVN',
    'Solomon Islands': 'SLB',
    'Somalia': 'SOM',
    'South Africa': 'ZAF',
    'South America': '',  # No ISO code provided
    'South Georgia And The South Sandwich Isla': '',  # No ISO code provided
    'South Korea': 'KOR',
    'Spain': 'ESP',
    'Sri Lanka': 'LKA',
    'Sudan': 'SDN',
    'Suriname': 'SUR',
    'Svalbard And Jan Mayen': 'SJM',
    'Swaziland': 'SWZ',
    'Sweden': 'SWE',
    'Switzerland': 'CHE',
    'Syria': 'SYR',
    'Taiwan': 'TWN',
    'Tajikistan': 'TJK',
    'Tanzania': 'TZA',
    'Thailand': 'THA',
    'Timor Leste': 'TLS',
    'Togo': 'TGO',
    'Tonga': 'TON',
    'Trinidad And Tobago': 'TTO',
    'Tunisia': 'TUN',
    'Turkey': 'TUR',
    'Turkmenistan': 'TKM',
    'Turks And Caicas Islands': '',  # No ISO code provided
    'Uganda': 'UGA',
    'Ukraine': 'UKR',
    'United Arab Emirates': 'ARE',
    'United Kingdom (Europe)': 'GBR',
    'United Kingdom': 'GBR',
    'United States': 'USA',
    'Uruguay': 'URY',
    'Uzbekistan': 'UZB',
    'Venezuela': 'VEN',
    'Vietnam': 'VNM',
    'Virgin Islands': '',  # No ISO code provided
    'Western Sahara': 'ESH',
    'Yemen': 'YEM',
    'Zambia': 'ZMB',
    'Zimbabwe': 'ZWE'
}

st.set_page_config(page_title="Climate Dashboard", page_icon=":earth:", layout="wide", initial_sidebar_state="expanded")

#Add custom CSS FONT
st.write("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@200&display=swap');
html, body, [class*="css"]  {
   font-family: 'Lexend', sans-serif;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_global_temperature_data():
    global_temperature_df = pd.read_csv("GlobalTemperatures.csv").dropna()
    global_temperature_df["datetime"] = pd.to_datetime(global_temperature_df["dt"])
    global_temperature_df.drop("dt", axis=1, inplace=True)
    global_temperature_df["datetime"] = pd.to_datetime(global_temperature_df["datetime"])
    global_temperature_df = global_temperature_df.set_index("datetime", drop=True)
    return global_temperature_df

@st.cache_data
def load_temperature_by_country_data():
    temperature_by_country_df = pd.read_csv("GlobalLandTemperaturesByCountry.csv").dropna()
    temperature_by_country_df["datetime"] = pd.to_datetime(temperature_by_country_df["dt"])
    temperature_by_country_df.drop("dt", axis=1, inplace=True)
    temperature_by_country_df = temperature_by_country_df.set_index("datetime", drop=False)
    # Add a new column 'ISO_code' to the dataset and map the ISO codes
    temperature_by_country_df['ISO_code'] = temperature_by_country_df['Country'].map(iso_codes)
    return temperature_by_country_df

@st.cache_data
def load_global_temperature_by_city_data():
    global_temperature_by_city_df = pd.read_csv("GlobalLandTemperaturesByMajorCity.csv").dropna()
    global_temperature_by_city_df["datetime"] = pd.to_datetime(global_temperature_by_city_df["dt"])
    global_temperature_by_city_df.drop("dt", axis=1, inplace=True)
    global_temperature_by_city_df = global_temperature_by_city_df.set_index("datetime", drop=True)
    return global_temperature_by_city_df

# Load global temperature data
global_temperature_df = load_global_temperature_data()
# Load temperature by country data
temperature_by_country_df = load_temperature_by_country_data()

# Load global temperature by city data
global_temperature_by_city_df = load_global_temperature_by_city_data()

# Load natural disaster dataset
natural_disaster_df = pd.read_csv("1900_2021_DISASTERS.xlsx - emdat data.csv")

## Global Temperature Analysis Page
def global_temperature_analysis():
    st.title('🌍 Global Temperature Analysis')
    plot_average_land_temperature()
    plot_maximum_land_temperature()
    plot_minimum_land_temperature()

  # Add explanation about polynomial regression
    st.markdown(
        """
        ## The trend line on each visual was obtained using Polynomial Regression

        Polynomial regression is a type of regression analysis where the relationship between the independent variable 
        (time in this case) and the dependent variable (temperature, average max and min) is modeled as an nth degree polynomial.

        Here's why it was chosen:

        - **Flexibility**: Polynomial regression can capture a wide range of relationships between variables. 
          It's not limited to linear relationships, so it can capture more complex trends that might be present in 
          temperature data, such as curves or nonlinear trends.

        - **Simple Implementation**: Polynomial regression is relatively simple to implement and understand compared 
          to more complex techniques like spline regression or neural networks.

        - **Interpretability**: The coefficients of the polynomial equation can provide insights into the rate of change 
          of temperature over time, making it easier to interpret the trend line.
        """
    )


# Plot average land temperature
def plot_average_land_temperature():
    st.subheader('🌡️ Average Land Temperature')
    avg_land_temp_chart = plot_temperature(global_temperature_df, 'LandAverageTemperature', 'Monthly Average Land Temperature', 'datetime', 'Average Land Temperature in Celsius', 'blue')
    st.altair_chart(avg_land_temp_chart, use_container_width=True)

# Plot maximum land temperature
def plot_maximum_land_temperature():
    st.subheader('🔥 Maximum Land Temperature')
    max_land_temp_chart = plot_temperature(global_temperature_df, 'LandMaxTemperature', 'Maximum Land Temperature', 'datetime', 'Maximum Land Temperature in Celsius', 'red')
    st.altair_chart(max_land_temp_chart, use_container_width=True)

# Plot minimum land temperature
def plot_minimum_land_temperature():
    st.subheader('❄️ Minimum Land Temperature')
    min_land_temp_chart = plot_temperature(global_temperature_df, 'LandMinTemperature', 'Minimum Land Temperature', 'datetime', 'Minimum Land Temperature in Celsius', 'green')
    st.altair_chart(min_land_temp_chart, use_container_width=True)


# Plot temperature data without uncertainty
def plot_temperature(data, column, title, x_title, y_title, color):
    chart_data = data.reset_index()  # Use the original DataFrame without any aggregation
    chart_data = chart_data.rename(columns={'datetime': 'Year'})  # Rename the datetime column to 'Year'
    chart_data['Year'] = pd.to_datetime(chart_data['Year'], format='%Y')
    chart = alt.Chart(chart_data).mark_line().encode(
        x=alt.X('Year:T', title=x_title),  # Specify datetime format for x-axis
        y=alt.Y(f'{column}:Q', title=y_title),
        color=alt.value(color)
    ).properties(
        title=title,
        width=700,
        height=400)

    # Add a trend line
    trend_line = chart.transform_regression(
        'Year', column,
        method='poly',  # Use polynomial regression
        order=1  # Linear trend line
    ).mark_line(color='black')  # Color of the trend line

    # Combine the main chart and the trend line
    chart_with_trend = (chart + trend_line).interactive()

    # Add a text mark for the trend line
    text_mark = alt.Chart({'values': [{}]}).mark_text(
        align='left', baseline='top', dx=5, dy=-5,  # Adjust text alignment and position
        text='Trend Line', fontSize=10, fontWeight='bold', color='black'  # Specify text properties
    ).encode(
        x=alt.value(10), y=alt.value(10)  # Position of the text mark
    )

    # Combine the chart with the trend line and the text mark
    combined_chart = (chart_with_trend + text_mark)

    return combined_chart

##Temp by country

def temperature_by_country_analysis(temperature_by_country_df):
    st.title('🌍 Temperature History Across Time')
    # Dropdown for selecting countries
    selected_countries = st.multiselect('Select Countries', temperature_by_country_df['Country'].unique())
    # Checkbox for selecting display mode
    display_yearly_avg = st.checkbox('Display Yearly Average Temperature', value=True, key='yearly_avg_checkbox')
    # Filter data for selected countries
    selected_countries_df = temperature_by_country_df[temperature_by_country_df['Country'].isin(selected_countries)]

    # Plot line plot
    if not selected_countries_df.empty:
        # Check if 'datetime' column exists and drop it if it does
        if 'datetime' in selected_countries_df.columns:
            selected_countries_df = selected_countries_df.drop(columns=['datetime'])

        # Reset the index to ensure 'datetime' becomes a regular column
        selected_countries_df = selected_countries_df.reset_index()

        # Pivot the DataFrame to have the years as index and each country's average temperature as a separate column
        pivoted_df = selected_countries_df.pivot_table(index='datetime', columns='Country', values='AverageTemperature')
        # Melt the DataFrame to have a single 'AverageTemperature' column and a 'Country' column
        melted_df = pd.melt(pivoted_df, ignore_index=False, var_name='Country', value_name='AverageTemperature')

        # Filter the data to display yearly averages if selected
        if display_yearly_avg:
            # Ensure 'datetime' column is properly named
            melted_df['Year'] = melted_df.index.year
            melted_df = melted_df.groupby(['Year', 'Country']).mean().reset_index()
            # Plot yearly average temperature
            chart = alt.Chart(melted_df).mark_line().encode(
                x=alt.X('Year:O', title='Year'),
                y=alt.Y('AverageTemperature:Q', title='Average Temperature (°C)'),
                color=alt.Color('Country:N', title='Country')
            ).properties(
                width=700,
                height=400
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            # Plot temperature for each month
            chart = alt.Chart(melted_df).mark_line().encode(
                x='datetime:T',
                y='AverageTemperature:Q',
                color='Country:N'
            ).properties(
                width=700,
                height=400
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
    else:
        st.write('Select one or more countries to visualize their temperature history.')



# World Map View Page

def world_map_view(temperature_by_country_df):
    st.title('🗺️ World Map View')

    # Allow user to select the year or range of years
    year_range = st.slider('📅 Select Year or Range of Years', min_value=1750, max_value=2014, value=(1750, 2014), key='world_map_year_range')

    # Filter data for the selected year or range of years
    filtered_data = temperature_by_country_df.loc[
        (temperature_by_country_df.index.year >= year_range[0]) &
        (temperature_by_country_df.index.year <= year_range[1])
        ].groupby('Country')['AverageTemperature'].mean().reset_index()


    data = go.Choropleth(
        locations=filtered_data['Country'],
        z=filtered_data['AverageTemperature'],
        locationmode='country names',
        text=filtered_data['Country'],
        colorscale='RdYlBu',
        reversescale=True,
        colorbar=dict(
            title='Average Temperature (°C)'
        )
    )

    # Define the layout
    layout = go.Layout(
        title='Average land temperature in countries',
        geo=dict(
            showframe=False,
            showocean=True,
            oceancolor='rgb(164, 230, 245)',
            projection=dict(
                type='natural earth',
                rotation=dict(
                    lon=60,
                    lat=10
                ),
            ),
            lonaxis=dict(
                showgrid=False,
                gridcolor='rgb(102, 102, 102)'
            ),
            lataxis=dict(
                showgrid=False,
                gridcolor='rgb(102, 102, 102)'
            )
        )
    )

    # Create the figure
    fig = go.Figure(data=data, layout=layout)

    # Display the figure using Plotly in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def temperature_increase_view(temperature_by_country_df):
    st.title('📈 Temperature Increase View')

    # Filter the DataFrame to include only the records for the years 1750 and 2014
    df_1850 = temperature_by_country_df.loc[temperature_by_country_df['datetime'].dt.year == 1850]
    df_2013 = temperature_by_country_df.loc[temperature_by_country_df['datetime'].dt.year == 2013]

    # Calculate the average temperature for each country in the years 1750 and 2014
    avg_temp_1850 = df_1850.groupby('Country')['AverageTemperature'].mean()
    avg_temp_2013 = df_2013.groupby('Country')['AverageTemperature'].mean()

    # Calculate the temperature difference between 1750 and 2014 for each country
    temp_difference = avg_temp_2013 - avg_temp_1850

    data = go.Choropleth(
        locations=temp_difference.index,
        z=temp_difference.values,
        locationmode='country names',
        text=temp_difference.index,
        colorscale='RdYlBu',
        reversescale=True,
        colorbar=dict(
            title='Average Temperature Increase (°C)'
        )
    )

    # Define the layout
    layout = go.Layout(
        title='Temperature Increase by Country (1850-2014)',
        geo=dict(
            showframe=False,
            showocean=True,
            oceancolor='rgb(164, 230, 245)',
            projection=dict(
                type='natural earth'
            ),
            lonaxis=dict(
                showgrid=False,
                gridcolor='rgb(102, 102, 102)'
            ),
            lataxis=dict(
                showgrid=False,
                gridcolor='rgb(102, 102, 102)'
            )
        )
    )

    # Create the figure
    fig = go.Figure(data=data, layout=layout)

    # Display the figure using Plotly in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Define function to plot top 10 countries for occurrence of natural disasters
def plot_top_countries_for_disasters():
    st.subheader("🌐 Top 10 Countries for Occurrence of Natural Disasters")
    top_countries = natural_disaster_df['Country'].value_counts().nlargest(10)
    st.bar_chart(top_countries)

# Define function to plot line graph of natural disasters by type over time
def plot_disasters_by_type_over_time():
    st.subheader("🌀 Natural Disasters by Type Over Time")
    disasters_by_type = natural_disaster_df.groupby(['Year', 'Disaster Type']).size().reset_index(name='Count')
    line_chart = alt.Chart(disasters_by_type).mark_line().encode(
        x='Year',
        y='Count',
        color='Disaster Type'
    ).properties(
        width=700,
        height=400
    ).interactive()
    st.altair_chart(line_chart, use_container_width=True)


# Define function to plot line graph with user-selected countries
def plot_user_selected_countries():
    st.subheader("🌋 Natural Disasters by country")
    selected_countries = st.multiselect("Select Countries", natural_disaster_df['Country'].unique())
    selected_data = natural_disaster_df[natural_disaster_df['Country'].isin(selected_countries)]

    if not selected_data.empty:
        # Group by Year, Country, and Disaster Type, count occurrences, and reset index
        selected_data_grouped = selected_data.groupby(['Year', 'Country', 'Disaster Type']).size().reset_index(
            name='Count')

        # Plot line chart if data is available
        line_chart = alt.Chart(selected_data_grouped).mark_line().encode(
            x=alt.X('Year:T', title='Year', axis=alt.Axis(format='%Y')),  # Specify Year as a temporal field
            y='Count:Q',  # Specify Count as a quantitative field
            color='Disaster Type:N',  # Specify Disaster Type as a nominal field for the legend
            tooltip=['Year:T', 'Count:Q', 'Country:N', 'Disaster Type:N']  # Tooltip fields
        ).properties(
            width=700,
            height=400
        ).interactive()

        st.altair_chart(line_chart, use_container_width=True)
    else:
        st.write("No data available for the selected countries.")

    # Define function for the Natural Disasters page
def natural_disaster_analysis():
    st.title("🌪️ Natural Disasters Analysis")
    plot_top_countries_for_disasters()
    plot_disasters_by_type_over_time()
    plot_user_selected_countries()


def forecast_temperature_by_city():
    st.title('🔮 Temperature Forecast for Major Cities')

    # Select major cities
    selected_cities = st.multiselect("Select Cities", global_temperature_by_city_df['City'].unique())

    for city in selected_cities:
        city_data = global_temperature_by_city_df[global_temperature_by_city_df['City'] == city]

        if not city_data.empty:
            # Filter the last 20 years of data
            last_20_years_data = city_data[city_data.index >= '1994-01-01']

            # Split data into train and validation sets
            train_data, val_data = train_test_split(last_20_years_data['AverageTemperature'], test_size=0.2, shuffle=False)

            # Fit SARIMA model on the training data
            sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            sarima_fit = sarima_model.fit(disp=False)

            # Forecast on validation data
            val_forecast = sarima_fit.get_forecast(steps=len(val_data))
            val_predicted_mean = val_forecast.predicted_mean
            val_conf_int = val_forecast.conf_int()

            # Concatenate actual and validation data for plotting
            combined_data = pd.concat([train_data, val_data])
            combined_data.index = pd.to_datetime(combined_data.index)

            # Create a DataFrame for the predicted values
            val_predicted_mean.index = val_data.index

            # Plot the actual and validation data
            st.subheader(f"Temperature Forecast for {city} (Last 20 Years)")

            # Plot actual and predicted data
            actual_data_df = pd.DataFrame({'Actual': combined_data})
            predicted_data_df = pd.DataFrame({'Predicted': val_predicted_mean})

            st.line_chart(actual_data_df.join(predicted_data_df, how='outer'))


def forecast_disasters_by_country():
    st.title('🔮 Natural Disasters Forecast by Country')
    st.markdown(
        """
The model used is SARIMAX. The SARIMAX model extends ARIMA by adding support for exogenous variables (covariates) and seasonality. It is particularly
 useful for time series data with seasonal patterns. The SARIMAX model can capture both the trend and seasonal components,
 making it suitable for predicting temperature data that exhibit these patterns.
        """
    )

    # Allow user to select countries
    selected_countries = st.multiselect('Select Countries', natural_disaster_df['Country'].unique())

    for country in selected_countries:
        country_data = natural_disaster_df[natural_disaster_df['Country'] == country]

        if not country_data.empty:
            # Ensure 'Year' is treated as a datetime type
            country_data['Year'] = pd.to_datetime(country_data['Year'], format='%Y')
            country_data = country_data.set_index('Year')

            # Resample data by year and count occurrences
            yearly_data = country_data.resample('Y').size()

            # Split data into train and validation sets
            train_data, val_data = train_test_split(yearly_data, test_size=0.2, shuffle=False)

            # Fit SARIMA model on the training data
            sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            sarima_fit = sarima_model.fit(disp=False)

            # Forecast on validation data
            val_forecast = sarima_fit.get_forecast(steps=len(val_data))
            val_predicted_mean = val_forecast.predicted_mean
            val_conf_int = val_forecast.conf_int()

            # Concatenate actual and validation data for plotting
            combined_data = pd.concat([train_data, val_data])
            combined_data.index = pd.to_datetime(combined_data.index)

            # Create a DataFrame for the predicted values
            val_predicted_mean.index = val_data.index

            # Plot the actual and validation data
            st.subheader(f"Natural Disasters Forecast for {country}")

            # Plot actual and predicted data
            actual_data_df = pd.DataFrame({'Actual': combined_data})
            predicted_data_df = pd.DataFrame({'Predicted': val_predicted_mean})

            st.line_chart(actual_data_df.join(predicted_data_df, how='outer'))


# Future Temperature Prediction Page with Model Comparison
def future_temperature_prediction_with_model_comparison():
    st.title('🔮 Future Temperature Prediction with Model Comparison')

    # Sidebar for selecting model and country
    model_type = st.sidebar.selectbox('Select Model', ('SARIMAX', 'ARIMA', 'Exponential Smoothing'))
    st.write(f'Selected Model: {model_type}')
    selected_country = st.sidebar.selectbox('Select Country', temperature_by_country_df['Country'].unique())
    st.write(f'Selected Country: {selected_country}')

    # Filter the temperature data for the selected country
    data = temperature_by_country_df[temperature_by_country_df['Country'] == selected_country]
    data = data['AverageTemperature'].dropna()

    # Split the data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

    # Fit the selected model
    if model_type == 'SARIMAX':
        model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    elif model_type == 'ARIMA':
        model = ARIMA(train_data, order=(1, 1, 1))
    elif model_type == 'Exponential Smoothing':
        model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=12)
    fitted_model = model.fit()

    # Make predictions
    predictions = fitted_model.forecast(len(test_data))

    # Calculate evaluation metrics
    mse = mean_squared_error(test_data, predictions)
    rmse = np.sqrt(mse)

    st.write(f'Mean Squared Error (MSE): {mse:.2f}')
    st.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

    # Plot the results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_data.index, y=train_data, mode='lines', name='Train Data'))
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data, mode='lines', name='Test Data'))
    fig.add_trace(go.Scatter(x=test_data.index, y=predictions, mode='lines', name='Predictions'))
    fig.update_layout(title=f'Future Temperature Prediction using {model_type}', xaxis_title='Date', yaxis_title='Temperature (°C)')
    st.plotly_chart(fig)

    # Add explanation about the selected model
    if model_type == 'SARIMAX':
        st.markdown(
            """
            ## Seasonal ARIMA with Exogenous Variables (SARIMAX)

            The SARIMAX model extends ARIMA by adding support for exogenous variables (covariates) and seasonality. It is particularly
            useful for time series data with seasonal patterns. The SARIMAX model can capture both the trend and seasonal components,
            making it suitable for predicting temperature data that exhibit these patterns.

            - **Order (p, d, q)**: Specifies the autoregressive (AR) order, differencing order, and moving average (MA) order.
            - **Seasonal Order (P, D, Q, s)**: Specifies the seasonal components for AR, differencing, and MA along with the seasonal period.

            ### Why SARIMAX?

            - **Handles Seasonality**: SARIMAX can capture and model seasonal effects in the data.
            - **Includes Exogenous Variables**: Allows incorporating other relevant variables that might influence the time series.
            - **Flexibility**: Can model a wide range of time series behaviors, making it suitable for complex datasets.
            """
        )
    elif model_type == 'ARIMA':
        st.markdown(
            """
            ## Autoregressive Integrated Moving Average (ARIMA)

            The ARIMA model is a widely used time series forecasting method that combines autoregressive (AR) and moving average (MA) components,
            along with differencing to make the time series stationary. ARIMA models are particularly effective for data with trends but without
            strong seasonal patterns.

            - **Order (p, d, q)**: Specifies the autoregressive order (p), differencing order (d), and moving average order (q).

            ### Why ARIMA?

            - **Simplicity**: ARIMA is relatively simple to implement and understand.
            - **Effective for Trend Data**: Suitable for datasets with trends and no strong seasonality.
            - **Widespread Use**: A well-established method with extensive resources and community support.
            """
        )
    elif model_type == 'Exponential Smoothing':
        st.markdown(
            """
            ## Exponential Smoothing

            Exponential Smoothing models are a class of forecasting methods that weigh past observations with exponentially decreasing weights.
            These models are particularly useful for data with trends and seasonality.

            - **Level, Trend, Seasonality**: The model can include components for the level, trend, and seasonal patterns.

            ### Why Exponential Smoothing?

            - **Handles Trends and Seasonality**: Effective for time series data with both trend and seasonal components.
            - **Smooth Predictions**: Produces smooth and stable forecasts.
            - **Easy to Interpret**: The model's components are intuitive and easy to understand.
            """
        )


# Function to preprocess and merge the datasets for the prediction page
def preprocess_data(temperature_by_country_df, natural_disaster_df):
    # Clean country names to ensure a proper merge
    temperature_by_country_df['Country'] = temperature_by_country_df['Country'].str.strip()
    natural_disaster_df['Country'] = natural_disaster_df['Country'].str.strip()

    # Get the common countries in both datasets
    common_countries = set(temperature_by_country_df['Country']).intersection(set(natural_disaster_df['Country']))

    # Debug: Check the common countries
    st.write("Common countries in both datasets:")
    st.write(common_countries)

    temperature_by_country_df = temperature_by_country_df[temperature_by_country_df['Country'].isin(common_countries)]
    natural_disaster_df = natural_disaster_df[natural_disaster_df['Country'].isin(common_countries)]

    # Aggregate temperature data by country and year
    temperature_by_country_df['Year'] = temperature_by_country_df['datetime'].dt.year
    avg_temp_by_country_year = temperature_by_country_df.groupby(['Country', 'Year'])[
        'AverageTemperature'].mean().reset_index()

    # Debug: Check the intermediate data
    st.write("Average Temperature by Country and Year:")
    st.dataframe(avg_temp_by_country_year.head())

    st.write("Natural Disaster Data:")
    st.dataframe(natural_disaster_df.head())

    # Merge temperature data with disaster data
    merged_df = pd.merge(natural_disaster_df, avg_temp_by_country_year, on=['Country', 'Year'], how='left')
    merged_df.dropna(inplace=True)  # Drop rows with missing values

    # Debug: Check the merged data
    st.write("Merged Data:")
    st.dataframe(merged_df.head())
    st.write("Merged Data Shape:")
    st.write(merged_df.shape)

    return merged_df


# Page for predicting natural disaster type and count
def predict_disasters():
    st.title("Predicting Natural Disasters 🌪️🌡️")

    # Create the merged dataset
    merged_df = preprocess_data(temperature_by_country_df, natural_disaster_df)

    if merged_df.empty:
        st.write("Merged data is empty after preprocessing.")
        return

    selected_country = st.selectbox('Select Country', merged_df['Country'].unique())
    country_data = merged_df[merged_df['Country'] == selected_country]

    if country_data.empty:
        st.write(f"No data available for {selected_country}.")
        return

    # Features and target for type prediction
    features_type = ['AverageTemperature']
    X_type = country_data[features_type]
    y_type = country_data['Disaster Type']

    # Features and target for count prediction
    features_count = ['AverageTemperature']
    X_count = country_data[features_count]
    y_count = country_data.groupby('Year')['Dis No'].count().reset_index(drop=True)

    # Train-test split
    X_type_train, X_type_test, y_type_train, y_type_test = train_test_split(X_type, y_type, test_size=0.2,
                                                                            random_state=42)
    X_count_train, X_count_test, y_count_train, y_count_test = train_test_split(X_count, y_count, test_size=0.2,
                                                                                random_state=42)

    # Classification model for disaster type prediction
    st.subheader("Predicting Disaster Type")
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_type_train, y_type_train)
    y_type_pred = clf.predict(X_type_test)
    st.write("Classification Report:")
    st.text(classification_report(y_type_test, y_type_pred))

    # Regression model for disaster count prediction
    st.subheader("Predicting Disaster Count")
    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_count_train, y_count_train)
    y_count_pred = reg.predict(X_count_test)
    mse = mean_squared_error(y_count_test, y_count_pred)
    rmse = np.sqrt(mse)
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Plot actual vs predicted counts
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_count_test.index, y=y_count_test, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=y_count_test.index, y=y_count_pred, mode='lines', name='Predicted'))
    fig.update_layout(title='Actual vs Predicted Disaster Counts', xaxis_title='Year', yaxis_title='Disaster Count')
    st.plotly_chart(fig)

# Main function to run the Streamlit app
def main():
    st.sidebar.title('Climate Dashboard 🍃')
    st.sidebar.header('Navigation')
    options = ['Global Temperature Analysis', 'Temperature by Country Analysis', 'World map of temperature',
               'World map of temperature increase', 'Natural Disasters Analysis', 'Forecast Disasters By Country',
               'Future Temperature Prediction By City','Future Temperature Prediction with Model Comparison', 'Disaster Type Prediction']
    choice = st.sidebar.radio('Select Page ⬇️', options)

    if choice == 'Global Temperature Analysis':
        global_temperature_analysis()
    elif choice == 'Temperature by Country Analysis':
        temperature_by_country_analysis(temperature_by_country_df)
    elif choice == 'World map of temperature':
        world_map_view(temperature_by_country_df)
    elif choice == 'World map of temperature increase':
        temperature_increase_view(temperature_by_country_df)
    elif choice == 'Natural Disasters Analysis':
        natural_disaster_analysis()
    elif choice == 'Forecast Disasters By Country':
        forecast_disasters_by_country()
    elif choice == 'Future Temperature Prediction By City':
        forecast_temperature_by_city()
    elif choice == 'Future Temperature Prediction with Model Comparison':
        future_temperature_prediction_with_model_comparison()
    elif choice == 'Disaster Type Prediction':
        predict_disasters()

main()
