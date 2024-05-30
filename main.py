import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import altair as alt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split

# I keep the ISO dict in case needed for maps
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

# Load 
global_temperature_df = load_global_temperature_data()
temperature_by_country_df = load_temperature_by_country_data()
global_temperature_by_city_df = load_global_temperature_by_city_data()
natural_disaster_df = pd.read_csv("1900_2021_DISASTERS.xlsx - emdat data.csv")

## Global Temperature Analysis Page
def global_temperature_analysis():
    st.title('Global Temperature Analysis')
    plot_average_land_temperature()
    plot_maximum_land_temperature()
    plot_minimum_land_temperature()

  # Add explanation 
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


def plot_average_land_temperature():
    st.subheader('Average Land Temperature')
    avg_land_temp_chart = plot_temperature(global_temperature_df, 'LandAverageTemperature', 'Monthly Average Land Temperature', 'datetime', 'Average Land Temperature in Celsius', 'blue')
    st.altair_chart(avg_land_temp_chart, use_container_width=True)

def plot_maximum_land_temperature():
    st.subheader('Maximum Land Temperature')
    max_land_temp_chart = plot_temperature(global_temperature_df, 'LandMaxTemperature', 'Maximum Land Temperature', 'datetime', 'Maximum Land Temperature in Celsius', 'red')
    st.altair_chart(max_land_temp_chart, use_container_width=True)

def plot_minimum_land_temperature():
    st.subheader('Minimum Land Temperature')
    min_land_temp_chart = plot_temperature(global_temperature_df, 'LandMinTemperature', 'Minimum Land Temperature', 'datetime', 'Minimum Land Temperature in Celsius', 'green')
    st.altair_chart(min_land_temp_chart, use_container_width=True)


# Plot temperature data without uncertainty (we drop uncertainty bc it's not essential in recent years)
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

    # Add a text mark for the trend line (not really working)
    text_mark = alt.Chart({'values': [{}]}).mark_text(
        align='left', baseline='top', dx=5, dy=-5,  # Adjust text alignment and position
        text='Trend Line', fontSize=10, fontWeight='bold', color='black'  # Specify text properties
    ).encode(
        x=alt.value(10), y=alt.value(10)  # Position of the text mark, but this doesn't wwork that well
    )

    # Combine the chart with the trend line and the text mark
    combined_chart = (chart_with_trend + text_mark)

    return combined_chart

##Temp by country

def temperature_by_country_analysis(temperature_by_country_df):
    st.title('Temperature History Across Time')
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
    st.title('World Map View')

    # select the year or range of years
    year_range = st.slider('Select Year or Range of Years', min_value=1750, max_value=2014, value=(1750, 2014), key='world_map_year_range')

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

   
    fig = go.Figure(data=data, layout=layout)

    # Display
    st.plotly_chart(fig, use_container_width=True)

def temperature_increase_view(temperature_by_country_df):
    st.title('Temperature Increase View')

    # Filter to include only the records for the years 1750 and 2014
    df_1850 = temperature_by_country_df.loc[temperature_by_country_df['datetime'].dt.year == 1850]
    df_2013 = temperature_by_country_df.loc[temperature_by_country_df['datetime'].dt.year == 2013]

    # cc average temperature for each country in the years 1750 and 2014
    avg_temp_1850 = df_1850.groupby('Country')['AverageTemperature'].mean()
    avg_temp_2013 = df_2013.groupby('Country')['AverageTemperature'].mean()

    # cc difference between 1750 and 2014 for each country
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

    
    fig = go.Figure(data=data, layout=layout)

    # Display 
    st.plotly_chart(fig, use_container_width=True)

# Define function 
def plot_top_countries_for_disasters():
    st.subheader("Top 10 Countries for Occurrence of Natural Disasters")
    top_countries = natural_disaster_df['Country'].value_counts().nlargest(10)
    st.bar_chart(top_countries)

# Define function 
def plot_disasters_by_type_over_time():
    st.subheader("Natural Disasters by Type Over Time")
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


# Define function t
def plot_user_selected_countries():
    st.subheader("Natural Disasters by country")
    selected_countries = st.multiselect("Select Countries", natural_disaster_df['Country'].unique())
    selected_data = natural_disaster_df[natural_disaster_df['Country'].isin(selected_countries)]

    if not selected_data.empty:
        # Group by Year, Country, and Disaster Type, count occurrences, and reset index
        selected_data_grouped = selected_data.groupby(['Year', 'Country', 'Disaster Type']).size().reset_index(
            name='Count')

        # Plot line chart if data is available
        line_chart = alt.Chart(selected_data_grouped).mark_line().encode(
            x=alt.X('Year:T', title='Year', axis=alt.Axis(format='%Y')),  # Specify Year as a temporal field
            y='Count:Q',  # Specify count as a quantitative field
            color='Disaster Type:N',  # Specify Disaster Type as a nominal field for the legend
            tooltip=['Year:T', 'Count:Q', 'Country:N', 'Disaster Type:N']  # Tooltip fields
        ).properties(
            width=700,
            height=400
        ).interactive()

        st.altair_chart(line_chart, use_container_width=True)
    else:
        st.write("No data available for the selected countries.")

    # Define 
def natural_disaster_analysis():
    st.title("Natural Disasters Analysis")
    plot_top_countries_for_disasters()
    plot_disasters_by_type_over_time()
    plot_user_selected_countries()



def forecast_temperature_by_city():
    st.title('Temperature Forecast for Major Cities')

    # Select
    selected_cities = st.multiselect("Select Cities", global_temperature_by_city_df['City'].unique())

    for city in selected_cities:
        city_data = global_temperature_by_city_df[global_temperature_by_city_df['City'] == city]

        if not city_data.empty:
            # Filter the last 20 years of data
            last_20_years_data = city_data[city_data.index >= '1994-01-01']

            # Split data into train and validation sets
            train_data, val_data = train_test_split(last_20_years_data['AverageTemperature'], test_size=0.2, shuffle=False)

            # Fit SARIMA model on training data
            sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            sarima_fit = sarima_model.fit(disp=False)

            # Forecast on validation data
            val_forecast = sarima_fit.get_forecast(steps=len(val_data))
            val_predicted_mean = val_forecast.predicted_mean
            val_conf_int = val_forecast.conf_int()

            # actual and validation data for plotting
            combined_data = pd.concat([train_data, val_data])
            combined_data.index = pd.to_datetime(combined_data.index)

            # DataFrame for the predicted values
            val_predicted_mean.index = val_data.index

            # Plot the actual and validation data
            st.subheader(f"Temperature Forecast for {city} (Last 20 Years)")

            # Plot actual and predicted data
            actual_data_df = pd.DataFrame({'Actual': combined_data})
            predicted_data_df = pd.DataFrame({'Predicted': val_predicted_mean})

            st.line_chart(actual_data_df.join(predicted_data_df, how='outer'))

            

def forecast_disasters_by_country():
    st.title('Natural Disasters Forecast by Country')

    # select countries
    selected_countries = st.multiselect('Select Countries', natural_disaster_df['Country'].unique())

    for country in selected_countries:
        country_data = natural_disaster_df[natural_disaster_df['Country'] == country]

        if not country_data.empty:
            # datetime type
            country_data['Year'] = pd.to_datetime(country_data['Year'], format='%Y')
            country_data = country_data.set_index('Year')

            # Resample data by year and count occurrences
            yearly_data = country_data.resample('Y').size()

            # Split data, train and validation sets
            train_data, val_data = train_test_split(yearly_data, test_size=0.2, shuffle=False)

            # Fit SARIMA model on the training data
            sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            sarima_fit = sarima_model.fit(disp=False)

            # Forecast on validation data
            val_forecast = sarima_fit.get_forecast(steps=len(val_data))
            val_predicted_mean = val_forecast.predicted_mean
            val_conf_int = val_forecast.conf_int()

            #  actual and validation data for plotting
            combined_data = pd.concat([train_data, val_data])
            combined_data.index = pd.to_datetime(combined_data.index)

            # DataFrame for the predicted values
            val_predicted_mean.index = val_data.index

            st.subheader(f"Natural Disasters Forecast for {country}")

            # Plot actual and predicted data
            actual_data_df = pd.DataFrame({'Actual': combined_data})
            predicted_data_df = pd.DataFrame({'Predicted': val_predicted_mean})

            st.line_chart(actual_data_df.join(predicted_data_df, how='outer'))

# sidebar
pages = {
    "Global Temperature Analysis": global_temperature_analysis,
    "Temperature by country analysis": temperature_by_country_analysis,
    "World map": world_map_view,
    "Increase in temperature": temperature_increase_view,
    "Natural Disasters": natural_disaster_analysis,
    "Forecast Disasters by Country": forecast_disasters_by_country,
    "Temperature Forecast": forecast_temperature_by_city
}

# Sidebar navigation
st.sidebar.title('Climate Dashboard')
selection = st.sidebar.radio("Go to", list(pages.keys()))


# Display 
if selection == "Temperature by country analysis":
    pages[selection](temperature_by_country_df)  # Pass temperature_by_country_df when calling the selected function
elif selection == "World map":
    pages[selection](temperature_by_country_df)
elif selection == "Increase in temperature":
    pages[selection](temperature_by_country_df)
else:
    pages[selection]()
