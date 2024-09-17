import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression

st.title("Brazil Debt to GDP Ratio Forecast")
st.write("This dashboard forecasts Brazil's debt-to-GDP ratio based on various economic parameters.")

# --- Data Loading (Replace with your actual path) ---
data = pd.read_csv('C:/Users/UT3P5T/Downloads/data_percent - Copy (2).csv')

# --- Sidebar for User Input ---
st.sidebar.header("Economic Parameter Assumptions")
forecast_start_year = 2024
# Example: Add slider for BCB_GDP_growth
bcb_gdp_growth_input = st.sidebar.number_input(f"BCB GDP Growth (%) (from {forecast_start_year} onwards)", min_value=-5.0, max_value=5.0, value=0.0, step=0.01)
bcb_igpm_input = st.sidebar.number_input(f"BCB IGP-M (%) (from {forecast_start_year} onwards)", min_value=-5.0, max_value=20.0, value=0.0, step=0.01)  # Adjust range as needed
bcb_ipca_input = st.sidebar.number_input(f"BCB IPCA (%) (from {forecast_start_year} onwards)", min_value=-5.0, max_value=20.0, value=0.0, step=0.01)  # Adjust range as needed
bcb_selic_input = st.sidebar.number_input(f"BCB Selic (%) (from {forecast_start_year} onwards)", min_value=0.0, max_value=20.0, value=0.0, step=0.01)  # Adjust range as needed

data = pd.read_csv('C:/Users/UT3P5T/Downloads/data_percent - Copy (2).csv')
print(data.head())
print(data.columns)

for year in range(forecast_start_year, 2030):  # Assuming your forecast goes up to 2029
    data.loc[data['Year'] == year, 'BCB_GDP_growth'] = bcb_gdp_growth_input
    data.loc[data['Year'] == year, 'BCB_IGP-M'] = bcb_igpm_input
    data.loc[data['Year'] == year, 'BCB_IPCA'] = bcb_ipca_input
    data.loc[data['Year'] == year, 'BCB_Selic'] = bcb_selic_input

# set values for the variables 
# values chosen by the user 




# Evolution equations:
def debt_to_GDP_ratio(debt, nominal_GDP):
    return debt/nominal_GDP

    
# Real GDP(t):
def next_Real_GDP(t, Real_GDP):
    growth_t = data['BCB_GDP_growth'][t]
    return Real_GDP*(1 + growth_t)

# Nominal GDP(t) formula:
def compute_nominal_GDP(t, Real_GDP):
    # depends on real GDP(t), deflator(t)
    deflator = data['GDP_deflator'][t]
    return Real_GDP*deflator/818.456


# Regression for domestic interest rate, we take the coefficients from data:
# make a real regression that computes coeff and then estimate variable 
# look at excel file and formulas for each 
from sklearn.linear_model import LinearRegression

# Assuming 'data' is your DataFrame with columns: '%debtIGP-M', 'BCB_IGP-M', 
# '%debtIPCA', 'BCB_IPCA', '%debtSelic', 'BCB_Selic', '%debtFixed-rate', 
# 'BCB_exchge_av', and your target variable (which you need to specify).


# 4. Define the function using the fitted coefficients
def it_d2(t):
    regression_equ = intercept 
    regression_equ += coefficients[0] * data['%debtIGP-M'][16] * data['BCB_IGP-M'][t] * 100
    regression_equ += coefficients[1] * data['%debtIPCA'][16] * data['BCB_IPCA'][t] * 100
    regression_equ += coefficients[2] * data['%debtSelic'][16] * data['BCB_Selic'][t] * 100
    regression_equ += coefficients[3] * data['%debtFixed-rate'][16] * data['BCB_exchge_av'][t] * 100
    return regression_equ

# Now you can use the 'it_d(t)' function with the dynamically calculated coefficients.
def it_d(t):
    # refaire reg
    Intercept = -0.010072
    regression_equ = Intercept -0.017233*data['%debtIGP-M'][16]*data['BCB_IGP-M'][t]*100
    regression_equ += 0.008549*data['%debtIPCA'][16]*data['BCB_IPCA'][t]*100
    regression_equ += 0.008041*data['%debtSelic'][16]*data['BCB_Selic'][t]*100
    regression_equ += 0.025354*data['%debtFixed-rate'][16]*data['BCB_exchge_av'][t]*100
    return regression_equ
    
    
# Total new issue debt NI_t

def NI(t, debt_f, debt_d, debt, nominal_GDP):
    #depends on if(t), id(t), debt_f(t-1), debt_d(t-1), debt(t-1), eavg(t),eop(t-1), nominalGDP(t), PB(t)
    
    eop = data['BCB_exchge_max'][t-1]
    eavg = data['BCB_exchge_av'][t]
    Nominal_interest = it_f*debt_f + it_d(t)*debt_d
    Amortization = c*debt - a*debt_f + a*debt_f*(eavg/eop)
    PB = data['BCB_PB_%GDP'][t]*nominal_GDP
    return Nominal_interest + Amortization - PB
# End of period stock of foreign currency debt:

def next_debt_f(t, debt_f, NI):
    #depends on debt_f(t-1), DMat12_f(t-1), NI(t)
    eop = data['BCB_exchge_max'][t]
    #if t == 17:

    eop_previous = data['BCB_exchge_max'][t-1]
    eavg = data['BCB_exchge_av'][t]
    return (1/eop_previous)*(debt_f*eop - a*debt_f*eavg)+ b*NI


# End of period stock of public debt denominated in local currency:
def next_debt_d(debt_d, NI, debt_f, debt):
    # depends on debt_d(t-1), debt_f(t-1), DMat12_d(t-1), debt(t-1), NI(t)
    Dmat12_d = c*debt - a*debt_f
    NI_d = (1-b)*NI
    return debt_d - Dmat12_d + NI_d


# Equation to compute current total debt in terms of debt_f and debt_d:
def debt_equation(debt_f, debt_d):
    # depends on debt_f(t), debt_d(t)
    # eop = data['Exchange rate - end of period'][t]
    #eavg = data['Exchange rate - average'][t]
    return debt_f + debt_d



    # We initialize the parameters of the evolution equations
a = 0.071800
b = 0.25
c = 0.061900
#We initialize the values before the recursive equations:
# Annualized foreign interest rate:
it_f = data['Nominal_Interest_FX'][16]/10#/(data['Balance'][1]*data['%debtFX'][1])*12
debt_d = data['Debt_Local_Currency'][16]
debt_f = data['Debt_Foreign_Currency'][16]
debt = data['Debt_Gross_Total'][16]
# we have verified that h = debt_f/(debt_d+debt_f)
nominal_GDP = data['GDP_nominal'][6]
Real_GDP = data['GDP_real'][16]
# Initialize with first two values:
ratio_over_time = []
foreign_ratios = []
local_ratios = []



for t in range(0,17): # initial values are for t=0, t=1:
    ratio_over_time.append(data['Debt_Gross_Total'][t]/data['GDP_nominal'][t])
    foreign_ratios.append(data['Debt_Foreign_Currency'][t]/data['GDP_nominal'][t])
    local_ratios.append(data['Debt_Local_Currency'][t]/data['GDP_nominal'][t])
    years = data['Year'][0:10]



# Loop to compute the values:
for t in range(17,23):
    Real_GDP = next_Real_GDP(t, Real_GDP)
    #print(Real_GDP) OK 
    nominal_GDP = compute_nominal_GDP(t,Real_GDP)

    # domestic interest rate:
    i_d = it_d(t)
    #print(i_d) OK 
    nominal_interest = NI(t, debt_f, debt_d, debt, nominal_GDP)
    # debt_f doesn't depend on debt_d, thus we compute it first.
    # foreign currecncy debt:
    new_debt_f = next_debt_f(t, debt_f, nominal_interest)
    # local currency debt (careful, we use foreign debt at t-1 for the␣forecast):
    debt_d = next_debt_d(debt_d, nominal_interest, debt_f, debt)
    debt_f = new_debt_f # now we update debt_f
    #New debt:
    debt = debt_equation(debt_f, debt_d)
    # print(debt) OK 
    # Now compute the ratio debt to GDP:
    ratio = debt_to_GDP_ratio(debt, nominal_GDP)
    foreign_ratio = debt_f/nominal_GDP
    local_ratio = debt_d/nominal_GDP
    ratio_over_time.append(ratio)
    foreign_ratios.append(foreign_ratio)
    #print(foreign_ratios)
    local_ratios.append(local_ratio)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


st.subheader("Debt to GDP Forecast")


years = list(range(2007, 2030)) # Years from 2022 to 2031
df = pd.DataFrame({'Years': years, 'Debt Over Time': ratio_over_time, 'Foreign Ratio': foreign_ratios, 'Local Ratio': local_ratios})
df['Years'] = pd.to_datetime(df['Years'], format='%Y')
# Define a function to format the y-axis tick labels as percentages
def percentage_formatter(x, pos):
    return '{:.0f}%'.format(x * 100)
sns.set()
plt.figure(figsize=(15, 8), facecolor='white')
x = df['Years'].dt.year.to_numpy() 
y = df['Debt Over Time'].to_numpy()
sns.set_style("darkgrid") # Set the overall style
fig, ax = plt.subplots(figsize=(15, 8), facecolor='white')

local_ratio = df['Local Ratio']
foreign_ratio = df['Foreign Ratio']
# Plotting the stacked bars
bars1 = ax.bar(x, local_ratio, width=0.4, color='darkcyan', label='Local currency-denominated debt % of GDP')
bars2 = ax.bar(x, foreign_ratio, bottom=local_ratio, width=0.4, color='grey', label='Foreign currency-denominated debt % of GDP')

for bar, year in zip(bars1[16:], years[16:]): # Starting from index 2 (year 2024)
    bar.set_hatch('xxxxx')
for bar, year in zip(bars2[16:], years[16:]): # Starting from index 2 (year 2024)
    bar.set_hatch('xxxxx')
for bar in bars1[15:]:  # Assuming index 15 corresponds to 2023
    bar.set_hatch('xxxxx')
for bar in bars2[15:]:
    bar.set_hatch('xxxxx')
ax.plot(x, y, linestyle='--', color='black', label='Debt to GDP')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.0f}%'.format(x * 100)))
ax.set_xticks(years)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(x, y, linestyle='--', color='black', label='Debt to GDP')  # Use the NumPy arrays
ax.set_xlabel('\nYear', fontsize=20, color='blue')
ax.set_ylabel('Debt to GDP ratio\n', fontsize=20, color='#2980b9')
ax.set_title("Debt to GDP ratio\n", fontsize=18, color='#3742fa')
plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
plt.xticks(years) # Set the x-axis ticks explicitly
plt.legend()
plt.tight_layout()
plt.show()


st.pyplot(fig)
if st.checkbox("Show Data Table"):
    st.write(df)
