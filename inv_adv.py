# Step 8: Add Performance Metrics and Chart Visualization with Improved Layout
# Objective: Visualize the portfolio’s performance using charts and metrics in a tile-like, organized layout.

# Import necessary libraries
import streamlit as st
import yfinance as yf
from notion_client import Client
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import openai
import json
from langchain_core.output_parsers import JsonOutputParser
import re


# Hardcoded Notion token, database ID, and Groq API key
NOTION_TOKEN = "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"
DATABASE_ID = "129d3e63b59c80319b8cd5df54f36b9f"
GROQ_API_KEY = "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"

# Initialize Notion client
notion = Client(auth=NOTION_TOKEN)
# Set Groq client using OpenAI compatibility
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)





# Function to check if stock data is already loaded for the current day by reading from Notion
def is_data_loaded_today():
    today = datetime.date.today().isoformat()
    try:
        results = notion.databases.query(database_id=DATABASE_ID, filter={
            "property": "Date Loaded",
            "date": {
                "equals": today
            }
        }).get("results")
        return len(results) > 0
    except Exception as e:
        st.error(f"Error checking if data is loaded today: {e}")
        return False

def populate_stock_data():
    # List of stock tickers (example: Nifty50, mid-cap, small-cap)
    stock_tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ITC.NS',
                     'BAJFINANCE.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'MARUTI.NS']
    
    # Fetch stock data from yfinance
    for ticker in stock_tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="1d")
        closing_price = history['Close'].iloc[-1] if not history.empty else 0.0
        
        # Extract relevant information
        name = info.get('shortName', 'N/A')
        beta = info.get('beta', 0.0)
        market_cap = info.get('marketCap', 0)
        dividend_yield = info.get('dividendYield', 0.0)
        pe_ratio = info.get('trailingPE', 0.0)
        sector = info.get('sector', 'N/A')
        
        # Create properties for Notion
        properties = {
            "Name": {"rich_text": [{"text": {"content": name}}]},
            "Ticker": {"title": [{"text": {"content": ticker}}]},
            "Beta": {"number": beta},
            "Market Cap": {"select": {"name": str(market_cap)}},
            "Dividend Yield": {"number": dividend_yield},
            "P/E Ratio": {"number": pe_ratio},
            "Sector": {"rich_text": [{"text": {"content": sector}}]},
            "Price": {"number": closing_price},
            "Date Loaded": {"date": {"start": datetime.date.today().isoformat()}}
        }
        
        # Insert data into Notion database
        try:
            notion.pages.create(
                parent={"database_id": DATABASE_ID},
                properties=properties
            )
        except Exception as e:
            st.error(f"Failed to add stock {name} to Notion. Error: {e}")




def classify_risk_profile(investment_performance, age_group, dependants, investment_percentage, income_sources, investment_loss_reaction, portfolio_protection, market_fluctuation, vacation_job_loss, unexpected_investment):
    # Improved logic to classify user into a risk profile
    # if risk_tolerance == "Conservative" or investment_horizon == "Short-term (1-3 years)" or income < 50000 or experience_level == "Beginner" or market_reaction == "Anxious":
    #     return "Conservative"
    # elif risk_tolerance == "Moderate" or investment_horizon == "Medium-term (3-7 years)" or (50000 <= income < 200000) or experience_level == "Intermediate" or market_reaction == "Neutral":
    #     return "Moderate"
    # else:
    #     return "Aggressive"
    prompt = f"""You are a financial expert. Based on the answers to a questionaire, you are supposed to classify the investor into a category of 'Conservative', 'Aggressive', or 'Moderate'.
    Your response should be in proper json format with two keys. The first key should 'profile' with values as a single word: Conservative, Moderate or Aggressive.
    The second key should be 'explanation' containing the reasoning for classifying the investor into the profile.
    Here are the results of the questionaire:
    Which of these statements best describes your attitudes about the next three years' performance of your investment?
    {investment_performance}
    What is your age group
    {age_group}
    How many dependants do you have (including spouse, children, dependent parents)?
    {dependants}
    What percentage of your monthly income can you invest?
    {investment_percentage}
    How many sources of income do you have?
    {income_sources}
    If your investment makes a 10% loss next year, what will you do?
    {investment_loss_reaction}
    Protecting my portfolio is more important to me than high returns.
    {portfolio_protection}
    When the market goes down, my preference would be to sell some riskier assets and put the money in safer assets.
    {market_fluctuation}
    You have just finished saving for a 'once-in-a-lifetime' vacation. Three weeks before you plan to leave, you lose your job. You would:
    {vacation_job_loss}
    If you unexpectedly received Rs. 10,00,000 to invest, what would you do?
    {unexpected_investment}
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful financial advisor."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating investment recommendations: {e}")
        return ""


def fetch_filtered_stocks(risk_profile):
    # Retrieve stock data from Notion database
    try:
        results = notion.databases.query(database_id=DATABASE_ID).get("results")
    except Exception as e:
        st.error(f"Error fetching data from Notion: {e}")
        return []
    
    # Filter stocks based on risk profile
    filtered_stocks = []
    for page in results:
        properties = page["properties"]
        ticker = properties["Ticker"]["title"][0]["text"]["content"]
        beta = properties["Beta"]["number"]
        dividend_yield = properties["Dividend Yield"]["number"]
        
        # Filtering logic based on risk profile
        if risk_profile == "Conservative" and beta < 1 and dividend_yield >= 0.02:
            filtered_stocks.append(ticker)
        elif risk_profile == "Moderate" and 1 <= beta <= 1.5:
            filtered_stocks.append(ticker)
        elif risk_profile == "Aggressive" and beta > 1.5:
            filtered_stocks.append(ticker)
    
    return filtered_stocks

def generate_investment_recommendations(risk_profile, filtered_stocks):
    # Use Groq API to generate personalized investment recommendations
    prompt = f"You are a financial advisor. Based on the user's risk profile ({risk_profile}) and the following stocks: {', '.join(filtered_stocks)}, provide a personalized investment recommendation. Explain why these stocks are suitable for the user's risk profile."
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful financial advisor."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating investment recommendations: {e}")
        return ""

def plot_stock_performance(filtered_stocks):
    # Plot historical performance of the filtered stocks
    if not filtered_stocks:
        st.warning("No stocks available to plot performance.")
        return
    
    st.write("\n### Historical Performance of Recommended Stocks")
    data = pd.DataFrame()
    for ticker in filtered_stocks:
        stock = yf.Ticker(ticker)
        history = stock.history(period="6mo")
        if not history.empty:
            data[ticker] = history["Close"]
    
    if not data.empty:
        st.line_chart(data)
    else:
        st.warning("No data available to plot.")

def display_portfolio_metrics(filtered_stocks):
    # Display relevant metrics for the recommended portfolio
    st.write("\n### Portfolio Metrics")
    if not filtered_stocks:
        st.warning("No stocks available to display metrics.")
        return
    
    metrics_data = []
    for ticker in filtered_stocks:
        stock = yf.Ticker(ticker)
        info = stock.info
        metrics_data.append({
            "Stock": ticker,
            "Beta": info.get("beta", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "Market Cap": info.get("marketCap", "N/A")
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df)

def main():
    # Welcome page setup
    st.title("InvestiGenie: Personalized AI Investment Guide")
    st.write("Welcome to InvestiGenie! Let's create a personalized investment plan based on your financial goals and risk tolerance.")
    
    # Check if stock data has been loaded today
    if not is_data_loaded_today():
        st.write("\nFetching stock data for today...")
        populate_stock_data()
        st.write("Stock data has been successfully fetched and stored in the Notion database!")
    
    # Navigation: Get Started Button
    if "start" not in st.session_state:
        st.session_state.start = False
    
    if not st.session_state.start:
        if st.button("Get Started"):
            st.session_state.start = True
    
    if st.session_state.start:
        # User input for risk profile
        # st.sidebar.header("Tell us about yourself")
        # age_group = st.sidebar.selectbox("What is your age group?", ["20 to 35", "36 to 50", "Above 50"])
        # dependants = st.sidebar.selectbox("How many dependants do you have (including spouse, children, dependent parents)?", ["0", "Medium-term (3-7 years)", "Long-term (7+ years)"])
        # income = st.sidebar.number_input("What is your annual income (in INR)?", min_value=0, step=50000)
        # experience_level = st.sidebar.selectbox("What is your investment experience level?", ["Beginner", "Intermediate", "Expert"])
        # market_reaction = st.sidebar.selectbox("How do you react to market volatility?", ["Anxious", "Neutral", "Excited"])
     
        

        age_group = st.sidebar.selectbox(
            "What is your age group?",
            ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65 and above"]
        )

        dependants = st.sidebar.selectbox(
            "How many dependants do you have (including spouse, children, dependent parents)?",
            ["None", "1", "2", "3", "4 or more"]
        )

        investment_percentage = st.sidebar.selectbox(
            "What percentage of your monthly income can you invest?",
            ["0-20%", "20-40%","40% or more"]
        )

        income_sources = st.sidebar.selectbox(
            "How many sources of income do you have?",
            ["1", "2", "3", "4 or more"]
        )

        

        investment_performance = st.sidebar.selectbox(
            "Which of these statements best describes your attitudes about the next three years' performance of your investment?",
            ["Expect consistent growth", "Expect some fluctuations", "Expect significant fluctuations", "Unsure"]
        )

        portfolio_protection = st.sidebar.selectbox(
            "Protecting my portfolio is more important to me than high returns.",
            ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
        )

        investment_loss_reaction = st.sidebar.selectbox(
            "If your investment makes a 10% loss next year, what will you do?",
            ["Sell some assets", "Hold on", "Buy more", "Seek advice"]
        )

        market_fluctuation = st.sidebar.selectbox(
            "When the market goes down, my preference would be to sell some riskier assets and put the money in safer assets.",
            ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
        )


        vacation_job_loss = st.sidebar.selectbox(
            "You have just finished saving for a 'once-in-a-lifetime' vacation. Three weeks before you plan to leave, you lose your job. You would:",
            ["Go anyway", "Postpone the trip", "Cancel the trip", "Look for a new job immediately"]
        )

        unexpected_investment = st.sidebar.selectbox(
            "If you unexpectedly received Rs. 10,00,000 to invest, what would you do?",
            ["Invest in stocks", "Invest in bonds", "Save for future needs", "Consult a financial advisor"]
        )

    
        if st.sidebar.button("Submit"):
            # Classify risk profile
            result = classify_risk_profile(investment_performance, age_group, dependants, investment_percentage, income_sources, investment_loss_reaction, portfolio_protection, market_fluctuation, vacation_job_loss, unexpected_investment)
            result = re.sub(r'^```json\n|```$', '', result, flags=re.MULTILINE)
            result = result.strip()
            risk_profile = json.loads(result)['profile']
            
            
            # Fetch filtered stocks based on risk profile
            filtered_stocks = fetch_filtered_stocks(risk_profile)
            # Generate investment recommendations
            recommendations = generate_investment_recommendations(risk_profile, filtered_stocks)
            st.write(f"\n### Personalized Investment Recommendations:\n{recommendations}")
            
            # Improved layout using columns
            col1, col2 = st.columns(2)
            with col1:
                plot_stock_performance(filtered_stocks)
            with col2:
                display_portfolio_metrics(filtered_stocks)

if __name__ == "__main__":
    main()
