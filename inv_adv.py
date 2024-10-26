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
from openai import OpenAI
import json
import re
from pydantic import BaseModel
from dotenv import load_dotenv


## Load environment variables
load_dotenv()

# Fetch tokens and keys from environment variables
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DATABASE_ID = os.getenv("DATABASE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Initialize Notion client
notion = Client(auth=NOTION_TOKEN)
# Set Groq client using OpenAI compatibility
client = OpenAI(api_key=OPENAI_API_KEY)


# Define structured output class for allocation data
class AllocationData(BaseModel):
    tickers: list[str]
    allocations: list[float]


# Function to check if stock data is already loaded for the current day by reading from Notion
def is_data_loaded_today():
    today = datetime.date.today().isoformat()
    try:
        results = notion.databases.query(
            database_id=DATABASE_ID,
            filter={"property": "Date Loaded", "date": {"equals": today}},
        ).get("results")
        return len(results) > 0
    except Exception as e:
        st.error(f"Error checking if data is loaded today: {e}")
        return False


def populate_stock_data():
    # Read stock tickers from CSV file
    try:
        stock_tickers_df = pd.read_csv("stock_tickers.csv")
        stock_tickers = stock_tickers_df["Ticker"].tolist()
    except Exception as e:
        st.error(f"Error reading stock tickers from CSV: {e}")
        return

    # Fetch stock data from yfinance
    for ticker in stock_tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="1d")
        closing_price = history["Close"].iloc[-1] if not history.empty else 0.0

        # Extract relevant information
        name = info.get("shortName", "N/A")
        beta = info.get("beta", 0.0)
        market_cap = info.get("marketCap", 0)
        dividend_yield = info.get("dividendYield", 0.0)
        pe_ratio = info.get("trailingPE", 0.0)
        sector = info.get("sector", "N/A")

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
            "Date Loaded": {"date": {"start": datetime.date.today().isoformat()}},
        }

        # Insert data into Notion database
        try:
            notion.pages.create(
                parent={"database_id": DATABASE_ID}, properties=properties
            )
        except Exception as e:
            st.error(f"Failed to add stock {name} to Notion. Error: {e}")


def classify_risk_profile(
    investment_performance,
    age_group,
    dependants,
    investment_percentage,
    income_sources,
    investment_loss_reaction,
    portfolio_protection,
    market_fluctuation,
    vacation_job_loss,
    unexpected_investment,
):
    # Improved logic to classify user into a risk profile
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
        with st.spinner("Classifying your risk profile..."):
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful financial advisor.",
                    },
                    {"role": "user", "content": prompt},
                ],
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
    prompt = (
        f"You are a financial advisor. Based on the user's risk profile ({risk_profile}) and the following stocks: {', '.join(filtered_stocks)}, "
        "provide a personalized investment recommendation in the following markdown format:\n\n"
        "### Investment Recommendation\n"
        "- **Risk Profile**: {risk_profile}\n"
        "- **Recommended Stocks**:\n"
        "  - Ticker: {{ticker}}, Allocation: {{allocation_percentage}}%, Reason: {{reason}}\n\n"
        "### Allocation\n"
        "- Allocate stocks based on the risk profile with percentage allocation for each stock.\n"
        "### Portfolio Metrics\n"
        "| Stock | Beta | Dividend Yield | P/E Ratio | Market Cap | % Allocation |\n"
        "|-------|------|----------------|-----------|------------|--------------|\n"
        "{{table_rows}}\n\n"
        "### Additional Considerations\n"
        "- **Economic Outlook**: Consider current market conditions and how they may impact the recommended stocks.\n"
        "- **Sector Diversification**: Ensure diversification across different sectors to reduce risk.\n"
        "- **Income Generation**: Highlight potential for dividend income, if applicable, and how it contributes to overall returns.\n"
    )
    try:
        with st.spinner("Generating investment recommendations..."):
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful financial advisor.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating investment recommendations: {e}")
        return ""


# New function to extract structured data from recommendations
def extract_allocation_data(recommendations):
    try:
        with st.spinner("Extracting allocation data..."):
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract the stock tickers and their percentage allocation.",
                    },
                    {"role": "user", "content": recommendations},
                ],
                response_format=AllocationData,
            )
        return completion.choices[0].message.parsed
    except Exception as e:
        st.error(f"Error extracting allocation data: {e}")
        return ""


def plot_stock_performance(allocation_data):
    # Plot historical performance of the portfolio compared to Nifty50
    if not allocation_data:
        st.warning("No allocation data available to plot performance.")
        return

    tickers = allocation_data.tickers
    allocations = allocation_data.allocations

    st.write("\n### Historical Performance of Portfolio vs Nifty50")
    data = pd.DataFrame()
    nifty50 = yf.Ticker("^NSEI").history(period="6mo")["Close"]

    if nifty50.empty:
        st.warning("No benchmark data available to plot.")
        return

    data["Nifty50"] = (nifty50 / nifty50.iloc[0] - 1) * 100

    portfolio_returns = pd.Series(0, index=nifty50.index)

    for ticker, allocation in zip(tickers, allocations):
        stock = yf.Ticker(ticker)
        history = stock.history(period="6mo")["Close"]
        if not history.empty:
            stock_returns = (history / history.iloc[0] - 1) * 100
            portfolio_returns += stock_returns * allocation / 100

    data["Portfolio"] = portfolio_returns

    if not data.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data["Nifty50"], label="Nifty50", color="blue")
        plt.plot(data.index, data["Portfolio"], label="Portfolio", color="green")
        plt.xlabel("Date")
        plt.ylabel("Percentage Return")
        plt.title("Historical Performance: Portfolio vs Nifty50")
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("No data available to plot.")


def main():
    # Welcome page setup
    st.title("InvestiGenie: Personalized AI Investment Guide")
    st.write(
        "Welcome to InvestiGenie! Let's create a personalized investment plan based on your financial goals and risk tolerance."
    )

    # Check if stock data has been loaded today
    if not is_data_loaded_today():
        with st.spinner("Fetching stock data for today..."):
            populate_stock_data()
        st.write(
            "Stock data has been successfully fetched and stored in the Notion database!"
        )

    # Navigation: Get Started Button
    if "start" not in st.session_state:
        st.session_state.start = False

    if not st.session_state.start:
        if st.button("Get Started"):
            with st.spinner("Setting up..."):
                st.session_state.start = True

    if st.session_state.start:
        # User input for risk profile
        age_group = st.sidebar.selectbox(
            "What is your age group?",
            ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65 and above"],
        )

        dependants = st.sidebar.selectbox(
            "How many dependants do you have (including spouse, children, dependent parents)?",
            ["None", "1", "2", "3", "4 or more"],
        )

        investment_percentage = st.sidebar.selectbox(
            "What percentage of your monthly income can you invest?",
            ["0-20%", "20-40%", "40% or more"],
        )

        income_sources = st.sidebar.selectbox(
            "How many sources of income do you have?", ["1", "2", "3", "4 or more"]
        )

        investment_performance = st.sidebar.selectbox(
            "Which of these statements best describes your attitudes about the next three years' performance of your investment?",
            [
                "Expect consistent growth",
                "Expect some fluctuations",
                "Expect significant fluctuations",
                "Unsure",
            ],
        )

        portfolio_protection = st.sidebar.selectbox(
            "Protecting my portfolio is more important to me than high returns.",
            ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
        )

        investment_loss_reaction = st.sidebar.selectbox(
            "If your investment makes a 10% loss next year, what will you do?",
            ["Sell some assets", "Hold on", "Buy more", "Seek advice"],
        )

        market_fluctuation = st.sidebar.selectbox(
            "When the market goes down, my preference would be to sell some riskier assets and put the money in safer assets.",
            ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
        )

        vacation_job_loss = st.sidebar.selectbox(
            "You have just finished saving for a 'once-in-a-lifetime' vacation. Three weeks before you plan to leave, you lose your job. You would:",
            [
                "Go anyway",
                "Postpone the trip",
                "Cancel the trip",
                "Look for a new job immediately",
            ],
        )

        unexpected_investment = st.sidebar.selectbox(
            "If you unexpectedly received Rs. 10,00,000 to invest, what would you do?",
            [
                "Invest in stocks",
                "Invest in bonds",
                "Save for future needs",
                "Consult a financial advisor",
            ],
        )

        if st.sidebar.button("Submit"):
            with st.spinner("Processing your information..."):
                # Classify risk profile
                result = classify_risk_profile(
                    investment_performance,
                    age_group,
                    dependants,
                    investment_percentage,
                    income_sources,
                    investment_loss_reaction,
                    portfolio_protection,
                    market_fluctuation,
                    vacation_job_loss,
                    unexpected_investment,
                )
                result = re.sub(r"^```json\n|```$", "", result, flags=re.MULTILINE)
                result = result.strip()
                risk_profile = json.loads(result)["profile"]
                risk_explanation = json.loads(result)["explanation"]

                # Fetch filtered stocks based on risk profile
                filtered_stocks = fetch_filtered_stocks(risk_profile)
                # Generate investment recommendations

                recommendations = generate_investment_recommendations(
                    risk_profile, filtered_stocks
                )
                st.markdown(
                    f"""
                <div style="border: 1px solid #ccc; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                    <h2>Risk Profiling</h2>
                    <p><strong>Age Group:</strong> {age_group}</p>
                    <p><strong>Risk Profile:</strong> {risk_profile}</p>
                    <p><strong>Risk Profile Explanation:</strong> {risk_explanation}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                st.write(
                    f"\n### Personalized Investment Recommendations:\n{recommendations}"
                )

                # Extract allocation data from recommendations
                allocation_data = extract_allocation_data(recommendations)
                # st.write(f"\n### Extracted Allocation Data:\n{allocation_data}")

                plot_stock_performance(allocation_data)


if __name__ == "__main__":
    main()
