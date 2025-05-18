from TP4 import regression_pipeline_one_company
import traceback

companies = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "005930.KS",
    "TCEHY", "BABA", "IBM", "INTC", "ORCL", "SONY", "ADBE",
    "NFLX", "AMD", "QCOM", "CSCO", "JPM", "GS", "V", "JNJ", "PFE",
    "XOM", "ASML.AS", "SAP.DE", "SIE.DE", "MC.PA", "TTE.PA",
    "SHEL.L", "BIDU", "JD", "BYDDY", "1398.HK", "TM", "9984.T", "NTDOY",
    "HYMTF", "RELIANCE.NS", "TCS.NS"
]

for company in companies:
    regression_pipeline_one_company(symbol=company)