import pandas as pd
from urllib.parse import urlparse

# Load source credibility from CSV
credible_sources_df = pd.read_csv("sources.csv")
credible_sources = dict(zip(credible_sources_df["domain"], credible_sources_df["credibility"]))

def check_source_credibility(url):
    try:
        domain = urlparse(url).netloc.lower()
        domain = domain.replace("www.", "")  # Normalize domain
        return credible_sources.get(domain, "Unknown")
    except:
        return "Invalid URL"