


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
plt.switch_backend('agg')



def wiki_page_views_hist():
    """plot the distribution of the page views"""
    df = pd.read_csv('dataset/wiki_page_views.csv')
    plt.figure(figsize=(10,6))
    plot = sns.histplot(df['page_views'], bins=50, log_scale=(False, True), color='skyblue', edgecolor='black')
    plt.xlabel('Number of Page Views')
    plt.ylabel('Number of Pages')
    plt.title('Distribution of Number of Pages Compared to View Count')

    plot.xaxis.set_major_formatter(ticker.EngFormatter())
    # save to file
    plt.savefig('evaluation/plotting/plots/insights/wiki_page_views_hist.png', dpi=300, bbox_inches='tight')

def wiki_page_edits_hist():
    """plot the distribution of the page edits"""
    df = pd.read_csv('dataset/wiki_page_edits.csv')
    plt.figure(figsize=(10,6))
    plot = sns.histplot(df['page_edits'], bins=50, log_scale=(False, True), color='skyblue', edgecolor='black')
    plt.xlabel('Number of Page Edits')
    plt.ylabel('Number of Pages')
    plt.title('Distribution of Number of Pages Compared to Edit Count')

    plot.xaxis.set_major_formatter(ticker.EngFormatter())
    # save to file
    plt.savefig('evaluation/plotting/plots/insights/wiki_page_edits_hist.png', dpi=300, bbox_inches='tight')


wiki_page_views_hist()
wiki_page_edits_hist()