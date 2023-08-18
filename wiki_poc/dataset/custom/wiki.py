# helpers to download, process and use wiki data

from SPARQLWrapper import SPARQLWrapper, JSON


# finds the given count of people from the english wikipedia
# returns a list of their names
def query_wiki_persons(count=10):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    # find pages with english title from uk or en
    sparql.setQuery("""
    SELECT DISTINCT ?page_titleEN WHERE {
    ?item wdt:P31 wd:Q5.
    ?article schema:about ?item;
        schema:isPartOf <https://en.wikipedia.org/>;
        schema:name ?page_titleEN.
    }
    LIMIT %d
    """ % (count))
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # get the page titles of the queries persons
    return [page['page_titleEN']['value'] for page in results['results']['bindings']]


# Extract the wiki article for every Wiki page title from the names list
# returns: articles list of format: [{id: dataset-id, text: wiki-text, title: wiki-page-title, url: link-to-wiki-page}, ...]
def extract_text(dataset, persons):
    # sort the dataset for faster index retrieval
    # sorting is cached automatically by dataset library after first run
    sortedDataset = dataset.sort('title')
    titles = sortedDataset['title']

    # find the indices of each person in the wiki dataset
    indices = {}
    for name in persons:
        try:
            indices[name] = titles.index(name)
        except:
            print("{} could not be found, skipping.".format(name))
            continue

    # find the corresponding articles (for every index of a known person create a list of their wiki pages)
    articles = []
    for name in indices.keys():
        articles.append(dataset[indices[name]])
    # strip all new line characters for easier processing
    for article in articles:
        article['text'] = article['text'].replace('\n', ' ')

    return articles
