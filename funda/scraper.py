#!/opt/homebrew/bin/python3

from funda_scraper import FundaScraper


scraper = FundaScraper(area="amsterdam", want_to="rent", find_past=False, page_start=1, n_pages=3, min_price=500, max_price=2000)
df = scraper.run(raw_data=False, save=True, filepath="test.csv")
df.head()



