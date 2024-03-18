from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json
import re

class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        # TODO
        self.crawling_threshold = crawling_threshold
        self.not_crawled = deque()
        self.crawled = []
        self.added_ids = set()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        # TODO
        URL_arr = URL.split('/')
        id = None
        for idx, part in enumerate(URL_arr):
                if part == 'title':
                        id = URL_arr[idx + 1]
                        break
        return id
        # return URL.split('/')[4]

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        # TODO
        with open('IMDB_crawled.json', 'w') as f:
            json.dump(self.crawled, f)
        pass

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        # TODO
        with open('IMDB_crawled.json', 'r') as f:
            self.crawled = json.load(f)

        with open('IMDB_not_crawled.json', 'w') as f: #?
            self.not_crawled = deque(json.load(f))

        self.added_ids = set([movie['id'] for movie in self.crawled])

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        # TODO
        return get(URL, headers=self.headers)

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        # TODO update self.not_crawled and self.added_ids
        response = self.crawl(self.top_250_URL)
        soup = BeautifulSoup(response.text, 'html.parser')
        movie_links = soup.select('#__next > main > div > div.ipc-page-content-container.ipc-page-content-container--center > section > div > div.ipc-page-grid.ipc-page-grid--bias-left > div > ul > li:nth-child(n) > div.ipc-metadata-list-summary-item__c > div > div > div.ipc-title.ipc-title--base.ipc-title--title.ipc-title-link-no-icon.ipc-title--on-textPrimary.sc-b0691f29-9.klOwFB.cli-title > a')
        # print(movie_links[-1].text)
        for link in movie_links:
           movie_id = self.get_id_from_URL(link['href'])
           if movie_id:
                  splited = link['href'].split('/')
                  self.not_crawled.append('https://www.imdb.com/' + splited[1] + '/' + splited[2] + '/')
                  self.added_ids.add(movie_id)

        

    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        TODO: 
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """

        # help variables
        WHILE_LOOP_CONSTRAINTS = None
        NEW_URL = None
        THERE_IS_NOTHING_TO_CRAWL = None

        self.extract_top_250()
        futures = []
        crawled_counter = 0
        lock = Lock()
        with ThreadPoolExecutor(max_workers=20) as executor:
            while self.not_crawled and crawled_counter < self.crawling_threshold:
                URL = self.not_crawled.popleft()
                futures.append(executor.submit(self.crawl_page_info, URL))
                with lock:
                    crawled_counter += 1
                if not self.not_crawled:
                    wait(futures)
                    futures = []

    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        """
        print("new iteration")
        # TODO
        response = self.crawl(URL)
        movie = self.get_imdb_instance()
        self.extract_movie_info(response, movie, URL)
        movie['id'] = self.get_id_from_URL(URL) ### ? ? ? 

        with self.add_list_lock:
            self.crawled.append(movie)
            print("Crawled movie:", movie['title'])
        for related_link in movie['related_links']:
            movie_id = self.get_id_from_URL(related_link)
            if movie_id in self.added_ids:
                continue
            with self.add_list_lock:
                self.not_crawled.append(related_link)
            with self.add_queue_lock:
                self.added_ids.add(movie_id)
        return
        pass

    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        # TODO
        soup = BeautifulSoup(res.text, 'html.parser')
        soup_summary = BeautifulSoup(self.crawl(self.get_summary_link(URL)).text, 'html.parser')
        soup_reviews = BeautifulSoup(self.crawl(self.get_review_link(URL)).text, 'html.parser')
        movie['title'] = IMDbCrawler.get_title(soup)
        movie['first_page_summary'] = IMDbCrawler.get_first_page_summary(soup)
        movie['release_year'] = IMDbCrawler.get_release_year(soup)
        movie['mpaa'] = IMDbCrawler.get_mpaa(soup)
        movie['budget'] = IMDbCrawler.get_budget(soup)
        movie['gross_worldwide'] = IMDbCrawler.get_gross_worldwide(soup)
        movie['directors'] = IMDbCrawler.get_director(soup)
        movie['writers'] = IMDbCrawler.get_writers(soup)
        movie['stars'] = IMDbCrawler.get_stars(soup)
        movie['related_links'] = IMDbCrawler.get_related_links(soup)
        movie['genres'] = IMDbCrawler.get_genres(soup)
        movie['languages'] = IMDbCrawler.get_languages(soup)
        movie['countries_of_origin'] = IMDbCrawler.get_countries_of_origin(soup)
        movie['rating'] = IMDbCrawler.get_rating(soup)
        movie['summaries'] = IMDbCrawler.get_summary(soup_summary)
        movie['synopsis'] = IMDbCrawler.get_synopsis(soup_summary)
        movie['reviews'] = IMDbCrawler.get_reviews_with_scores(soup_reviews)
        

    def get_summary_link(self, url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            # TODO
            summary_url = url + 'plotsummary'
            get(summary_url)
            return summary_url
            pass
        except:
            print("failed to get summary link")
            return ''

    def get_review_link(self, url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            # TODO
            review_url = url + "reviews"
            get(review_url)
            return review_url
            pass
        except:
            print("failed to get review link")
            return ''

    def get_title(soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            # TODO
            title = soup.select('#__next > main > div > section.ipc-page-background.ipc-page-background--base.sc-304f99f6-0.fSJiHR > section > div:nth-child(5) > section > section > div.sc-491663c0-3.bdjVSf > div.sc-67fa2588-0.cFndlt > h1 > span')
            return title[0].text
            pass
        except:
            print("failed to get title")
            return 'not-present'

    def get_first_page_summary(soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            # TODO
            summary = soup.find('span', role='presentation', class_='sc-466bb6c-0 hlbAws')
            return summary.text
            pass
        except:
            print("failed to get first page summary")
            return 'not-present'

    def get_director(soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            # TODO
            all_lists = soup.find_all('li')
            directors_set = set()
            for li in all_lists:
                text = li.text.strip()
                if text.startswith('Director') or text.startswith('Directors'):
                    director_names = [a.text.strip() for a in li.find_all('a')]
                    directors_set.update(director_names)
            return list(directors_set)
            pass
        except:
            print("failed to get director")
            return ['not-present']

    def get_stars(soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            # TODO
            all_lists = soup.find_all('li')
            stars_set = set()
            for li in all_lists:
                text = li.text.strip()
                if text.startswith('Stars'):
                    stars_names = [a.text.strip() for a in li.find_all('a')]
                    stars_set.update(stars_names)
            stars_set.discard('')
            stars_set.discard('Stars')
            return (list(stars_set))
            pass
        except:
            print("failed to get stars")
            return ['not-present']

    def get_writers(soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            # TODO
            all_lists = soup.find_all('li')
            writers_set = set()
            for li in all_lists:
                text = li.text.strip()
                if text.startswith('Writers') or text.startswith('Writer'):
                    writers_names = [a.text.strip() for a in li.find_all('a')]
                    writers_set.update(writers_names)
            writers_set.discard('')
            writers_set.discard('Writers')
            return (list(writers_set))
            pass
        except:
            print("failed to get writers")
            return ['not-present']

    def get_related_links(soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        def get_id_from_URL(url):
            URL_arr = url.split('/')
            id = None
            for idx, part in enumerate(URL_arr):
                if part == 'title':
                        id = URL_arr[idx + 1]
                        break
            return id
        try:
            # TODO
            all_links = soup.find(attrs={'data-testid' : 'MoreLikeThis'})
            related_links = []
            movie_ids = []
            if all_links:
                for link in all_links.find_all('a'):
                    href = link.get('href')
                    if get_id_from_URL(href) is not None and get_id_from_URL(href) not in movie_ids:
                        movie_ids.append(get_id_from_URL(href))  
                        splited = href.split('/')
                        final_link = 'https://www.imdb.com' + '/' +splited[1] + '/' + splited[2] + '/'
                        related_links.append(final_link)
            return related_links
            pass
        except:
            print("failed to get related links")
            return ['not-present']

    def get_summary(soup):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            # TODO
            summaries = []
            li_elements = soup.select('div[data-testid="sub-section-summaries"] li')
            for li in li_elements:
                summaries.append(li.text)
            return summaries 
            pass
        except:
            print("failed to get summary")
            return ['not-present']

    def get_synopsis(soup):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:

            # TODO
            li_elements = soup.select('div[data-testid="sub-section-synopsis"] li')
            # print(li_elements)
            synopsis = li_elements[0].get_text(separator='<br/><br/>').strip()
            return synopsis.split("<br/><br/>")
            pass
        except:
            print("failed to get synopsis")
            return ['not-present']

    def get_reviews_with_scores(soup):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            # TODO
            review_blocks = soup.find_all('div', class_="review-container")
            reviews_with_scores = []
            for review_block in review_blocks:
                try:
                    score = review_block.find('span', class_='rating-other-user-rating').get_text(strip=True)
                    review = review_block.find('div', class_='text show-more__control').get_text(strip=True)
                except:
                    continue
                reviews_with_scores.append([score, review])
            return reviews_with_scores
            pass
        except:
            print("failed to get reviews")
            return [['not-present']]

    def get_genres(soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            # TODO
            genres = []
            genre_labels = soup.find('div', {'data-testid': 'genres'})
            for a in genre_labels.find_all('a'):
                 genres.append(a.text.strip())
            return genres
            pass
        except:
            print("Failed to get generes")
            return ['not-present']

    def get_rating(soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            # TODO
            rating = None
            rating_label = soup.find('div', {'data-testid': 'hero-rating-bar__aggregate-rating__score'})
            return rating_label.text.strip()
            pass
        except:
            print("failed to get rating")
            return 'not-present'

    def get_mpaa(soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            # TODO
            mpaa = soup.select('#__next > main > div > section.ipc-page-background.ipc-page-background--base.sc-304f99f6-0.fSJiHR > section > div:nth-child(5) > section > section > div.sc-491663c0-3.bdjVSf > div.sc-67fa2588-0.cFndlt > ul > li:nth-child(2) > a')
            return mpaa[0].text
            pass
        except:
            print("failed to get mpaa")
            return 'not-present'

    def get_release_year(soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            # TODO
            releaseinfo_links = soup.find_all('a', href=lambda href: href and 'releaseinfo' in href)
            releaseinfo_texts = [link.get_text(strip=True) for link in releaseinfo_links]
            return releaseinfo_texts[0]
            pass
        except:
            print("failed to get release year")
            return 'not-present'

    def get_languages(soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            # TODO
            language_li = soup.find('li', {'data-testid': 'title-details-languages'})
            languages = language_li.text.strip()
            languages = re.findall('[A-Z][^A-Z]*', languages)
            return languages[1:]
            pass
        except:
            print("failed to get languages")
            return ['not-present']

    def get_countries_of_origin(soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            # TODO
            origin_tags = soup.find_all('a', href=lambda href: href and 'country_of_origin' in href)
            origins = [tag.get_text(strip=True) for tag in origin_tags]
            return origins
            pass
        except:
            print("failed to get countries of origin")
            return ['not-present']

    def get_budget(soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            # TODO
            budget_li = soup.find('li', {'data-testid': 'title-boxoffice-budget'})
            if budget_li:
                spans = budget_li.find_all('span')
                if len(spans) > 1:
                     parts = spans[1].get_text(strip=True).split('(')
                     if parts[0]:
                        return parts[0].strip()
                     else:
                         return 'not-resent'
            return 'not-present'
            pass
        except:
            print("failed to get budget")
            return 'not-present'

    def get_gross_worldwide(soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            # TODO
            gross_li = soup.find('li', {'data-testid' : 'title-boxoffice-cumulativeworldwidegross'})
            if gross_li:
                gross_amount = gross_li.find_all('span')[1].get_text(strip=True)
                if gross_amount is not None:
                    return gross_amount
                else:
                    return 'not-present'
            return 'not-present'
            pass
        except:
            print("failed to get gross worldwide")
            return 'not-present'


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=1500)
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()
    
    ids = [movie['id'] for movie in imdb_crawler.crawled]
    unique_ids = set(ids)
    num_unique_ids = len(unique_ids)
    print(num_unique_ids)
if __name__ == '__main__':
    main()
