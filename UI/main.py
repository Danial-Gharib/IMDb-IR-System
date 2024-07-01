import streamlit as st
import sys
import os
import time
from enum import Enum
import random

# Add your necessary imports here
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from Logic import utils
from Logic.core.utility.snippet import Snippet
from Logic.core.link_analysis.analyzer import LinkAnalyzer
from Logic.core.indexer.index_reader import Index_reader
from Logic.core.search import Indexes

snippet_obj = Snippet()

class color(Enum):
    RED = "#FF6347"
    GREEN = "#32CD32"
    BLUE = "#1E90FF"
    YELLOW = "#FFD700"
    WHITE = "#F8F8FF"
    CYAN = "#00CED1"
    MAGENTA = "#FF00FF"
    PURPLE = "#8A2BE2"
    ORANGE = "#FFA500"
    PINK = "#FF69B4"

def get_top_x_movies_by_rank(x: int, results: list):
    path = "indexes_standard/"
    document_index = Index_reader(path, Indexes.DOCUMENTS)
    corpus = []
    root_set = []
    for movie_id, movie_detail in document_index.index.items():
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        corpus.append({"id": movie_id, "title": movie_title, "stars": stars})

    for element in results:
        movie_id = element[0]
        movie_detail = document_index.index[movie_id]
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        root_set.append({"id": movie_id, "title": movie_title, "stars": stars})
    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=x)
    return actors, movies

def get_summary_with_snippet(movie_info, query):
    summary = movie_info["first_page_summary"]
    snippet, not_exist_words = snippet_obj.find_snippet(summary, query)
    if "***" in snippet:
        snippet = snippet.split()
        for i in range(len(snippet)):
            current_word = snippet[i]
            if current_word.startswith("***") and current_word.endswith("***"):
                current_word_without_star = current_word[3:-3]
                summary = summary.lower().replace(
                    current_word_without_star,
                    f"<b><font size='4' color={random.choice(list(color)).value}>{current_word_without_star}</font></b>",
                )
    return summary

def search_time(start, end):
    st.success(f"Search took: {(end - start) * 1e3:.2f} milliseconds")

def search_handling(
    search_button,
    search_term,
    search_max_num,
    search_weights,
    search_method,
    unigram_smoothing,
    alpha,
    lamda,
    filter_button,
    num_filter_results,
):
    if filter_button:
        if "search_results" in st.session_state:
            top_actors, top_movies = get_top_x_movies_by_rank(
                num_filter_results, st.session_state["search_results"]
            )
            st.markdown(f"**Top {num_filter_results} Actors:**")
            actors_ = ", ".join(top_actors)
            st.markdown(
                f"<span style='color:{random.choice(list(color)).value};font-size:1.2em;'>{actors_}</span>",
                unsafe_allow_html=True,
            )
            st.divider()

        st.markdown(f"**Top {num_filter_results} Movies:**")
        for i in range(len(top_movies)):
            card = st.columns([3, 1])
            info = utils.get_movie_by_id(top_movies[i], utils.movies_dataset)
            with card[0].container():
                st.title(info["title"])
                st.markdown(f"[Link to movie]({info['URL']})")
                st.markdown(
                    f"<b><font size='4'>Summary:</font></b> {get_summary_with_snippet(info, search_term)}",
                    unsafe_allow_html=True,
                )

            with st.container():
                st.markdown("**Directors:**")
                num_authors = len(info["directors"])
                for j in range(num_authors):
                    st.text(info["directors"][j])

            with st.container():
                st.markdown("**Stars:**")
                num_authors = len(info["stars"])
                stars = "".join(star + ", " for star in info["stars"])
                st.text(stars[:-2])

                topic_card = st.columns(1)
                with topic_card[0].container():
                    st.write("Genres:")
                    num_topics = len(info["genres"])
                    for j in range(num_topics):
                        st.markdown(
                            f"<span style='color:{random.choice(list(color)).value};font-size:1.1em;'>{info['genres'][j]}</span>",
                            unsafe_allow_html=True,
                        )
            with card[1].container():
                st.image(info["Image_URL"], use_column_width=True)

            st.divider()
        return

    if search_button:
        corrected_query = search_term
        # utils.correct_text(search_term, utils.movies_dataset)

        if corrected_query != search_term:
            st.warning(f"Your search terms were corrected to: {corrected_query}")
            search_term = corrected_query

        with st.spinner("Searching..."):
            time.sleep(0.5)  # for showing the spinner! (can be removed)
            start_time = time.time()
            result = utils.search(
                query=search_term,
                max_result_count=search_max_num,
                method=search_method,
                weights=search_weights,
                unigram_smoothing_method=unigram_smoothing,
                alpha=alpha,
                lamda=lamda,
            )
            if "search_results" in st.session_state:
                st.session_state["search_results"] = result
            print(f"Result: {result}")
            end_time = time.time()
            if len(result) == 0:
                st.warning("No results found!")
                return

            search_time(start_time, end_time)

        for i in range(len(result)):
            card = st.columns([3, 1])
            info = utils.get_movie_by_id(result[i][0], utils.movies_dataset)
            with card[0].container():
                st.title(info["title"])
                st.markdown(f"[Link to movie]({info['URL']})")
                st.write(f"Relevance Score: {result[i][1]}")
                st.markdown(
                    f"<b><font size='4'>Summary:</font></b> {get_summary_with_snippet(info, search_term)}",
                    unsafe_allow_html=True,
                )

            with st.container():
                st.markdown("**Directors:**")
                if info["directors"] is None:
                    st.text("No directors")
                else:
                    num_authors = len(info["directors"])
                    for j in range(num_authors):
                        st.text(info["directors"][j])

            with st.container():
                st.markdown("**Stars:**")
                num_authors = len(info["stars"])
                stars = "".join(star + ", " for star in info["stars"])
                st.text(stars[:-2])

                topic_card = st.columns(1)
                with topic_card[0].container():
                    st.write("Genres:")
                    num_topics = len(info["genres"])
                    for j in range(num_topics):
                        st.markdown(
                            f"<span style='color:{random.choice(list(color)).value};font-size:1.1em;'>{info['genres'][j]}</span>",
                            unsafe_allow_html=True,
                        )
            with card[1].container():
                st.image(info["Image_URL"], use_column_width=True)

            st.divider()

        st.session_state["search_results"] = result
        if "filter_state" in st.session_state:
            st.session_state["filter_state"] = (
                "search_results" in st.session_state
                and len(st.session_state["search_results"]) > 0
            )

def main():
    st.set_page_config(
        page_title="Movie Search Engine",
        page_icon=":clapper:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for enhanced UI
    st.markdown("""
        <style>
            @media (max-width: 600px) {
                .stButton button {
                    width: 100%;
                }
                .stTextInput input {
                    width: 100%;
                }
                .stMarkdown span {
                    font-size: 1em;
                }
            }
            body {
                background-image: url();
                background-size: cover;
                font-family: 'Arial', sans-serif;
            }
            .stButton button {
                background-color: #FFA07A;
                color: white;
                border-radius: 5px;
                font-size: 16px;
                font-weight: bold;
            }
            .stTextInput input {
                border: 2px solid #FFA07A;
                border-radius: 5px;
            }
            .stSlider .stSliderThumb {
                background-color: #FFA07A;
            }
            .stSlider .stSliderTrack {
                background-color: #FFA07A;
            }
            .stMarkdown span {
                font-size: 1.1em;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎬 Movie Search Engine")
    st.write(
        "Welcome to the Movie Search Engine! Search through the IMDB dataset to find the most relevant movies based on your search terms."
    )
    st.markdown(
        '<span style="color:#FFA07A;font-size:1.2em;">Developed By: MIR Team at Sharif University</span>',
        unsafe_allow_html=True,
    )

    search_term = st.text_input("🔍 Search Term")
    with st.expander("🔧 Advanced Search"):
        search_max_num = st.number_input(
            "Maximum number of results", min_value=5, max_value=100, value=10, step=5
        )
        weight_stars = st.slider(
            "Weight of stars in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )

        weight_genres = st.slider(
            "Weight of genres in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )

        weight_summary = st.slider(
            "Weight of summary in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )
        slider_ = st.slider("Select the number of top movies to show", 1, 10, 5)

        search_weights = [weight_stars, weight_genres, weight_summary]
        search_method = st.selectbox(
            "Search method", ("ltn.lnn", "ltc.lnc", "OkapiBM25", "unigram")
        )

        unigram_smoothing = None
        alpha, lamda = None, None
        if search_method == "unigram":
            unigram_smoothing = st.selectbox(
                "Smoothing method",
                ("naive", "bayes", "mixture"),
            )
            if unigram_smoothing == "bayes":
                alpha = st.slider(
                    "Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )
            if unigram_smoothing == "mixture":
                alpha = st.slider(
                    "Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )
                lamda = st.slider(
                    "Lambda",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )

    if "search_results" not in st.session_state:
        st.session_state["search_results"] = []

    search_button = st.button("🔍 Search!")
    filter_button = st.button("🎬 Filter movies by ranking")

    search_handling(
        search_button,
        search_term,
        search_max_num,
        search_weights,
        search_method,
        unigram_smoothing,
        alpha,
        lamda,
        filter_button,
        slider_,
    )

if __name__ == "__main__":
    main()
