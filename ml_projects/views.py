from django.shortcuts import render
from django.http import HttpResponse
from urllib import request as url_request
from bs4 import BeautifulSoup as bs
import re
import nltk
import heapq
from textblob import TextBlob
import tweepy
from django.conf import settings
from django.contrib.staticfiles.finders import find
from django.templatetags.static import static
from sklearn.externals import joblib
#import sklearn.external.joblib as extjoblib
#import joblib

from sklearn.feature_extraction.text import CountVectorizer

# Create your views here.

def landing_page(request):
    return render(request, 'index.html')

def web_summarizer(request):
    summarized_sentence=''

    if request.GET.get('website',False) is not False:
        website=request.GET['website']

        all_content=""
        if 'http' in website:
            # data preparation
            modified_request = url_request.Request(website, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'})
            html = url_request.urlopen(modified_request)

            soupObject = bs(html, 'html.parser')
            paragraphs = soupObject.findAll('p')

            for paragraph in paragraphs:
                all_content += paragraph.text

            # clean data
            cleaned_all_content = re.sub(r'\[[0-9]*\]', '', all_content)
            cleaned_all_content = re.sub(r'\s+', ' ', cleaned_all_content)

            # create sentence token
            sentences__tokens = nltk.sent_tokenize(cleaned_all_content)

            cleaned_all_content = re.sub(r'[^a-zA-Z]', ' ', cleaned_all_content)
            cleaned_all_content = re.sub(r'\s+', ' ', cleaned_all_content)

            # create word token
            words_tokens = nltk.word_tokenize(cleaned_all_content)

            stopwords = nltk.corpus.stopwords.words('english')

            word_frequencies = {}

            for word in words_tokens:
                if word not in stopwords:
                    if word not in word_frequencies.keys():
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1

            # calculate weighted frequency
            try:
                maximum_frequency_word = max(word_frequencies.values())
            except ValueError:
                maximum_frequency_word = 0

            for word in word_frequencies.keys():
                word_frequencies[word] = (word_frequencies[word] / maximum_frequency_word)

            # calculate sentence score with each word
            sentences_scores = {}

            for sentence in sentences__tokens:
                for word in nltk.word_tokenize(sentence.lower()):
                    if word in word_frequencies.keys():
                        if (len(sentence.split(' '))) < 30:
                            if sentence not in sentences_scores.keys():
                                sentences_scores[sentence] = word_frequencies[word]
                            else:
                                sentences_scores[sentence] += word_frequencies[word]
            summary = heapq.nlargest(5, sentences_scores, key=sentences_scores.get)

            summarized_sentence = ''

            for sentence in summary:
                summarized_sentence += sentence

        else:
            summarized_sentence='Error!!! Kindly enter a valid string. Ensure you include HTTP(S)://'


    return render(request, 'summary.html', {'website': summarized_sentence})

def twitter(request):
    #initialize variables
    tweet_topic = ''
    positive_tweets = 0
    negative_tweets = 0
    neutral_tweets = 0
    polarity = 0
    tweet=''

    if request.GET.get('tweet', False) is not False:
        tweet = request.GET['tweet']


        if tweet is not '':
            # authenicate API
            CONSUMER_KEY = '*****************'
            CONSUMER_SECRET = '*****************'
            ACCESS_TOKEN = '*****************'
            ACCESS_TOKEN_SECRET = '*****************'

            auth = tweepy.OAuthHandler(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET)
            auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

            api = tweepy.API(auth)
            # status = "Testing!"
            # api.update_status(status=status)

            no_of_tweets_analyse = 0

            # tweet_content=tweepy.Cursor(api.search,q=tweet,lang="English")
            tweet_content = api.search(tweet)

            for twt in tweet_content:
                analysis = TextBlob(twt.text)
                polarity += analysis.sentiment.polarity

                if (analysis.sentiment.polarity == 0):
                    neutral_tweets += 1
                elif (analysis.sentiment.polarity < 0.00):
                    negative_tweets += 1
                elif (analysis.sentiment.polarity > 0.00):
                    positive_tweets += 1

                no_of_tweets_analyse += 1

            # work on perentage
            positive_tweets = format((float(positive_tweets) / float(no_of_tweets_analyse)) * 100, '.2f')
            negative_tweets = format((float(negative_tweets) / float(no_of_tweets_analyse)) * 100, '.2f')
            neutral_tweets = format((float(neutral_tweets) / float(no_of_tweets_analyse)) * 100, '.2f')

            tweet_topic = positive_tweets
        else:
            tweet='Error!!! (Empty string)'


    return render(request, 'twitter.html',{'sentiment': tweet_topic,'tweet':tweet,'positive':positive_tweets,'negative':negative_tweets,'neutral':neutral_tweets})

def name_classify(request):
    model_url = get_static('assets/models/nigerian_region_names_model.pkl')
    vectorizer_url=get_static('assets/models/vectorizer.pkl')
    region=''
    fullname=''
    if request.GET.get('fullname', False) is not False:
        fullname = request.GET['fullname']


        if fullname is not '':
            loaded_model = joblib.load(model_url)
            vectorizer = joblib.load(vectorizer_url)

            vector = vectorizer.transform([fullname]).toarray()
            if loaded_model.predict(vector) == 0:
                region="Eastern"
            elif loaded_model.predict(vector) == 1:
                region="Western"
            else:
                region="Northern"
        else:
          region='Error!!! (Empty string- Enter Your Fullname)'

    return render(request, 'names.html',{'fullname': fullname,'region':region})

def get_static(path):
    if settings.DEBUG:
        return find(path)
    else:
        return static(path)