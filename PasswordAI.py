import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words1
from stop_words import get_stop_words
import re
import mmap
import pprint
import joblib
from nltk.corpus import stopwords
import yake
import re
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import spacy
import os


class Context_extractor:

    def __init__(self) -> None:
        pass

    def keyword_extractor(self, convo, max_ngram_size, deduplication_thresh, num_of_keywords):
        """ This function will extract the keywords from within a provided text

        Args:
            convo (str): conversation to analyse
            max_ngram_size (int): max size of keyword ngram
            deduplication_thresh (float): duplication threshold between 0 and 1
            num_of_keywords (int): number of keywords to be produced

        Returns:
            list : list of keywords
        """
        
        # The lower the score, the more relevant the keyword is
        language = "en"
        
        # creating keyword extractor
        custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresh, top=num_of_keywords, features=None)
        
        # extracting and sorting keywords
        keywords = custom_kw_extractor.extract_keywords(convo.lower())
    
        return keywords

    def topic_analysis(self, convo_list, stop_words, topic_extract = False):
        """This function analyses the topics within a provideed text using LDA

        Args:
            convo_list (str): conversation to be analysed
            stop_words (list): list of stop words
            topic_extract (bool, optional): determines if topic analysis should occur. Defaults to False.

        Returns:
            list: corpus and LDA topics
        """
       
        # creating bow
        # Remove new line characters
        data = [re.sub('\s+', ' ', sent) for sent in convo_list]

        # Remove distracting single quotes
        sentences = [re.sub("\'", "", sent) for sent in data]

        data_words = []
        for sentence in sentences:
            data_words.append(gensim.utils.simple_preprocess(str(sentence), deacc=True)) 

        # Build the bigram model
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        bigram_mod = gensim.models.phrases.Phraser(bigram)

        # Remove Stop Words
        stop_words = stopwords.words('english')
        stop_words.extend(['user', 'assistant', 'person'])
        data_words_nostops = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in data_words]

        # Form Bigrams
        data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]
        
        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = []
        
        # python3 -m spacy download en
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        
        for sent in data_words_bigrams:
            doc = nlp(" ".join(sent)) 
            data_lemmatized.append([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']])

        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        # Build LDA model
        lda_model = gensim.models.LdaMulticore(corpus, num_topics=10, id2word=id2word, passes=10, workers=2)
        
        return corpus, lda_model

    def display_context_analysis(self, corpus, lda_model, keywords, convo_list):
        """This function will display the context analysis variables

        Args:
            corpus (list): corpus of conversation
            lda_model (lda_model): lda model of the conversation
            keywords (list): list of keywords
            convo_list (list): list of conversations
        """

        for i in range(0,len(corpus)):
            print('-------------------------------------------')
            print('Key words and topics for sentence:')
            print('==================================')
            print(convo_list[i], '\n')
            print('Key topics:')
            print('===========')
            for index, score in sorted(lda_model[corpus[i]], key=lambda tup: -1*tup[1]):
                print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
            print()
            print('Keywords:')
            print('=========')
            keywords[i] = self.my_sort(keywords[i])
            for kw in keywords[i]:
                print(kw)
            print('-------------------------------------------', '\n')

        return

    def context_filter(self, convo_list, stop_words, check_context = True, context = ['password']):
        """This function conducts a contextual analysis of a text and looks for search words in topics and keywords

        Args:
            convo_list (str): conversation to be analysed
            stop_words (list): list of stop words
            check_context (bool, optional): determines if the context filter should be applied. Defaults to True.
            context (list, optional): list of search words used in the filter. Defaults to ['password'].

        Returns:
            list: bool list of whether the conversation(s) have passed the filter
        """
        
        # gathering the csv document topics 
        corpus, lda_model = self.topic_analysis(convo_list, stop_words)

        # gathering the csv keywords
        keywords = []
        for convo in convo_list:
            keywords.append(self.keyword_extractor(convo,2, 0.9, 20))

        # displaying results
        #self.display_context_analysis(corpus, lda_model, keywords, convo_list)
        
        # finding topics within the conversations
        topics = {}
        for i in range(0,len(corpus)):
            topics[i] = []
            for index, score in sorted(lda_model[corpus[i]]):
                
                # splitting up the topics for the sentence
                topics[i].append(lda_model.print_topic(index, 20))

        # checking if the context needs to be filtered
        if check_context:

            # init further analysis boolean list
            further_analysis = [False] * len(corpus)
        
            # checking topics for 'password'
            for i in range(0,len(corpus)):
                # isolating each topic
                for w in topics[i]:
                    for topic in context:
                        # checking to see if specified context is present
                        if topic in w:
                            further_analysis[i] = True
                            break
        
            # checking keywords for context parameter     
            for i in range(0,len(keywords)):
                for word in keywords[i]:
                    
                    # checking to see if password is present in keywords
                    for keyw in context:
                        if keyw in word[0]:
                            further_analysis[i] = True
                            break
        else:
            # init further analysis boolean list
            further_analysis = [True] * len(corpus)

        # filtering out conversations which are not suspicious   
        sus_convos = []
        for i in range(0,len(further_analysis)):
            
            if further_analysis[i]:
                sus_convos.append(i)
                print('convo ', i , ' registered as suspicious')
                
        return sus_convos, keywords, topics

class Detail_extractor:

    def __init__(self) -> None:
        pass

    def get_emails(self, convo):
        """This function get the emails from within a conversation

        Args:
            convo (str): conversation to be analysed

        Returns:
            set : set of emails found
        """

        # Removing lines that start with '//' because the regular expression
        # mistakenly matches patterns like 'http://foo@bar.com' as '//foo@bar.com'.
        regex = re.compile(("([a-z0-9!#$%&'*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`"
                        "{|}~-]+)*(@|\sat\s)(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(\.|"
                        "\sdot\s))+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)"))

        # extracting all found emails
        found_emails = []
        for word in convo:
            for email in re.findall(regex, word.lower()):
                if not email[0].startswith('//'):
                    found_emails.append(email[0])

        # returning all emails  
        return found_emails

    def get_intrinsic_analysis(self, model, stop_words, convo_storage):
        """This function performs intrinsic analysis of each token to determine whether it is a password

        Args:
            model (model): ia_model
            stop_words (list): list of stop words
            convo_storage (dict): dictionary containing information on the conversation

        Returns:
            list: list of detected passwords
        """

        # creating dictionaries for password candidate storage
        potential_passwords = {}
        detected_passwords = {}
        sus_convos = convo_storage['tokenised_convos']

        # creating dataframe of words
        df = pd.DataFrame(sus_convos, columns=['words'])

        # fitting tokenizer 
        tokenizer = Tokenizer(num_words=None, filters=None, lower=False, char_level=True, oov_token='<UNK>')
        tokenizer.fit_on_texts(df.words)

        # tokenize and pad everything, max len of 32
        text_sequences = tokenizer.texts_to_sequences(df.words)
        text_padded = pad_sequences(text_sequences, padding='post', truncating='post', maxlen=32)

        # predicting potential passwords
        pred_test = np.rint(model.predict(text_padded))
        
        # gathering potential passwords
        potential_passwords = []
        for x in range(0,len(sus_convos)):
            if pred_test[x] == 1:
                potential_passwords.append(sus_convos[x])

        # initializing punctuations string
        punc = ''';'",'''

        # iterating over all potenital passwords to determine thier viability as a real password
        detected_passwords = []
        for word in potential_passwords:
            
            #for word in potential_passwords:
        
            # Removing punctuations in string
            for ele in word:
                if ele in punc:
                    word = word.replace(ele, "")

            # checking to see if potential password is a false postive within the stop word list
            if str(word[0].lower() + word[1:]) not in stop_words and str(word[0].lower() + word[1:-1]) not in stop_words:
               
                # checking the potential password is a valid length # ADD VALIDITY CHECKS HERE
                if len(word) > 3 and len(word) < 32:
                        
                        # filtering out any emails, postcodes, dates, website urls, directories and any obvious non-password entities
                        if word not in convo_storage['emails'] and word not in convo_storage['postcodes'] and word not in convo_storage['dates'] \
                        and word not in convo_storage['urls'] and word not in convo_storage['directories'] and not re.search(r'^[^a-zA-Z0-9]+$', word) \
                        and word not in convo_storage['ip_addresses'] and word not in convo_storage['domains'] and word not in convo_storage['timestamps'] \
                        and'\\' not in word and '{' not in word and '}' not in word and '(' not in word and ')' not in word and ':' not in word and '/' not in word:
            
                            detected_passwords.append(word)


        return detected_passwords

    def get_password_contexts(self, convo, passwords, num_words_surround):
        """This function gathers the surrounding n words from a target word

        Args:
            convo (str): conversation to be analysed
            passwords (list): list of target words
            num_words_surround (int): number of words surrounding to be gathered

        Returns:
            list : list of target words and surrounding contexts
        """

        # internal function for finding password contexts
        def password_context_search(sentence,target, n):
            # Searches for text, and retrieves n words either side of the text, which are retuned seperatly
        
            indices = (i for i,word in enumerate(sentence) if target in word)
            neighbors = []
            for ind in indices:
                neighbors.append(sentence[ind-n:ind]+[str("{" + target + "}")]+sentence[ind+1:ind+n+2])

            try:
                return (" ".join(neighbors[0]))
            except:
                return None

        # finding porived password contexts
        password_contexts = set()

        for pswd in passwords:
                result = password_context_search(convo, pswd, num_words_surround)
                if result:
                    password_contexts.add(result)

        return password_contexts

    def get_postcodes(self, convo): 
        """This function gets the postcodes from a provided conversation

        Args:
            convo (str): conversation to be analysed

        Returns:
            set: set of all postcodes 
        """

        # finding all UK postcodes in the converation
        found_postcodes = set(re.findall(r'[A-Z]{1,2}[0-9R][0-9A-Z]? [0-9][A-Z]{2}', convo))
               
        # returning all postcodes 
        return found_postcodes

    def get_regex_passwords(self, convo):
        """This function gets passwords which follow common regex rules from a provided text

        Args:
            convo (str): conversation to be analysed

        Returns:
            set: set of regex passwords
        """

        # patterns from https://dev.to/rasaf_ibrahim/write-regex-password-validation-like-a-pro-5175

        # creating a set to store passwords and list of password regex 
        regex_passwords = set()
        regex_password_patterns = [r'^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z])(?=.*\W)(?!.* ).{8,16}$',  r'^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z])(?=.*_)(?!.*\W)(?!.* ).{8,16}$', 
                                  r'^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z])(?=.*_)(?!.* ).{8,16}$', r'^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z]).{8,16}$']
        
        # variable used in uneeded special character removal
        punc = ''';:'",.'''

        # checking every word in the conversation to see if it matches a regex pattern
        for pattern in regex_password_patterns:
            for word in convo:

                # Removing punctuations in string
                for ele in word:
                    if ele in punc:
                        word = word.replace(ele, "")

                # checking regex patterns
                result = re.findall(pattern, word)

                # adding results to set
                for pswd in result:
                    regex_passwords.add(pswd)

        return regex_passwords

    def get_dates(self, convo):
        """This function gets any dates from the conversation

        Args:
            convo (str): conversation to be analysed

        Returns:
            set: set of dates
        """

        # patterns from https://medium.com/analytics-vidhya/how-to-extract-all-date-variants-from-real-world-data-using-python-c9dc7413954c
        dates = set()
        date_patterns =  ['\d{1,2}[/-]\d{1,2}[/-]\d{2,4}','\d{4}[/-]\d{1,2}[/-]\d{1,2}', r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-zA-Z.,-]*[\s-]?(\d{1,2})?[,\s-]?[\s]?\d{4}',
                          r'\d{1,2}[/-]\d{4}']
        
        # checking every word in the conversation to see if it matches a regex pattern
        for pattern in date_patterns:
            # checking regex patterns
            result = re.findall(pattern, convo)

            # adding results to set
            for date in result:
                dates.add(date)
    
        return dates

    def get_URLs(self, convo):
        """This function gets any urls from within the conversation

        Args:
            convo (str): conversation to be analysed

        Returns:
            set: set of urls
        """

        # pattern from https://www.geeksforgeeks.org/python-check-url-string/
        urls = set()
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        
        for url in re.findall(regex, convo):

            urls.add(url[0])

        return urls

    def get_directories(self, convo):
        """This function gets all directories from within the text

        Args:
            convo (str): conversation to be analysed

        Returns:
            set: set of directories 
        """

        # pattern from https://www.appsloveworld.com/coding/python3x/168/python-regex-to-find-directory-path-location-location-location?expand_article=1
        dirs = set()
        regex =   r'[A-Z]:[\\\/].*\.[\w:]+'

        for dir in re.findall(regex, convo):

            dirs.add(dir)

        return dirs

    def get_ip_addresses(self, convo):
        """This function gets any IPv4 and IPv6 addresses from within the conversation

        Args:
            convo (str): conversation to be analysed

        Returns:
            set: set of IPv4 and IPv6 addresses
        """

        # pattern from https://gist.github.com/dfee/6ed3a4b05cfe7a6faf40a2102408d5d8
        ips = set()

        # Make a regular expression for validating an Ipv6 and Ipv4
        IPV4SEG  = r'(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])'
        IPV4ADDR = r'(?:(?:' + IPV4SEG + r'\.){3,3}' + IPV4SEG + r')'
        IPV6SEG  = r'(?:(?:[0-9a-fA-F]){1,4})'
        IPV6GROUPS = (
            r'(?:' + IPV6SEG + r':){7,7}' + IPV6SEG,                  # 1:2:3:4:5:6:7:8
            r'(?:' + IPV6SEG + r':){1,7}:',                           # 1::                                 1:2:3:4:5:6:7::
            r'(?:' + IPV6SEG + r':){1,6}:' + IPV6SEG,                 # 1::8               1:2:3:4:5:6::8   1:2:3:4:5:6::8
            r'(?:' + IPV6SEG + r':){1,5}(?::' + IPV6SEG + r'){1,2}',  # 1::7:8             1:2:3:4:5::7:8   1:2:3:4:5::8
            r'(?:' + IPV6SEG + r':){1,4}(?::' + IPV6SEG + r'){1,3}',  # 1::6:7:8           1:2:3:4::6:7:8   1:2:3:4::8
            r'(?:' + IPV6SEG + r':){1,3}(?::' + IPV6SEG + r'){1,4}',  # 1::5:6:7:8         1:2:3::5:6:7:8   1:2:3::8
            r'(?:' + IPV6SEG + r':){1,2}(?::' + IPV6SEG + r'){1,5}',  # 1::4:5:6:7:8       1:2::4:5:6:7:8   1:2::8
            IPV6SEG + r':(?:(?::' + IPV6SEG + r'){1,6})',             # 1::3:4:5:6:7:8     1::3:4:5:6:7:8   1::8
            r':(?:(?::' + IPV6SEG + r'){1,7}|:)',                     # ::2:3:4:5:6:7:8    ::2:3:4:5:6:7:8  ::8       ::
            r'fe80:(?::' + IPV6SEG + r'){0,4}%[0-9a-zA-Z]{1,}',       # fe80::7:8%eth0     fe80::7:8%1  (link-local IPv6 addresses with zone index)
            r'::(?:ffff(?::0{1,4}){0,1}:){0,1}[^\s:]' + IPV4ADDR,     # ::255.255.255.255  ::ffff:255.255.255.255  ::ffff:0:255.255.255.255 (IPv4-mapped IPv6 addresses and IPv4-translated addresses)
            r'(?:' + IPV6SEG + r':){1,4}:[^\s:]' + IPV4ADDR,          # 2001:db8:3:4::192.0.2.33  64:ff9b::192.0.2.33 (IPv4-Embedded IPv6 Address)
        )
        IPV6ADDR = '|'.join(['(?:{})'.format(g) for g in IPV6GROUPS[::-1]])  # Reverse rows for greedy match
        patterns = [IPV4ADDR, IPV6ADDR]

        # checking every word in the conversation to see if it matches a regex pattern
        for pattern in patterns:
            # checking regex patterns
            result = re.findall(pattern, convo)

            # adding results to set
            for ip in result:
                ips.add(ip)
    
        return ips

    def get_domains(self, convo, tld_list):
        """This function gets all domains from within the text

        Args:
            convo (str): conversation to analyse
            tld_list (list): list of top level domains to check

        Returns:
            set: set of domains
        """

        # finding all domains within the convo, registed in tld list
        domains = set()
        for word in convo:
            for tld in tld_list:
                if tld in word and len(word) < 32: # NEED TO CHECK IF THIS IS VALID VALUE
                    domains.add(word)
       
        return domains

    def get_timestamps(self, convo):
        """This function gets all timestamps from within the text

        Args:
            convo (str): conversation to be analysed

        Returns:
            set: set of timestamps
        """

        # finding timestamps int a conversation
        timestamps = set()
        regex =   r'\b\d{10}\b'

        for ts in re.findall(regex, convo):

            timestamps.add(ts)

        return timestamps

    def analyse_cookie_file(self, convo_data):
        """This function extracts cookie information from a text

        Args:
            convo_data (str): text to be analysed

        Returns:
            dict: dictionary containing cookie information
        """

        # splitting the convo list to be processed
        convo_data = convo_data.split("\n") 

        # creating a nested dictionary of all cookie domains
        cookie_domains = {}
        for line in convo_data:
            line = line.split("\t")

            if len(line) > 5:
                cookie_domains[line[0]] = {}

        

        # populating each cookie domain dictionary with its recorded cookies
        for line in convo_data:
            line = line.split("\t")
        
            if len(line) > 5:
                domain = line[0]
                cookie_uid = hash(str(line[4] + line[5]))
                cookie_domains[domain][cookie_uid] = {} # creating a unique id of the cookie stored
                cookie_domains[domain][cookie_uid]['file_paths'] = line[2]
                cookie_domains[domain][cookie_uid]['cookie_names'] = line[5]
                cookie_domains[domain][cookie_uid]['cookie_value'] = line[6]

        return cookie_domains


# Utility functions 
def analyse_dir(dir, search_keywords = ['CDS'], context_keywords = ['password'], check_context = True):
    """This function performs analysis of a given directory

    Args:
        dir (str): directory to search
        search_keywords (list, optional): words to be searched for in entire document to determine further analysis. Defaults to ['CDS'].
        context_keywords (list, optional): word to be searched for in context analysis to determine further analysis Defaults to ['password'].
        check_context (bool, optional): determines if context filter is applied to files in directory. Defaults to True.
    """
    
    # load ia_model in preperation for analysis
    ia_model = load_ia_model()

    for filename in os.listdir(dir):
        if filename.endswith('.txt') or filename.endswith('.csv'):
            print('\n######## START OF FILE SEARCH ########')
            analyse_file(str(dir + '/' + filename), search_keywords, context_keywords, check_context, ia_model)
            print('\n######## END OF FILE ########\n')
            
def search_word_filter(filename, keywords):
    """This function searches for keywords in files

    Args:
        filename (str): name of file
        keywords (list): list of keywords to search

    Returns:
        bool : bool determining if keyword(s) are present
    """

    # converting search term to bytes
    for i in range(0, len(keywords)):
        if type(keywords[i]) != bytes:
            keywords[i] = keywords[i].encode('utf-8')

    # checks a provided file for a set of key words
    with open(filename, 'rb', 0) as file:
        s = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        for term in keywords:
            if s.find(term) != -1:
                print('**Keyword: ' , term, ' exists in file**')
                return True
            
        print('--> Keywords not found in document')
        return False

def prep_file_data(filename):
    """This function produces all the neccasary variables for the PasswordAI functionality

    Args:
        filename (str): name of file to be analysed

    Returns:
        var: various veriables used in later processing of files
    """

    # initilising variables
    convo_list = []
    convo_ids = []
    convo_uids = []
    convo_storage = {}
    i = 0

    if '.csv' in filename:

        # import data from csv conversation file
        convo_data = pd.read_csv(filename)

        # separating out all converation samples
        for convo in convo_data.columns:
            convo_str = convo_data[convo].values[0]
            convo_list.append(convo_str)
            convo_uids.append(hash(convo_str))
            convo_storage[i] = {}
            convo_ids.append(i)
            i += 1

    elif '.txt' in filename:

        # importing text file data
        convo_data = open(filename,'r').read()
        convo_list.append(convo_data)
        convo_uids.append(hash(convo_data))
        convo_storage[i] = {}
        convo_ids.append(i)
        i += 1

    else:
        assert 'Invalid file type'

    # building stopword list
    stop_words = stopwords.words('english')
    stop_words.extend(stop_words1)
    stop_words2 = get_stop_words('en')
    stop_words.extend(stop_words2)

    # extending stopword list
    my_file = open("word lists/wlist_match7.txt", "r") 
    data = my_file.read() 
    data_into_list = data.split("\n") 
    stop_words_extended = stop_words
    stop_words_extended.extend(data_into_list)

    # preparing tld file
    tld_file = open('word lists/tlds.txt', 'r')
    tld_file = tld_file.read()
    tld_file = tld_file.split("\n") 
    tld_file_new = []
    for tld in tld_file:
        tld_file_new.append(str('.' + tld))

    # removing final element of tld '.'
    tld_file_new.pop()

    # spliting conversations into seperate sentances
    for i in range(0,len(convo_list)):
        convo_storage[i]['convo'] = convo_list[i]
        convo_storage[i]['u_id'] = convo_uids[i]
        convo_storage[i]['split_sentences'] = (re.compile('[.!?] ').split(convo_list[i]))
        convo_storage[i]['tokenised_convos'] = (convo_list[i].split())

    return convo_storage, stop_words, stop_words_extended, convo_list, tld_file_new

def analyse_file(filename, search_keywords = ['CDS'], context_keywords = ['password', 'passwords'], check_context = True, ia_model = None):
    """This function performs analysis of a file extracting key information where possible

    Args:
        filename (str): name of file to be analysed
        search_keywords (list, optional): list of search words for document search. Defaults to ['CDS'].
        context_keywords (list, optional): list of search words for context filter. Defaults to ['password', 'passwords'].
        check_context (bool, optional): determines whether the context filter is applied. Defaults to True.
        ia_model (model, optional): ia_model. Defaults to None.
    """

    cont_ex = Context_extractor()
    deat_ex = Detail_extractor()
    print(filename)
    print('--> Searching document for Keywords')

    if search_word_filter(filename, search_keywords):

        # prepare data to be used in later functions
        print('--> Preparing data')
        convo_storage, stop_words, stop_words_extended, convo_list, tld_file = prep_file_data(filename)

        # finding out which conversations discuss specified contexts FIX THIS VALIDATION CHECK
        print('--> Applying context Filter')
        try:
            suspicious_convo_indexes, keywords, topics = cont_ex.context_filter(convo_list, stop_words, check_context=check_context ,context=context_keywords)
        except:
            print('Error in context filter')
            return

        # find the context in which potential passwords are discussed and details within conversations
        print('--> Extracting text context and details')
        for i in suspicious_convo_indexes: 

            # if there are cookies present, analyse the file for cookies
            convo_storage[i]['cookies'] = {}
            if search_word_filter(filename, keywords=['cookie', 'Cookie', 'TRUE']):
                convo_storage[i]['cookies'] = deat_ex.analyse_cookie_file(convo_list[i])
         
            # getting the keywords for the conversation
            convo_storage[i]['keywords'] = set(keywords[i])

            # getting the topics for a conversation
            convo_storage[i]['topics'] = set(topics[i])

            # finding emails within the conversation
            convo_storage[i]['emails'] = deat_ex.get_emails(convo_storage[i]['tokenised_convos'])

            # finding postcodes within the conversation
            convo_storage[i]['postcodes'] = deat_ex.get_postcodes(convo_list[i])

            # finding dates within the conversation
            convo_storage[i]['dates'] = deat_ex.get_dates(convo_list[i])

            # finding all urls within the conversation
            convo_storage[i]['urls'] = deat_ex.get_URLs(convo_list[i])

            # finding all directories within the conversation
            convo_storage[i]['directories'] = deat_ex.get_directories(convo_list[i])

            # finding all IP addresses within the conversation
            convo_storage[i]['ip_addresses'] = deat_ex.get_ip_addresses(convo_list[i])

            # finding all domains within the conversation
            convo_storage[i]['domains'] = deat_ex.get_domains(convo_storage[i]['tokenised_convos'],tld_file)

            # finding all timestamps within the conversation
            convo_storage[i]['timestamps'] =  deat_ex.get_timestamps(convo_list[i])

            # getting the AI detected passwords, these need to be calculated after all other details 
            convo_storage[i]['passwords'] = set(deat_ex.get_intrinsic_analysis(ia_model, stop_words_extended, convo_storage[i])) 

            # finding passwords which match common password requirement rules
            convo_storage[i]['regex_passwords'] = deat_ex.get_regex_passwords(convo_storage[i]['tokenised_convos'])

            # find all password contexts within the conversation
            convo_storage[i]['password_contexts'] = deat_ex.get_password_contexts(convo_storage[i]['tokenised_convos'], convo_storage[i]['passwords'], 3)

        # removing uneeded fields
        for i in convo_storage:
            convo_storage[i].pop('tokenised_convos')
            convo_storage[i].pop('split_sentences')

        for i in range(0,len(convo_storage)):
            if i in suspicious_convo_indexes:
                print('----------------------------------------------------------------')
                pprint.pprint(convo_storage[i])

def load_ia_model():
    """This function loads the intrinsic analysis model

    Returns:
        model: ia_model
    """

    # load ML model for intrinsic analysis
    print('--> Loading Intrinsic Analysis Model')
    ia_model = joblib.load("./random_forest.joblib")#tf.keras.models.load_model('bilstm_full', compile=False) # need to check versions of tensorflow are the same for training and loading

    return ia_model


if __name__ == '__main__':

    #filename = 'manual_convo_list.csv'
    #ia_model = load_ia_model()
    #analyse_file(filename, ['CDS'], ['password'], True, ia_model)
    
    dir = '/home/ben/Desktop/OvertAI/CDS work/credential finder/misc/demo_files'
    analyse_dir(dir,['Password:', 'Username:', 'email', 'UserName:', 'TRUE'], ['password', 'passwords', 'credentials'], False)

    # ISSUES:
    # context filter will sometimes fail if file has little data
    # some non passwords are still included in AI identified passswords
    """
        Use:
            - uf using analyse_dir() function, dir is the directory containing files which will be explored (.txt and .csv only at the minute)
            - the search words will be searched for in each document in an exact manner to determine whether the file needs to be further analysed
            - the context words will be searched for in the keywords and topics of the document if it passes the search word filter
            - the check context bool determines if the file should undergo the context analysis filter, or be analysed directly after the search word filter
            - the ia_model is required when using the analyse_file() function, and should be loaded via the load_ia_model() function and passed into the analyse_file() function
    """


    
        