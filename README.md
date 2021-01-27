# COVID-19-Tweets-Analysis-Jan-Oct

These are data and codes used to support our findings in the paper on public sentiment on COVID-19 and its relation to influence.

Tweets are hydrated from ids obtained from a large-scale COVID-19 Twitter chatter dataset for open scientific research.

Banda, J. M., Tekumalla, R., Wang, G., Yu, J., Liu, T., Ding, Y., … Chowell, G.. (2020). A large-scale COVID-19 Twitter chatter dataset for open scientific research - an international collaboration (Version 36) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.4274912

Code files (.py):
“Eng extraction.py”: extract data of tweets in English language from the full dataset. Due to the large amount of data since March and the limitation of computing power, the tweet information of every 6 or 7 days was saved in a separate file. The output data is saved in “original_eng_dataset” folder.
“id collection.py”: select 5% of the tweet info saved in each file of the folder “original_eng_dataset” and extract the id of each tweet. The output data is saved in the folder “collected_id”.
“exported tweets cleaning.py”: only save the text and influence-related variables of each tweet hydrated. The output data is saved in the folder “hydrated_tweet”.

"0. Sentiment analysis training.py": train model for sentiment classification (feature selection)
1.1-2 - 1.10: process tweets from each month
