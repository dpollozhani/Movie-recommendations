from dataextraction import chrome, download_own_ratings, download_movie_data
from datatransformation import load_files, merge_files, prepare_model
from recommendation import generate_recommendations, prettify
from slack_post import post_message, slack
from datetime import datetime
import sys

accepted_arguments = ['-partial', '-test', '-download'] #-test to run everything except posting to Slack
given_arguments = sys.argv[1:]
arguments = [None]
for arg in given_arguments:
    if arg in accepted_arguments:
        arguments.append(arg)

if '-download' in arguments:
    arguments = ['-download']

print(*arguments[1:])

browser = chrome()

def main():
    
    start_time = datetime.now()
    print('Script started at', start_time)

    if not any(arguments) or '-test' in arguments or '-download' in arguments:
        #### Collect all data ####
        print('-----------------------------')
        print('Downloading data...')
        print('-----------------------------')
        download_movie_data()
        download_own_ratings(browser)

        elapsed_time = datetime.now()-start_time
        print(f'Elapsed time: {elapsed_time}')

    if not any(arguments) or '-partial' in arguments or '-test' in arguments:
        #### Clean all data ####
        print('-----------------------------')
        print('Cleaning, merging, and preparing data...')
        print('-----------------------------')
        loaded_files = load_files()
        merged_data = merge_files(loaded_files)
        prepare_model(merged_data)

        elapsed_time = datetime.now()-start_time
        print(f'Elapsed time: {elapsed_time}')
    
        #### Generate recommendations ####
        print('-----------------------------')
        print('Generating recommendations...')
        print('-----------------------------')
        recommendations = generate_recommendations()
        
        elapsed_time = datetime.now()-start_time
        print(f'Elapsed time: {elapsed_time}')

    if not any(arguments) or '-partial' in arguments:
        #### Post recommendations to Slack ####
        print('-----------------------------')
        print('Posting recommendations to Slack...')
        print('-----------------------------')
        channel_id = slack['channel id']
        post_message(channel=channel_id, comment="today's recommendations :smile:", content=prettify(recommendations,1))

    end_time = datetime.now()
    elapsed_time = end_time-start_time
    print('-----------------------------')
    print('Script finished at', end_time)

if __name__ == '__main__':
    main()