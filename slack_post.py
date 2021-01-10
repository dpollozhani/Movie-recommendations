import os
import json
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

with open('json/slack.json', 'r') as f:
    slack = json.load(f)
    f.close()

def authenticate_client():
    return WebClient(token=os.environ.get('SLACK_TOKEN'))

def post_message(channel, comment, content, mention='@channel'):
    
    client = authenticate_client()

    try:
        result = client.chat_postMessage(
            channel=channel,
            text= f"Hi {mention}, here are {comment}",
            blocks = [
                {
			    "type": "section",
			    "text": {"type": "mrkdwn",
				        "text": f"Here are {comment}"
			            }
                },
                {
			    "type": "section",
			    "text": {"type": "mrkdwn",
				        "text": f"{content}"
			            }
                }
            ]
        )
        assert result['ok'] is True
        print(f"Post to channel {channel} was successful!")  
    except SlackApiError as e:
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")

def post_dm(user, comment, content):
    pass    

if __name__ == '__main__':
    import recommendation
    from recommendation import generate_recommendations, prettify, table_to_image
    
    df = prettify(generate_recommendations(),1)
    channel_id = slack['channel id']
    post_message(channel_id, comment="today's recommendations :smile:", content=df)
    

    