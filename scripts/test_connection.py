import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("RIOT_API_KEY")
url = "https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/PistaciovyBandit/LIDL"

r = requests.get(url, headers={"X-Riot-Token": api_key})
print(r.status_code)
print(r.text)