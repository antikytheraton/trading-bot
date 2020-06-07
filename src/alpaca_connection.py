import os
from alpaca_trade_api import REST


class AlpacaPaperSocket(REST):
    def __init__(self):
        super().__init__(
            key_id=os.getenv("APCA_API_KEY_ID"),
            secret_key=os.getenv("APCA_API_SECRET_KEY"),
            base_url=os.getenv("APCA_API_BASE_URL"),
        )
