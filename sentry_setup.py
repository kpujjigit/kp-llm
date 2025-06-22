import os
from dotenv import load_dotenv
import sentry_sdk

def init_sentry():
    load_dotenv()
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        environment=os.getenv("ENVIRONMENT"),
        traces_sample_rate=1.0,
    )
