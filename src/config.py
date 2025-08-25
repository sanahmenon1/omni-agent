from pydantic import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    openai_api_key: str = Field(alias="OPENAI_API_KEY")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
