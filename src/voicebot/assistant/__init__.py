"""The office services the voice bot talks to, English and Urdu.

The recognisers and how the language is decided from them, the voices and
how a reply is made speakable for them, the voice activity model that finds
where speech starts and ends, and who the bot is.

    python -m voicebot.assistant setup     # download the voice activity model
    python -m voicebot.assistant check     # configuration and every endpoint

Every service it talks to is in configs/assistant.yaml.
"""
