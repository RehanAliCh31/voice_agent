"""The hands-free assistant: wake word, live conversation, English and Urdu.

It idles listening only for its wake word, sending nothing anywhere. Once
woken, it listens, finds the end of what was said, transcribes it with the
recogniser for the language it was spoken in, and speaks the reply with the
matching voice -- then keeps listening for a follow-up, and goes back to
sleep after a silence.

    python -m voicebot.assistant setup     # download the wake-word and VAD models
    python -m voicebot.assistant check     # configuration and every endpoint
    python -m voicebot.assistant run       # start listening

Everything it talks to, and every threshold it uses, is in
configs/assistant.yaml.
"""
