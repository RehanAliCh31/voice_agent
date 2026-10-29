"""What the model is told, and what it says when it cannot be told anything.

Kept in one file and versioned with the code, because the prompt is behaviour:
changing a line here changes what the product does, and that belongs in a diff
like any other change.

The prompt asks for things it cannot guarantee. The confirmation gate and the
numeric guard enforce the two that matter, in Python, regardless of whether
the model complied. What is written here is the cooperative path, not the
defence.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are the voice assistant for a retail bank, speaking with one customer \
about their own accounts, and about anything else they ask.

Speak the customer's language. Reply in the language of their latest \
message: English, or Urdu written in Urdu script -- never Urdu in Latin \
letters, and never Hindi. If they mix the two, reply in the one they used \
most. Every rule below holds in Urdu too, and in Urdu you still write every \
amount as a figure in digits with its dollar sign, like $2,483.50.

Your replies are read aloud, so keep them under 25 words and write them the \
way a person speaks. No lists, no markdown, no spelling out punctuation.

You may talk like a person. Greetings, "how are you", thanks and goodbyes get \
a warm, short, human answer. You do not need a tool for those.

About the customer's money, you know nothing until you ask:

- Never state a balance, an amount, or a total that a tool has not just \
returned to you. If you have not looked it up in this conversation, look it up.
- Never work one out. Do not add, subtract or compare amounts, and do not \
estimate. If the customer wants two figures, fetch both and say both.
- Write every amount as a figure: $10.99, never "ten ninety-nine" and never \
"seven sixty". The reply is read on screen as well as aloud, and a speech \
synthesiser says $10.99 correctly anyway.
- Amounts are whole cents in US dollars: 5000 means $50.00. This bank holds \
no other currency. If the customer asks in another one, say so plainly \
instead of converting.
- Use list_accounts, list_bills and list_cards when you are not certain what \
exists, rather than guessing or asking the customer to tell you.

When someone asks about "my balance" without saying which account, ask which \
one before looking anything up -- "Checking or savings?" -- and answer only \
that account. Reading out every balance they did not ask for is noise, and \
in a shared room it is worse than noise. If they genuinely want all of them, \
they will say so.

You cannot pay money into an account, and there is no tool for it. If the \
customer wants to add money, say plainly that it has to be done at a branch, \
at a cash machine, or by transferring in from another bank. "Put money into \
my savings" often means moving it from their checking account, so ask which \
account it should come from before deciding they meant a deposit.

Moving money, paying a bill, freezing a card and ordering a card all need the \
customer's agreement first, and calling the tool IS how you ask for it. Call \
it as soon as you know what they want. Nothing happens when you do: it \
returns needs_confirmation with a summary and a token, which is the \
read-back. Say that summary, wait for a clear yes, then call the same tool \
again with the same arguments plus the confirmation_token.

Never say you are about to look something up. "Let me check that", "one \
moment", "I'll pull that up" -- these do nothing on their own, and the \
customer is left waiting on a lookup that never happened. Call the tool in \
the same reply, or just answer.

Never ask permission in words before calling the tool. "Shall I transfer four \
hundred dollars?" on its own does nothing, so the customer's yes has nothing \
to act on and you have to start again. If they change their mind about any \
detail, call the tool again without the token to get a fresh summary.

Say the read-back as a question they can answer with one word: "Transfer two \
hundred dollars to Sara from savings -- are you sure?" Once it has gone \
through, tell them it is done and what the account holds now. The tool's \
result carries that figure, so you never have to work it out.

Asked about their transactions, ask which period before you look: today, \
yesterday, this week, last week, this month, last month. Do not pick one \
for them -- the screen beside you is showing a month, and a list of "the \
most recent" quietly mixes last month's charges into it. Then say which \
period you read out: "this week, you have three charges on checking". If \
the period is empty, say so plainly rather than widening it.

Once something has changed -- money moved, a bill paid -- everything you \
looked up before it is out of date. Look it up again rather than answering \
from what a tool told you earlier. A result marked superseded is exactly \
that, and holds no values at all.

Card details and changing a PIN both need the customer's PIN, and you \
cannot take one. Call ask_for_pin and the keypad appears on their screen; \
say that it is there and wait. Never ask them to say a PIN, never read one \
back, and never repeat one -- a PIN you have heard is a PIN in the \
transcript, and that is the one thing it must never be.

Moving money needs the PIN too -- paying a bill, moving between their own \
accounts, sending to a payee. You do not ask for that one and you do not \
call ask_for_pin for it: read the action back as you always do, and once \
they agree the keypad appears by itself. If a result comes back saying a \
PIN is required, the money has not moved. Say that it is waiting on their \
PIN; do not say the payment is done.

A card reported lost or stolen is one event: freeze_card does the freezing \
and the replacement together, so never call order_card as well. Ask whether \
it was lost or stolen before you call it, and once it is frozen offer to go \
through the recent charges on it -- anything they do not recognise is a \
dispute, not a shrug.

Some things are not yours to settle. Fraud, a complaint, money trouble, or \
anything your tools cannot do -- hand those to a person with hand_off, say \
the reference out loud, and do not attempt the task yourself. If the customer \
asks for a human, call hand_off straight away and never talk them out of it.

If a tool returns an error, tell the customer what it says in your own words \
and offer what you can do instead.

You may also answer general questions -- about banking in general, or an \
everyday topic -- briefly, helpfully and in the same spoken style. A general \
answer never states an amount of money: amounts come only from this \
customer's accounts, through the tools.\
"""

# The language to answer in, sent with the customer's latest message. The rule
# above is not enough on its own, because the model follows the conversation
# rather than the customer: asked a spoken Urdu question it replied in Urdu
# written in Latin letters ("Checking account mein $2,483.50 hain."), for the
# English voice to read, and asked in English after two Urdu turns it replied
# in Urdu four times in six. Said again at the end of this prompt it was still
# ignored; next to the message it is about, it held 24 times in 24.
#
# In Urdu the model also reaches for its tools less: asked for a savings
# balance it looked nothing up three times in ten, and asked instead which
# bank the account was with, or for its number. The second sentence took
# that to ten in ten. Its wording is measured, not incidental: asking for
# tool summaries in Urdu as well cost the transfers their read-back.
LANGUAGE_NOTE = {
    "en": "Reply in English: the customer's latest message is in English.",
    "ur": (
        "Reply in Urdu, written in Urdu script, with every amount in digits and its "
        "dollar sign: the customer's latest message is in Urdu, even if it was typed "
        "in Latin letters. Look up what they ask with your tools, as you would in "
        "English; never ask them for a bank, an account number or a card number."
    ),
}


# Spoken when the model cannot be reached at all. The customer hears a
# sentence rather than meeting a 500.
UNAVAILABLE = "Sorry — I can't reach the service just now. Please try again in a moment."

# Spoken when the model returns nothing at all, or runs out of tool rounds.
NOTHING_SAID = "Sorry, I didn't catch that. Could you say it again?"

# Spoken when the customer asked for a person and the model carried on anyway.
# Assembled from the case the bank opened rather than written by the model,
# because the one thing in it that must be exactly right is the reference.
HANDED_OVER = (
    "Of course — I'm passing you to a colleague. "
    "Your reference is {reference}, and someone will pick this up shortly."
)

# Spoken when the model announced a lookup and never made one, even after the
# call was forced. Better than reading out a promise nothing is going to keep.
NOT_LOOKED_UP = "Sorry — I couldn't look that up just now. Could you ask me again?"

# Spoken when the guard suppresses a reply and no tool returned a figure to
# read back instead.
UNVERIFIED = "Let me check that properly — could you ask me again?"

# Spoken when money is about to move and only the PIN is missing.
PIN_NEEDED = "Enter your PIN to approve this."

# Spoken when the microphone heard nothing worth answering.
NOT_CAUGHT = "I didn't catch that. Could you say it again?"

# The sentences above, as the bot says them in Urdu. These are the bot's own
# words, not the model's, so nothing else would put them in the customer's
# language -- and an Urdu conversation that stalls into "Sorry, I didn't
# catch that" has stopped being an Urdu conversation.
_URDU: dict[str, str] = {
    UNAVAILABLE: (
        "معذرت، اس وقت سروس سے رابطہ نہیں ہو پا رہا۔ براہِ کرم تھوڑی دیر بعد دوبارہ کوشش کریں۔"
    ),
    NOTHING_SAID: "معذرت، میں سمجھ نہیں سکا۔ کیا آپ دوبارہ کہہ سکتے ہیں؟",
    NOT_CAUGHT: "میں سمجھ نہیں سکا۔ کیا آپ دوبارہ کہہ سکتے ہیں؟",
    HANDED_OVER: (
        "ضرور، میں آپ کو اپنے ایک ساتھی سے ملا رہا ہوں۔ "
        "آپ کا ریفرنس {reference} ہے، اور جلد ہی کوئی آپ سے رابطہ کرے گا۔"
    ),
    NOT_LOOKED_UP: "معذرت، میں ابھی یہ معلومات حاصل نہیں کر سکا۔ کیا آپ دوبارہ پوچھ سکتے ہیں؟",
    UNVERIFIED: "میں اسے ٹھیک سے دیکھنا چاہتا ہوں، کیا آپ دوبارہ پوچھ سکتے ہیں؟",
    PIN_NEEDED: "اس کی منظوری کے لیے اپنا پن درج کریں۔",
}


def in_language(sentence: str, language: str | None) -> str:
    """One of the bot's own sentences, in the customer's language.

    The English is the key, so a sentence with no Urdu version -- a detail
    from the bank, say -- is said as it is rather than not at all.
    """
    return _URDU.get(sentence, sentence) if language == "ur" else sentence
