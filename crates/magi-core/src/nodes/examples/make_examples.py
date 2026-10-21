"""Writes the example datasets a new text model starts with (the files beside
this script). They are original, short and general: plain English, everyday
conversation, step-by-step reasoning at three effort levels, using tools, and
naming conversations – and, from make_agent_data.py, thinking when it helps,
tools one after another, errors, small talk and longer answers. Run it again
after changing either:

    python crates/magi-core/src/nodes/examples/make_examples.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def obj(**properties):
    return {"type": "object", "properties": {k: {"type": v} for k, v in properties.items()},
            "required": list(properties)}


WEB = {"name": "web_search", "description": "Searches the web and returns short results.", "parameters": obj(query="string")}
FETCH = {"name": "fetch_page", "description": "Reads the text of a web page.", "parameters": obj(url="string")}
CALC = {"name": "calculator", "description": "Works out a math expression.", "parameters": obj(expression="string")}
WEATHER = {"name": "get_weather", "description": "Gets the current weather for a place.", "parameters": obj(location="string")}
TIME = {"name": "get_time", "description": "Gets the current date and time for a place.", "parameters": obj(location="string")}
NOTE = {"name": "save_note", "description": "Saves a short note for the user.", "parameters": obj(text="string")}
READ = {"name": "read_file", "description": "Reads a text file.", "parameters": obj(path="string")}
LIST = {"name": "list_files", "description": "Lists the files in a folder.", "parameters": obj(folder="string")}

# The tools the example model offers in MacroMagi (Text & Tools).
DEFAULT_TOOLS = [WEB, CALC, WEATHER, TIME]


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def result(value):
    return {"role": "tool", "content": json.dumps(value, ensure_ascii=False)}


def user(text):
    return {"role": "user", "content": text}


def say(text, thinking=None, calls=None):
    message = {"role": "assistant", "content": text}
    if thinking:
        message["thinking"] = thinking
    if calls:
        message["tool_calls"] = calls
    return message


# ---- plain English, to pretrain on --------------------------------------------------

LANGUAGE = """
A sentence is a group of words that says something complete. It starts with a capital letter and ends with a full stop, a question mark or an exclamation mark.

The sun rises in the morning and sets in the evening. In between, the sky is light. At night it is dark, and you can often see the moon and the stars.

Water can be a liquid, a solid or a gas. When it gets very cold, it freezes into ice. When it gets very hot, it boils and turns into steam.

A week has seven days: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday and Sunday. Many people work or go to school on the first five days and rest at the weekend.

A year has twelve months. January is the first month and December is the last. Most months have thirty or thirty-one days, and February has twenty-eight, or twenty-nine in a leap year.

There are sixty seconds in a minute, sixty minutes in an hour and twenty-four hours in a day.

Numbers help us count and measure. Adding puts numbers together, and taking away makes a number smaller. Multiplying is adding the same number many times, and dividing shares a number into equal parts.

An even number can be split into two equal whole parts, like 2, 4, 6 and 8. An odd number cannot, like 1, 3, 5 and 7.

Red, blue and yellow are colors. Mixing blue and yellow makes green. Mixing red and yellow makes orange. Mixing red and blue makes purple.

Plants need light, water and air to grow. Their leaves use sunlight to make food. Most plants grow from seeds.

Dogs, cats, horses and cows are animals that live with people. Lions, wolves and eagles are wild animals. Fish live in water and breathe with gills.

People eat food to have energy. Fruit and vegetables are good for the body. Drinking water helps the body work well.

When you meet someone, you can say "Hello" or "Good morning". When you leave, you can say "Goodbye" or "See you later". Saying "please" and "thank you" is polite.

A question asks for information. Questions often begin with who, what, where, when, why or how. "Where is the station?" asks for a place. "When does it open?" asks for a time.

A good answer is clear and short. It says the main point first. If something is not known, a good answer says so instead of guessing.

A computer follows instructions. A program is a list of instructions. A file holds information, such as text or a picture, and a folder holds files.

The internet connects computers around the world. A web page is a document on the internet, and a search engine helps people find web pages about a topic.

A map shows where places are. North is usually at the top. East is to the right, west is to the left and south is at the bottom.

Weather is what the air is like outside. It can be sunny, cloudy, rainy, windy or snowy. Temperature tells how hot or cold it is.

A recipe tells you how to cook something. It lists the ingredients first, then the steps in order. Following the steps one by one gives the best result.

To solve a problem, first understand what is asked. Then break it into small steps, do each step, and check the answer at the end.

Comparing things means finding how they are alike and how they are different. A cat and a dog are both pets, but a cat often climbs while a dog often fetches.

If one thing causes another, we say "because" or "so". It rained, so the ground is wet. The ground is wet because it rained.

Lists put things in order. A shopping list might say: bread, milk, eggs and apples. A to-do list says what to do and can be checked off one item at a time.

Money is used to buy things. Prices are written with a number and a currency, such as 5 dollars or 3 euros. A discount makes the price lower.

Being kind means helping others, listening carefully and speaking politely. A kind word can make someone's day better.

Sleep helps the body and the mind rest. Most adults need around seven to nine hours of sleep each night.

A tool is something that helps you do a job. A hammer drives nails, scissors cut paper, and a calculator works out sums quickly.

Reading every day makes reading easier. New words become familiar, and sentences become easier to follow.

Shapes have names. A circle is round. A triangle has three sides. A square has four equal sides, and a rectangle has four sides with opposite sides equal.

The area of a rectangle is its length times its width. The perimeter is the distance all the way around its edges.

A percent is a part of one hundred. Fifty percent is half, twenty-five percent is a quarter, and ten percent is one tenth.

Time zones mean that the clock shows different times in different places. When it is morning in one country, it can be evening in another.

Opposites are words with different meanings: hot and cold, big and small, fast and slow, open and closed, early and late.

A friendly message might say: "Hi Sam, thanks for your help yesterday. See you on Friday!" It greets the person, says the main point and ends warmly.

Instructions are clearer when each step starts with an action: open the box, take out the parts, then read the guide.

Summaries are short versions of longer texts. They keep the main ideas and leave out small details.

If you are not sure about a fact, you can look it up, ask someone who knows, or say that you are not sure.
"""


# ---- everyday conversation ----------------------------------------------------------

CHAT = [
    ("Hello!", "Hello! How can I help you today?"),
    ("Hi there", "Hi! What would you like to do?"),
    ("Good morning", "Good morning! What can I do for you?"),
    ("Thanks!", "You're welcome!"),
    ("Thank you for your help.", "Happy to help. Is there anything else?"),
    ("Who are you?", "I'm Magi, a small assistant made in MacroMagi. I can answer simple questions, think through problems and use tools."),
    ("What can you do?", "I can answer questions, explain simple ideas, work through small problems step by step, and use tools such as search when they are available."),
    ("How are you?", "I'm doing well, thank you. How can I help?"),
    ("Goodbye", "Goodbye! Have a good day."),
    ("What is the opposite of hot?", "Cold."),
    ("What is the opposite of early?", "Late."),
    ("How many days are in a week?", "Seven."),
    ("How many months are in a year?", "Twelve."),
    ("How many minutes are in an hour?", "Sixty."),
    ("What color do you get when you mix blue and yellow?", "Green."),
    ("What do plants need to grow?", "Light, water and air – and most also need soil."),
    ("What is a noun?", "A noun is a word for a person, place or thing, such as teacher, city or apple."),
    ("What is a verb?", "A verb is an action or state word, such as run, read or be."),
    ("What is an adjective?", "An adjective describes a noun, such as big, red or happy."),
    ("Explain what a computer program is.", "A program is a list of instructions a computer follows, step by step, to do a task."),
    ("What is the internet?", "The internet is a network that connects computers around the world so they can share information."),
    ("What does a search engine do?", "It finds web pages about a topic you type in."),
    ("Give me three fruits.", "Apple, banana and orange."),
    ("Name four shapes.", "Circle, triangle, square and rectangle."),
    ("Write a short greeting for a friend.", "Hi! It's great to hear from you. I hope you're having a lovely day."),
    ("Make this sentence shorter: I am writing to you today in order to ask whether you might be able to help me.", "Could you help me?"),
    ("Make this more polite: Give me the report.", "Could you please send me the report?"),
    ("Fix the grammar: she go to school every day.", "She goes to school every day."),
    ("Fix the spelling: I recieved the leter yesterday.", "I received the letter yesterday."),
    ("Summarize: The meeting moved from Monday to Wednesday because the room was booked. Everyone should bring their notes.", "The meeting is now on Wednesday; bring your notes."),
    ("Turn this into a list: eggs, milk, bread and butter", "- Eggs\n- Milk\n- Bread\n- Butter"),
    ("What is a good way to start solving a problem?", "Make sure you understand what is asked, then break it into small steps."),
    ("Is the sun a star?", "Yes. The sun is a star."),
    ("Do fish breathe air?", "Most fish breathe through gills, which take oxygen from water."),
    ("What does 'summary' mean?", "A summary is a short version of something longer that keeps the main ideas."),
    ("Tell me a fun fact.", "Octopuses have three hearts."),
    ("What should I do if I don't know an answer?", "Say you're not sure, then look it up or ask someone who knows."),
    ("What's the capital of the country I live in?", "I don't know where you live. Tell me the country and I'll help."),
    ("What time is it?", "I can't see a clock here. With a time tool I could check; otherwise your device's clock shows it."),
    ("Can you remember things between chats?", "No. I only see the current conversation."),
    ("Write one sentence about rain.", "Rain falls from clouds and helps plants grow."),
    ("Describe a cat in five words.", "Soft, curious, playful, quiet, independent."),
    ("Say yes or no: is ice cold?", "Yes."),
    ("What is 5 plus 7?", "12."),
    ("What is 9 times 3?", "27."),
    ("What is 20 divided by 4?", "5."),
    ("Spell the word 'friend'.", "F-R-I-E-N-D."),
    ("Answer in one word: what do bees make?", "Honey."),
]

CONVERSATIONS = [
    [user("Hi! Can you help me plan my morning?"),
     say("Of course. What do you need to get done, and when do you start?"),
     user("I start work at 9. I need to shower, eat and walk the dog."),
     say("Here's a simple plan:\n- 7:30 Walk the dog\n- 8:00 Shower\n- 8:20 Breakfast\n- 8:45 Leave for work")],
    [user("I want to learn a new word every day."),
     say("Great idea. Today's word: 'curious' – wanting to know or learn something."),
     user("Use it in a sentence."),
     say("The curious child asked how rainbows form.")],
    [user("Can you count to five?"),
     say("1, 2, 3, 4, 5."),
     user("Now backwards."),
     say("5, 4, 3, 2, 1.")],
    [user("What is a rectangle?"),
     say("A shape with four sides and four right angles; opposite sides are equal."),
     user("And a square?"),
     say("A rectangle whose four sides are all equal.")],
    [user("I'm feeling tired."),
     say("Sorry to hear that. A short break, a glass of water or some fresh air can help. Did you sleep well?"),
     user("Not really."),
     say("Then an early night could help most. Try to rest for seven to nine hours.")],
]


# ---- reasoning, at three effort levels ----------------------------------------------

# (question, answer, low, medium, high): the thinking at each effort.
REASONING = [
    ("Tom has 3 apples and buys 5 more. Then he eats 2. How many apples does he have now?", "6 apples.",
     "3 + 5 - 2 = 6.",
     "He starts with 3. Buying 5 more makes 3 + 5 = 8. Eating 2 leaves 8 - 2 = 6.",
     "Start: 3 apples. He buys 5 more: 3 + 5 = 8. He eats 2: 8 - 2 = 6. Check: 6 + 2 eaten = 8, and 8 - 5 bought = 3, the start. So 6."),
    ("Which is bigger, 0.7 or 0.65?", "0.7 is bigger.",
     "0.70 > 0.65.",
     "Write both with two decimals: 0.70 and 0.65. 70 hundredths is more than 65 hundredths, so 0.7 is bigger.",
     "Compare place by place. Tenths: 0.7 has 7, 0.65 has 6. 7 > 6, so 0.7 is bigger whatever comes after. Check with hundredths: 0.70 vs 0.65, and 70 > 65. Same answer."),
    ("A train leaves at 2:15 pm and the trip takes 50 minutes. When does it arrive?", "At 3:05 pm.",
     "2:15 + 50 min = 3:05.",
     "From 2:15, 45 minutes gets to 3:00. 5 minutes are left, so 3:05 pm.",
     "Split 50 minutes into 45 + 5. 2:15 + 45 minutes = 3:00. 3:00 + 5 minutes = 3:05. Check: from 2:15 to 3:05 is 45 + 5 = 50 minutes. It arrives at 3:05 pm."),
    ("A shirt costs $20 and is 25% off. What does it cost now?", "$15.",
     "25% of 20 is 5; 20 - 5 = 15.",
     "25% is a quarter. A quarter of $20 is $5. $20 - $5 = $15.",
     "25% means 25 out of 100, which is one quarter. 20 / 4 = 5, so the discount is $5. The new price is 20 - 5 = 15. Check: 15 is 75% of 20, because 20 x 0.75 = 15. So $15."),
    ("All cats are animals. Milo is a cat. Is Milo an animal?", "Yes, Milo is an animal.",
     "Cats are animals; Milo is a cat.",
     "Every cat is an animal. Milo is a cat, so Milo is one of the cats, and so an animal.",
     "Rule: every member of 'cats' is in 'animals'. Fact: Milo is in 'cats'. So Milo is in 'animals'. Nothing else is needed, and nothing says otherwise. Yes."),
    ("What comes next: 2, 4, 8, 16, ...?", "32.",
     "Each doubles: 16 x 2 = 32.",
     "2 to 4 is x2, 4 to 8 is x2, 8 to 16 is x2. So the next is 16 x 2 = 32.",
     "Look at the steps: 4/2 = 2, 8/4 = 2, 16/8 = 2. Each number is double the one before. Adding does not fit: the gaps are 2, 4, 8, not equal. So the next is 16 x 2 = 32."),
    ("Sara is older than Ben. Ben is older than Lia. Who is the youngest?", "Lia is the youngest.",
     "Sara > Ben > Lia.",
     "Sara is older than Ben, and Ben is older than Lia, so the order is Sara, Ben, Lia. Lia is youngest.",
     "Write it as an order by age: Sara > Ben (first fact), Ben > Lia (second fact). Together: Sara > Ben > Lia. The one at the end is the youngest: Lia. Check: Lia is younger than Ben, and Ben is younger than Sara, so no one is younger than Lia."),
    ("If today is Monday, what day will it be in 10 days?", "Thursday.",
     "10 = 7 + 3; Monday + 3 = Thursday.",
     "7 days later it is Monday again. 3 more days: Tuesday, Wednesday, Thursday.",
     "A week is 7 days, so after 7 days it is Monday again. 10 - 7 = 3 days remain. Monday + 1 = Tuesday, + 2 = Wednesday, + 3 = Thursday. Check by counting all 10: Tue, Wed, Thu, Fri, Sat, Sun, Mon, Tue, Wed, Thu. Thursday."),
    ("How many letters are in the word 'banana', and how many of them are 'a'?", "6 letters, 3 of them 'a'.",
     "b-a-n-a-n-a: 6 letters, 3 a's.",
     "Spell it: b, a, n, a, n, a. That is 6 letters. The a's are in places 2, 4 and 6: 3 of them.",
     "Write each letter with its place: 1 b, 2 a, 3 n, 4 a, 5 n, 6 a. Total: 6. The a's: places 2, 4, 6, so 3. The others: b once, n twice, and 1 + 2 + 3 = 6, which matches the total."),
    ("Which weighs more: a kilogram of feathers or a kilogram of stones?", "They weigh the same: one kilogram each.",
     "Both are 1 kg.",
     "Both are one kilogram. The stones are smaller, but the weight is the same.",
     "The question gives the weight of both: one kilogram. Feathers take up more space, which can make them seem lighter, but a kilogram is a kilogram. They weigh the same."),
    ("It takes 5 minutes to boil one egg. How long does it take to boil 3 eggs together in the same pot?", "5 minutes.",
     "Together, at once: still 5 minutes.",
     "The eggs boil at the same time in one pot, so it still takes 5 minutes, not 15.",
     "Boiling time depends on each egg being in hot water, not on how many there are. All 3 are in the pot together, so they cook at the same time. 15 would be right only if they were boiled one after another. So 5 minutes."),
    ("I have 12 cookies to share equally among 4 friends. How many does each friend get?", "3 cookies each.",
     "12 / 4 = 3.",
     "Sharing equally means dividing: 12 / 4 = 3.",
     "Equal sharing is division: 12 / 4. 4 x 3 = 12, so each gets 3. Check: 3 + 3 + 3 + 3 = 12, and nothing is left over."),
]

# Questions answered with medium thinking only.
MORE_REASONING = [
    ("Is 17 a prime number?", "Yes, 17 is prime.",
     "17 is not even. 17 / 3 is not whole, and 5 x 5 = 25 is already more than 17, so no number below it divides it. It is prime."),
    ("A rectangle is 6 cm long and 4 cm wide. What is its area?", "24 square centimetres.",
     "Area is length times width: 6 x 4 = 24, in square centimetres."),
    ("Put these numbers in order from smallest to largest: 9, 2, 7, 4.", "2, 4, 7, 9.",
     "The smallest is 2, then 4, then 7, and 9 is the largest."),
    ("Count the vowels in the word 'education'.", "5 vowels: e, u, a, i, o.",
     "Letters: e, d, u, c, a, t, i, o, n. Vowels are a, e, i, o, u: e, u, a, i, o. That is 5."),
    ("What is 15% of 80?", "12.",
     "10% of 80 is 8 and 5% is half of that, 4. 8 + 4 = 12."),
    ("Half of a number is 9. What is the number?", "18.",
     "If half is 9, the whole is 9 x 2 = 18."),
    ("How many minutes are there in 3 hours?", "180 minutes.",
     "One hour is 60 minutes, so 3 hours is 3 x 60 = 180."),
    ("If you have 2 red socks and 2 blue socks in a drawer, how many must you take out without looking to be sure of a matching pair?", "3 socks.",
     "With 2 you could get one of each color. The third sock must match one of them, since there are only two colors. So 3."),
    ("A box holds 6 eggs. How many boxes do you need for 20 eggs?", "4 boxes.",
     "20 / 6 = 3 with 2 left over. The 2 extra eggs need another box, so 3 + 1 = 4."),
    ("Anna reads 10 pages a day. How many days does she need for a 75-page book?", "8 days.",
     "75 / 10 = 7.5. She can't read half a day's pages as a separate day, so after 7 days 5 pages remain: day 8 finishes it."),
]


# ---- tools -------------------------------------------------------------------------

TOOLS = [
    # A search, then an answer from the result.
    {"tools": [WEB, CALC], "messages": [
        user("How many legs does a spider have?"),
        say("Spiders have eight legs.", thinking="I know this: spiders have eight legs. No tool is needed.")]},
    {"tools": [WEB, FETCH], "messages": [
        user("Find out what the boiling point of water is at sea level."),
        say("", thinking="The user asks me to look it up. I'll search.", calls=[call("web_search", query="boiling point of water at sea level")]),
        result({"results": [{"title": "Boiling point", "snippet": "At sea level, pure water boils at 100 °C (212 °F)."}]}),
        say("At sea level, water boils at 100 °C (212 °F).")]},
    {"tools": [WEATHER, TIME], "messages": [
        user("What's the weather like in Oslo right now?"),
        say("", thinking="I can't know the current weather; get_weather can.", calls=[call("get_weather", location="Oslo")]),
        result({"location": "Oslo", "temperature_c": 4, "conditions": "light rain", "wind_kmh": 12}),
        say("It's 4 °C in Oslo with light rain and a light wind. Take an umbrella.")]},
    {"tools": [WEATHER], "messages": [
        user("Is it warm in Madrid today?"),
        say("", calls=[call("get_weather", location="Madrid")]),
        result({"location": "Madrid", "temperature_c": 27, "conditions": "sunny"}),
        say("Yes – it's sunny and 27 °C in Madrid.")]},
    {"tools": [TIME, WEATHER], "messages": [
        user("What time is it in Tokyo?"),
        say("", calls=[call("get_time", location="Tokyo")]),
        result({"location": "Tokyo", "time": "21:40", "date": "Tuesday"}),
        say("It's 21:40 (9:40 pm) on Tuesday in Tokyo.")]},
    {"tools": [CALC], "messages": [
        user("What is 487 times 26?"),
        say("", thinking="Big multiplication: the calculator is safer.", calls=[call("calculator", expression="487 * 26")]),
        result({"result": 12662}),
        say("487 × 26 = 12,662.")]},
    {"tools": [CALC, WEB], "messages": [
        user("What is 2 + 2?"),
        say("4.", thinking="Simple enough to do myself.")]},
    {"tools": [CALC], "messages": [
        user("If I save $35 a week, how much will I have after a year?"),
        say("", thinking="A year has 52 weeks: 35 x 52.", calls=[call("calculator", expression="35 * 52")]),
        result({"result": 1820}),
        say("After 52 weeks you'll have $1,820.")]},
    # Two tools at once.
    {"tools": [WEATHER, TIME], "messages": [
        user("What's the time and weather in Paris?"),
        say("", calls=[call("get_time", location="Paris"), call("get_weather", location="Paris")]),
        result({"location": "Paris", "time": "14:05", "date": "Saturday"}),
        result({"location": "Paris", "temperature_c": 18, "conditions": "cloudy"}),
        say("In Paris it's 14:05 on Saturday, 18 °C and cloudy.")]},
    # Search, then read a page.
    {"tools": [WEB, FETCH], "messages": [
        user("Look up how to make a cup of green tea."),
        say("", calls=[call("web_search", query="how to make green tea")]),
        result({"results": [{"title": "Brewing green tea", "url": "https://example.com/green-tea", "snippet": "Use water below boiling..."}]}),
        say("", thinking="The snippet is short; I'll read the page.", calls=[call("fetch_page", url="https://example.com/green-tea")]),
        result({"text": "Heat water to about 80 °C. Add one teaspoon of leaves per cup. Steep for 2 to 3 minutes, then strain."}),
        say("1. Heat water to about 80 °C (not boiling).\n2. Add one teaspoon of green tea per cup.\n3. Steep for 2–3 minutes, then strain.")]},
    # A tool fails.
    {"tools": [WEATHER], "messages": [
        user("What's the weather in Atlantis?"),
        say("", calls=[call("get_weather", location="Atlantis")]),
        result({"error": "location not found"}),
        say("I couldn't find weather for Atlantis – the weather service doesn't know that place. Is there another city you meant?")]},
    {"tools": [WEB], "messages": [
        user("Search for the opening hours of the city library."),
        say("", calls=[call("web_search", query="city library opening hours")]),
        result({"results": []}),
        say("The search found nothing. Which city's library do you mean? With the name I can search again.")]},
    # No tool fits.
    {"tools": [CALC], "messages": [
        user("What's the weather in Rome?"),
        say("I can't check the weather – I only have a calculator here. A weather app or website will have it.")]},
    {"tools": [WEATHER], "messages": [
        user("Send an email to my boss saying I'll be late."),
        say("I can't send emails with the tools I have. Here's a message you can send:\n\nHi, I'm running late this morning and will be in as soon as I can. Sorry for the trouble.")]},
    # Notes and files.
    {"tools": [NOTE], "messages": [
        user("Remind me to buy milk."),
        say("", calls=[call("save_note", text="Buy milk")]),
        result({"saved": True}),
        say("Saved a note: buy milk.")]},
    {"tools": [LIST, READ], "messages": [
        user("What's in my notes folder?"),
        say("", calls=[call("list_files", folder="notes")]),
        result({"files": ["shopping.txt", "ideas.txt"]}),
        say("Your notes folder has two files: shopping.txt and ideas.txt.")]},
    {"tools": [LIST, READ], "messages": [
        user("Read my shopping list."),
        say("", thinking="It is probably notes/shopping.txt.", calls=[call("read_file", path="notes/shopping.txt")]),
        result({"text": "eggs\nrice\ntomatoes"}),
        say("Your shopping list: eggs, rice and tomatoes.")]},
    # Asks before guessing.
    {"tools": [WEATHER], "messages": [
        user("Will it rain today?"),
        say("Where are you? Tell me the town and I'll check the weather.")]},
    {"tools": [WEATHER], "messages": [
        user("Will it rain today?"),
        say("Where are you? Tell me the town and I'll check the weather."),
        user("Dublin"),
        say("", calls=[call("get_weather", location="Dublin")]),
        result({"location": "Dublin", "temperature_c": 11, "conditions": "showers"}),
        say("Yes – showers are expected in Dublin today, 11 °C.")]},
    # A follow-up uses earlier results.
    {"tools": [WEB, CALC], "messages": [
        user("Find how many meters are in a mile."),
        say("", calls=[call("web_search", query="meters in a mile")]),
        result({"results": [{"title": "Mile", "snippet": "One mile is 1,609.344 meters."}]}),
        say("One mile is about 1,609 meters (1,609.344)."),
        user("So how many meters are 3 miles?"),
        say("", calls=[call("calculator", expression="3 * 1609.344")]),
        result({"result": 4828.032}),
        say("3 miles is about 4,828 meters.")]},
    {"tools": [WEB], "messages": [
        user("Who are you and what tools do you have?"),
        say("I'm Magi, a small assistant. Here I can search the web with web_search.")]},
    {"tools": [WEB, WEATHER, CALC, TIME], "messages": [
        user("Hello!"),
        say("Hello! I can search the web, check the weather and time, and do math. What do you need?")]},
    {"tools": [WEB, WEATHER, CALC, TIME], "messages": [
        user("What is 15 + 27?"),
        say("42.")]},
    {"tools": [WEB, WEATHER, CALC, TIME], "messages": [
        user("How cold is it in Helsinki?"),
        say("", calls=[call("get_weather", location="Helsinki")]),
        result({"location": "Helsinki", "temperature_c": -6, "conditions": "clear"}),
        say("It's -6 °C and clear in Helsinki – very cold.")]},
    {"tools": [WEB, WEATHER, CALC, TIME], "messages": [
        user("Search the web for easy dinner ideas."),
        say("", calls=[call("web_search", query="easy dinner ideas")]),
        result({"results": [{"title": "Quick dinners", "snippet": "Pasta with tomato sauce, vegetable stir-fry, omelette, rice bowls."}]}),
        say("Some easy dinners: pasta with tomato sauce, a vegetable stir-fry, an omelette, or a rice bowl.")]},
]


# With all four default tools listed, as MacroMagi asks: each kind of request, several ways.
CITIES = [("Oslo", 4, "light rain"), ("Madrid", 27, "sunny"), ("Cairo", 33, "clear"), ("Toronto", -2, "snow"),
          ("Lima", 19, "cloudy"), ("Seoul", 12, "windy")]
WEATHER_ASKS = ["What's the weather in {}?", "Is it cold in {} today?", "Will it rain in {}?", "How warm is it in {}?"]
TIME_ASKS = ["What time is it in {}?", "What's the date in {} right now?"]
CALCS = [("What is 487 times 26?", "487 * 26", 12662, "487 × 26 = 12,662."),
         ("Work out 1250 divided by 25.", "1250 / 25", 50, "1250 ÷ 25 = 50."),
         ("What's 18% of 340?", "0.18 * 340", 61.2, "18% of 340 is 61.2."),
         ("Add 3,456 and 7,891.", "3456 + 7891", 11347, "3,456 + 7,891 = 11,347."),
         ("How much is 99 squared?", "99 ** 2", 9801, "99² = 9,801."),
         ("What is 7.5 times 12.4?", "7.5 * 12.4", 93.0, "7.5 × 12.4 = 93.")]
SEARCHES = [("Search for the tallest mountain in Africa.", "tallest mountain in Africa",
             "Kilimanjaro, in Tanzania, is the tallest mountain in Africa at 5,895 m.",
             "The tallest mountain in Africa is Kilimanjaro, in Tanzania – 5,895 m."),
            ("Look up how long it takes to hard-boil an egg.", "how long to hard boil an egg",
             "Hard-boiled eggs take about 10 to 12 minutes in boiling water.",
             "About 10–12 minutes in boiling water."),
            ("Find out what a group of crows is called.", "group of crows name",
             "A group of crows is called a murder.", "A group of crows is called a murder."),
            ("What's a good stretch for a stiff neck? Search it.", "stretch for stiff neck",
             "Slowly tilt your ear toward your shoulder and hold for 20 seconds, then switch sides.",
             "Try a side tilt: ear toward shoulder, hold 20 seconds, then switch sides.")]
for n, (city, temp, conditions) in enumerate(CITIES):
    ask = WEATHER_ASKS[n % len(WEATHER_ASKS)].format(city)
    rain = "rain" in conditions or "snow" in conditions
    reply = "It's %d °C and %s in %s." % (temp, conditions, city)
    if ask.startswith("Will it rain"):
        reply = ("Yes – " if rain else "No – ") + reply[0].lower() + reply[1:]
    TOOLS.append({"tools": DEFAULT_TOOLS, "messages": [
        user(ask),
        say("", thinking="Current weather: get_weather." if n % 2 else None, calls=[call("get_weather", location=city)]),
        result({"location": city, "temperature_c": temp, "conditions": conditions}),
        say(reply)]})
for n, (city, _, _) in enumerate(CITIES[:4]):
    clock = ["08:15", "13:50", "19:05", "23:30"][n]
    day = ["Monday", "Wednesday", "Friday", "Sunday"][n]
    TOOLS.append({"tools": DEFAULT_TOOLS, "messages": [
        user(TIME_ASKS[n % 2].format(city)),
        say("", calls=[call("get_time", location=city)]),
        result({"location": city, "time": clock, "date": day}),
        say("In %s it's %s on %s." % (city, clock, day))]})
for n, (ask, expression, value, reply) in enumerate(CALCS):
    TOOLS.append({"tools": DEFAULT_TOOLS, "messages": [
        user(ask),
        say("", thinking="Math: the calculator." if n % 2 == 0 else None, calls=[call("calculator", expression=expression)]),
        result({"result": value}),
        say(reply)]})
for n, (ask, query, snippet, reply) in enumerate(SEARCHES):
    TOOLS.append({"tools": DEFAULT_TOOLS, "messages": [
        user(ask),
        say("", thinking="I should look this up." if n % 2 else None, calls=[call("web_search", query=query)]),
        result({"results": [{"title": query.capitalize(), "snippet": snippet}]}),
        say(reply)]})
# More of each, varied, so the model learns which tool fits a request rather than the requests themselves.
import random  # noqa: E402

rng = random.Random(7)
PLACES = ["Paris", "Berlin", "Tokyo", "Nairobi", "Sydney", "Chicago", "Mumbai", "Rome", "Dublin", "Mexico City",
          "Bangkok", "Vienna", "Lagos", "Denver", "Lisbon", "Hanoi", "Prague", "Quito", "Perth", "Reykjavik"]
CONDITIONS = ["sunny", "cloudy", "light rain", "heavy rain", "snow", "fog", "clear", "windy", "showers"]
WEATHER_WAYS = ["What's the weather in {}?", "Is it raining in {}?", "How hot is it in {} right now?",
                "Do I need an umbrella in {} today?", "weather {}", "Tell me the temperature in {}.",
                "Is it sunny in {}?", "How cold is it in {}?"]
TIME_WAYS = ["What time is it in {}?", "Current time in {}?", "What day is it in {}?", "time in {}",
             "Is it morning or evening in {} now?"]
SEARCH_TOPICS = [("how to fix a squeaky door", "Spray the hinge with a little oil, then open and close the door a few times.",
                  "Put a little oil on the hinge and swing the door a few times."),
                 ("how many hours puppies sleep", "Puppies sleep 18 to 20 hours a day.", "Puppies sleep about 18–20 hours a day."),
                 ("easy bread recipe", "Mix flour, water, yeast and salt; let it rise an hour; bake at 220 °C for 30 minutes.",
                  "Mix flour, water, yeast and salt, let it rise for an hour, then bake at 220 °C for about 30 minutes."),
                 ("why leaves change color", "Leaves change color in autumn as green chlorophyll breaks down.",
                  "As days get colder, the green chlorophyll breaks down and other colors show."),
                 ("best time to water plants", "Water plants early in the morning so less water evaporates.",
                  "Early morning – less water evaporates."),
                 ("how to remove a coffee stain", "Blot it, rinse with cold water, then wash with a little soap.",
                  "Blot it, rinse with cold water, then wash with a little soap."),
                 ("what do pandas eat", "Pandas eat mostly bamboo.", "Pandas eat mostly bamboo."),
                 ("how far a marathon is", "A marathon is 42.195 km (26.2 miles).", "A marathon is 42.195 km (26.2 miles).")]
SEARCH_WAYS = ["Search the web for {}.", "Look up {}.", "Can you find {}?", "Google {}", "Find out {}."]
OPS = [("+", "plus", lambda a, b: a + b), ("-", "minus", lambda a, b: a - b), ("*", "times", lambda a, b: a * b),
       ("/", "divided by", lambda a, b: a / b)]
CALC_WAYS = ["What is {a} {word} {b}?", "Calculate {a} {sym} {b}.", "{a} {sym} {b} = ?", "Work out {a} {word} {b} for me.",
             "How much is {a} {word} {b}?"]


def number(value):
    return int(value) if float(value).is_integer() else round(value, 4)


for n in range(32):
    city = rng.choice(PLACES)
    temp, conditions = rng.randint(-8, 36), rng.choice(CONDITIONS)
    TOOLS.append({"tools": DEFAULT_TOOLS, "messages": [
        user(rng.choice(WEATHER_WAYS).format(city)),
        say("", thinking="This needs the current weather." if n % 3 == 0 else None,
            calls=[call("get_weather", location=city)]),
        result({"location": city, "temperature_c": temp, "conditions": conditions}),
        say("In %s it's %d °C with %s." % (city, temp, conditions) if conditions != "clear" and conditions != "sunny"
            else "It's %s and %d °C in %s." % (conditions, temp, city))]})
for n in range(16):
    city = rng.choice(PLACES)
    hour, minute = rng.randint(0, 23), rng.choice([0, 5, 10, 20, 30, 45, 55])
    day = rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    clock = "%02d:%02d" % (hour, minute)
    part = "morning" if 5 <= hour < 12 else "afternoon" if 12 <= hour < 18 else "evening" if 18 <= hour < 22 else "night"
    TOOLS.append({"tools": DEFAULT_TOOLS, "messages": [
        user(rng.choice(TIME_WAYS).format(city)),
        say("", calls=[call("get_time", location=city)]),
        result({"location": city, "time": clock, "date": day}),
        say("It's %s on %s in %s – %s there." % (clock, day, city, part))]})
for n in range(32):
    sym, word, fn = rng.choice(OPS)
    a, b = rng.randint(12, 9999), rng.randint(2, 999)
    value = number(fn(a, b))
    ask = rng.choice(CALC_WAYS).format(a=a, b=b, word=word, sym=sym)
    TOOLS.append({"tools": DEFAULT_TOOLS, "messages": [
        user(ask),
        say("", thinking="Arithmetic: the calculator is exact." if n % 3 == 0 else None,
            calls=[call("calculator", expression="%d %s %d" % (a, sym, b))]),
        result({"result": value}),
        say("%d %s %d = %s." % (a, {"*": "×", "/": "÷"}.get(sym, sym), b, "{:,}".format(value)))]})
for n in range(16):
    query, snippet, reply = SEARCH_TOPICS[n % len(SEARCH_TOPICS)]
    TOOLS.append({"tools": DEFAULT_TOOLS, "messages": [
        user(rng.choice(SEARCH_WAYS).format(query)),
        say("", thinking="I'll search for it." if n % 3 == 0 else None, calls=[call("web_search", query=query)]),
        result({"results": [{"title": query.capitalize(), "snippet": snippet}]}),
        say(reply)]})

# Answered straight away, tools or not.
for ask, reply in [("What is 6 times 7?", "42."), ("Thanks!", "You're welcome!"), ("What color is the sky on a clear day?", "Blue."),
                   ("Who are you?", "I'm Magi, a small assistant. I can search the web, check the weather and time, and do math."),
                   ("Say hello in French.", "Bonjour!")]:
    TOOLS.append({"tools": DEFAULT_TOOLS, "messages": [user(ask), say(reply)]})


# ---- naming a conversation (as chat programs ask) -----------------------------------

TITLE_JSON = ("Based on the conversation above, can you please come up with a 2-5 word title for this conversation? "
              "Reply in JSON format: {\"title\": \"your generated title here...\"}. Use no other fields other than "
              "'title'. Do not wrap the output in backticks. Thanks.")
TITLE_TAGS = ("Based on the conversation above, can you please come up with a 2-5 word title for this conversation? "
              "Put your answer in <title> tags, like this: <title>Your Title Here</title>. Do not explain anything. "
              "Just return the title in the specified format.")
TITLE_PLAIN = "Write a short title (2-5 words) for this conversation. Reply with the title only."

TITLES = [
    ("Hello!", "Hello! How can I help you today?", "Friendly Greeting"),
    ("What's the weather in Oslo?", "It's 4 °C with light rain in Oslo.", "Weather in Oslo"),
    ("How do I make green tea?", "Heat water to 80 °C, steep the leaves 2–3 minutes, then strain.", "Making Green Tea"),
    ("What is 487 times 26?", "487 × 26 = 12,662.", "Multiplication Help"),
    ("Can you help me plan my morning?", "Sure: walk the dog, shower, eat, then leave at 8:45.", "Morning Routine Plan"),
    ("Fix the grammar: she go to school.", "She goes to school.", "Grammar Fix"),
    ("Is 17 a prime number?", "Yes, 17 is prime.", "Prime Number Question"),
    ("Give me easy dinner ideas.", "Pasta, stir-fry, an omelette or a rice bowl.", "Easy Dinner Ideas"),
    ("What time is it in Tokyo?", "It's 21:40 in Tokyo.", "Time in Tokyo"),
    ("Tell me a fun fact.", "Octopuses have three hearts.", "Fun Animal Fact"),
    ("How do I say thank you politely?", "You can say: Thank you very much, I really appreciate it.", "Polite Thank You"),
    ("Remind me to buy milk.", "Saved a note: buy milk.", "Milk Reminder"),
]


# ---- answers not to give ------------------------------------------------------------

AVOID = [
    # Making up what a tool would say.
    {"tools": [WEATHER], "prompt": "What's the weather in Oslo?", "rejected": "It's sunny and 25 °C in Oslo."},
    {"tools": [TIME], "prompt": "What time is it in Lima?", "rejected": "It's 3:00 pm in Lima."},
    # Tools that are not offered, or calls written wrong.
    {"tools": [CALC], "prompt": "Search the web for cheap flights.",
     "rejected": "<tool_call>{\"name\": \"web_search\", \"arguments\": {\"query\": \"cheap flights\"}}</tool_call>"},
    {"tools": [WEB], "prompt": "Look up the size of the moon.",
     "rejected": "<tool_call>{\"name\": \"web_search\", \"query\": \"size of the moon\"}</tool_call>"},
    {"tools": [WEB], "prompt": "Look up the size of the moon.", "rejected": "<tool_call>web_search(size of the moon)</tool_call>"},
    {"prompt": "Hello!", "rejected": "<tool_call>{\"name\": \"type\", \"arguments\": {\"text\": \"Hello!\"}}</tool_call>"},
    # Titles that are not titles.
    {"prompt": "Hello!\n\n" + TITLE_JSON, "rejected": "{\"title\": \"type\"}"},
    {"prompt": "What's 2 + 2?\n\n" + TITLE_JSON, "rejected": "type"},
    # Repeating, or saying nothing.
    {"prompt": "Tell me about cats.", "rejected": "Cats cats cats cats cats cats cats cats."},
    {"prompt": "What is 5 plus 7?", "rejected": "5 plus 7 is 5 plus 7 is 5 plus 7 is"},
    {"prompt": "Who are you?", "rejected": "Who are you?"},
]


# ---- simple facts in sentences, to pretrain on -------------------------------------

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


def facts():
    import random
    rng = random.Random(11)
    out = []
    for _ in range(120):
        a, b = rng.randint(0, 10), rng.randint(0, 10)
        kind = rng.randrange(4)
        if kind == 0:
            out.append("%s plus %s is %s. In numbers, %d + %d = %d." % (NUMBER_WORDS[a].capitalize(), NUMBER_WORDS[b], NUMBER_WORDS[a + b] if a + b <= 20 else a + b, a, b, a + b))
        elif kind == 1 and a >= b:
            out.append("%s minus %s is %s, because %d - %d = %d." % (NUMBER_WORDS[a].capitalize(), NUMBER_WORDS[b], NUMBER_WORDS[a - b], a, b, a - b))
        elif kind == 2:
            out.append("%d is %s than %d." % (a, "bigger" if a > b else "smaller" if a < b else "the same as", b) if a != b else "%d is equal to %d." % (a, b))
        else:
            out.append("%d times %d is %d." % (a, b, a * b))
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for i, day in enumerate(days):
        out.append("The day after %s is %s, and the day before it is %s." % (day, days[(i + 1) % 7], days[(i - 1) % 7]))
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    for i, month in enumerate(months):
        out.append("%s is month number %d. The month after it is %s." % (month, i + 1, months[(i + 1) % 12]))
    pairs = [("hot", "cold"), ("big", "small"), ("fast", "slow"), ("open", "closed"), ("early", "late"),
             ("light", "dark"), ("up", "down"), ("full", "empty"), ("happy", "sad"), ("easy", "hard")]
    for a, b in pairs:
        out.append("The opposite of %s is %s, and the opposite of %s is %s." % (a, b, b, a))
    return out


# ---- knowing what it knows ---------------------------------------------------------
#
# A small model knows little. These teach it to act well anyway: greet back and
# keep small talk short (with a short thought, so a greeting is never taken for a
# problem to work out), say plainly when it does not know, search when a search
# tool is offered instead of guessing, and ask when a request is unclear.

SMALL_TALK = [
    (["Hello", "hello", "Hi", "hi", "Hey", "hey there", "Hello!", "Hi!", "Heya", "hiya", "Yo", "Hello Magi"],
     "Someone greets me. I greet back and offer help.",
     ["Hello! How can I help?", "Hi! What can I do for you?", "Hey! What would you like to do?"]),
    (["Good morning", "good morning!", "Morning"],
     "A morning greeting. I answer in kind.", ["Good morning! What can I do for you today?"]),
    (["Good evening", "Good night"],
     "An evening greeting. I answer in kind.", ["Good evening! How can I help?", "Good night! Sleep well."]),
    (["How are you?", "how are you doing", "How's it going?", "What's up?", "How are things?"],
     "Small talk about how I am. A short, friendly answer, then back to them.",
     ["I'm doing well, thanks. How can I help?", "All good here. What can I do for you?"]),
    (["Thanks", "thank you", "Thanks a lot!", "Thank you so much", "thx", "Cheers"],
     "They thank me. I say they're welcome.", ["You're welcome!", "Happy to help!", "Any time."]),
    (["Bye", "Goodbye", "See you later", "bye!", "I have to go"],
     "They are leaving. A short goodbye.", ["Goodbye! Have a good day.", "See you later!", "Bye! Take care."]),
    (["Who are you?", "What are you?", "What's your name?", "Are you a person?", "Are you an AI?"],
     "They ask about me. I'm Magi, a small AI assistant.",
     ["I'm Magi, a small AI assistant made in MacroMagi. I can chat, work through simple problems and use tools.",
      "I'm Magi, an AI assistant. I'm small, so I know only a little, but I can use tools like search."]),
    (["What can you do?", "How can you help me?", "What are you good at?"],
     "They ask what I can do. I list it briefly and honestly.",
     ["I can chat, explain simple things, work through small problems step by step, write a little Python, and use tools such as search and a calculator when they're available. I'm small, so for facts I often need to look things up."]),
    (["Are you smart?", "Do you know everything?", "How much do you know?"],
     "They ask how much I know. I'm small: I should be honest about it.",
     ["Not everything – I'm a small model, so I know only basic things. When I'm not sure, I'll say so or look it up."]),
    (["ok", "OK", "okay", "Alright", "Cool", "Nice", "Great"],
     "A short acknowledgement. Nothing to work out; I keep it short.",
     ["Great! Anything else?", "Alright. Let me know if you need anything."]),
    (["lol", "haha", ":)", "😂"],
     "They laugh. A light, short reply.", ["Glad that made you smile!", "Ha! What else can I do for you?"]),
    (["I'm bored", "I'm tired", "I'm sad"],
     "They share how they feel. A kind, short reply.",
     ["Sorry to hear that. Want to talk about it, or should I suggest something to do?"]),
]

UNCLEAR = ["asdfgh", "jkjkjk", "??", "hmm", "the thing", "do it", "blue seven run", "what about that one", "x", "...",
           "can you", "and then?"]
UNCLEAR_REPLIES = ["I'm not sure what you mean. Could you say a bit more?",
                   "Sorry, I didn't understand that. What would you like me to do?",
                   "Could you tell me a little more about what you need?"]

# Facts a small model cannot know, or know reliably: specific, recent, personal or made up.
UNKNOWN_FACTS = [
    ("What is the population of Ulaanbaatar?", "population of Ulaanbaatar", "About 1.7 million people live in Ulaanbaatar."),
    ("Who won the 2018 Tour de France?", "2018 Tour de France winner", "Geraint Thomas won the 2018 Tour de France."),
    ("How tall is the Burj Khalifa?", "Burj Khalifa height", "The Burj Khalifa is 828 metres tall."),
    ("When was the University of Oslo founded?", "University of Oslo founded", "It was founded in 1811."),
    ("What is the price of gold today?", "gold price today", "Gold is trading at about 2,350 dollars an ounce today."),
    ("What's the latest news about electric cars?", "latest news electric cars", "The top results are about new, cheaper models and more charging stations."),
    ("Who is the mayor of Lyon?", "mayor of Lyon", "Grégory Doucet is the mayor of Lyon."),
    ("How many moons does Neptune have?", "how many moons does Neptune have", "Neptune has 16 known moons."),
    ("What year did the Berlin Wall fall?", "year Berlin Wall fell", "The Berlin Wall fell in 1989."),
    ("What is the deepest lake in Africa?", "deepest lake in Africa", "Lake Tanganyika is the deepest lake in Africa."),
    ("Who wrote the novel Things Fall Apart?", "Things Fall Apart author", "Chinua Achebe wrote Things Fall Apart."),
    ("What is the boiling point of ethanol?", "boiling point of ethanol", "Ethanol boils at about 78 °C."),
    ("How long is the Danube?", "length of the Danube river", "The Danube is about 2,850 km long."),
    ("Which company makes the Model Y?", "Model Y car maker", "Tesla makes the Model Y."),
    ("What is the speed of sound in water?", "speed of sound in water", "Sound travels about 1,480 metres a second in water."),
    ("When is the next full moon?", "next full moon date", "The next full moon is in 9 days."),
    ("What's the score of the game last night?", "score of last night's game", "The search found several games last night; which team do you mean?"),
    ("Who invented the telephone?", "who invented the telephone", "Alexander Graham Bell is credited with inventing the telephone in 1876."),
    ("What is the GDP of Kenya?", "GDP of Kenya", "Kenya's GDP is about 110 billion dollars."),
    ("How many people speak Swahili?", "number of Swahili speakers", "Around 200 million people speak Swahili."),
]
NO_TOOL_UNKNOWN = [
    "I don't know that for sure, and I can't look it up right now. I'd rather not guess.",
    "I'm not sure. I'm a small model and don't have that fact; a search would find it.",
    "I don't know. If you give me a search tool, I can look it up.",
]
UNKNOWABLE = [
    ("What am I thinking right now?", "I can't know what you're thinking – you'll have to tell me!"),
    ("How old is my neighbour?", "I don't know your neighbour, so I can't tell. You could ask them."),
    ("What's my name?", "I don't know your name unless you tell me."),
    ("What did I have for breakfast?", "I can't know that – I only see what you write here."),
    ("What's the password to my email?", "I don't know it, and I shouldn't. If you've forgotten it, use the email service's reset option."),
    ("Where did I leave my keys?", "I can't know that. Try retracing your steps: the last place you used them, pockets, bags and the door."),
    ("What is the capital of Zandoria?", "I don't know a country called Zandoria. Did you mean another name?"),
    ("Who won the match tomorrow?", "Nobody knows yet – it hasn't happened."),
    ("What will the weather be next year?", "Nobody can know that far ahead. I can check the forecast for the next few days if you like."),
]


def knowing():
    samples = []
    for asks, thought, replies in SMALL_TALK:
        for n, ask in enumerate(asks):
            reply = replies[n % len(replies)]
            samples.append({"prompt": ask, "thinking": thought, "response": reply})
            if n % 2 == 0:
                # With tools offered, small talk still needs none.
                samples.append({"tools": DEFAULT_TOOLS, "prompt": ask, "thinking": thought + " No tool is needed.",
                                "response": reply})
    for n, ask in enumerate(UNCLEAR):
        samples.append({"prompt": ask, "thinking": "I can't tell what they want. I'll ask, not guess.",
                        "response": UNCLEAR_REPLIES[n % len(UNCLEAR_REPLIES)]})
    for n, (ask, query, found) in enumerate(UNKNOWN_FACTS):
        # Without a search tool: say so, do not make it up.
        samples.append({"tools": [], "prompt": ask,
                        "thinking": "A specific fact. I don't know it reliably, and I have no search tool.",
                        "response": NO_TOOL_UNKNOWN[n % len(NO_TOOL_UNKNOWN)]})
        # With one: search, then answer from the result.
        samples.append({"tools": DEFAULT_TOOLS, "messages": [
            user(ask),
            say("", "A specific fact I don't know reliably. I'll search instead of guessing.",
                [call("web_search", query=query)]),
            result({"results": [{"title": query.capitalize(), "snippet": found}]}),
            say(found),
        ]})
    for ask, reply in UNKNOWABLE:
        samples.append({"tools": [], "prompt": ask, "thinking": "I have no way to know this. I say so plainly.",
                        "response": reply})
        samples.append({"tools": DEFAULT_TOOLS, "prompt": ask,
                        "thinking": "No tool can tell me this. I say so plainly.", "response": reply})
    return samples


# ---- compaction: a long conversation, kept short -------------------------------------
#
# When a chat grows past what the model can read, MacroMagi asks the model to
# write the earlier part short and goes on with that. Each sample is a
# conversation written as "User: … / Magi: …" lines and what to keep of it:
# who the user is, what they want, and the facts found – not the small talk.

# Many names, places, likes and tasks, so that the only way to write a chat short
# is to copy what it says – not to remember a few names from the samples.
NAMES = ["Sam", "Lena", "Omar", "Mia", "Kenji", "Ana", "Tom", "Priya", "Luca", "Zoe", "Farid", "Hana", "Diego",
         "Ingrid", "Kwame", "Aiko", "Mateus", "Freya", "Rashid", "Elif", "Joon", "Amara", "Pavel", "Sofia", "Tariq",
         "Noor", "Ewan", "Chiara", "Bilal", "Yara", "Henrik", "Lucía", "Femi", "Mei", "Arjun", "Greta", "Tomás",
         "Leila", "Kofi", "Sakura", "Nikolai", "Inés", "Dmitri", "Asha", "Bruno", "Olga", "Hugo", "Nadia", "Rafael",
         "Keiko", "Jonah", "Selin", "Marek", "Ayla", "Theo", "Rosa", "Imran", "Clara", "Viktor", "Esther", "Ravi",
         "Wanjiru", "Oskar", "Lina", "Emeka", "Paula", "Haruto", "Mira", "Andrés", "Fatima", "Jakob", "Ines", "Cyrus",
         "Daria", "Malik", "Nora", "Seo-yeon", "Gabriel", "Thandi", "Ilya", "Beatriz", "Kai", "Layla", "Ruben"]
LIKES = ["tea", "jazz", "hiking", "chess", "cooking", "old films", "cycling", "painting", "gardening", "photography",
         "knitting", "rock climbing", "board games", "poetry", "baking bread", "birdwatching", "swimming", "sci-fi books",
         "pottery", "running", "camping", "tango", "sushi", "astronomy", "woodwork", "yoga", "football", "the piano",
         "crosswords", "travel vlogs", "sourdough", "skateboarding", "ceramics", "surfing", "opera", "video games"]
TRIP_CITIES = ["Rome", "Tokyo", "Lisbon", "Oslo", "Cairo", "Toronto", "Vienna", "Seoul", "Lima", "Hanoi", "Dublin",
               "Marrakesh", "Kyoto", "Prague", "Nairobi", "Reykjavik", "Mexico City", "Istanbul", "Sydney", "Bergen",
               "Porto", "Krakow", "Bali", "Cape Town", "Montreal", "Edinburgh", "Buenos Aires", "Athens", "Hamburg"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
TASKS = ["call the dentist", "buy milk", "send the report", "water the plants", "book the train tickets",
         "pay the electricity bill", "call the bank", "renew my passport", "email Professor Lee", "pick up the parcel",
         "fix the bike tyre", "book a haircut", "return the library books", "order new glasses", "back up my laptop",
         "buy a birthday present for Jo", "cancel the gym membership", "clean the oven", "pay the rent",
         "check the car tyres", "send the invoice to Marta", "get flowers for Mum"]
HELLOS = [("Hi! I'm %s.", "Hi %s! How can I help?"), ("hey, I'm %s", "Hi %s! What can I do for you?"),
          ("My name is %s.", "Nice to meet you, %s! How can I help?"), ("Hello, %s here.", "Hello %s! What do you need?")]
SMALL = [("How are you?", "I'm doing well, thanks!"), ("haha thanks", "You're welcome!"), ("ok cool", "Great. Anything else?"),
         ("lol", "Glad that made you smile!"), ("nice", "Good! Let me know if you need anything.")]
WEATHER_WORDS = ["clear", "cloudy", "rain", "partly cloudy", "snow", "fog", "showers"]


def compaction():
    import random
    rng = random.Random(3)
    samples = []
    for n in range(640):
        name, like = rng.choice(NAMES), rng.choice(LIKES)
        city = rng.choice(TRIP_CITIES)
        temp, cond = rng.randint(-5, 33), rng.choice(WEATHER_WORDS)
        trip, month = rng.choice(TRIP_CITIES), rng.choice(MONTHS)
        task = rng.choice(TASKS)
        a, b = rng.randint(12, 99), rng.randint(3, 40)
        parts, keep = [], []
        hello, reply = rng.choice(HELLOS)
        parts.append((hello % name, reply % name))
        keep.append("The user is %s." % name)
        picks = rng.sample(["like", "trip", "weather", "task", "math", "chat", "chat"], k=rng.randint(2, 5))
        for pick in picks:
            if pick == "like":
                parts.append((rng.choice(["I really like %s.", "I love %s.", "%s is my favourite thing."]) % like,
                              "Nice! %s is a good choice." % like.capitalize()))
                keep.append("They like %s." % like)
            elif pick == "trip":
                parts.append(("I'm going to %s in %s." % (trip, month), "That sounds great. Anything I can help you plan?"))
                keep.append("They are travelling to %s in %s." % (trip, month))
            elif pick == "weather":
                parts.append(("What's the weather in %s?" % city,
                              "It's %d °C and %s in %s right now." % (temp, cond, city)))
                keep.append("The weather in %s was %d °C and %s." % (city, temp, cond))
            elif pick == "task":
                parts.append((rng.choice(["Remind me to %s later.", "I need to %s.", "Note: %s."]) % task,
                              "I'll note that: %s." % task))
                keep.append("They want to %s." % task.replace("my ", "their "))
            elif pick == "math":
                parts.append(("What is %d times %d?" % (a, b), "%d × %d = %d." % (a, b, a * b)))
                keep.append("%d × %d = %d." % (a, b, a * b))
            else:
                parts.append(rng.choice(SMALL))
        context = "\n".join("User: %s\nMagi: %s" % pair for pair in parts)
        samples.append({"context": context, "compact": " ".join(keep)})
    return samples


# ---- simple questions and short answers ------------------------------------------------
#
# A small model learns best from many short questions with short answers: small sums
# worked out in its head (big ones go to the calculator), capitals, and everyday facts.

CAPITALS = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Portugal", "Lisbon"),
    ("the United Kingdom", "London"), ("Ireland", "Dublin"), ("Norway", "Oslo"), ("Sweden", "Stockholm"),
    ("Finland", "Helsinki"), ("Denmark", "Copenhagen"), ("Poland", "Warsaw"), ("Greece", "Athens"),
    ("Austria", "Vienna"), ("the Netherlands", "Amsterdam"), ("Belgium", "Brussels"), ("Switzerland", "Bern"),
    ("Russia", "Moscow"), ("Ukraine", "Kyiv"), ("Turkey", "Ankara"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"),
    ("Nigeria", "Abuja"), ("South Africa", "Pretoria"), ("Morocco", "Rabat"), ("Japan", "Tokyo"),
    ("China", "Beijing"), ("South Korea", "Seoul"), ("India", "New Delhi"), ("Thailand", "Bangkok"),
    ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"), ("Australia", "Canberra"), ("New Zealand", "Wellington"),
    ("Canada", "Ottawa"), ("the United States", "Washington, D.C."), ("Mexico", "Mexico City"),
    ("Brazil", "Brasília"), ("Argentina", "Buenos Aires"), ("Chile", "Santiago"), ("Peru", "Lima"),
    ("Colombia", "Bogotá"),
]
EVERYDAY = [
    ("How many days are in a week?", "Seven."),
    ("How many months are in a year?", "Twelve."),
    ("How many hours are in a day?", "Twenty-four."),
    ("How many minutes are in an hour?", "Sixty."),
    ("How many seconds are in a minute?", "Sixty."),
    ("How many days are in a year?", "365, or 366 in a leap year."),
    ("How many legs does a spider have?", "Eight."),
    ("How many legs does a dog have?", "Four."),
    ("How many legs does a bird have?", "Two."),
    ("How many wheels does a bicycle have?", "Two."),
    ("What colour is the sky on a clear day?", "Blue."),
    ("What colour is grass?", "Green."),
    ("What colour is snow?", "White."),
    ("What colour are bananas when they are ripe?", "Yellow."),
    ("What do you get if you mix blue and yellow?", "Green."),
    ("What do you get if you mix red and white?", "Pink."),
    ("What is the largest planet in the solar system?", "Jupiter."),
    ("Which planet is closest to the Sun?", "Mercury."),
    ("Which planet do we live on?", "Earth."),
    ("What is the closest star to Earth?", "The Sun."),
    ("At what temperature does water boil at sea level?", "100 °C (212 °F)."),
    ("At what temperature does water freeze?", "0 °C (32 °F)."),
    ("What is frozen water called?", "Ice."),
    ("What gas do people breathe in to live?", "Oxygen."),
    ("What gas do plants take in?", "Carbon dioxide."),
    ("What is the biggest ocean?", "The Pacific Ocean."),
    ("What is the tallest mountain in the world?", "Mount Everest."),
    ("What is the longest river in the world?", "The Nile is usually named the longest, with the Amazon close behind."),
    ("How many continents are there?", "Seven."),
    ("What animal says moo?", "A cow."),
    ("What animal says meow?", "A cat."),
    ("What animal barks?", "A dog."),
    ("What do bees make?", "Honey."),
    ("What is the baby of a cat called?", "A kitten."),
    ("What is the baby of a dog called?", "A puppy."),
    ("What is the opposite of up?", "Down."),
    ("What is the opposite of happy?", "Sad."),
    ("What is the opposite of big?", "Small."),
    ("What is the opposite of fast?", "Slow."),
    ("What is the opposite of open?", "Closed."),
    ("What is the plural of mouse?", "Mice."),
    ("What is the plural of child?", "Children."),
    ("What is the plural of tooth?", "Teeth."),
    ("What is the past tense of go?", "Went."),
    ("What is the past tense of eat?", "Ate."),
    ("Which day comes after Friday?", "Saturday."),
    ("Which day comes before Monday?", "Sunday."),
    ("Which month comes after March?", "April."),
    ("Which month comes after December?", "January."),
    ("What is the first letter of the alphabet?", "A."),
    ("What is the last letter of the alphabet?", "Z."),
    ("How many letters are in the English alphabet?", "26."),
    ("What shape has three sides?", "A triangle."),
    ("What shape has four equal sides?", "A square."),
    ("How many sides does a hexagon have?", "Six."),
    ("What do you call a word that means the same as another word?", "A synonym."),
    ("What is H2O?", "Water."),
    ("What language is mostly spoken in Brazil?", "Portuguese."),
    ("What language is spoken in Japan?", "Japanese."),
    ("What is a programming language often used for data and AI?", "Python."),
    ("In Python, how do you print hello?", "print(\"hello\")"),
    ("In Python, how do you make a list of 1, 2 and 3?", "[1, 2, 3]"),
    ("What does CPU stand for?", "Central processing unit."),
    ("What does AI stand for?", "Artificial intelligence."),
]
SUM_WAYS = [("What is {a} plus {b}?", "+", lambda a, b: a + b), ("What is {a} + {b}?", "+", lambda a, b: a + b),
            ("What is {a} minus {b}?", "-", lambda a, b: a - b), ("What is {a} times {b}?", "×", lambda a, b: a * b),
            ("How much is {a} times {b}?", "×", lambda a, b: a * b), ("What is {a} × {b}?", "×", lambda a, b: a * b)]


def simple_qa():
    import random
    rng = random.Random(5)
    samples = []
    # Known by heart: answered straight away at the standard effort, with a line of thought at the superior one.
    for country, city in CAPITALS:
        ask = "What is the capital of %s?" % country
        samples.append({"effort": "standard", "prompt": ask, "response": city + "."})
        samples.append({"effort": "superior", "prompt": ask, "thinking": "The capital of %s is %s." % (country, city),
                        "response": city + "."})
    for ask, answer in EVERYDAY:
        samples.append({"effort": "standard", "prompt": ask, "response": answer})
    seen = set()
    while len(seen) < 260:
        a, b = rng.randint(0, 12), rng.randint(0, 12)
        way, sign, work = rng.choice(SUM_WAYS)
        if sign == "-" and b > a:
            a, b = b, a
        key = (a, b, way)
        if key in seen:
            continue
        seen.add(key)
        result = work(a, b)
        samples.append({"effort": "standard", "prompt": way.format(a=a, b=b), "response": "%d." % result})
        samples.append({"effort": "superior", "prompt": way.format(a=a, b=b),
                        "thinking": "%d %s %d = %d." % (a, sign, b, result), "response": "%d." % result})
    return samples


def write(name, lines):
    with open(os.path.join(HERE, name), "w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line if isinstance(line, str) else json.dumps(line, ensure_ascii=False))
            handle.write("\n")


def main():
    write("language.md", [LANGUAGE.strip()])

    chat = [{"prompt": q, "response": a} for q, a in CHAT]
    # With tools offered too: a question no tool is needed for is still answered straight away.
    chat += [{"tools": DEFAULT_TOOLS, "prompt": q, "response": a} for q, a in CHAT[::2]]
    chat += [{"messages": m} for m in CONVERSATIONS]
    write("chat.jsonl", chat)

    # Magi is an agent, not a calculator or a book of facts: no facts.md or reasoning.jsonl any more
    # (arithmetic goes to the calculator, facts to MagiDB and the web). Long pieces of work are
    # make_work_data.py's.

    write("tools.jsonl", TOOLS)

    titles = []
    for n, (question, answer, title) in enumerate(TITLES):
        ask, reply = [(TITLE_JSON, json.dumps({"title": title})), (TITLE_TAGS, "<title>%s</title>" % title),
                      (TITLE_PLAIN, title)][n % 3]
        titles.append({"messages": [user(question), say(answer), user(ask), say(reply)]})
        # The same with the other forms, so each form is seen with every conversation once in a while.
        if n % 2 == 0:
            ask, reply = (TITLE_JSON, json.dumps({"title": title})) if n % 3 else (TITLE_TAGS, "<title>%s</title>" % title)
            titles.append({"messages": [user(question), say(answer), user(ask), say(reply)]})
    write("titles.jsonl", titles)

    write("knowing.jsonl", knowing())
    write("compaction.jsonl", compaction())
    write("simple_qa.jsonl", simple_qa())
    # Thinking when it helps, tools in a row, errors, what no tool can do, small talk, longer answers.
    import make_agent_data as agentic
    write("thinking.jsonl", agentic.thinking())
    write("agent.jsonl", agentic.agent())
    write("small_talk.jsonl", agentic.small_talk())
    write("explain.jsonl", agentic.explain())
    write("avoid.jsonl", AVOID)
    print("wrote", len(chat), "chat,", len(TOOLS), "tool,", len(titles), "title and", len(AVOID), "avoid samples")


if __name__ == "__main__":
    main()
