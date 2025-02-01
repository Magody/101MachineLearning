import os
from flask import Flask, request, jsonify
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from dotenv import load_dotenv
from openai import OpenAI
import re

# Function to clean an ID by removing non-alphanumeric characters and stripping whitespace
def clean_id(id):
    return re.sub(r"[^A-z\d]", "", id).strip()

# Function to write role and content to a local text file
def write_local_txt(filename, role, content):
    with open(f"./{filename}.txt", "a") as f_in:
        f_in.write(role + "#SEP#" + content.replace("\n", " ") + "\n")

# Function to recover chat history from a local text file
def recover_chat(filename):

    system_prompt = """You are a supportive and empathetic companion specializing in mental health and well-being.
    Your primary goal is to listen attentively, provide compassionate responses, and offer sound advice that promotes mental and emotional health. Always prioritize the safety and well-being of the person you are interacting with. Avoid giving any medical diagnosis or treatment advice, and instead, encourage seeking professional help when needed.
    Keep your language positive, respectful, and non-judgmental, fostering a safe space for open and honest conversation.
    Your answer should keep short
    """
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    try:
        with open(f"./{filename}.txt", "r") as f_in:
            lines = f_in.readlines()
            for line in lines:
                if len(line) < 0:
                    continue

                line_split = line.split("#SEP#")
                role = line_split[0].lower().strip()
                content = line_split[1].strip()
                messages.append({"role": role, "content": content})

    except Exception as error:
        print(error)
    return messages

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')

# Initialize Twilio client
twilio_client = Client(os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN'))
twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')

print(os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN'))

# Define route for home page
@app.route('/home', methods=['GET'])
def test():
    return "<b> Hello home </b>"

# Define route for handling incoming voice calls
@app.route('/voice', methods=['POST'])
def voice():
    response = VoiceResponse()
    from_number = request.form.get('From', '')
    print("FROM", from_number)

    # Initial message
    gather = Gather(input='speech', action='/gather', method='POST', timeout=60, speechTimeout='auto', finishOnKey='*')
    gather.say('Hello, tell me how can i help you?')
    response.append(gather)
    response.pause(length=10)
    response.say('We did not receive any input. Goodbye!')
    return str(response)

# Define route for handling gathered speech input
@app.route('/gather', methods=['POST'])
def gather():
    from_number = request.form.get('From', '')
    response = VoiceResponse()
    transcription = request.form.get('SpeechResult', '')
    if transcription:
        filename = clean_id(from_number)

        messages = []
        if os.path.isfile(f"./{filename}.txt"):
            messages = recover_chat(filename)
            
        messages.append({"role": "user", "content": transcription})
        write_local_txt(filename, "user", transcription)

        client = OpenAI()

        messages_classifier = [
            {
                "role": "system",
                "content": """Clasify if the user would like to continue or end the interaction.
                Answer 'end' if the user wants to end the conversation or 'continue' in every other case.
                For instance: If user say 'stop' or 'end' you must answer 'end'
                If the user continues to answer coherently, then you must answer 'continue'

                The following will be the input of user
                """
            },
            {
                "role": "user", "content": transcription
            }
        ]

        try:

            completion_classifier = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages_classifier
            )
            response_openai_classifier = str(completion_classifier.choices[0].message.content).lower()

            if 'continue' in response_openai_classifier:

                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )

                response_openai = str(completion.choices[0].message.content)
                print(f'Transcription: {transcription}. {response_openai}')
                response.say(response_openai)
                write_local_txt(filename, "assistant", response_openai)
                
                # Continue with conversation
                gather = Gather(input='speech', action='/gather', method='POST', timeout=60, speechTimeout='auto', finishOnKey='*')
                gather.say('You can continue talking to me, if you want to stop just say stop or end.')
                response.append(gather)

            else:
                message_contents = ""
                for message in messages:
                    if message['role'] == 'assistant':
                        message_contents += message['content']

                messages_summary = [
                    {
                        "role": "system",
                        "content": f"""The following are advices made by a LLM specialized in mental health, create a summary of the most relevant: {message_contents}"""
                    },
                ]

                completion_summary = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages_summary
                )
                response_openai_summary = str(completion_summary.choices[0].message.content).lower()

                print(response_openai_summary)
                send_sms(from_number, response_openai_summary)
                response.say("Goodbye, call me if you need more help")
                response.pause(length=2)
        except Exception as error:
            print(error)
            response.say("Sorry something happened on my program")
    else:
        response.say('No message received. Goodbye!')

    return str(response)

# Function to send SMS using Twilio
def send_sms(to_number, message):
    twilio_client.messages.create(
        body=message,
        from_=os.getenv('TWILIO_PHONE_NUMBER'),
        to=to_number
    )

# Function to process audio with OpenAI Whisper (currently not used)
def process_audio_with_whisper(audio_file_path):
    client = OpenAI()
    with open(audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )
    return transcription['text']

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
