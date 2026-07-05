import os
import gradio as gr
from fpdf import FPDF
import datetime
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_elevenlabs
from pydub import AudioSegment

# Convert mp3 to wav safely
def convert_mp3_to_wav(mp3_path, wav_path):
    try:
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"Audio conversion failed: {e}")
        return None

system_prompt = """You have to act as a professional doctor, I know you are not but this is for learning purpose. 
With what I see, I think you have .... 
If you make a differential, suggest some remedies for them. Do not add any numbers or special characters in your response. 
Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Do not say 'In the image I see' but say 'With what I see, I think you have ....'
Don't respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot. 
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please."""

neon_css = """
body {
    background-color: #0f0f1a;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #e0e0ff;
    margin: 0;
    padding: 0;
}

h1 {
    font-weight: 900;
    font-size: 3rem;
    text-align: center;
    margin: 0;
    color: #00fff7;
    text-shadow:
       0 0 5px #00fff7,
       0 0 10px #00fff7,
       0 0 20px #00fff7,
       0 0 40px #00fff7,
       0 0 80px #00fff7;
}

h3 {
    color: #33fff6;
    font-weight: 700;
    margin-bottom: 1rem;
    text-shadow: 0 0 10px #33fff6;
}

.gradio-container {
    max-width: 900px;
    margin: 3rem auto 4rem;
    background: #111126;
    border-radius: 20px;
    box-shadow:
        0 0 15px #00fff7,
        inset 0 0 10px #00fff7;
    padding: 2rem 3rem;
}

.gr-button-primary {
    background: linear-gradient(90deg, #00fff7, #00b3e6);
    border: none;
    box-shadow:
        0 0 8px #00fff7,
        0 0 20px #00b3e6;
    color: #000;
    font-weight: 700;
    font-size: 1.25rem;
    padding: 0.8rem 1.8rem;
    border-radius: 15px;
    transition: all 0.3s ease;
    cursor: pointer;
}

.gr-button-primary:hover {
    background: linear-gradient(90deg, #00b3e6, #00fff7);
    box-shadow:
        0 0 20px #00fff7,
        0 0 40px #00b3e6;
    color: #000;
}

.gr-textbox, .gr-textarea {
    background-color: #0a0a1d;
    border: 2px solid #00fff7;
    border-radius: 12px;
    color: #e0e0ff;
    font-size: 1.1rem;
    padding: 0.6rem 1rem;
    box-shadow:
        0 0 8px #00fff7 inset;
    transition: border-color 0.3s ease;
}

.gr-textbox:focus, .gr-textarea:focus {
    border-color: #00b3e6;
    outline: none;
    box-shadow:
        0 0 12px #00b3e6 inset;
}

.gr-audio {
    border-radius: 12px;
    border: 2px solid #00fff7;
    box-shadow: 0 0 12px #00b3e6;
}

.gr-row {
    gap: 2rem;
}

.gr-column {
    background: #121229;
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow:
        0 0 20px #00b3e6 inset;
}

.gr-markdown {
    color: #e0e0ff;
}

hr {
    border: 1px solid #00fff7;
    opacity: 0.2;
}

footer {
    color: #33fff6;
    font-size: 0.9rem;
    text-align: center;
    margin-top: 3rem;
    text-shadow: 0 0 10px #33fff6;
}

@media (max-width: 768px) {
    .gradio-container {
        padding: 1.5rem 1.5rem;
    }
    h1 {
        font-size: 2rem;
    }
}
"""

def clean_transcription(text):
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
    return text

def save_chat_to_pdf(name, age, transcription, doctor_response):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 255, 255)

    pdf.cell(0, 10, f"Patient Name: {name}", ln=True)
    pdf.cell(0, 10, f"Patient Age: {age}", ln=True)
    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(224, 224, 255)

    pdf.multi_cell(0, 10, "Patient Said:")
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, transcription)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(102, 255, 249)
    pdf.multi_cell(0, 10, "Doctor's Advice and Diagnosis:")
    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(224, 224, 255)
    pdf.multi_cell(0, 10, doctor_response)
    pdf.ln(10)

    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Note: This is an AI-generated consultation. For medical emergencies, please consult a professional doctor.", align="C")

    filename = f"chat_{name.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    filepath = os.path.join("outputs", filename)
    os.makedirs("outputs", exist_ok=True)
    pdf.output(filepath)
    return filepath

def process_inputs(name, age, audio_filepath, image_filepath):
    if not name or age is None:
        return "Please enter your name and age.", "", None, None

    try:
        speech_to_text_output = transcribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3"
        )
        speech_to_text_output = clean_transcription(speech_to_text_output)
    except Exception as e:
        return "Speech-to-text failed", f"Transcription error: {e}", None, None

    if image_filepath:
        try:
            encoded_img = encode_image(image_filepath)
            doctor_response = analyze_image_with_query(
                query=system_prompt + speech_to_text_output,
                encoded_image=encoded_img,
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
        except Exception as e:
            doctor_response = f"Image analysis failed: {e}"
    else:
        doctor_response = "No image provided for analysis."

    try:
        text_to_speech_with_elevenlabs(
            input_text=doctor_response,
            output_filepath="final.mp3"
        )
        wav_path = convert_mp3_to_wav("final.mp3", "final.wav")
    except Exception as e:
        wav_path = None
        print(f"Text-to-speech failed: {e}")

    pdf_file = save_chat_to_pdf(name, age, speech_to_text_output, doctor_response)
    return speech_to_text_output, doctor_response, wav_path, pdf_file

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("""
    <h1 style="text-align:center; color:#00fff7;">AI Doctor Assistant</h1>
    <p style="text-align:center; font-size:1.1rem; color:#66fff9;">Speak your symptoms. Show your condition. Get a doctor's-like opinion — instantly.</p>
    <hr style="border-color:#00fff7;">
    """)

    with gr.Row():
        with gr.Column(scale=1):
            name_input = gr.Textbox(label="Your Name", placeholder="Enter your full name")
            age_input = gr.Slider(label="Your Age", minimum=0, maximum=120, step=1)
            voice_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak Your Symptoms")
            image_input = gr.Image(type="filepath", label="Upload Symptom Image (Optional)")
            submit_btn = gr.Button("Analyze and Diagnose", variant="primary")

        with gr.Column(scale=1):
            stt_output = gr.Textbox(label="Transcription", interactive=False, lines=2)
            doc_response = gr.Textbox(label="Doctor's Diagnosis", interactive=False, lines=5)
            audio_output = gr.Audio(label="Doctor's Voice Response", autoplay=True)
            pdf_download = gr.File(label="Download Consultation PDF")

    submit_btn.click(
        fn=process_inputs,
        inputs=[name_input, age_input, voice_input, image_input],
        outputs=[stt_output, doc_response, audio_output, pdf_download]
    )

    gr.Markdown("""
    <footer style="text-align:center; color:#33fff6; font-size:0.9rem; margin-top:2rem;">This is an AI-powered tool for learning. Not a substitute for real medical consultation.</footer>
    """)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), debug=True)
