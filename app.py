# ── BLOCK 1: IMPORTS & SETUP ──
# This block handles all the necessary imports for running the Flask app,
# handling models, managing audio files, and integrating the RAG manager.
import os
import torch
import torchaudio
import sqlite3
import uuid
import datetime
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from rag_manager import RAGManager

# Initialize the Flask application and enable Cross-Origin Resource Sharing (CORS)
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ── BLOCK 2: INDIC CONFORMER ASR MODEL LOADING ──
# Loads the automatic speech recognition (ASR) model to transcribe spoken audio.
print("Loading IndicConformer model... This may take a moment (it's around 600MB).")
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)
print("Model loaded successfully.")

# Mapping from our simple 2-letter language codes to the FLORES codes required by the IndicTrans2 model
flores_codes = {
    "as": "asm_Beng",
    "bn": "ben_Beng",
    "brx": "brx_Deva",
    "doi": "doi_Deva",
    "gu": "guj_Gujr",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "ks": "kas_Arab",
    "kok": "gom_Deva",
    "mai": "mai_Deva",
    "ml": "mal_Mlym",
    "mni": "mni_Beng",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "sa": "san_Deva",
    "sat": "sat_Olck",
    "sd": "snd_Arab",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
}

# ── BLOCK 3: INDIC TRANS 2 TRANSLATION MODEL LOADING ──
# Loads the translation model used to translate Indian languages into English
# for the RAG query, ensuring any compatibility issues with Transformers v5 are patched.
import sys
import types
# Mock the transformers.onnx module as it's sometimes missing or moved in newer transformer versions
if 'transformers.onnx' not in sys.modules:
    sys.modules['transformers.onnx'] = types.ModuleType('transformers.onnx')
    sys.modules['transformers.onnx'].OnnxConfig = object
    sys.modules['transformers.onnx'].OnnxSeq2SeqConfigWithPast = object
    sys.modules['transformers.onnx.utils'] = types.ModuleType('transformers.onnx.utils')
    sys.modules['transformers.onnx.utils'].compute_effective_axis_dimension = lambda *args, **kwargs: 1

print("Loading IndicTrans2 translation model... This may take a moment.")
translation_model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name, trust_remote_code=True)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name, trust_remote_code=True)

# Fix: transformers v5 post_init() corrupts sinusoidal positional embedding buffers.
# Force recomputation of the mathematical (non-learned) positional embeddings.
for module in [translation_model.model.encoder, translation_model.model.decoder]:
    ep = module.embed_positions
    ep.make_weights(ep.weights.shape[0], ep.embedding_dim, ep.padding_idx)

translation_model.eval()
ip = IndicProcessor(inference=True)
print("Translation model loaded successfully.")

# Directory where uploaded audio files are temporarily stored before processing
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('data', exist_ok=True)

# ── SIMPLE DATABASE SETUP ──
def init_db():
    conn = sqlite3.connect('data/history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recordings (
            id TEXT PRIMARY KEY,
            filename TEXT,
            transcription TEXT,
            translation TEXT,
            rag_answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ── BLOCK 4: MAIN TRANSCRIBE ENDPOINT ──
# This is the primary API endpoint that receives audio, transcribes it,
# translates it to English, and passes it to the RAG system to generate an answer.
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # 1. Validate and save the uploaded audio file
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Generate a unique ID for this recording
    file_id = uuid.uuid4().hex
    filename = f"{file_id}.mp3"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Check if a specific language was requested (e.g., 'hi' for Hindi)
        language = request.form.get('language') or 'hi'
        
        print(f"Transcribing {filepath}... (Language: {language})")
        
        # 2. Resample the audio if needed and run Transcription (ASR)
        wav, sr = torchaudio.load(filepath)
        print(f"[DEBUG] Audio loaded: shape={wav.shape}, sr={sr}")
        wav = torch.mean(wav, dim=0, keepdim=True)
        target_sample_rate = 16000
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
            wav = resampler(wav)
            
        transcription_rnnt = model(wav, language, "rnnt")
        print(f"[DEBUG] ASR raw result: type={type(transcription_rnnt)}, value={transcription_rnnt}")
            
        # Safely extract text from ASR result
        if isinstance(transcription_rnnt, list) and len(transcription_rnnt) > 0:
            text = transcription_rnnt[0] if transcription_rnnt[0] is not None else ""
        elif transcription_rnnt is not None:
            text = str(transcription_rnnt)
        else:
            text = ""
        print("Transcription result:", text)
        
        # 3. Translate the transcribed text to English
        flores_src_lang = flores_codes.get(language)
        translation = ""
        if flores_src_lang and text and text.strip():
            print(f"Translating to English from {flores_src_lang}...")
            batch = [text.strip()]
            batch = ip.preprocess_batch(batch, src_lang=flores_src_lang, tgt_lang="eng_Latn")
            print(f"Preprocessed batch: {batch}")

            translation_tokenizer._switch_to_input_mode()
            inputs = translation_tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            )
            print(f"Tokenized input_ids shape: {inputs['input_ids'].shape}")

            with torch.no_grad():
                generated_tokens = translation_model.generate(
                    **inputs,
                    use_cache=False,  # DynamicCache in transformers v5 is incompatible with IndicTrans2's old tuple-based cache
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )
            print(f"Generated tokens shape: {generated_tokens.shape}")

            translation_tokenizer._switch_to_target_mode()
            generated_tokens = translation_tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            translation_tokenizer._switch_to_input_mode()
            print(f"Decoded tokens: {generated_tokens}")

            translations = ip.postprocess_batch(generated_tokens, lang="eng_Latn")
            translation = translations[0]
            print("Translation result:", translation)

        # 4. RAG: answer the query using the English text (or original text if no translation)
        query_text = translation if translation else (text.strip() if text else "")
        rag_answer = ""
        if rag_manager and query_text:
            print(f"Querying RAG with: {query_text}")
            rag_result = rag_manager.query(query_text)
            rag_answer = rag_result.get("answer", "")
            print(f"RAG answer: {rag_answer[:200]}")

        # 5. Save the results to our simple SQLite database
        safe_text = text.strip() if text else ""
        conn = sqlite3.connect('data/history.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO recordings (id, filename, transcription, translation, rag_answer)
            VALUES (?, ?, ?, ?, ?)
        ''', (file_id, filename, safe_text, translation, rag_answer))
        conn.commit()
        conn.close()

        # Return all outputs to the client
        return jsonify({
            'id': file_id,
            'filename': filename,
            'text': safe_text,
            'translation': translation,
            'rag_answer': rag_answer,
        })
    except Exception as e:
        import traceback
        print(f"Error during transcription: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── BLOCK 5: RAG INITIALIZATION ──
# Initialize RAG at module level so it is guaranteed to be ready before Flask
# serves any request (eliminates the race with Flask's debug-mode reloader).
# Run 'python ingest.py' first if the vector database hasn't been built yet.
rag_manager = RAGManager()

# ── BLOCK 6: RAG QUERY ENDPOINT ──
# An endpoint to directly test text-based querying without going through the audio pipeline.
@app.route('/query', methods=['POST'])
def query_rag():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Please provide a "question" field in JSON body'}), 400

    question = data['question'].strip()
    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400

    result = rag_manager.query(question)
    return jsonify(result)


# ── SIMPLE HISTORY AND AUDIO ENDPOINTS ──
# Returns all past recordings from the database (only those whose audio files still exist)
@app.route('/history', methods=['GET'])
def get_history():
    conn = sqlite3.connect('data/history.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM recordings ORDER BY timestamp ASC')
    rows = cursor.fetchall()

    # Filter out entries whose audio files have been deleted from disk
    history = []
    orphan_ids = []
    for row in rows:
        filepath = os.path.join(UPLOAD_FOLDER, row['filename'])
        if os.path.exists(filepath):
            history.append(dict(row))
        else:
            orphan_ids.append(row['id'])

    # Clean up orphaned database entries
    if orphan_ids:
        cursor.executemany('DELETE FROM recordings WHERE id = ?',
                           [(rid,) for rid in orphan_ids])
        conn.commit()

    conn.close()
    return jsonify(history)

# Clears all history from the database and deletes audio files
@app.route('/history', methods=['DELETE'])
def clear_history():
    conn = sqlite3.connect('data/history.db')
    conn.execute('DELETE FROM recordings')
    conn.commit()
    conn.close()
    # Also remove any leftover audio files
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))
    return jsonify({'status': 'History cleared'})

# Serves the saved mp3 files back to the frontend so they can be played
@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ── BLOCK 7: APPLICATION EXECUTION ──
# Main entrypoint to start the Flask application.
if __name__ == '__main__':
    # Start the Flask web server on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
