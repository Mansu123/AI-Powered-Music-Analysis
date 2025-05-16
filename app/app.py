
# app.py
# app.py
from music_llm_agent import MusicAnalysisAgent, AudioFeatureExtractor
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import warnings
import librosa
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import json
import os

warnings.filterwarnings("ignore")

# Set default environment variables if not present
os.environ.setdefault("GROQ_API_KEY", "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")

class EnhancedInstrumentDetector:
    def __init__(self):
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        map_path = tf.keras.utils.get_file(
            'yamnet_class_map.csv',
            'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
        )
        class_map = pd.read_csv(map_path)
        self.class_names = class_map['display_name'].tolist()
        self.category_map = {
            'Percussion': ['Drum', 'Snare drum', 'Hi-hat', 'Drum kit', 'Tabla', 'Tambourine', 'Percussion'],
            'Vocals': ['Singing', 'Voice', 'Vocal music', 'Vocalist', 'Singer'],
            'High-pitched': ['Flute', 'Violin', 'Trumpet', 'Saxophone', 'Clarinet'],
            'Piano': ['Piano', 'Keyboard (musical)', 'Electric piano', 'Synthesizer'],
            'Guitar': ['Guitar', 'Acoustic guitar', 'Electric guitar', 'Guitar strum', 'Bass guitar'],
            'Electronic': ['Synthesizer', 'Electronic music', 'Techno', 'House music'],
            'Strings': ['Violin', 'Cello', 'Viola', 'String section', 'Orchestral music']
        }
        self.index_to_category = {
            i: cat for cat, names in self.category_map.items()
            for i, name in enumerate(self.class_names) if name in names
        }

    def detect_instruments(self, file):
        waveform, sr = librosa.load(file, sr=16000)
        scores, _, _ = self.yamnet_model(waveform)  # Fixed unpacking
        scores_np = scores.numpy()
        counts = {cat: 0 for cat in self.category_map}
        for frame in scores_np:
            for idx, val in enumerate(frame):
                if val > 0.1 and idx in self.index_to_category:  
                    counts[self.index_to_category[idx]] += 1
        total = sum(counts.values())
        return {k: (v / total) * 100 for k, v in counts.items() if v > 0} if total > 0 else {}

    def visualize_results(self, proportions):
        fig, ax = plt.subplots(figsize=(8, 5))
        if not proportions:
            ax.text(0.5, 0.5, "No instruments detected", ha='center', va='center')
            return fig
        labels = list(proportions.keys())
        values = list(proportions.values())
        
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(labels)))
        bars = ax.bar(labels, values, color=colors)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('% presence', fontsize=12)
        ax.set_title('Detected Instrument Categories', fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()
        return fig

def plot_virality(energy, tempo):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    boost = 1.0
    if energy > 0.22 and tempo > 115:
        boost = 1.15
    elif energy > 0.15 and 100 <= tempo <= 115:
        boost = 1.05
        
    virality_score = min(((tempo / 180.0) * 0.6 + (energy / 0.35) * 0.4) * boost, 1.0) * 100
    
    if virality_score < 40:
        color = '#3498db'
    elif virality_score < 70:
        color = '#f39c12'
    else:
        color = '#e74c3c'
    
    ax.bar(['Virality Score'], [virality_score], color=color)
    ax.axhline(y=30, color='r', linestyle='--', alpha=0.3, label='Low Virality')
    ax.axhline(y=60, color='g', linestyle='--', alpha=0.3, label='Medium Virality')
    ax.axhline(y=85, color='b', linestyle='--', alpha=0.3, label='High Virality')
    ax.text(0, virality_score + 2, f'{virality_score:.1f}%', ha='center', fontsize=14, fontweight='bold')
    
    ax.set_ylim([0, 100])
    ax.set_ylabel('Virality Rate (%)', fontsize=12)
    ax.set_title('Estimated Virality Potential', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    fig.tight_layout()
    return fig

def process_audio(audio_path, llm_provider="groq"):
    extractor = AudioFeatureExtractor(audio_path)
    if not extractor.extract_all_features():
        return "Failed to extract features.", None, "Virality prediction error", "LLM insight unavailable", None
    
    try:
        features_json = extractor.to_json()
        summary = json.loads(features_json)
        detector = EnhancedInstrumentDetector()
        proportions = detector.detect_instruments(audio_path)
        plot = detector.visualize_results(proportions)
        instruments = list(proportions.keys())
        
        try:
            agent = MusicAnalysisAgent(model="llama3-70b-8192", provider=llm_provider)
        except Exception as e:
            print(f"Error initializing MusicAnalysisAgent: {e}")
            return "Agent initialization error", None, "Could not initialize music analysis agent", "LLM unavailable", None
        
        try:
            virality_result = agent.analyze_song_features(features_json)
        except Exception as e:
            print("LLM analysis error:", e)
            virality_result = f"LLM Analysis Error: {str(e)}"
            
        try:
            improvement = agent.get_song_improvement_suggestions(features_json)
        except Exception as e:
            print("Improvement error:", e)
            improvement = f"Improvement Suggestion Error: {str(e)}"
            
        try:
            workout_fit = agent.assess_workout_playlist_fit(features_json)
        except Exception as e:
            print("Workout Fit error:", e)
            workout_fit = f"Workout Fit Suggestion Error: {str(e)}"
            
        try:
            marketing = agent.suggest_marketing_channels(features_json)
        except Exception as e:
            print("Marketing error:", e)
            marketing = f"Marketing Suggestion Error: {str(e)}"
            
        try:
            genre = agent.recommend_genre_classification(features_json)
        except Exception as e:
            print("Genre classification error:", e)
            genre = f"Genre Classification Error: {str(e)}"
            
        try:
            mood_type = agent.recommend_music_category(features_json)
        except Exception as e:
            print("Music type error:", e)
            mood_type = f"Music Type Unavailable: {str(e)}"
            
        try:
            lyric_suggestions = agent.analyze_lyric_improvement(features_json)
        except Exception as e:
            print("Lyric suggestion error:", e)
            lyric_suggestions = f"Lyric suggestions unavailable: {str(e)}"
            
        try:
            commercial_potential = agent.analyze_commercial_potential(features_json)
        except Exception as e:
            print("Commercial potential analysis error:", e)
            commercial_potential = f"Commercial potential analysis unavailable: {str(e)}"
        
        llm_output = f"""
## 🎧 Instrument-Based Analysis
Detected Instruments: {', '.join(instruments) if instruments else 'None Detected'}

## 🎼 Genre Classification
{genre}

## 💖 Music Type (e.g., Romantic, Chill, Party)
{mood_type}

## 🎛 Overall Analysis
{virality_result}

## 🎚 Suggestions for Improvement
{improvement}

## 🖋 Lyric Improvement Suggestions
{lyric_suggestions}

## 🏋️ Workout Fit Assessment
{workout_fit}

## 📈 Marketing Recommendations
{marketing}

## 💰 Commercial Potential Analysis
{commercial_potential}
"""
        
        virality_text = f"""🎼 Track Overview
- File: {summary['file_name']}
- Tempo: {summary['tempo']:.2f} BPM
- Key: {summary['key']}
- Mood: {', '.join(summary['mood_indicators'])}
- Energy: {summary['energy']:.4f}"""
        
        virality_chart = plot_virality(summary['energy'], summary['tempo'])
        
        return ", ".join(instruments), plot, virality_text, llm_output, virality_chart
    except Exception as e:
        print("Full audio processing error:", e)
        return f"Error: {str(e)}", None, "Error", f"LLM Error: {str(e)}", None

# ✅ Fixed CSS block
custom_css = """
body {
    background-color: #87CEEB;
    color: #333333;
    font-family: 'Poppins', sans-serif;
}
.gradio-container {
    background-color: #a9d6f5;
    border-radius: 15px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
}
.output-markdown {
    line-height: 1.7;
    background-color: #c3e1f7;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
h2 {
    color: #555555;
    border-bottom: 2px solid #555555;
    padding-bottom: 10px;
    margin-top: 20px;
}
.block-container {
    background-color: #b8dcf7;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.block-container label {
    color: #555555;
    font-weight: bold;
}
button {
    background-color: #ffb6c1 !important;
    color: #333333 !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    box-shadow: 0 4px 8px rgba(255, 182, 193, 0.3) !important;
    transition: all 0.3s ease !important;
}
button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 12px rgba(255, 182, 193, 0.4) !important;
}
"""

with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="rose")
) as demo:
    gr.Markdown("""
    # 🎵 AI-Powered Music Analysis Suite

    Upload your track to discover its musical characteristics, commercial potential, and receive expert recommendations.
    """)
    
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Song (MP3/WAV)", show_label=True, elem_classes=["audio-component"])
    
    with gr.Row():
        with gr.Column(scale=3):
            llm_choice = gr.Dropdown(
                choices=["groq", "openai", "huggingface"], 
                label="Choose LLM Provider",
                value="groq"
            )
        with gr.Column(scale=1):
            analyze_btn = gr.Button("🔍 Analyze Audio", variant="primary", size="lg")
    
    with gr.Group(elem_classes=["block-container"]):
        gr.Markdown("### 🎸 Detected Instruments")
        instrument_text = gr.Textbox(label="Instruments Found", interactive=False)
        instrument_plot = gr.Plot(label="Instrument Distribution")
    
    with gr.Group(elem_classes=["block-container"]):
        gr.Markdown("### 📊 Track Overview & Virality")
        virality_output = gr.Textbox(label="Audio Summary", lines=5, interactive=False)
        virality_plot = gr.Plot(label="Virality Score")
    
    with gr.Group(elem_classes=["block-container"]):
        gr.Markdown("### 🤖 AI Music Analysis")
        llm_output = gr.Markdown(label="AI Insight")
    
    analyze_btn.click(
        fn=process_audio, 
        inputs=[audio_input, llm_choice], 
        outputs=[instrument_text, instrument_plot, virality_output, llm_output, virality_plot]
    )
    
    gr.Markdown("""
    ---
    ### 🎧 About This Tool

    This AI-powered music analysis tool combines powerful audio feature extraction with LLM insights to help musicians, producers, and marketers understand their tracks better.

    **Features:**
    - Instrument detection with TensorFlow YAMNet
    - Audio feature extraction (tempo, key, energy, mood)
    - LLM-powered creative insights
    - Commercial potential analysis
    - Marketing recommendations

    © 2025 Music AI Suite | Built with TensorFlow Hub + LangChain + Gradio
    """)

if __name__ == "__main__":
    demo.launch()
