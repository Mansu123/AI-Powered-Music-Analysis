
import os
import json
import numpy as np
import librosa
import warnings
import pandas as pd
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from huggingface_hub import InferenceClient

warnings.filterwarnings("ignore")

# Enhanced LLM support with multiple providers
def generate_music_insight(analysis_text, provider="groq", model="llama3-70b-8192"):
    try:
        prompt_template = PromptTemplate(
            input_variables=["analysis_text"],
            template="""
            You're an AI music expert. Based on the musical analysis below, provide insight on:
            - Genre and mood
            - Instrumentation and music type
            - Suggestions for improvement
            - Commercial/viral potential
            - Suitable platforms or audience
            Musical Analysis: {analysis_text}
            Answer in a concise paragraph.
            """
        )
        final_prompt = prompt_template.format(analysis_text=analysis_text)
        
        # Support for multiple LLM providers
        if provider.lower() == "groq":
            llm = ChatGroq(
                model_name=model, 
                groq_api_key=os.getenv("GROQ_API_KEY", "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")
            )
        elif provider.lower() == "openai":
            llm = ChatOpenAI(
                model_name=model,
                openai_api_key=os.getenv("OPENAI_API_KEY", "sk-demo-key")
            )
        elif provider.lower() == "huggingface":
            # For HuggingFace models, we'll use their Inference API
            client = InferenceClient(
                model=model,
                token=os.getenv("HF_API_TOKEN", "hf_demo_token")
            )
            return client.text_generation(
                final_prompt,
                max_new_tokens=512,
                temperature=0.7,
                return_full_text=False
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
        return llm.invoke(final_prompt)
    except Exception as e:
        return f"LLM Error: {str(e)}"

# AudioFeatureExtractor (unchanged)
class AudioFeatureExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.y = None
        self.sr = None
        self.features = {}

    def load_audio(self):
        try:
            self.y, self.sr = librosa.load(self.file_path, sr=None)
            return True
        except Exception as e:
            print("Audio loading error:", e)
            return False

    def extract_basic_features(self):
        duration = len(self.y) / self.sr
        tempo, beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)
        rms = librosa.feature.rms(y=self.y)[0]
        zcr = librosa.feature.zero_crossing_rate(self.y)[0]
        y_harmonic, y_percussive = librosa.effects.hpss(self.y)
        
        self.features['basic'] = {
            'duration': float(duration),
            'tempo': float(tempo),
            'num_beats': len(beat_frames),
            'avg_rms_energy': float(np.mean(rms)),
            'avg_zero_crossing_rate': float(np.mean(zcr)),
            'harmonic_percussive_ratio': float(np.sum(np.abs(y_harmonic)) / (np.sum(np.abs(y_percussive)) + 1e-10))
        }

    def extract_spectral_features(self):
        centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        contrast = librosa.feature.spectral_contrast(y=self.y, sr=self.sr)
        rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)[0]
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)
        
        self.features['spectral'] = {
            'avg_spectral_centroid': float(np.mean(centroid)),
            'avg_spectral_contrast': [float(np.mean(band)) for band in contrast],
            'avg_spectral_rolloff': float(np.mean(rolloff)),
            'avg_mfccs': [float(np.mean(m)) for m in mfccs]
        }

    def extract_harmonic_features(self):
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        tonnetz = librosa.feature.tonnetz(y=self.y, sr=self.sr)
        chroma_avg = np.mean(librosa.feature.chroma_cqt(y=self.y, sr=self.sr), axis=1)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        self.features['harmonic'] = {
            'chroma_energy': [float(np.mean(c)) for c in chroma],
            'tonnetz_features': [float(np.mean(t)) for t in tonnetz],
            'estimated_key': key_names[np.argmax(chroma_avg)],
            'key_strength': float(np.max(chroma_avg) / (np.sum(chroma_avg) + 1e-10))
        }

    def detect_instruments(self):
        perc_ratio = self.features['basic']['harmonic_percussive_ratio']
        centroid = self.features['spectral']['avg_spectral_centroid']
        mfccs = self.features['spectral']['avg_mfccs']
        
        instruments = []
        if perc_ratio < 0.8:
            instruments.append("Drums/Percussion")
        if centroid < 1000:
            instruments.append("Bass")
        if 1000 <= centroid <= 3000:
            instruments.append("Guitar/Piano")
        if centroid > 3000:
            instruments.append("High-pitched instruments")
        if 1500 < centroid < 4000 and abs(mfccs[2]) > 5:
            instruments.append("Vocals")
            
        self.features['detected_instruments'] = instruments

    def analyze_mood(self):
        tempo = self.features['basic']['tempo']
        energy = self.features['basic']['avg_rms_energy']
        centroid = self.features['spectral']['avg_spectral_centroid']
        
        moods = []
        if tempo < 80:
            moods.append("Slow/Relaxed")
        elif tempo <= 120:
            moods.append("Moderate/Balanced")
        else:
            moods.append("Fast/Energetic")
            
        if energy < 0.1:
            moods.append("Calm")
        elif energy > 0.2:
            moods.append("Intense")
            
        if centroid < 1500:
            moods.append("Dark/Warm")
        elif centroid > 3000:
            moods.append("Bright/Sharp")
            
        self.features['mood_indicators'] = moods

    def extract_all_features(self):
        if not self.load_audio():
            return False
        self.extract_basic_features()
        self.extract_spectral_features()
        self.extract_harmonic_features()
        self.detect_instruments()
        self.analyze_mood()
        return True

    def to_json(self):
        summary = {
            "file_name": os.path.basename(self.file_path),
            "duration": self.features['basic']['duration'],
            "tempo": self.features['basic']['tempo'],
            "key": self.features['harmonic']['estimated_key'],
            "energy": self.features['basic']['avg_rms_energy'],
            "detected_instruments": self.features['detected_instruments'],
            "mood_indicators": self.features['mood_indicators'],
            "spectral_centroid": self.features['spectral']['avg_spectral_centroid'],
            "harmonic_percussive_ratio": self.features['basic']['harmonic_percussive_ratio']
        }
        return json.dumps(summary, indent=2)

class MusicAnalysisAgent:
    def __init__(self, model="llama3-70b-8192", provider="groq"):
        self.model = model
        self.provider = provider
        
        # Configure LLM based on provider
        if provider.lower() == "groq":
            groq_api_key = os.getenv("GROQ_API_KEY", "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")
            self.llm = ChatGroq(model_name=model, groq_api_key=groq_api_key)
        elif provider.lower() == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY", "sk-demo-key")
            self.llm = ChatOpenAI(model_name=model, openai_api_key=openai_api_key)
        elif provider.lower() == "huggingface":
            # For Hugging Face, we'll initialize during the chain execution
            self.hf_client = InferenceClient(
                model=model,
                token=os.getenv("HF_API_TOKEN", "hf_demo_token")
            )
            self.llm = None
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _run_chain(self, template, features):
        prompt = PromptTemplate.from_template(template)
        
        if self.provider.lower() == "huggingface":
            # For Hugging Face, use direct inference
            try:
                full_prompt = prompt.format(features=features)
                return self.hf_client.text_generation(
                    full_prompt,
                    max_new_tokens=512,
                    temperature=0.7,
                    return_full_text=False
                )
            except Exception as e:
                print(f"HuggingFace inference error: {str(e)}")
                return f"Error with HuggingFace inference: {str(e)}"
        else:
            # For other providers, use LangChain
            chain = prompt | self.llm | StrOutputParser()
            try:
                return chain.invoke({"features": features})
            except Exception as e:
                print(f"Chain execution error: {str(e)}")
                return f"Error in LLM chain: {str(e)}"

    def analyze_song_features(self, features):
        try:
            return self._run_chain("""
            You are a music producer. Analyze: {features}
            Discuss sound profile, genre, emotional tone, and how features interact.
            """, features)
        except Exception as e:
            print(f"Song feature analysis error: {str(e)}")
            return f"Song analysis error. Please check the {self.provider} API connection."

    def get_song_improvement_suggestions(self, features):
        try:
            return self._run_chain("""
            Suggest improvements in production, instrumentation, mix, structure, and effects: {features}
            """, features)
        except Exception as e:
            print(f"Improvement suggestions error: {str(e)}")
            return f"Improvement suggestions unavailable. Please check the {self.provider} API connection."

    def assess_workout_playlist_fit(self, features):
        try:
            return self._run_chain("""
            Evaluate if suitable for workout playlists. Consider tempo, energy, emotion, instruments, context: {features}
            """, features)
        except Exception as e:
            print(f"Workout playlist assessment error: {str(e)}")
            return f"Workout playlist assessment unavailable. Please check the {self.provider} API connection."

    def suggest_marketing_channels(self, features):
        try:
            return self._run_chain("""
            Recommend marketing channels: platforms, social media, audience targeting, playlist strategy: {features}
            """, features)
        except Exception as e:
            print(f"Marketing channels suggestion error: {str(e)}")
            return f"Marketing channel suggestions unavailable. Please check the {self.provider} API connection."

    def recommend_genre_classification(self, features):
        try:
            return self._run_chain("""
            Suggest a musical genre based on the features below. Be specific and explain: {features}
            """, features)
        except Exception as e:
            print(f"Genre classification error: {str(e)}")
            return f"Genre classification unavailable. Please check the {self.provider} API connection."

    def recommend_music_category(self, features):
        try:
            return self._run_chain("""
            Classify this music into a category like Chill, Romantic, Party, Study, Sad, Workout, etc. {features}
            """, features)
        except Exception as e:
            print(f"Music category recommendation error: {str(e)}")
            return f"Music category classification unavailable. Please check the {self.provider} API connection."

    # New enhanced methods for more specific music analysis
    def analyze_lyric_improvement(self, features, current_lyrics=None):
        """Suggests improvements for song lyrics based on the audio analysis"""
        try:
            return self._run_chain("""
            Based on this audio analysis: {features}
            
            Provide recommendations for lyrical content and style that would complement 
            the musical qualities. Consider the mood, tempo, and emotional tone.
            """+ (f"\n\nCurrent lyrics: {current_lyrics}" if current_lyrics else ""), features)
        except Exception as e:
            print(f"Lyric improvement analysis error: {str(e)}")
            return f"Lyric improvement suggestions unavailable. Please check the {self.provider} API connection."
            
    def analyze_commercial_potential(self, features):
        """Analyzes commercial potential of the track"""
        try:
            return self._run_chain("""
            Analyze the commercial potential of this track based on: {features}
            
            Consider:
            1. Current music market trends
            2. Similar successful tracks
            3. Target audience size and engagement
            4. Streaming potential
            5. Licensing opportunities
            
            Provide a detailed assessment with specific recommendations.
            """, features)
        except Exception as e:
            print(f"Commercial potential analysis error: {str(e)}")
            return f"Commercial potential analysis unavailable. Please check the {self.provider} API connection."