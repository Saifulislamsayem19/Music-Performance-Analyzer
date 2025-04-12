import os
import numpy as np
import mido
import music21
from fractions import Fraction
from sklearn.preprocessing import StandardScaler
from openai import OpenAI
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import json
import plotly.io as pio
import datetime

# Load environment variables
load_dotenv()

# Initialize OpenAI client with validation
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable")

client = OpenAI(api_key=openai_api_key)

# Helper function to round values to 2 decimal places
def round_to_2dp(value):
    """Round numerical values to 2 decimal places"""
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    return value

class MusicPerformanceAnalyzer:
    def __init__(self, midi_file_path):
        self.midi_file_path = midi_file_path
        self.midi_data = mido.MidiFile(midi_file_path)
        self.music21_score = music21.converter.parse(midi_file_path)
    
    # All the analyzer methods remain unchanged
    def analyze_pitch_accuracy(self):
        """
        Advanced pitch analysis using music theory and statistical methods
        """
        # Extract all pitches
        pitches = [note.pitch.midi for note in self.music21_score.flatten().notesAndRests if note.isNote]
        
        if not pitches:
            return {
                'pitch_range': 0,
                'mean_pitch': 0,
                'pitch_variance': 0,
                'out_of_scale_notes': 0
            }
        
        # Calculate pitch-related metrics with rounding
        pitch_analysis = {
            'pitch_range': int(max(pitches) - min(pitches)),
            'mean_pitch': round_to_2dp(np.mean(pitches)),
            'pitch_variance': round_to_2dp(np.var(pitches)),
            'out_of_scale_notes': int(self._detect_out_of_scale_notes())
        }
        
        return pitch_analysis
    
    def _detect_out_of_scale_notes(self):
        """
        Detect notes that are outside the expected musical scale
        """
        out_of_scale_count = 0
        try:
            key = self.music21_score.analyze('key')
            scale = key.getScale()
            scale_pitches = scale.getPitches()
            scale_pitch_classes = {p.midi % 12 for p in scale_pitches}
            
            for note in self.music21_score.flatten().notesAndRests:
                if note.isNote:
                    note_pitch_class = note.pitch.midi % 12
                    if note_pitch_class not in scale_pitch_classes:
                        out_of_scale_count += 1
        except Exception as e:
            logging.warning(f"Error detecting out of scale notes: {str(e)}")
        
        return out_of_scale_count
    
    def analyze_rhythm_precision(self):
        """
        Advanced rhythm and timing analysis
        """
        # Analyze note durations and timing
        durations = [note.duration.quarterLength for note in self.music21_score.flatten().notesAndRests]
        
        if not durations:
            return {
                'average_note_duration': 0,
                'duration_variance': 0,
                'timing_consistency': {'mean_timing_deviation': 0, 'timing_variance': 0},
                'tempo_variations': {'tempo_count': 0, 'tempo_variation': 0}
            }
        
        rhythm_analysis = {
            'average_note_duration': round_to_2dp(np.mean(durations)),
            'duration_variance': round_to_2dp(np.var(durations)),
            'timing_consistency': self._calculate_timing_consistency(),
            'tempo_variations': self._detect_tempo_changes()
        }
        
        return rhythm_analysis
    
    def _calculate_timing_consistency(self):
        """
        Calculate the consistency of note timings
        """
        offsets = [note.offset for note in self.music21_score.flatten().notesAndRests]
        
        if len(offsets) <= 1:
            return {'mean_timing_deviation': 0, 'timing_variance': 0}
            
        timing_differences = np.diff(offsets)
        
        return {
            'mean_timing_deviation': round_to_2dp(np.mean(np.abs(timing_differences))),
            'timing_variance': round_to_2dp(np.var(timing_differences))
        }
    
    def _detect_tempo_changes(self):
        """
        Detect tempo changes throughout the performance
        """
        tempos = []
        for track in self.midi_data.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempos.append(mido.tempo2bpm(msg.tempo))
        
        return {
            'tempo_count': len(tempos),
            'tempo_variation': round_to_2dp(np.var(tempos)) if tempos else 0
        }
    
    def analyze_dynamics(self):
        """
        Analyze dynamic variations and velocity
        """
        velocities = []
        for track in self.midi_data.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    velocities.append(msg.velocity)
        
        if not velocities:
            return {
                'velocity_range': 0,
                'mean_velocity': 0,
                'velocity_variance': 0,
                'dynamic_changes': 0
            }
        
        dynamics_analysis = {
            'velocity_range': max(velocities) - min(velocities),
            'mean_velocity': round_to_2dp(np.mean(velocities)),
            'velocity_variance': round_to_2dp(np.var(velocities)),
            'dynamic_changes': self._detect_dynamic_changes(velocities)
        }
        
        return dynamics_analysis
    
    def _detect_dynamic_changes(self, velocities):
        """
        Detect sudden dynamic changes
        """
        if len(velocities) <= 1:
            return 0
            
        velocity_changes = np.diff(velocities)
        sudden_changes = np.sum(np.abs(velocity_changes) > np.std(velocity_changes) * 2)
        
        return int(sudden_changes)
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive performance analysis report
        """
        pitch_analysis = self.analyze_pitch_accuracy()
        rhythm_analysis = self.analyze_rhythm_precision()
        dynamics_analysis = self.analyze_dynamics()
        
        # Calculate an overall performance score
        performance_score = self._calculate_performance_score(
            pitch_analysis, rhythm_analysis, dynamics_analysis
        )
        
        return {
            'pitch_analysis': pitch_analysis,
            'rhythm_analysis': rhythm_analysis,
            'dynamics_analysis': dynamics_analysis,
            'performance_score': round_to_2dp(performance_score)
        }
    
    def _calculate_performance_score(self, pitch, rhythm, dynamics):
        """
        Calculate an overall performance score based on different musical aspects
        """
        # Weighted scoring system
        pitch_score = self._normalize_metric(pitch['pitch_variance'], 0, 100)
        rhythm_score = self._normalize_metric(rhythm['duration_variance'], 0, 100)
        dynamics_score = self._normalize_metric(dynamics['velocity_variance'], 0, 100)
        
        # Weighted average with more importance to pitch and rhythm
        performance_score = (
            0.4 * pitch_score + 
            0.4 * rhythm_score + 
            0.2 * dynamics_score
        )
        
        return round_to_2dp(performance_score)
    
    def _normalize_metric(self, value, min_val, max_val):
        """
        Normalize a metric to a 0-100 scale
        """
        # Ensure float conversion
        value = float(value)
        min_val = float(min_val)
        max_val = float(max_val)
        
        # Avoid division by zero
        if max_val == min_val:
            return 50.0  # Return middle value
            
        normalized = max(min(((value - min_val) / (max_val - min_val)) * 100, 100), 0)
        return round_to_2dp(normalized)
    
    def analyze_specific_errors(self):
        """
        Detect specific common performance errors
        """
        errors = []
        
        try:
            # Detect pitch errors
            key = self.music21_score.analyze('key')
            for note in self.music21_score.flatten().notesAndRests:
                if note.isNote:
                    # Check if note is out of key
                    if note.pitch.pitchClass not in [p.pitchClass for p in key.pitches]:
                        errors.append({
                            'type': 'pitch_error',
                            'description': f"Note {note.pitch.nameWithOctave} at measure {note.measureNumber} is out of key ({key.tonic.name} {key.mode})",
                            'measure': int(note.measureNumber),
                            'severity': 'medium'
                        })
        except Exception as e:
            logging.warning(f"Error analyzing pitch errors: {str(e)}")
        
        try:
            # Detect rhythm errors
            time_signatures = self.music21_score.getTimeSignatures()
            if time_signatures:
                time_signature = time_signatures[0]
                for note in self.music21_score.flatten().notesAndRests:
                    # Check for notes that cross barlines inappropriately
                    if note.duration.quarterLength > 4 and time_signature.numerator == 4:
                        errors.append({
                            'type': 'rhythm_error',
                            'description': f"Unusually long note ({round_to_2dp(note.duration.quarterLength)} beats) at measure {note.measureNumber}",
                            'measure': int(note.measureNumber),
                            'severity': 'low'
                        })
        except Exception as e:
            logging.warning(f"Error analyzing rhythm errors: {str(e)}")
        
        try:
            # Detect dynamic inconsistencies
            velocities = []
            for track in self.midi_data.tracks:
                for msg in track:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        velocities.append(msg.velocity)
            
            if velocities:
                avg_velocity = np.mean(velocities)
                for i, velocity in enumerate(velocities):
                    if abs(velocity - avg_velocity) > 30:
                        errors.append({
                            'type': 'dynamic_error',
                            'description': f"Unusual dynamic change at note {i+1}",
                            'position': i+1,
                            'severity': 'low'
                        })
        except Exception as e:
            logging.warning(f"Error analyzing dynamic errors: {str(e)}")
        
        return errors
    
    def visualize_performance_errors(self, errors):
        """
        Create a visualization of performance errors and return the figure
        with improved display and levels
        """
        # Extract notes and their properties
        notes = []
        for note in self.music21_score.flatten().notesAndRests:
            if note.isNote:
                # Convert all values to native Python types
                notes.append({
                    'measure': int(note.measureNumber),
                    'offset': round_to_2dp(float(note.offset)),
                    'pitch': int(note.pitch.midi),
                    'duration': round_to_2dp(float(note.duration.quarterLength)),
                    'error': False,
                    'error_type': None,
                    'severity': None
                })
        
        if not notes:
            # Return empty figure if no notes
            return go.Figure()
        
        # Mark errors
        for error in errors:
            if 'measure' in error:
                for note in notes:
                    if note['measure'] == error['measure']:
                        note['error'] = True
                        note['error_type'] = error['type']
                        note['severity'] = error['severity']
        
        # Create data for visualization - ensure all values are native Python types
        measures = [int(note['measure']) for note in notes]
        pitches = [int(note['pitch']) for note in notes]
        durations = [float(note['duration']) * 30 for note in notes]  # Scale up for better visibility
        
        # Prepare color mapping based on error type and severity
        colors = []
        for note in notes:
            if not note['error']:
                colors.append('blue')
            elif note['severity'] == 'low':
                colors.append('yellow')
            elif note['severity'] == 'medium':
                colors.append('orange')
            else:
                colors.append('red')
        
        # Define pitch levels/ranges for better visualization
        pitch_levels = {
            'Very Low': (0, 48),    # 0-48 (C0-C4)
            'Low': (48, 60),        # 48-60 (C4-C5)
            'Medium': (60, 72),     # 60-72 (C5-C6)
            'High': (72, 84),       # 72-84 (C6-C7)
            'Very High': (84, 128)  # 84-127 (C7+)
        }
        
        # Create subplots
        fig = make_subplots(rows=2, cols=1, 
                            subplot_titles=("Pitch Distribution", "Performance Timeline"),
                            specs=[[{"type": "scatter"}], [{"type": "scatter"}]],
                            vertical_spacing=0.15)
        
        # Add pitch scatter plot
        fig.add_trace(
            go.Scatter(
                x=measures, 
                y=pitches,
                mode='markers',
                marker=dict(
                    size=durations,
                    color=colors,
                    line=dict(width=1, color='black')
                ),
                name='Notes',
                hovertemplate='Measure: %{x}<br>Pitch: %{y}<br>Duration: %{text} beats',
                text=[str(round_to_2dp(d/30)) for d in durations]  # Fixed duration display
            ),
            row=1, col=1
        )
        
        # Add timeline representation
        offsets = [float(note['offset']) for note in notes]
        fig.add_trace(
            go.Scatter(
                x=offsets,
                y=[1] * len(offsets),  # Constant y for timeline
                mode='markers',
                marker=dict(
                    size=durations,
                    color=colors,
                    line=dict(width=1, color='black')
                ),
                name='Timeline',
                hovertemplate='Offset: %{x:.2f}<br>Duration: %{text} beats',
                text=[str(round_to_2dp(d/30)) for d in durations]  # Fixed duration display
            ),
            row=2, col=1
        )
        
        # Add pitch level ranges as horizontal regions
        colors_alpha = {'Very Low': 'rgba(135, 206, 250, 0.2)',  # Light blue
                      'Low': 'rgba(144, 238, 144, 0.2)',       # Light green
                      'Medium': 'rgba(255, 255, 224, 0.2)',    # Light yellow
                      'High': 'rgba(255, 218, 185, 0.2)',      # Light peach
                      'Very High': 'rgba(255, 182, 193, 0.2)'} # Light pink
        
        # Add the pitch level ranges
        for level_name, (low, high) in pitch_levels.items():
            fig.add_shape(
                type="rect",
                xref="paper",
                yref="y",
                x0=0,
                y0=low,
                x1=1,
                y1=high,
                fillcolor=colors_alpha[level_name],
                layer="below",
                line_width=0,
                row=1, col=1,
            )
            
            # Add label for each pitch range
            fig.add_annotation(
                x=0,
                y=(low + high) / 2,
                xref="paper",
                yref="y",
                text=level_name,
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                opacity=0.8,
                row=1, col=1
            )
        
        # Add legend for error severity
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=15, color='blue'),
                name='No Error',
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=15, color='yellow'),
                name='Low Severity Error',
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=15, color='orange'),
                name='Medium Severity Error',
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=15, color='red'),
                name='High Severity Error',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Performance Analysis Visualization",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=80, r=80, t=100, b=80),
        )
        
        # Update axes labels and ranges
        fig.update_yaxes(title_text="Pitch (MIDI note number)", row=1, col=1)
        fig.update_xaxes(title_text="Measure Number", row=1, col=1)
        fig.update_yaxes(title_text="Timeline", row=2, col=1, showticklabels=False)
        fig.update_xaxes(title_text="Time Offset", row=2, col=1)
        
        return fig
    

def generate_ai_feedback(analysis_report, specific_errors):
    """
    Generate detailed, personalized AI feedback with specific error references
    """
    try:
        # Format specific errors for the prompt
        error_descriptions = "\n".join([f"- {error['type']}: {error['description']}" for error in specific_errors[:5]])
        
        if not error_descriptions:
            error_descriptions = "- No specific errors detected."
        
        prompt = f"""
        Analyze this detailed music performance report:
        
        Pitch Analysis:
        - Pitch Range: {analysis_report['pitch_analysis']['pitch_range']}
        - Out of Scale Notes: {analysis_report['pitch_analysis']['out_of_scale_notes']}
        
        Rhythm Analysis:
        - Average Note Duration: {analysis_report['rhythm_analysis']['average_note_duration']}
        - Tempo Variations: {analysis_report['rhythm_analysis']['tempo_variations']['tempo_count']}
        
        Dynamics Analysis:
        - Velocity Range: {analysis_report['dynamics_analysis']['velocity_range']}
        - Dynamic Changes: {analysis_report['dynamics_analysis']['dynamic_changes']}
        
        Overall Performance Score: {analysis_report['performance_score']}
        
        Specific Errors Detected:
        {error_descriptions}

        FORMATTING REQUIREMENTS:
        1. DO NOT use any special formatting markers like # or * unless specifically requested
        2. DO NOT use symbols like -> or => in your responses
        3. DO NOT start sentences with symbols or special characters
        
        Provide a detailed, constructive, and encouraging musical performance feedback. 
        Highlight strengths and suggest specific improvements for pitch accuracy, 
        rhythmic precision, and dynamic expression.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  
            messages=[
                {"role": "system", "content": "You are an expert music instructor providing detailed, supportive, and actionable feedback, explaining music concepts in simple terms appropriate for students."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logging.error(f"Error generating AI feedback: {str(e)}")
        return f"Error generating feedback: {str(e)}"

def setup_logging(app):
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Configure file handler
    file_handler = RotatingFileHandler(
        'logs/music_analyzer.log', 
        maxBytes=10240, 
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)

# Improved JSON encoder that handles more types including fractions
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        
        if isinstance(obj, Fraction):
            return round_to_2dp(float(obj))
        elif hasattr(obj, '__float__'):
            return round_to_2dp(float(obj))
        elif hasattr(obj, '__int__'):
            return int(obj)
        elif hasattr(obj, '__str__'):
            return str(obj)
        # Handle music21 specific types if needed
        elif isinstance(obj, music21.base.Music21Object):
            return str(obj)
        # Handle any other special types here
        return super().default(obj)

# Flask App Configuration
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configure Flask to use the custom encoder
app.json_encoder = CustomJSONEncoder

setup_logging(app)

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Ensure the static directory exists
os.makedirs('static', exist_ok=True)
os.makedirs(os.path.join('static', 'visualizations'), exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/static/<path:filename>')  
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/visualization/<path:vis_id>')
def serve_visualization(vis_id):
    vis_path = os.path.join(app.static_folder, 'visualizations', f"{vis_id}.html")
    if os.path.exists(vis_path):
        return send_from_directory(os.path.join(app.static_folder, 'visualizations'), f"{vis_id}.html")
    return "Visualization not found", 404

@app.route('/status')
def status():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.datetime.now().isoformat(),
        'service': 'Music Performance Analyzer',
        'version': '1.0.0'
    })

@app.route('/analyze_music', methods=['POST'])
def analyze_music():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Validate file extension
    if not file.filename.lower().endswith(('.mid', '.midi')):
        return jsonify({'error': 'Invalid file format. Please upload a MIDI file.'}), 400
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_file:
            file_path = temp_file.name
            file.save(file_path)
        
        try:
            # Perform analysis
            analyzer = MusicPerformanceAnalyzer(file_path)
            analysis_report = analyzer.generate_comprehensive_report()
            
            # Detect specific errors
            specific_errors = analyzer.analyze_specific_errors()
            
            # Generate visualization if there are errors
            visualization_url = None
            if specific_errors:
                try:
                    fig = analyzer.visualize_performance_errors(specific_errors)
                    
                    # Ensure visualization directory exists
                    os.makedirs(os.path.join(app.static_folder, 'visualizations'), exist_ok=True)
                    
                    # Generate unique ID for this visualization
                    vis_id = f"vis_{hash(file.filename)}_{hash(str(specific_errors))}"
                    vis_path = os.path.join(app.static_folder, 'visualizations', f"{vis_id}.html")
                    
                    # Save visualization as standalone HTML
                    pio.write_html(fig, file=vis_path, auto_open=False)
                    
                    visualization_url = f"/visualization/{vis_id}"
                except Exception as vis_error:
                    app.logger.error(f"Error creating visualization: {str(vis_error)}")
                    # Continue without visualization if it fails
            
            # Generate AI feedback
            try:
                ai_feedback = generate_ai_feedback(analysis_report, specific_errors)
            except Exception as ai_error:
                app.logger.error(f"Error generating AI feedback: {str(ai_error)}")
                ai_feedback = "Sorry, there was an error generating AI feedback for your performance."
            
            response_data = {
                'analysis': analysis_report,
                'errors': specific_errors,
                'visualization': visualization_url,
                'feedback': ai_feedback
            }
            
            # Clean up the temporary file
            os.unlink(file_path)
            
            return jsonify(response_data)
            
        except Exception as analysis_error:
            app.logger.error(f"Error analyzing MIDI file: {str(analysis_error)}")
            return jsonify({'error': f'Error analyzing MIDI file: {str(analysis_error)}'}), 500
    
    except Exception as e:
        app.logger.error(f"General error processing request: {str(e)}")
        # Clean up in case of error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.unlink(file_path)
        return jsonify({'error': f'Error processing MIDI file: {str(e)}'}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
