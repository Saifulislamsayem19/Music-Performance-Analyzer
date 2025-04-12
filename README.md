# Music Performance Analyzer

A sophisticated application that analyzes MIDI performances, providing detailed feedback on pitch accuracy, rhythm precision, and dynamic expression.

![image](https://github.com/user-attachments/assets/e388fd76-7f48-41f1-9025-249449610f21)

## 🎵 Overview

This Music Performance Analyzer is a Flask-based web application that leverages advanced music theory, statistical analysis, and AI to provide comprehensive feedback on musical performances. It processes MIDI files and generates detailed reports, visualizations, and personalized coaching feedback to help musicians improve their playing.

## ✨ Features

- **Comprehensive Musical Analysis**
  - Pitch accuracy and scale adherence
  - Rhythm precision and timing consistency
  - Dynamic expression and velocity variations
  
- **Specific Error Detection**
  - Identifies notes outside the musical key
  - Highlights unusual note durations
  - Detects inconsistent dynamics

- **Interactive Visualizations**
  - Pitch distribution graphs
  - Performance timeline visualization
  - Color-coded error identification

- **AI-Powered Feedback**
  - Generates personalized, constructive feedback
  - Highlights strengths and areas for improvement
  - Provides actionable suggestions for practice

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Analysis**: Music21, Mido, NumPy, SciKit-Learn
- **AI Integration**: OpenAI API
- **Visualization**: Plotly
- **Frontend**: HTML, CSS, JavaScript

##  Requirements

- Python 3.8+
- OpenAI API key
- Python packages listed in requirements.txt

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/music-performance-analyzer.git
   cd music-performance-analyzer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

5. Run the application:
   ```bash
   python app.py
   ```

6. Access the web interface at `http://localhost:5000`

## 📊 How It Works

1. **Upload MIDI File**: Submit a MIDI recording of your musical performance
2. **Analysis**: The application analyzes pitch, rhythm, and dynamics
3. **Error Detection**: Specific performance errors are identified and categorized
4. **Visualization**: Interactive graphs illustrate performance characteristics
5. **AI Feedback**: Personalized coaching feedback is generated based on the analysis

## 🖼️ Screenshots

### Analysis Dashboard
![image](https://github.com/user-attachments/assets/848fc9da-1120-4bce-b08f-5285df880b91)
![image](https://github.com/user-attachments/assets/1a0a39e8-918f-4af8-89fc-117635da7d09)

### Performance Visualization
![image](https://github.com/user-attachments/assets/c315f2bd-25e5-4f82-8f5a-2a5dfb95775c)

### AI Feedback
![image](https://github.com/user-attachments/assets/70a18618-36bb-4a41-8c6b-db51ab24c26c)

## 🔍 Usage Example

```python
analyzer = MusicPerformanceAnalyzer('your_performance.mid')
analysis_report = analyzer.generate_comprehensive_report()
specific_errors = analyzer.analyze_specific_errors()
feedback = generate_ai_feedback(analysis_report, specific_errors)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- Music21 for music theory and analysis capabilities
- OpenAI for the natural language processing capabilities
- Plotly for the interactive visualizations
