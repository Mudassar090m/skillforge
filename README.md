# skillforge
# 🎓 SkillForge - AI Course Generator

Transform any idea into a comprehensive learning experience with AI-powered course generation.

## 🌟 Features

- **📚 Course Generation**: Create complete courses with modules, lessons, examples, and key takeaways
- **🗺️ Course Blueprint**: Visual ASCII mindmap of course structure
- **📝 Interactive Quizzes**: Auto-generated MCQs and True/False questions with explanations
- **📥 Download**: Export courses as text files
- **🎨 Beautiful UI**: Modern, gradient-based interface with animations

## 🚀 Live Demo

Try it here: [SkillForge on Hugging Face Spaces](#)

## 💻 Local Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/skillforge.git
cd skillforge
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up API keys**

Choose one of the following providers:

**Option A: Groq (Recommended - Fast & Free)**
```bash
export API_PROVIDER=groq
export API_KEY=your_groq_api_key_here
```
Get your free API key: https://console.groq.com/keys

**Option B: Google Gemini (Free)**
```bash
export API_PROVIDER=gemini
export API_KEY=your_gemini_api_key_here
```
Get your free API key: https://aistudio.google.com/app/apikey

4. **Run the application**
```bash
python app.py
```

5. **Open in browser**
Navigate to: `http://localhost:7860`

## 🔧 Deployment

### Deploy to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Choose "Gradio" as the SDK
3. Upload `app.py` and `requirements.txt`
4. Add secrets in Space settings:
   - `API_PROVIDER` = `groq` or `gemini`
   - `API_KEY` = your API key
5. Your app will automatically deploy!

### Deploy to GitHub

1. Create a new repository on GitHub
2. Push your code:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/skillforge.git
git push -u origin main
```

## 📖 Usage

### Generate a Course

1. Navigate to the **Course Generator** tab
2. Enter your topic (e.g., "Machine Learning Basics")
3. Select difficulty level (Beginner/Intermediate/Advanced)
4. Click "🚀 Generate Course"
5. Download the course using the download button

### View Course Structure

1. Go to the **Course Blueprint** tab
2. View the ASCII mindmap of your course structure

### Take a Quiz

1. Navigate to the **Quiz Generator** tab
2. Select a module from the dropdown
3. Click "🎯 Generate Quiz"
4. Answer all questions
5. Click "📊 Submit Quiz" to see results with explanations

## 🛠️ Technology Stack

- **Frontend**: Gradio 4.44.0
- **AI Models**: 
  - Groq (Llama 3.3 70B)
  - Google Gemini 1.5 Flash
- **Language**: Python 3.9+

## 📊 System Architecture

```
User Interface (Gradio)
         ↓
  Controller Logic
         ↓
    API Handler
         ↓
   AI Services (Groq/Gemini)
         ↓
   JSON Parser & Validator
         ↓
  In-Memory Storage
         ↓
   HTML Renderer
         ↓
  Display to User
```

## 🎯 Real-World Applications

- **Students**: Free access to personalized courses
- **Professionals**: Quick upskilling for career growth
- **Teachers**: Save 20+ hours on course creation
- **Companies**: Reduce training costs by 80-90%
- **Entrepreneurs**: Learn multiple skills fast
- **Hobbyists**: Free courses for passion projects

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request




## 🙏 Acknowledgments

- Groq for providing fast AI inference
- Google for Gemini API
- Gradio for the amazing UI framework

## 📧 Contact

For questions or support, please open an issue or contact: mudassarijaz.tech@gmail.com

---

Made with ❤️ by [Mudassar ijaz]
