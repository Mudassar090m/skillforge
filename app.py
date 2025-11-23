"""
SkillForge - AI-Powered Course Generator
Author: Mudassar ijaz
Description: Transform any idea into a comprehensive learning experience
"""

import gradio as gr
import json
import os
import re
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# API Configuration
def get_api_config():
    """Get API configuration from environment"""
    api_provider = os.environ.get('API_PROVIDER', 'groq').lower()
    api_key = os.environ.get('API_KEY', '')

    if api_provider == 'groq':
        api_key = api_key or os.environ.get('GROQ_API_KEY')
    elif api_provider == 'gemini':
        api_key = api_key or os.environ.get('GEMINI_API_KEY')

    if not api_key:
        raise ValueError(f"Please set API_KEY or {api_provider.upper()}_API_KEY environment variable")

    return api_provider, api_key

def extract_json_from_text(text):
    """Extract and clean JSON from response text"""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return text

def call_ai_api(prompt, system_prompt=""):
    """Make API call to selected AI provider"""
    try:
        api_provider, api_key = get_api_config()

        if api_provider == 'groq':
            from groq import Groq
            client = Groq(api_key=api_key)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.5,
                max_tokens=4096
            )
            return chat_completion.choices[0].message.content

        elif api_provider == 'gemini':
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')

            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = model.generate_content(full_prompt)
            return response.text

        else:
            return f"Unknown API provider: {api_provider}"

    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            return "❌ RATE LIMIT: Please wait and try again or switch API provider."
        return f"❌ API Error: {error_msg}"

def generate_course(topic, difficulty, progress=gr.Progress()):
    """Generate complete course structure"""
    progress(0.1, desc="🚀 Initializing course generation...")

    system_prompt = """You are SkillForge AI. Create comprehensive courses. Respond with ONLY valid JSON."""

    prompt = f"""Create a course for: "{topic}" at {difficulty} level.

Return ONLY this JSON:
{{
  "title": "Course Title",
  "difficulty": "{difficulty}",
  "learning_outcomes": ["Outcome 1", "Outcome 2", "Outcome 3"],
  "modules": [
    {{
      "module_number": 1,
      "module_title": "Module Title",
      "lessons": [
        {{
          "lesson_number": 1,
          "lesson_title": "Lesson Title",
          "explanation": "Detailed explanation",
          "examples": ["Example 1", "Example 2"],
          "key_takeaways": ["Takeaway 1", "Takeaway 2"]
        }}
      ]
    }}
  ]
}}

Create 4-5 modules with 3-4 lessons each."""

    progress(0.3, desc="🧠 AI is designing your course...")
    response = call_ai_api(prompt, system_prompt)

    progress(0.7, desc="📚 Structuring course content...")

    try:
        json_str = extract_json_from_text(response)
        course_data = json.loads(json_str)

        if not isinstance(course_data, dict) or 'modules' not in course_data:
            raise ValueError("Invalid course structure")

        progress(1.0, desc="✅ Course generated successfully!")
        return course_data, create_course_display(course_data), create_mindmap(course_data)

    except Exception as e:
        progress(1.0, desc="❌ Error occurred")
        error_msg = f"<div style='color: red; padding: 20px;'><h3>Error: {str(e)}</h3><pre>{response[:500]}</pre></div>"
        return None, error_msg, "Error"

def create_course_display(course_data):
    """Create beautiful HTML display of course"""
    if not course_data:
        return "<p style='color: red;'>No course data available</p>"

    html = f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 1200px; margin: 0 auto;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; border-radius: 20px; margin-bottom: 30px;">
            <h1 style="color: white; margin: 0; font-size: 2.5em;">📚 {course_data.get('title', 'Course')}</h1>
            <p style="color: #f0f0f0; margin-top: 10px; font-size: 1.2em;">Difficulty: {course_data.get('difficulty', 'N/A')}</p>
        </div>
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 30px; border-radius: 15px; margin-bottom: 30px;">
            <h2 style="color: white; margin-top: 0;">🎯 Learning Outcomes</h2>
            <ul style="color: white; font-size: 1.1em; line-height: 1.8;">
    """

    for outcome in course_data.get('learning_outcomes', []):
        html += f"<li>{outcome}</li>"

    html += "</ul></div><div>"

    colors = [
        "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
        "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
        "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
        "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
    ]

    for idx, module in enumerate(course_data.get('modules', [])):
        color = colors[idx % len(colors)]
        html += f"""
        <div style="background: {color}; padding: 25px; border-radius: 15px; margin-bottom: 25px;">
            <h2 style="color: white;">📖 Module {module.get('module_number', idx+1)}: {module.get('module_title', '')}</h2>
        """

        for lesson in module.get('lessons', []):
            html += f"""
            <div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 10px; margin-top: 15px;">
                <h3 style="color: white;">💡 {lesson.get('lesson_title', '')}</h3>
                <p style="color: white; line-height: 1.7;">{lesson.get('explanation', '')}</p>
                <h4 style="color: white;">📌 Examples:</h4>
                <ul style="color: white;">
            """
            for example in lesson.get('examples', []):
                html += f"<li>{example}</li>"
            html += "</ul><h4 style='color: white;'>✅ Key Takeaways:</h4><ul style='color: white;'>"
            for takeaway in lesson.get('key_takeaways', []):
                html += f"<li>{takeaway}</li>"
            html += "</ul></div>"

        html += "</div>"

    html += "</div></div>"
    return html

def create_mindmap(course_data):
    """Generate ASCII mindmap"""
    if not course_data:
        return "No course data"

    mindmap = f"""
🎓 {course_data.get('title', 'Course')}
📊 Difficulty: {course_data.get('difficulty', 'N/A')}

"""

    for idx, module in enumerate(course_data.get('modules', [])):
        is_last = idx == len(course_data.get('modules', [])) - 1
        prefix = "└──" if is_last else "├──"
        mindmap += f"{prefix} 📚 Module {module.get('module_number', idx+1)}: {module.get('module_title', '')}\n"

        lessons = module.get('lessons', [])
        for l_idx, lesson in enumerate(lessons):
            is_last_lesson = l_idx == len(lessons) - 1
            l_prefix = "    └──" if is_last_lesson else "    ├──" if is_last else "│   └──" if is_last_lesson else "│   ├──"
            mindmap += f"{l_prefix} 💡 {lesson.get('lesson_title', '')}\n"

        if not is_last:
            mindmap += "│\n"

    return mindmap

def create_download_content(course_data):
    """Create downloadable text version"""
    if not course_data:
        return None

    content = f"""
{'='*80}
SKILLFORGE COURSE
{'='*80}

Title: {course_data.get('title', 'Course')}
Difficulty: {course_data.get('difficulty', 'N/A')}

LEARNING OUTCOMES
{'-'*80}
"""

    for idx, outcome in enumerate(course_data.get('learning_outcomes', []), 1):
        content += f"{idx}. {outcome}\n"

    content += f"\n{'='*80}\nCOURSE MODULES\n{'='*80}\n\n"

    for module in course_data.get('modules', []):
        content += f"\nMODULE {module.get('module_number', '')}: {module.get('module_title', '')}\n{'-'*80}\n"

        for lesson in module.get('lessons', []):
            content += f"\nLesson {lesson.get('lesson_number', '')}: {lesson.get('lesson_title', '')}\n\n"
            content += f"{lesson.get('explanation', '')}\n\n"
            content += "Examples:\n"
            for example in lesson.get('examples', []):
                content += f"  • {example}\n"
            content += "\nKey Takeaways:\n"
            for takeaway in lesson.get('key_takeaways', []):
                content += f"  ✓ {takeaway}\n"
            content += "\n"

    content += f"\n{'='*80}\nGenerated by SkillForge - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'='*80}\n"

    filename = f"course_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = os.path.join(os.getcwd(), filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    return filepath

def generate_quiz(course_data, module_number, progress=gr.Progress()):
    """Generate quiz for specific module"""
    progress(0.2, desc="🎯 Generating quiz...")

    if not course_data or not course_data.get('modules'):
        return None, "<div style='color: red;'>No course data</div>"

    module_idx = module_number - 1
    if module_idx >= len(course_data['modules']):
        return None, "<div style='color: red;'>Invalid module</div>"

    module = course_data['modules'][module_idx]
    lessons_titles = ", ".join([lesson.get('lesson_title', '') for lesson in module.get('lessons', [])])

    system_prompt = "Generate quiz in JSON only."

    prompt = f"""Quiz for: {module.get('module_title', '')}
Topics: {lessons_titles}

Return ONLY JSON:
{{
  "module_title": "{module.get('module_title', '')}",
  "mcqs": [
    {{"question": "Q?", "options": ["A) Op1", "B) Op2", "C) Op3", "D) Op4"], "correct_answer": "A", "explanation": "Why"}},
    {{"question": "Q?", "options": ["A) Op1", "B) Op2", "C) Op3", "D) Op4"], "correct_answer": "B", "explanation": "Why"}},
    {{"question": "Q?", "options": ["A) Op1", "B) Op2", "C) Op3", "D) Op4"], "correct_answer": "C", "explanation": "Why"}},
    {{"question": "Q?", "options": ["A) Op1", "B) Op2", "C) Op3", "D) Op4"], "correct_answer": "D", "explanation": "Why"}},
    {{"question": "Q?", "options": ["A) Op1", "B) Op2", "C) Op3", "D) Op4"], "correct_answer": "A", "explanation": "Why"}}
  ],
  "true_false": [
    {{"statement": "Statement", "correct_answer": true, "explanation": "Why"}},
    {{"statement": "Statement", "correct_answer": false, "explanation": "Why"}},
    {{"statement": "Statement", "correct_answer": true, "explanation": "Why"}}
  ]
}}"""

    progress(0.5, desc="🧠 Creating questions...")
    response = call_ai_api(prompt, system_prompt)

    try:
        json_str = extract_json_from_text(response)
        quiz_data = json.loads(json_str)

        if not isinstance(quiz_data, dict) or 'mcqs' not in quiz_data:
            raise ValueError("Invalid quiz format")

        progress(1.0, desc="✅ Quiz ready!")

        info_html = f"""
        <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 30px; border-radius: 15px; color: white; text-align: center;">
            <h2>📝 Quiz: {quiz_data.get('module_title', 'Module')}</h2>
            <p>8 Questions (5 MCQs + 3 True/False)</p>
        </div>
        """

        return quiz_data, info_html

    except Exception as e:
        progress(1.0, desc="❌ Error")
        error_html = f"<div style='color: red; padding: 20px;'>Error: {str(e)}</div>"
        return None, error_html

def display_quiz_questions(quiz_data):
    """Display quiz questions"""
    if not quiz_data:
        return []

    html_outputs = []

    for idx, mcq in enumerate(quiz_data.get('mcqs', [])[:5], 1):
        html = f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 25px; border-radius: 15px; color: white; margin-bottom: 15px;">
            <h3>❓ MCQ {idx}</h3>
            <p style="font-size: 1.15em;"><strong>{mcq.get('question', '')}</strong></p>
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
                <strong>Options:</strong><br>
        """
        for opt in mcq.get('options', []):
            html += f"<div>{opt}</div>"
        html += "</div></div>"
        html_outputs.append(html)

    while len(html_outputs) < 5:
        html_outputs.append("")

    for idx, tf in enumerate(quiz_data.get('true_false', [])[:3], 1):
        html = f"""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 25px; border-radius: 15px; color: white; margin-bottom: 15px;">
            <h3>✓ True/False {idx}</h3>
            <p style="font-size: 1.15em;"><strong>{tf.get('statement', '')}</strong></p>
        </div>
        """
        html_outputs.append(html)

    while len(html_outputs) < 8:
        html_outputs.append("")

    return html_outputs

def submit_quiz(quiz_data, *answers):
    """Calculate results"""
    if not quiz_data:
        return "<p style='color: red;'>No quiz data</p>"

    mcq_answers = list(answers[:5])
    tf_answers = list(answers[5:8])

    correct_count = 0
    total = 8

    for mcq, user_ans in zip(quiz_data.get('mcqs', [])[:5], mcq_answers):
        if user_ans and user_ans == mcq.get('correct_answer'):
            correct_count += 1

    for tf, user_ans in zip(quiz_data.get('true_false', [])[:3], tf_answers):
        correct_ans = "True" if tf.get('correct_answer') else "False"
        if user_ans and user_ans == correct_ans:
            correct_count += 1

    percentage = (correct_count / total * 100)

    if percentage >= 80:
        grade_color, grade_text, emoji = "#43e97b", "Excellent! 🎉", "🌟"
    elif percentage >= 60:
        grade_color, grade_text, emoji = "#fa709a", "Good Job! 👏", "👍"
    else:
        grade_color, grade_text, emoji = "#f5576c", "Keep Learning! 📚", "💪"

    html = f"""
    <div style="font-family: 'Segoe UI'; max-width: 1000px; margin: 0 auto;">
        <div style="background: {grade_color}; padding: 50px; border-radius: 20px; text-align: center; color: white;">
            <div style="font-size: 4em;">{emoji}</div>
            <h1>{grade_text}</h1>
            <p style="font-size: 3.5em; font-weight: bold;">{correct_count}/{total}</p>
            <p style="font-size: 1.8em;">Score: {percentage:.1f}%</p>
        </div>
        <h2 style="color: #667eea; margin: 30px 0;">📊 Detailed Results</h2>
    """

    for idx, (mcq, user_ans) in enumerate(zip(quiz_data.get('mcqs', [])[:5], mcq_answers), 1):
        is_correct = user_ans and user_ans == mcq.get('correct_answer')
        bg = "#43e97b" if is_correct else "#f5576c"
        icon = "✅" if is_correct else "❌"

        html += f"""
        <div style="background: {bg}; padding: 25px; border-radius: 15px; margin-bottom: 20px; color: white;">
            <h3>{icon} MCQ {idx}</h3>
            <p><strong>Q:</strong> {mcq.get('question', '')}</p>
            <p><strong>Your Answer:</strong> {user_ans or 'Not answered'}</p>
            <p><strong>Correct:</strong> {mcq.get('correct_answer', '')}</p>
            <p><strong>💡 Explanation:</strong> {mcq.get('explanation', '')}</p>
        </div>
        """

    for idx, (tf, user_ans) in enumerate(zip(quiz_data.get('true_false', [])[:3], tf_answers), 1):
        correct = "True" if tf.get('correct_answer') else "False"
        is_correct = user_ans and user_ans == correct
        bg = "#43e97b" if is_correct else "#f5576c"
        icon = "✅" if is_correct else "❌"

        html += f"""
        <div style="background: {bg}; padding: 25px; border-radius: 15px; margin-bottom: 20px; color: white;">
            <h3>{icon} T/F {idx}</h3>
            <p><strong>Statement:</strong> {tf.get('statement', '')}</p>
            <p><strong>Your Answer:</strong> {user_ans or 'Not answered'}</p>
            <p><strong>Correct:</strong> {correct}</p>
            <p><strong>💡 Explanation:</strong> {tf.get('explanation', '')}</p>
        </div>
        """

    html += "</div>"
    return html

# Global storage
course_storage = {}

def generate_and_store(topic, difficulty):
    """Generate and store course"""
    course_data, display, mindmap = generate_course(topic, difficulty)
    course_storage['current_course'] = course_data

    if course_data and course_data.get('modules'):
        download_path = create_download_content(course_data)
        module_options = [f"Module {m.get('module_number', i+1)}: {m.get('module_title', '')}"
                         for i, m in enumerate(course_data['modules'])]
        return (
            display, mindmap,
            gr.update(choices=module_options, value=module_options[0], visible=True),
            gr.update(value=download_path, visible=True),
            gr.update(visible=True)
        )
    return display, mindmap, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def generate_quiz_interface(module_selection):
    """Generate quiz interface"""
    if not module_selection or 'current_course' not in course_storage:
        return [""] + [""] * 8 + [gr.update(visible=False, value=None)] * 8 + [gr.update(visible=False), ""]

    try:
        module_number = int(module_selection.split(':')[0].replace('Module ', ''))
    except:
        return ["Error"] + [""] * 8 + [gr.update(visible=False, value=None)] * 8 + [gr.update(visible=False), ""]

    quiz_data, info_html = generate_quiz(course_storage['current_course'], module_number)
    course_storage['current_quiz'] = quiz_data

    if not quiz_data:
        return [info_html] + [""] * 8 + [gr.update(visible=False, value=None)] * 8 + [gr.update(visible=False), ""]

    question_htmls = display_quiz_questions(quiz_data)
    radio_updates = [gr.update(visible=True, value=None)] * 8

    return [info_html] + question_htmls + radio_updates + [gr.update(visible=True), ""]

def submit_quiz_answers(*answers):
    """Submit answers"""
    if 'current_quiz' not in course_storage:
        return "<div style='color: red;'>No quiz available</div>"
    return submit_quiz(course_storage['current_quiz'], *answers)

# CSS
custom_css = """
.gradio-container {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;}
h1 {text-align: center !important;}
.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important; color: white !important; font-weight: bold !important;
    transition: all 0.3s ease !important;
}
.primary:hover {transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;}
"""

# Gradio Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="purple", secondary_hue="pink")) as app:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="font-size: 3.5em; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: bold;">
            🎓 SkillForge
        </h1>
        <p style="font-size: 1.3em; color: #666;">Transform Any Idea Into a Comprehensive Learning Experience</p>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("📚 Course Generator"):
            gr.Markdown("### Generate Your Complete Course")
            with gr.Row():
                topic_input = gr.Textbox(label="Topic", placeholder="e.g., Python Programming", lines=2)
                difficulty_input = gr.Dropdown(choices=["Beginner", "Intermediate", "Advanced"], label="Difficulty", value="Intermediate")
            generate_btn = gr.Button("🚀 Generate Course", variant="primary", size="lg")
            course_output = gr.HTML()
            download_btn = gr.File(label="📥 Download Course", visible=False)

        with gr.Tab("🗺️ Course Blueprint"):
            gr.Markdown("### Visual Structure")
            mindmap_output = gr.Textbox(label="Course Mindmap", lines=20)

        with gr.Tab("📝 Quiz Generator"):
            gr.Markdown("### Test Your Knowledge")
            module_selector = gr.Dropdown(label="Select Module", choices=[], visible=False)
            quiz_btn = gr.Button("🎯 Generate Quiz", variant="primary", size="lg", visible=False)
            quiz_info = gr.HTML()
            gr.Markdown("---")

            with gr.Column():
                quiz_html_1 = gr.HTML()
                quiz_radio_1 = gr.Radio(choices=["A", "B", "C", "D"], label="Your Answer", visible=False)
                quiz_html_2 = gr.HTML()
                quiz_radio_2 = gr.Radio(choices=["A", "B", "C", "D"], label="Your Answer", visible=False)
                quiz_html_3 = gr.HTML()
                quiz_radio_3 = gr.Radio(choices=["A", "B", "C", "D"], label="Your Answer", visible=False)
                quiz_html_4 = gr.HTML()
                quiz_radio_4 = gr.Radio(choices=["A", "B", "C", "D"], label="Your Answer", visible=False)
                quiz_html_5 = gr.HTML()
                quiz_radio_5 = gr.Radio(choices=["A", "B", "C", "D"], label="Your Answer", visible=False)
                quiz_html_6 = gr.HTML()
                quiz_radio_6 = gr.Radio(choices=["True", "False"], label="Your Answer", visible=False)
                quiz_html_7 = gr.HTML()
                quiz_radio_7 = gr.Radio(choices=["True", "False"], label="Your Answer", visible=False)
                quiz_html_8 = gr.HTML()
                quiz_radio_8 = gr.Radio(choices=["True", "False"], label="Your Answer", visible=False)

            submit_quiz_btn = gr.Button("📊 Submit Quiz", variant="primary", size="lg", visible=False)
            quiz_results = gr.HTML()

    quiz_htmls = [quiz_html_1, quiz_html_2, quiz_html_3, quiz_html_4, quiz_html_5, quiz_html_6, quiz_html_7, quiz_html_8]
    quiz_radios = [quiz_radio_1, quiz_radio_2, quiz_radio_3, quiz_radio_4, quiz_radio_5, quiz_radio_6, quiz_radio_7, quiz_radio_8]

    generate_btn.click(generate_and_store, [topic_input, difficulty_input], [course_output, mindmap_output, module_selector, download_btn, quiz_btn])
    quiz_btn.click(generate_quiz_interface, [module_selector], [quiz_info] + quiz_htmls + quiz_radios + [submit_quiz_btn, quiz_results])
    submit_quiz_btn.click(submit_quiz_answers, quiz_radios, [quiz_results])

if __name__ == "__main__":
    app.launch()
