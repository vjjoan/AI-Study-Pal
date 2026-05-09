"""
AI Study Pal - Main Flask Application

"""

from flask import Flask, render_template, request, jsonify, send_file
import json
import csv
import io
import os
from datetime import datetime, timedelta

from modules.data_setup import load_and_clean_data, generate_visualization
from modules.ml_quiz import QuizGenerator
from modules.dl_summarizer import TextSummarizer
from modules.nlp_tips import StudyTipsGenerator
from modules.study_plan import StudyPlanGenerator

app = Flask(__name__)

# Initialize modules
quiz_gen = QuizGenerator()
summarizer = TextSummarizer()
tips_gen = StudyTipsGenerator()
plan_gen = StudyPlanGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/study-plan', methods=['POST'])
def generate_study_plan():
    data = request.get_json()
    subject = data.get('subject', 'General')
    hours = int(data.get('hours', 2))
    goal = data.get('goal', 'exam_prep')
    days = int(data.get('days', 7))

    plan = plan_gen.generate(subject, hours, goal, days)
    return jsonify({'success': True, 'plan': plan})

@app.route('/api/quiz', methods=['POST'])
def generate_quiz():
    data = request.get_json()
    subject = data.get('subject', 'General')
    difficulty = data.get('difficulty', 'easy')
    num_questions = int(data.get('num_questions', 5))

    quiz = quiz_gen.generate_quiz(subject, difficulty, num_questions)
    return jsonify({'success': True, 'quiz': quiz})

@app.route('/api/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'success': False, 'error': 'No text provided'})

    summary = summarizer.summarize(text)
    feedback = summarizer.get_feedback(subject=data.get('subject', 'your subject'))
    return jsonify({'success': True, 'summary': summary, 'feedback': feedback})

@app.route('/api/tips', methods=['POST'])
def get_study_tips():
    data = request.get_json()
    text = data.get('text', '')
    subject = data.get('subject', 'General')

    tips = tips_gen.generate_tips(text, subject)
    keywords = tips_gen.extract_keywords(text)
    return jsonify({'success': True, 'tips': tips, 'keywords': keywords})

@app.route('/api/resources', methods=['POST'])
def get_resources():
    data = request.get_json()
    subject = data.get('subject', 'General')
    resources = quiz_gen.get_resources(subject)
    return jsonify({'success': True, 'resources': resources})

@app.route('/api/download-schedule', methods=['POST'])
def download_schedule():
    data = request.get_json()
    plan = data.get('plan', [])
    subject = data.get('subject', 'Study')

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Day', 'Date', 'Subject', 'Topic', 'Duration (hours)', 'Activity', 'Goal'])

    for item in plan:
        writer.writerow([
            item.get('day', ''),
            item.get('date', ''),
            subject,
            item.get('topic', ''),
            item.get('hours', ''),
            item.get('activity', ''),
            item.get('goal', '')
        ])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'{subject}_study_schedule.csv'
    )

@app.route('/api/visualization', methods=['GET'])
def get_visualization():
    img_base64 = generate_visualization()
    return jsonify({'success': True, 'image': img_base64})

@app.route('/api/eda', methods=['GET'])
def get_eda():
    stats = load_and_clean_data()
    return jsonify({'success': True, 'stats': stats})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
