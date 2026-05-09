"""
Module 1: Python and Data Setup
- Loads educational text dataset
- Cleans data (deduplication, lowercasing)
- Performs EDA (subject counts)
- Generates basic visualizations
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import base64
import io
import json
import os

# ─────────────────────────────────────────────
# Sample educational dataset 
# ─────────────────────────────────────────────
RAW_DATA = [
    {"subject": "Mathematics", "topic": "Algebra", "text": "Algebra is a branch of mathematics dealing with symbols and the rules for manipulating those symbols. Variables represent numbers or quantities in algebraic expressions and equations."},
    {"subject": "Mathematics", "topic": "Calculus", "text": "Calculus is the mathematical study of continuous change. It has two major branches: differential calculus and integral calculus, which are related by the fundamental theorem of calculus."},
    {"subject": "Mathematics", "topic": "Geometry", "text": "Geometry is a branch of mathematics concerned with questions of shape, size, relative position of figures, and the properties of space."},
    {"subject": "Mathematics", "topic": "Statistics", "text": "Statistics is the discipline that concerns the collection, organization, analysis, interpretation, and presentation of data."},
    {"subject": "Science", "topic": "Physics", "text": "Physics is the natural science that studies matter, its motion and behavior through space and time, and the related entities of energy and force."},
    {"subject": "Science", "topic": "Chemistry", "text": "Chemistry is the scientific discipline involved with elements and compounds composed of atoms, molecules and ions: their composition, structure, properties, behavior and the changes they undergo during a reaction with other substances."},
    {"subject": "Science", "topic": "Biology", "text": "Biology is the natural science that studies life and living organisms, including their physical structure, chemical processes, molecular interactions, physiological mechanisms, development and evolution."},
    {"subject": "Science", "topic": "Astronomy", "text": "Astronomy is a natural science that studies celestial objects and phenomena. It uses mathematics, physics, and chemistry in order to explain their origin and evolution."},
    {"subject": "Computer Science", "topic": "Algorithms", "text": "An algorithm is a finite sequence of well-defined, computer-implementable instructions to solve a class of problems or to perform a computation."},
    {"subject": "Computer Science", "topic": "Data Structures", "text": "A data structure is a particular way of organizing and storing data in a computer so that it can be accessed and modified efficiently."},
    {"subject": "Computer Science", "topic": "Machine Learning", "text": "Machine learning is a method of data analysis that automates analytical model building. It is based on the idea that systems can learn from data and identify patterns."},
    {"subject": "Computer Science", "topic": "Networking", "text": "Computer networking is the practice of connecting computers and devices to share resources. Networks can be classified by geographic scale, such as LANs and WANs."},
    {"subject": "History", "topic": "World War II", "text": "World War II was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries forming two opposing military alliances."},
    {"subject": "History", "topic": "Ancient Rome", "text": "Ancient Rome was a civilization that grew from a small agricultural community on the Italian Peninsula to one of the largest empires in the ancient world."},
    {"subject": "History", "topic": "Industrial Revolution", "text": "The Industrial Revolution was the transition to new manufacturing processes in Europe and the United States from about 1760 to 1840."},
    {"subject": "English", "topic": "Grammar", "text": "English grammar is the set of structural rules of the English language. This includes the structure of words, phrases, clauses, sentences, and whole texts."},
    {"subject": "English", "topic": "Literature", "text": "Literature broadly refers to any collection of written work, but it is also used more narrowly for writings specifically considered to be an art form."},
    {"subject": "Mathematics", "topic": "Algebra", "text": "Algebra is a branch of mathematics dealing with symbols and the rules for manipulating those symbols."},  # duplicate
    {"subject": "Science", "topic": "Physics", "text": "Physics is the natural science that studies matter, its motion and behavior through space and time."},  # duplicate
]


def load_and_clean_data():
    """
    Load raw educational data, clean it, and return EDA statistics.
    """
    df = pd.DataFrame(RAW_DATA)

    # ── Cleaning ──────────────────────────────
    # Remove duplicates
    original_count = len(df)
    df.drop_duplicates(subset=['text'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Lowercase text
    df['text_clean'] = df['text'].str.lower().str.strip()
    df['topic_clean'] = df['topic'].str.lower().str.strip()
    df['word_count'] = df['text_clean'].apply(lambda x: len(x.split()))

    # ── EDA ───────────────────────────────────
    subject_counts = df['subject'].value_counts().to_dict()
    avg_word_count = round(df['word_count'].mean(), 1)
    topic_list = df['topic'].unique().tolist()

    stats = {
        'total_records': len(df),
        'duplicates_removed': original_count - len(df),
        'subjects': subject_counts,
        'unique_topics': len(topic_list),
        'avg_word_count': avg_word_count,
        'topics': topic_list
    }

    # Save cleaned data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/cleaned_dataset.csv', index=False)

    return stats


def generate_visualization():
    """
    Generate a subject distribution pie chart and return as base64 PNG.
    """
    df = pd.DataFrame(RAW_DATA)
    df.drop_duplicates(subset=['text'], inplace=True)
    subject_counts = df['subject'].value_counts()

    # ── Plot ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0f0f1a')

    # Pie Chart
    colors = ['#6366f1', '#22d3ee', '#f59e0b', '#10b981', '#f43f5e', '#a78bfa']
    wedges, texts, autotexts = axes[0].pie(
        subject_counts.values,
        labels=subject_counts.index,
        autopct='%1.1f%%',
        colors=colors[:len(subject_counts)],
        startangle=140,
        textprops={'color': 'white', 'fontsize': 10}
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(9)
    axes[0].set_facecolor('#0f0f1a')
    axes[0].set_title('Subject Distribution', color='white', fontsize=13, pad=15)

    # Bar Chart - word counts per subject
    topic_wc = df.groupby('subject')['text'].apply(lambda x: x.str.split().apply(len).mean())
    axes[1].bar(topic_wc.index, topic_wc.values, color=colors[:len(topic_wc)], edgecolor='none')
    axes[1].set_facecolor('#0f0f1a')
    axes[1].set_title('Avg. Words per Subject', color='white', fontsize=13, pad=15)
    axes[1].tick_params(colors='white')
    axes[1].spines['bottom'].set_color('#333')
    axes[1].spines['left'].set_color('#333')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].set_ylabel('Avg Words', color='white')
    for label in axes[1].get_xticklabels():
        label.set_rotation(15)
        label.set_color('white')
    for label in axes[1].get_yticklabels():
        label.set_color('white')

    plt.tight_layout(pad=2)

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0f0f1a')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_base64
