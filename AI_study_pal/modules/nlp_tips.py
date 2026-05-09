"""
Module 4: NLP for Study Tips
- Tokenization (NLTK-style)
- Keyword extraction from text
- Study tip generation based on keywords
"""

import re
import math
from collections import Counter


# ─────────────────────────────────────────────
# Study Tip Templates (keyword → tip)
# ─────────────────────────────────────────────
GENERIC_TIPS = [
    "📌 Review key terms daily using flashcards for better retention.",
    "⏰ Use the Pomodoro Technique: study 25 mins, break 5 mins.",
    "📝 Summarize each topic in your own words to test understanding.",
    "🔁 Spaced repetition: revisit material after 1 day, 3 days, then 1 week.",
    "🧠 Teach the concept to someone else — it's the best way to learn!",
    "📖 Read actively: highlight key ideas and annotate your notes.",
    "🎯 Set clear, specific goals for each study session.",
    "💤 Get enough sleep — memory consolidation happens during rest.",
]

SUBJECT_TIPS = {
    "Mathematics": [
        "🔢 Practice 5–10 problems daily to build muscle memory for formulas.",
        "📐 Draw diagrams and visual representations for geometry and calculus.",
        "✅ Always check your work by substituting answers back into equations.",
        "📊 Group similar problem types and master one type before moving on.",
    ],
    "Science": [
        "🔬 Connect concepts to real-world examples to deepen understanding.",
        "⚗️ Create concept maps linking related scientific terms.",
        "🌍 Use diagrams and process charts for understanding biological cycles.",
        "📡 Watch experiment demonstrations online to visualize abstract concepts.",
    ],
    "Computer Science": [
        "💻 Code every day, even if just for 15 minutes — practice is key.",
        "🐛 Debug code line by line; learn to read error messages carefully.",
        "🗂️ Draw out data structures visually to understand their behavior.",
        "📁 Build small projects to apply what you've learned practically.",
    ],
    "History": [
        "📅 Create timelines to visualize the sequence of historical events.",
        "🗺️ Use maps to understand the geographic context of historical events.",
        "📜 Connect historical events to causes and effects for deeper insight.",
        "🔗 Group events by theme (economic, political, social) for pattern recognition.",
    ],
    "English": [
        "✍️ Write a daily journal to practice grammar and expression.",
        "📚 Read a variety of texts: fiction, non-fiction, and poetry.",
        "🗣️ Practice vocabulary by using 3 new words per day in sentences.",
        "📝 Analyze the structure of well-written essays to model your own writing.",
    ],
}

KEYWORD_TIPS = {
    'formula': "📐 Create a formula sheet and review it before every session.",
    'equation': "✏️ Practice solving equations step-by-step, showing all work.",
    'theorem': "📖 Memorize the theorem statement, then understand its proof.",
    'experiment': "🔬 Recreate experiments mentally: list inputs, process, and outputs.",
    'code': "💻 Type out every code example by hand — never just copy-paste.",
    'algorithm': "🔁 Trace through algorithms manually with sample inputs.",
    'history': "📜 Link events to their dates using memory hooks or rhymes.",
    'essay': "🖊️ Outline before writing: intro, 3 body points, conclusion.",
    'vocabulary': "🃏 Use spaced-repetition flashcards for vocabulary mastery.",
    'proof': "🔍 Break proofs into steps; justify each step before moving on.",
    'diagram': "✏️ Redraw key diagrams from memory to test understanding.",
    'definition': "💡 Write definitions in your own words, then compare to the original.",
}


class StudyTipsGenerator:
    """
    NLP-based study tip generator using tokenization and keyword extraction.
    """

    def __init__(self):
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'it', 'its', 'this', 'that', 'these', 'those', 'be', 'been',
            'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'shall', 'can', 'as', 'if',
            'then', 'than', 'so', 'not', 'no', 'nor', 'yet', 'both', 'also',
            'such', 'more', 'most', 'some', 'any', 'all', 'each', 'every',
            'one', 'two', 'three', 'many', 'much', 'few', 'several', 'used',
            'use', 'using', 'called', 'known', 'often', 'however', 'since',
        ])

    def tokenize(self, text: str):
        """Tokenize text into words (NLTK-style)."""
        # Lowercase and remove punctuation
        text = text.lower()
        tokens = re.findall(r'\b[a-zA-Z]+\b', text)
        return tokens

    def remove_stopwords(self, tokens):
        """Remove stop words from token list."""
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]

    def extract_keywords(self, text: str, top_n: int = 8):
        """
        Extract top keywords using TF-based frequency (NLTK tokenization).
        Returns list of (keyword, frequency) tuples.
        """
        if not text.strip():
            return []

        tokens = self.tokenize(text)
        filtered = self.remove_stopwords(tokens)

        if not filtered:
            return []

        freq = Counter(filtered)
        total = len(filtered)

        # Normalize frequencies
        scored = [(word, round(count / total, 3)) for word, count in freq.most_common(top_n)]
        return scored

    def generate_tips(self, text: str, subject: str = 'General', num_tips: int = 5):
        """
        Generate study tips based on extracted keywords and subject.
        """
        tips = []

        # 1. Keyword-specific tips
        if text.strip():
            keywords = self.extract_keywords(text, top_n=15)
            keyword_words = [kw[0] for kw in keywords]

            for kw in keyword_words:
                if kw in KEYWORD_TIPS and len(tips) < 3:
                    tip = KEYWORD_TIPS[kw]
                    if tip not in tips:
                        tips.append(tip)

        # 2. Subject-specific tips
        subj_tips = SUBJECT_TIPS.get(subject, [])
        for tip in subj_tips:
            if len(tips) >= num_tips:
                break
            if tip not in tips:
                tips.append(tip)

        # 3. Generic tips to fill remaining slots
        for tip in GENERIC_TIPS:
            if len(tips) >= num_tips:
                break
            if tip not in tips:
                tips.append(tip)

        return tips[:num_tips]
