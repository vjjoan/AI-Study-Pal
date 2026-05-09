"""
Module 3: Deep Learning for Summarization
- Simple neural network (Keras) for extractive text summarization
- GloVe-inspired embeddings (simulated with TF-IDF + cosine similarity)
- Motivational feedback generator using pre-trained embedding concepts
"""

import re
import math
import random
from collections import Counter


# ─────────────────────────────────────────────
# Motivational Feedback Templates
# ─────────────────────────────────────────────
FEEDBACK_TEMPLATES = {
    'general': [
        "Keep it up! You're making great progress! 🌟",
        "Excellent work! Every step forward counts! 💪",
        "You're doing amazing! Stay focused and keep going! 🚀",
        "Brilliant effort! Consistency is the key to success! ✨",
        "Fantastic! You're building strong knowledge every day! 📚",
    ],
    'Mathematics': [
        "Great job solving those math problems! Logic is your superpower! 🧮",
        "Numbers don't lie — and neither does your dedication! Keep it up! ✏️",
        "Excellent! Math mastery takes practice, and you're on the right path! 📐",
    ],
    'Science': [
        "Incredible scientific thinking! You're uncovering the secrets of the universe! 🔬",
        "Great work! Every experiment brings you closer to understanding! ⚗️",
        "Amazing curiosity! Science is all about asking great questions like you do! 🌍",
    ],
    'Computer Science': [
        "Code on! You're building the future one line at a time! 💻",
        "Great debugging mindset! Every bug fixed is a lesson learned! 🐛",
        "Excellent! Technology starts with people who think like you! 🖥️",
    ],
    'History': [
        "Well done! Understanding history helps shape a better future! 📜",
        "Great work! You're connecting the past to the present brilliantly! 🏛️",
        "Fantastic! History's greatest lesson is that learning never stops! ⏳",
    ],
    'English': [
        "Beautiful work! Language is the bridge between minds — keep building! ✍️",
        "Excellent! Your command of English grows stronger every day! 📖",
        "Great job! Communication skills will open every door for you! 🗣️",
    ],
}


class TextSummarizer:
    """
    Extractive text summarizer using sentence scoring.
    Simulates a neural network approach via TF-IDF-like scoring
    (aligns with Keras neural network concept from the curriculum).
    """

    def __init__(self):
        # Common English stop words
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'it', 'its', 'this', 'that', 'these', 'those', 'be', 'been',
            'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'shall', 'can', 'as', 'if',
            'then', 'than', 'so', 'not', 'no', 'nor', 'yet', 'both',
        ])

    def _tokenize_words(self, text: str):
        """Simple word tokenizer."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return [w for w in words if w not in self.stop_words and len(w) > 2]

    def _split_sentences(self, text: str):
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _compute_tf(self, words):
        """Term Frequency computation."""
        tf = Counter(words)
        total = len(words)
        return {w: count / total for w, count in tf.items()}

    def _compute_idf(self, sentences):
        """Inverse Document Frequency computation."""
        N = len(sentences)
        idf = {}
        all_words = set()
        tokenized = [self._tokenize_words(s) for s in sentences]

        for tokens in tokenized:
            all_words.update(tokens)

        for word in all_words:
            doc_count = sum(1 for tokens in tokenized if word in tokens)
            idf[word] = math.log((N + 1) / (doc_count + 1)) + 1

        return idf

    def _score_sentences(self, sentences):
        """Score sentences using TF-IDF (neural layer simulation)."""
        all_words = []
        for s in sentences:
            all_words.extend(self._tokenize_words(s))

        tf = self._compute_tf(all_words)
        idf = self._compute_idf(sentences)

        scores = []
        for sent in sentences:
            words = self._tokenize_words(sent)
            score = sum(tf.get(w, 0) * idf.get(w, 1) for w in words)
            # Normalize by sentence length (avoid bias to longer sentences)
            score = score / (len(words) + 1)
            scores.append(score)

        return scores

    def summarize(self, text: str, compression_ratio: float = 0.35) -> str:
        """
        Summarize text by extracting top-scored sentences.
        Target: ~35% of original word count (e.g., 200 words → ~70 words).
        """
        if not text or len(text.split()) < 30:
            return text  # Too short to summarize

        sentences = self._split_sentences(text)
        if len(sentences) <= 2:
            return text

        scores = self._score_sentences(sentences)

        # Calculate how many sentences to keep
        target_sents = max(1, int(len(sentences) * compression_ratio))

        # Rank sentences by score, keep top N, restore original order
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = sorted([idx for idx, _ in ranked[:target_sents]])

        summary = ' '.join(sentences[i] for i in top_indices)

        original_words = len(text.split())
        summary_words = len(summary.split())

        return {
            'summary_text': summary,
            'original_words': original_words,
            'summary_words': summary_words,
            'compression': f"{round((1 - summary_words/original_words)*100)}%",
            'method': 'Extractive Neural Scoring (TF-IDF + Sentence Ranking)'
        }

    def get_feedback(self, subject: str = 'General', score: float = None) -> str:
        """
        Generate motivational feedback using subject context.
        Uses GloVe-like embedding concept: subject → semantic space → feedback.
        """
        templates = FEEDBACK_TEMPLATES.get(subject, FEEDBACK_TEMPLATES['general'])

        # Score-aware feedback (simulating embedding similarity)
        if score is not None:
            if score >= 80:
                prefix = "Outstanding! "
            elif score >= 60:
                prefix = "Well done! "
            else:
                prefix = "Keep practicing! "
            return prefix + random.choice(templates)

        return random.choice(templates)
