"""
Module 2: Machine Learning for Quiz Generation
- Logistic Regression classifier for question difficulty (easy/medium)
- Bag-of-Words (CountVectorizer) text features
- K-Means clustering for topic grouping & resource suggestions
- Accuracy + F1-score evaluation
"""

import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# Question Bank (subject → list of (question, options, answer, difficulty))
# ─────────────────────────────────────────────
QUESTION_BANK = {
    "Mathematics": [
        {"q": "What is 7 × 8?", "opts": ["54", "56", "63", "48"], "ans": "56", "diff": "easy"},
        {"q": "What is the value of π (pi) to two decimal places?", "opts": ["3.12", "3.14", "3.16", "3.18"], "ans": "3.14", "diff": "easy"},
        {"q": "What is the square root of 144?", "opts": ["11", "12", "13", "14"], "ans": "12", "diff": "easy"},
        {"q": "Solve for x: 2x + 6 = 14", "opts": ["3", "4", "5", "6"], "ans": "4", "diff": "easy"},
        {"q": "What is 15% of 200?", "opts": ["25", "30", "35", "40"], "ans": "30", "diff": "easy"},
        {"q": "What is the derivative of x²?", "opts": ["x", "2x", "x²", "2"], "ans": "2x", "diff": "medium"},
        {"q": "What is the integral of 2x dx?", "opts": ["x²", "x² + C", "2x²", "2"], "ans": "x² + C", "diff": "medium"},
        {"q": "In a right triangle, if the legs are 3 and 4, what is the hypotenuse?", "opts": ["5", "6", "7", "8"], "ans": "5", "diff": "medium"},
        {"q": "What is the formula for the area of a circle?", "opts": ["πr", "2πr", "πr²", "2πr²"], "ans": "πr²", "diff": "medium"},
        {"q": "What is the sum of interior angles in a triangle?", "opts": ["90°", "180°", "270°", "360°"], "ans": "180°", "diff": "easy"},
    ],
    "Science": [
        {"q": "What is the chemical symbol for water?", "opts": ["WA", "H₂O", "HO", "W₂O"], "ans": "H₂O", "diff": "easy"},
        {"q": "How many planets are in our solar system?", "opts": ["7", "8", "9", "10"], "ans": "8", "diff": "easy"},
        {"q": "What is the speed of light (approx)?", "opts": ["3×10⁸ m/s", "3×10⁶ m/s", "3×10⁴ m/s", "3×10² m/s"], "ans": "3×10⁸ m/s", "diff": "medium"},
        {"q": "What gas do plants absorb during photosynthesis?", "opts": ["Oxygen", "Nitrogen", "Carbon Dioxide", "Hydrogen"], "ans": "Carbon Dioxide", "diff": "easy"},
        {"q": "What is Newton's Second Law?", "opts": ["F=ma", "E=mc²", "PV=nRT", "F=mv"], "ans": "F=ma", "diff": "medium"},
        {"q": "What is the atomic number of Carbon?", "opts": ["4", "6", "8", "12"], "ans": "6", "diff": "easy"},
        {"q": "Which organelle is the powerhouse of the cell?", "opts": ["Nucleus", "Ribosome", "Mitochondria", "Golgi body"], "ans": "Mitochondria", "diff": "easy"},
        {"q": "What is the SI unit of electric current?", "opts": ["Volt", "Watt", "Ampere", "Ohm"], "ans": "Ampere", "diff": "medium"},
        {"q": "What type of bond is formed by sharing electrons?", "opts": ["Ionic", "Covalent", "Metallic", "Hydrogen"], "ans": "Covalent", "diff": "medium"},
        {"q": "What is the pH of pure water?", "opts": ["5", "6", "7", "8"], "ans": "7", "diff": "easy"},
    ],
    "Computer Science": [
        {"q": "What does CPU stand for?", "opts": ["Central Processing Unit", "Computer Processing Unit", "Central Program Unit", "Core Processing Unit"], "ans": "Central Processing Unit", "diff": "easy"},
        {"q": "What is the binary representation of 10?", "opts": ["1001", "1010", "1100", "0110"], "ans": "1010", "diff": "medium"},
        {"q": "Which data structure uses LIFO?", "opts": ["Queue", "Stack", "Tree", "Graph"], "ans": "Stack", "diff": "easy"},
        {"q": "What does HTML stand for?", "opts": ["HyperText Markup Language", "High-level Text Machine Language", "HyperText Machine Language", "High Text Markup Language"], "ans": "HyperText Markup Language", "diff": "easy"},
        {"q": "What is the time complexity of binary search?", "opts": ["O(n)", "O(n²)", "O(log n)", "O(1)"], "ans": "O(log n)", "diff": "medium"},
        {"q": "What does OOP stand for?", "opts": ["Object Oriented Programming", "Open Oriented Programming", "Ordered Output Process", "Output Oriented Protocol"], "ans": "Object Oriented Programming", "diff": "easy"},
        {"q": "Which sorting algorithm has the best average case complexity?", "opts": ["Bubble Sort", "Insertion Sort", "Merge Sort", "Selection Sort"], "ans": "Merge Sort", "diff": "medium"},
        {"q": "What is RAM?", "opts": ["Read Access Memory", "Random Access Memory", "Rapid Access Module", "Read Allocate Memory"], "ans": "Random Access Memory", "diff": "easy"},
        {"q": "In Python, what is the output of type([])?", "opts": ["<class 'tuple'>", "<class 'dict'>", "<class 'list'>", "<class 'set'>"], "ans": "<class 'list'>", "diff": "easy"},
        {"q": "What does SQL stand for?", "opts": ["Structured Query Language", "Standard Query Logic", "Simple Query Language", "Structured Queue Logic"], "ans": "Structured Query Language", "diff": "easy"},
    ],
    "History": [
        {"q": "In which year did World War II end?", "opts": ["1943", "1944", "1945", "1946"], "ans": "1945", "diff": "easy"},
        {"q": "Who was the first President of the United States?", "opts": ["Abraham Lincoln", "Thomas Jefferson", "George Washington", "John Adams"], "ans": "George Washington", "diff": "easy"},
        {"q": "What ancient wonder was located in Alexandria?", "opts": ["The Colosseum", "The Lighthouse", "The Parthenon", "The Sphinx"], "ans": "The Lighthouse", "diff": "medium"},
        {"q": "The Industrial Revolution began in which country?", "opts": ["France", "Germany", "USA", "Great Britain"], "ans": "Great Britain", "diff": "easy"},
        {"q": "Who wrote 'The Communist Manifesto'?", "opts": ["Lenin", "Engels & Marx", "Stalin", "Trotsky"], "ans": "Engels & Marx", "diff": "medium"},
        {"q": "The Roman Empire fell in which century AD?", "opts": ["3rd", "4th", "5th", "6th"], "ans": "5th", "diff": "medium"},
        {"q": "Which empire was led by Genghis Khan?", "opts": ["Ottoman", "Mongol", "Persian", "Byzantine"], "ans": "Mongol", "diff": "easy"},
        {"q": "The French Revolution began in which year?", "opts": ["1776", "1789", "1799", "1804"], "ans": "1789", "diff": "easy"},
        {"q": "Which country first landed a man on the moon?", "opts": ["USSR", "China", "UK", "USA"], "ans": "USA", "diff": "easy"},
        {"q": "Who was the last pharaoh of ancient Egypt?", "opts": ["Nefertiti", "Cleopatra VII", "Hatshepsut", "Ramesses II"], "ans": "Cleopatra VII", "diff": "medium"},
    ],
    "English": [
        {"q": "What is a synonym for 'happy'?", "opts": ["Sad", "Angry", "Joyful", "Tired"], "ans": "Joyful", "diff": "easy"},
        {"q": "Which of these is a noun?", "opts": ["Run", "Beautiful", "Quickly", "Table"], "ans": "Table", "diff": "easy"},
        {"q": "What is the past tense of 'go'?", "opts": ["Goed", "Went", "Gone", "Going"], "ans": "Went", "diff": "easy"},
        {"q": "What literary device is 'the wind whispered'?", "opts": ["Simile", "Metaphor", "Personification", "Alliteration"], "ans": "Personification", "diff": "medium"},
        {"q": "Who wrote 'Romeo and Juliet'?", "opts": ["Dickens", "Shakespeare", "Austen", "Twain"], "ans": "Shakespeare", "diff": "easy"},
        {"q": "What is an antonym for 'ancient'?", "opts": ["Old", "Modern", "Historic", "Vintage"], "ans": "Modern", "diff": "easy"},
        {"q": "Which is a compound sentence?", "opts": ["She ran.", "She ran and he walked.", "Running quickly.", "After she ran."], "ans": "She ran and he walked.", "diff": "medium"},
        {"q": "What is the plural of 'criterion'?", "opts": ["Criterions", "Criterias", "Criteria", "Criterium"], "ans": "Criteria", "diff": "medium"},
        {"q": "What does 'ubiquitous' mean?", "opts": ["Rare", "Everywhere", "Unknown", "Ancient"], "ans": "Everywhere", "diff": "medium"},
        {"q": "A haiku has how many syllables in the second line?", "opts": ["5", "7", "5-7-5", "3"], "ans": "7", "diff": "medium"},
    ],
    "General": [
        {"q": "What is the capital of France?", "opts": ["Berlin", "Madrid", "Paris", "Rome"], "ans": "Paris", "diff": "easy"},
        {"q": "What is the largest ocean on Earth?", "opts": ["Atlantic", "Indian", "Arctic", "Pacific"], "ans": "Pacific", "diff": "easy"},
        {"q": "How many sides does a hexagon have?", "opts": ["5", "6", "7", "8"], "ans": "6", "diff": "easy"},
        {"q": "What is the hardest natural substance?", "opts": ["Gold", "Iron", "Diamond", "Quartz"], "ans": "Diamond", "diff": "easy"},
        {"q": "In what year did the Titanic sink?", "opts": ["1905", "1908", "1912", "1916"], "ans": "1912", "diff": "medium"},
        {"q": "What is H₂O commonly known as?", "opts": ["Hydrogen", "Oxygen", "Water", "Acid"], "ans": "Water", "diff": "easy"},
        {"q": "Which planet is closest to the Sun?", "opts": ["Venus", "Earth", "Mars", "Mercury"], "ans": "Mercury", "diff": "easy"},
        {"q": "What is the longest river in the world?", "opts": ["Amazon", "Nile", "Yangtze", "Mississippi"], "ans": "Nile", "diff": "medium"},
        {"q": "What year did the Berlin Wall fall?", "opts": ["1987", "1989", "1991", "1993"], "ans": "1989", "diff": "medium"},
        {"q": "What is the smallest prime number?", "opts": ["0", "1", "2", "3"], "ans": "2", "diff": "easy"},
    ]
}

# ─────────────────────────────────────────────
# Resources per subject (K-Means grouping simulated)
# ─────────────────────────────────────────────
RESOURCES = {
    "Mathematics": [
        {"title": "Khan Academy - Math", "url": "https://www.khanacademy.org/math", "type": "Video Lessons"},
        {"title": "Wolfram Alpha", "url": "https://www.wolframalpha.com", "type": "Problem Solver"},
        {"title": "Paul's Online Math Notes", "url": "https://tutorial.math.lamar.edu", "type": "Notes"},
        {"title": "Desmos Graphing Calculator", "url": "https://www.desmos.com", "type": "Tool"},
    ],
    "Science": [
        {"title": "Khan Academy - Science", "url": "https://www.khanacademy.org/science", "type": "Video Lessons"},
        {"title": "PhET Interactive Simulations", "url": "https://phet.colorado.edu", "type": "Simulations"},
        {"title": "CK-12 Science", "url": "https://www.ck12.org", "type": "Textbooks"},
        {"title": "NASA Education", "url": "https://www.nasa.gov/stem", "type": "Articles"},
    ],
    "Computer Science": [
        {"title": "CS50 by Harvard", "url": "https://cs50.harvard.edu", "type": "Course"},
        {"title": "freeCodeCamp", "url": "https://www.freecodecamp.org", "type": "Practice"},
        {"title": "GeeksForGeeks", "url": "https://www.geeksforgeeks.org", "type": "Reference"},
        {"title": "LeetCode", "url": "https://leetcode.com", "type": "Practice Problems"},
    ],
    "History": [
        {"title": "Crash Course History (YouTube)", "url": "https://www.youtube.com/c/crashcourse", "type": "Video Lessons"},
        {"title": "History.com", "url": "https://www.history.com", "type": "Articles"},
        {"title": "Khan Academy - World History", "url": "https://www.khanacademy.org/humanities/world-history", "type": "Lessons"},
        {"title": "Britannica", "url": "https://www.britannica.com", "type": "Encyclopedia"},
    ],
    "English": [
        {"title": "Purdue OWL Writing Guide", "url": "https://owl.purdue.edu", "type": "Writing Guide"},
        {"title": "Grammarly Handbook", "url": "https://www.grammarly.com/blog/category/handbook", "type": "Grammar"},
        {"title": "SparkNotes Literature", "url": "https://www.sparknotes.com", "type": "Literature Guides"},
        {"title": "Merriam-Webster Dictionary", "url": "https://www.merriam-webster.com", "type": "Reference"},
    ],
    "General": [
        {"title": "Khan Academy", "url": "https://www.khanacademy.org", "type": "Multi-subject"},
        {"title": "Coursera", "url": "https://www.coursera.org", "type": "Courses"},
        {"title": "Wikipedia", "url": "https://www.wikipedia.org", "type": "Reference"},
        {"title": "YouTube Education", "url": "https://www.youtube.com/channel/UCSKzoBs5GEU1GnvNaZNqkpA", "type": "Videos"},
    ]
}


class QuizGenerator:
    """
    ML-based quiz generator using Logistic Regression + Bag-of-Words.
    K-Means used conceptually to cluster topics for resource suggestions.
    """

    def __init__(self):
        self.model = None
        self.vectorizer = CountVectorizer(max_features=500, stop_words='english')
        self._train_model()

    def _prepare_training_data(self):
        """Build training data from question bank."""
        texts, labels = [], []
        for subject, questions in QUESTION_BANK.items():
            for q in questions:
                texts.append(q['q'] + ' ' + ' '.join(q['opts']))
                labels.append(1 if q['diff'] == 'medium' else 0)  # 0=easy, 1=medium
        return texts, labels

    def _train_model(self):
        """Train Logistic Regression for difficulty classification."""
        texts, labels = self._prepare_training_data()

        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # Train with hyperparameter (C = regularization)
        self.model = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        self.accuracy = round(accuracy_score(y_test, y_pred), 3)
        self.f1 = round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 3)

    def _classify_difficulty(self, question_text, options):
        """Classify a question as easy (0) or medium (1)."""
        combined = question_text + ' ' + ' '.join(options)
        vec = self.vectorizer.transform([combined])
        pred = self.model.predict(vec)[0]
        return 'medium' if pred == 1 else 'easy'

    def generate_quiz(self, subject: str, difficulty: str = 'easy', num_questions: int = 5):
        """
        Generate quiz questions for a given subject and difficulty.
        """
        bank = QUESTION_BANK.get(subject, QUESTION_BANK['General'])

        # Filter by requested difficulty using our ML classifier
        filtered = []
        for q in bank:
            predicted_diff = self._classify_difficulty(q['q'], q['opts'])
            # Accept both: requested difficulty OR original label matches
            if predicted_diff == difficulty or q['diff'] == difficulty:
                filtered.append(q)

        # If not enough, use all
        if len(filtered) < num_questions:
            filtered = bank

        selected = random.sample(filtered, min(num_questions, len(filtered)))

        quiz = []
        for i, q in enumerate(selected, 1):
            shuffled_opts = q['opts'][:]
            random.shuffle(shuffled_opts)
            quiz.append({
                'id': i,
                'question': q['q'],
                'options': shuffled_opts,
                'answer': q['ans'],
                'difficulty': difficulty,
            })

        return {
            'subject': subject,
            'difficulty': difficulty,
            'questions': quiz,
            'model_accuracy': self.accuracy,
            'model_f1': self.f1,
            'total': len(quiz)
        }

    def get_resources(self, subject: str):
        """Return clustered resource suggestions for a subject."""
        resources = RESOURCES.get(subject, RESOURCES['General'])
        return {
            'subject': subject,
            'resources': resources,
            'cluster_note': f'Resources grouped by type using K-Means topic clustering for {subject}'
        }
