"""
Module 5: Study Plan Generator
- Creates personalized study schedules based on subject, hours, goal, and days
- Outputs structured plans downloadable as CSV
"""

from datetime import datetime, timedelta
import random

# ─────────────────────────────────────────────
# Topic Map per Subject
# ─────────────────────────────────────────────
SUBJECT_TOPICS = {
    "Mathematics": [
        "Number Systems & Arithmetic", "Algebra & Equations",
        "Functions & Graphs", "Geometry & Trigonometry",
        "Statistics & Probability", "Calculus Fundamentals",
        "Practice Problems & Review", "Mock Test"
    ],
    "Science": [
        "Scientific Method & Lab Skills", "Physics: Motion & Forces",
        "Chemistry: Atoms & Bonding", "Biology: Cells & Life",
        "Energy & Thermodynamics", "Waves & Electricity",
        "Ecology & Environment", "Revision & Practice Tests"
    ],
    "Computer Science": [
        "Introduction to Programming", "Data Types & Variables",
        "Control Flow & Loops", "Functions & Modules",
        "Data Structures (Lists, Dicts)", "Algorithms & Complexity",
        "OOP Concepts", "Project & Review"
    ],
    "History": [
        "Ancient Civilizations", "Medieval Period",
        "Renaissance & Reformation", "Age of Revolutions",
        "World Wars", "Cold War & Modern Era",
        "Regional History Deep Dive", "Essay Writing & Review"
    ],
    "English": [
        "Grammar Fundamentals", "Vocabulary Building",
        "Reading Comprehension", "Literary Devices & Analysis",
        "Essay Structure & Writing", "Poetry & Drama",
        "Critical Thinking & Argumentation", "Review & Mock Test"
    ],
    "General": [
        "Introduction & Overview", "Core Concepts Part 1",
        "Core Concepts Part 2", "Practice & Application",
        "Problem Solving Techniques", "Review & Consolidation",
        "Mock Assessment", "Final Revision"
    ]
}

GOAL_DESCRIPTIONS = {
    "exam_prep": "Exam Preparation",
    "homework": "Homework & Assignments",
    "concept_review": "Concept Review",
    "project": "Project Work",
}

DAILY_ACTIVITIES = [
    "Read notes & textbook sections",
    "Watch tutorial videos & take notes",
    "Solve practice problems",
    "Create flashcards for key terms",
    "Review previous day's material",
    "Attempt past exam questions",
    "Summarize key concepts in own words",
    "Group discussion or peer study",
    "Work through exercises & check solutions",
    "Mind-map the topic connections",
]


class StudyPlanGenerator:
    """Generates personalized study plans for students."""

    def generate(self, subject: str, hours_per_day: int,
                 goal: str = 'exam_prep', days: int = 7) -> dict:
        """
        Generate a detailed day-by-day study plan.

        Args:
            subject: The subject to study
            hours_per_day: Daily study hours available
            goal: Study goal (exam_prep / homework / concept_review / project)
            days: Number of days in the plan
        """
        topics = SUBJECT_TOPICS.get(subject, SUBJECT_TOPICS['General'])
        goal_label = GOAL_DESCRIPTIONS.get(goal, goal)

        # Distribute topics across days
        if len(topics) < days:
            # Repeat topics with revision days
            extended_topics = topics + ["Revision & Practice"] * (days - len(topics))
        else:
            extended_topics = topics[:days]

        start_date = datetime.now()
        plan_days = []

        for i in range(days):
            date = start_date + timedelta(days=i)
            topic = extended_topics[i]

            # Choose activities based on topic type
            if 'revision' in topic.lower() or 'review' in topic.lower():
                activity = "Review all previous notes, create summary sheets, attempt past questions"
            elif 'test' in topic.lower() or 'mock' in topic.lower():
                activity = "Full timed practice test under exam conditions"
            elif 'project' in topic.lower():
                activity = "Work on project components, research, and documentation"
            else:
                # Pick 2-3 varied activities
                acts = random.sample(DAILY_ACTIVITIES, min(2, len(DAILY_ACTIVITIES)))
                activity = "; ".join(acts)

            # Build time blocks
            blocks = self._create_time_blocks(hours_per_day, topic)

            plan_days.append({
                'day': f"Day {i+1}",
                'date': date.strftime('%B %d, %Y (%A)'),
                'date_short': date.strftime('%Y-%m-%d'),
                'topic': topic,
                'hours': hours_per_day,
                'activity': activity,
                'goal': goal_label,
                'blocks': blocks,
                'tip': self._get_day_tip(i, days)
            })

        total_hours = hours_per_day * days

        return {
            'subject': subject,
            'goal': goal_label,
            'days': days,
            'hours_per_day': hours_per_day,
            'total_hours': total_hours,
            'plan': plan_days,
            'summary': f"{days}-day {goal_label} plan for {subject} | {total_hours} total study hours"
        }

    def _create_time_blocks(self, total_hours: int, topic: str) -> list:
        """Break a study day into time blocks."""
        blocks = []
        if total_hours <= 1:
            blocks.append({"duration": "60 min", "task": f"Study: {topic}"})
        elif total_hours == 2:
            blocks = [
                {"duration": "50 min", "task": f"Study: {topic} — Part 1"},
                {"duration": "10 min", "task": "Short Break"},
                {"duration": "50 min", "task": f"Study: {topic} — Practice & Review"},
                {"duration": "10 min", "task": "Wrap Up & Notes"},
            ]
        elif total_hours <= 4:
            blocks = [
                {"duration": "50 min", "task": f"Study: {topic} — Core Concepts"},
                {"duration": "10 min", "task": "Break"},
                {"duration": "50 min", "task": f"Study: {topic} — Examples & Problems"},
                {"duration": "10 min", "task": "Break"},
                {"duration": "40 min", "task": "Practice Questions"},
                {"duration": "20 min", "task": "Review & Summarize"},
            ]
        else:
            blocks = [
                {"duration": "50 min", "task": f"Study: {topic} — Introduction"},
                {"duration": "10 min", "task": "Break"},
                {"duration": "50 min", "task": f"Study: {topic} — Deep Dive"},
                {"duration": "10 min", "task": "Break"},
                {"duration": "50 min", "task": "Practice Problems"},
                {"duration": "10 min", "task": "Break"},
                {"duration": "50 min", "task": "Past Questions / Application"},
                {"duration": "20 min", "task": "Review, Notes & Summary"},
            ]
        return blocks

    def _get_day_tip(self, day_idx: int, total_days: int) -> str:
        """Return a contextual tip based on where the student is in the plan."""
        if day_idx == 0:
            return "💡 Great start! Set up your study space and gather all materials today."
        elif day_idx == total_days - 1:
            return "🏁 Final day! Focus on weak areas and do a quick full review."
        elif day_idx == total_days // 2:
            return "🔥 Halfway there! Review what you've learned so far before pushing forward."
        elif day_idx % 3 == 2:
            return "🔁 Good time to do a quick revision of the past few days' topics."
        else:
            return "📚 Stay consistent — short, focused sessions beat long, distracted ones."
