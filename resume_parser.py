import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResumeParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.skills_db = ['Python', 'Java', 'SQL', 'Machine Learning', 'Data Analysis', 'Deep Learning',
                          'NLP', 'Tableau', 'Power BI', 'C++', 'AWS', 'Azure', 'TensorFlow', 'PyTorch']

    def extract_entities(self, text):
        doc = self.nlp(text)

        entities = {
            'name': None,
            'email': None,
            'phone': None,
            'skills': [],
            'education': []
        }

        # Extract name
        for ent in doc.ents:
            if ent.label_ == 'PERSON' and not entities['name']:
                entities['name'] = ent.text
                break

        # Extract email
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, text)
        entities['email'] = emails[0] if emails else None

        # Extract phone number
        phone_pattern = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        phones = re.findall(phone_pattern, text)
        entities['phone'] = phones[0] if phones else None

        # Extract skills
        lower_text = text.lower()
        for skill in self.skills_db:
            if skill.lower() in lower_text:
                entities['skills'].append(skill)

        # Extract education
        education_keywords = ['Bachelor', 'Master', 'B.Sc', 'M.Sc', 'B.Tech', 'M.Tech', 'PhD']
        for keyword in education_keywords:
            if keyword.lower() in lower_text:
                entities['education'].append(keyword)

        return entities

    def score_resume(self, entities, job_description):
        resume_text = ' '.join(entities['skills']) or ''
        corpus = [resume_text, job_description]

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

        feedback = f"TF-IDF similarity score: {round(similarity, 2)}%."
        if similarity < 50:
            missing = set(job_description.lower().split()) - set(resume_text.lower().split())
            feedback += f" Add or highlight relevant keywords like: {', '.join(list(missing)[:5])}"

        return {"score": round(similarity, 2), "feedback": feedback}
