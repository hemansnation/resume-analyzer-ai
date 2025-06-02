import spacy
import re
# ADDed THESE new IMPORTS 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResumeParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.skills_db = ['Python','Java', 'SQL', 'Machine Learning', 'Data Analysis']
    
    def extract_entities(self, text):
        doc = self.nlp(text)

        entities = {
            'name': None,
            'email': None,
            'phone': None,
            'skills': [],
            'education': []
        }

        for ent in doc.ents:
            if ent.label_ == 'PERSON' and not entities['name']:
                entities['name'] = ent.text
                break
        
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, text)
        entities['email'] = emails[0] if emails else None

        phone_pattern = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        phones = re.findall(phone_pattern, text)
        entities['phone'] = phones[0] if phones else None

        for skill in self.skills_db:
            if skill.lower() in text.lower():
                entities['skills'].append(skill)
            
        degree_keywords = ['bachelors', 'masters', 'phd', 'degree']

        for sent in doc.sents:
            if any(keyword in sent.text.lower() for keyword in degree_keywords):
                entities['education'].append(sent.text.strip())

        return entities

    # def score_resume(self, entities, job_skills=['Python', 'SQL']):
    #     matched_skills = [skill for skill in entities['skills'] if skill in job_skills]
    #     score = len(matched_skills) / len(job_skills) * 100

    #     feedback = f'Matched {len(matched_skills)} out of {len(job_skills)} required skills.'

    #     if score < 50:
    #         feedback += "Consider adding skills like: " + ', '.join(set(job_skills) - set(entities['skills']))

    #     return {"score": round(score, 2), "feedback": feedback}
    

def score_resume(self, entities, job_description):
    resume_text = ' '.join(entities['skills']) or ''
    corpus = [resume_text, job_description]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

    feedback = f"TF-IDF similarity score: {round(similarity, 2)}%."
    if similarity < 50:
        missing = set(job_description.lower().split()) - set(resume_text.lower().split())
        feedback += f" Add or highlight relevant keywords like: {', '.join(list(missing)[:5])}"

    return {"score": round(similarity, 2), "feedback": feedback}



